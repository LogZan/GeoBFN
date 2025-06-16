import argparse
import copy
import datetime
import os
import pytz
import pickle
import numpy as np
import pandas as pd
import logging as python_logging
from datetime import datetime, timedelta
from typing import Any, Optional, List

import pytorch_lightning as pl
import torch
from absl import logging
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch_geometric.data import Data, DataLoader  # Use torch_geometric DataLoader

# Assume these imports exist and are correct
from core.config.config import Config
from core.model.bfn.bfn_base import bfn4MolEGNN
from core.data.qm9_gen import QM9Gen
from core.data.data_gen_compete_condition import CompeteDataGen
from core.callbacks.basic import (
    NormalizerCallback,
    RecoverCallback,
    EMACallback,
)
from core.evaluation.validation_callback import (
    MolGenValidationCallback,
)

# Set precision
torch.set_float32_matmul_precision("high")


class BFN4MolSampler(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        # Initialize the core dynamics model
        self.dynamics = bfn4MolEGNN(
            in_node_nf=self.cfg.dynamics.in_node_nf,
            hidden_nf=self.cfg.dynamics.hidden_nf,
            n_layers=self.cfg.dynamics.n_layers,
            # condition_time=False,
            sigma1_coord=self.cfg.dynamics.sigma1_coord,
            sigma1_charges=self.cfg.dynamics.sigma1_charges,
            bins=self.cfg.dynamics.bins,
            beta1=self.cfg.dynamics.beta1,
            sample_steps=self.cfg.dynamics.sample_steps,
            no_diff_coord=self.cfg.dynamics.no_diff_coord,
            charge_discretised_loss=self.cfg.dynamics.charge_discretised_loss,
            charge_clamp=self.cfg.dynamics.charge_clamp,
            t_min=self.cfg.dynamics.t_min,
        )
        self.save_hyperparameters(logger=False)
        self.atomic_nb = self.cfg.dataset.atomic_nb
        self.remove_h = self.cfg.dataset.remove_h
        self.atom_type_num = len(self.atomic_nb) - self.remove_h

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        """
        Performs unconditional sampling based on the input batch structure.
        The input batch should define n_nodes, edge_index, and segment_ids.
        """
        edge_index, segment_ids, condition_value = (
            batch.edge_index,  # [2, edge_num]
            batch.batch,  # [n_nodes],
            batch.condition_value,  # [n_nodes, 1] or None
        )
        n_nodes = segment_ids.shape[0]

        logging.info(f"Sampling batch {batch_idx} with {n_nodes} total nodes.")
        # Perform the reverse diffusion process (sampling)
        # The dynamics model takes structural info and generates coordinates and features
        theta_chain = self.dynamics(
            n_nodes=n_nodes,
            edge_index=edge_index,
            segment_ids=segment_ids,
            condition=condition_value,  # [n_nodes, 1] or None
        )

        # Get the final state (t=0)
        x, h = theta_chain[-1] # x: positions [n_nodes, 3], h: features [n_nodes, feat_dim]

        # Decode charges/features into atom types (one-hot)
        atom_type = self.charge_decode(h[:, :1]) # Assuming first feature dim is charge-like

        # Create output Data objects
        out_batch = copy.deepcopy(batch) # Keep structural info (edge_index, batch, ptr)

        out_batch.x, out_batch.pos = atom_type, x
        _slice_dict = {
            "x": out_batch._slice_dict["zx"],
            "pos": out_batch._slice_dict["zpos"],
        }
        _inc_dict = {"x": out_batch._inc_dict["zx"], "pos": out_batch._inc_dict["zpos"]}
        out_batch._inc_dict.update(_inc_dict)
        out_batch._slice_dict.update(_slice_dict)
        out_data_list = out_batch.to_data_list()

        return out_data_list

    def charge_decode(self, charge: torch.Tensor) -> torch.Tensor:
        """
        Decodes continuous charge-like features into one-hot atom types.
        charge: [n_nodes, 1]
        """
        # Ensure atomic_nb is sorted if not already
        sorted_atomic_nb = sorted(self.atomic_nb[self.remove_h:])
        max_atomic_nb = max(sorted_atomic_nb) if sorted_atomic_nb else 1 # Avoid division by zero

        # Calculate anchor points based on sorted atomic numbers
        anchor = torch.tensor(
            [(2 * k - 1) / max_atomic_nb - 1 for k in sorted_atomic_nb],
            dtype=torch.float32,
            device=charge.device,
        )
        # Ensure anchor is broadcastable: [1, num_atom_types]
        anchor = anchor.unsqueeze(0)

        # Find the index of the closest anchor point for each node
        # charge is [n_nodes, 1], anchor is [1, num_atom_types]
        # Resulting abs diff is [n_nodes, num_atom_types]
        atom_type_indices = (charge - anchor).abs().argmin(dim=-1) # Result is [n_nodes]

        # Create one-hot encoding
        one_hot = torch.zeros(
            [charge.shape[0], self.atom_type_num],
            dtype=torch.float32,
            device=charge.device
        )
        one_hot[torch.arange(charge.shape[0]), atom_type_indices] = 1
        return one_hot


if __name__ == "__main__":
    # Record start time
    script_start_time = datetime.now()
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_condition", type=str, default="dataset/competition_round2/input_condition.csv", help="input_condition.csv"
    )
    parser.add_argument(
        "--config_file", type=str, default="logs/zengchuanlong_geobfn/compete_round2/config.yaml", help="Path to the config YAML file used for training."
    )
    parser.add_argument(
        "--num_samples", type=int, default=10000, help="Number of molecules to generate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for generation."
    )
    parser.add_argument("--exp_name", type=str, default="geobfn_sampling", help="Experiment name for logging.")
    parser.add_argument("--logging_level", type=str, default="info", choices=["debug", "info", "warning", "error", "fatal"])
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode (overrides some settings).")

    _args = parser.parse_args()

    # Load config and potentially override with command-line args
    cfg = Config(config_file=_args.config_file) # Load base config first
    cfg.debug = _args.debug

    if cfg.debug:
        cfg.exp_name = "debug_sampling"
        _args.num_samples = 50 # Reduce samples in debug mode
        cfg.evaluation.eval_data_num = 50
        cfg.dynamics.sample_steps = 50 # Faster sampling for debug

    print(f"--- Sampling Configuration ---")
    print(f"Number of samples to generate: {_args.num_samples}")
    print(f"Checkpoint path: {cfg.accounting.checkpoint_path}")
    print(f"-----------------------------")

    logging_level = {
        "info": logging.INFO, "debug": logging.DEBUG, "warning": logging.WARNING,
        "error": logging.ERROR, "fatal": logging.FATAL,
    }
    logging.set_verbosity(logging_level[_args.logging_level])

    # --- Setup Local Logging ---
    os.makedirs("logs", exist_ok=True)
    run_name = cfg.exp_name + f'_sampling_{datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")}'
    log_file_path = f"logs/{run_name}.log"
    
    # Setup Python logging
    python_logging.basicConfig(
        level=python_logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            python_logging.FileHandler(log_file_path),
            python_logging.StreamHandler()
        ]
    )
    python_logging.info(f"Local logging initialized. Log file: {log_file_path}")

    # Read and normalize condition values
    logging.info(f"Reading condition values from {_args.input_condition}")
    condition_df = pd.read_csv(_args.input_condition)
    condition_values = condition_df['Value'].values
    
    # Normalize the condition values to [0, 1]
    min_val = condition_values.min()
    max_val = condition_values.max()
    normalized_values = (condition_values - min_val) / (max_val - min_val)
    logging.info(f"Normalized {len(normalized_values)} condition values. Original range: [{min_val:.4f}, {max_val:.4f}]")

    # --- Data Loader for Sampling Input ---
    logging.info(f"Preparing sampling input loader for {_args.num_samples} molecules...")
    DataGenClass = QM9Gen if cfg.dataset.name == "qm9" else CompeteDataGen if cfg.dataset.name == "compete_condition" else None
    if DataGenClass is None:
        raise NotImplementedError(f"Dataset type '{cfg.dataset.name}' not recognized for sampling.")

    sampling_loader = DataGenClass.initiate_evaluation_dataloader(
        data_num=_args.num_samples,
        n_node_histogram=cfg.dataset.n_node_histogram,
        batch_size=cfg.evaluation.batch_size, # Use eval batch size from config
        num_workers=cfg.dataset.num_workers,
        max_n_nodes=60,
        condition_values=normalized_values,  # Pass normalized condition values
    )
    logging.info(f"Sampling loader created with {len(sampling_loader)} batches.")


    # --- Reference Dataset (for Evaluation Callback) ---
    logging.info("Loading reference dataset for evaluation metrics...")
    ref_DataGen = DataGenClass(
            datadir=cfg.dataset.datadir,
            batch_size=1, # Minimal batch size, just need the .ds object
            n_node_histogram=cfg.dataset.n_node_histogram,
            debug=cfg.debug,
            num_workers=0, # No workers needed just for .ds
            split="train", # Use training split for reference stats
            load_in_memory=False # Avoid loading full dataset if large
        )
    ref_dataset = ref_DataGen.ds
    logging.info(f"Reference dataset '{cfg.dataset.name}' loaded.")


    # --- Model Instantiation ---
    model = BFN4MolSampler(config=cfg)

    # --- Callbacks ---
    callbacks = [
        RecoverCallback(
                latest_ckpt=cfg.accounting.checkpoint_path,
                resume=cfg.optimization.resume or cfg.test,
                recover_trigger_loss=cfg.optimization.recover_trigger_loss,
                skip_count_limit=cfg.optimization.skip_count_limit,
            ),
        NormalizerCallback(normalizer_dict=cfg.dataset.normalizer_dict),
        MolGenValidationCallback(
            dataset=ref_dataset, # Provide reference dataset
            atom_type_one_hot=True, # As used in predict_step output
            single_bond=cfg.evaluation.single_bond,
        ),
        EMACallback(decay=0.9999, ema_device="cuda"),
    ]

    # --- Trainer Setup for Prediction ---
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=False,  # Disable PyTorch Lightning logger
        callbacks=callbacks,
        max_epochs=1, # Needs at least 1 epoch to run predict
        check_val_every_n_epoch=9999, # Disable validation loop
        num_sanity_val_steps=0,      # Disable sanity check
        enable_checkpointing=False, # Disable checkpointing during prediction
        enable_progress_bar=True,
        enable_model_summary=True,
    )

    # --- Run Sampling ---
    logging.info("Starting molecule generation...")
    results = trainer.predict(model, dataloaders=sampling_loader)

    logging.info("Molecule generation finished.")

    generated_molecules: List[Data] = [mol for batch_result in results for mol in batch_result]

    logging.info(f"Successfully generated {len(generated_molecules)} molecules.")

    # --- Post-processing ---

    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create logs directory
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Get timestamp for filename
    timestamp = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    import re
    try:
        epoch = re.search(r'epoch=(\d+)', cfg.accounting.checkpoint_path).group(1)
    except AttributeError:
        epoch = "last"
    output_file = os.path.join(output_dir, f"output_{timestamp}_epochs_{epoch}_ema.pkl")
    
    python_logging.info(f"Output will be saved to: {output_file}")
    
    # Process each molecule
    logging.info("Processing molecules for pickle file...")
    processed_molecules = []
    
    atomic_symbols = cfg.dataset.atom_decoder
    remove_h = cfg.dataset.remove_h  # Check if H is removed from consideration
    
    for mol_idx, mol in enumerate(generated_molecules):
        if mol_idx % 100 == 0:
            logging.info(f"Processing molecule {mol_idx}/{len(generated_molecules)}")
        
        natoms = mol.num_nodes
        
        x_onehot = mol.x.cpu().numpy()  # [num_nodes, atom_type_num]
        elements = []
        
        for atom_idx in range(natoms):
            atom_type_idx = np.argmax(x_onehot[atom_idx])
            element_symbol = atomic_symbols[atom_type_idx + remove_h]
            elements.append(element_symbol)
        
        coordinates = mol.pos.cpu().numpy().tolist()  # Convert to Python list
        
        molecule_dict = {
            'natoms': natoms,
            'elements': elements,
            'coordinates': coordinates
        }
        
        processed_molecules.append(molecule_dict)
    
    with open(output_file, 'wb') as f:
        pickle.dump(processed_molecules, f)
    
    logging.info(f"Saved {len(processed_molecules)} molecules to {output_file}")

    python_logging.info("Sampling completed successfully.")

    script_end_time = datetime.now()
    total_runtime = script_end_time - script_start_time
    
    hours, remainder = divmod(total_runtime.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    runtime_str = ""
    if hours > 0:
        runtime_str += f"{int(hours)}h "
    if minutes > 0 or hours > 0:
        runtime_str += f"{int(minutes)}m "
    runtime_str += f"{int(seconds)}s"
    
    print(f"Sampling script finished. Total runtime: {runtime_str}")
    logging.info(f"Total runtime: {runtime_str}")
    python_logging.info(f"Total runtime: {runtime_str}")