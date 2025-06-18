from typing import Any, Optional, List, Tuple
import pytorch_lightning as pl
import argparse
import copy
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
torch.set_float32_matmul_precision('high')
import os
import datetime, pytz
import pickle
import numpy as np
from swanlab.integration.pytorch_lightning import SwanLabLogger as WandbLogger
from torch.optim.optimizer import Optimizer
from core.config.config import Config
from core.model.bfn.bfn_base import bfn4MolEGNN
from core.data.qm9_gen import QM9Gen
from core.data.data_gen_compete import CompeteDataGen
from pytorch_lightning.callbacks import ModelCheckpoint
from core.callbacks.basic import (
    Gradient_clip,
    NormalizerCallback,
    RecoverCallback,
    EMACallback,
)
from core.evaluation.validation_callback import (
    MolGenValidationCallback,
    MolVisualizationCallback,
)
from absl import logging

# Imports for energy prediction
from rdkit import Chem
from tqdm import tqdm
from predict.qm9.property_prediction import prop_utils
from predict.qm9.property_prediction.models_property import EGNN as PropEGNN
from core.evaluation.utils import build_molecule

# Set precision
torch.set_float32_matmul_precision("high")

# =============================================================================
# Helper Functions for Energy Prediction (from predict_qm9_property.py)
# =============================================================================

# --通用配置--
MAX_ATOMS = 29
SPECIES = [1, 6, 7, 8, 9, 15, 16, 17, 35]
SPECIES_TO_IDX = {z: i for i, z in enumerate(SPECIES)}
IN_NODE_NF = len(SPECIES)

def _featurize_mol(mol: Chem.Mol, max_atoms: int = MAX_ATOMS) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    if n_atoms > max_atoms:
        raise ValueError(f"原子数 {n_atoms} > 允许上限 {max_atoms}")

    pos = np.zeros((max_atoms, 3), dtype=np.float32)
    coord = np.array([conf.GetAtomPosition(i) for i in range(n_atoms)], dtype=np.float32)
    coord -= coord.mean(axis=0, keepdims=True)
    pos[:n_atoms] = coord

    one_hot = np.zeros((max_atoms, IN_NODE_NF), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        z = atom.GetAtomicNum()
        if z not in SPECIES_TO_IDX:
            raise KeyError(f"原子类型 {z} 未知")
        one_hot[i, SPECIES_TO_IDX[z]] = 1.0

    atom_mask = np.zeros(max_atoms, dtype=np.float32)
    atom_mask[:n_atoms] = 1.0
    return pos, one_hot, atom_mask

def _mols_to_batch(mol_list: List[Chem.Mol], device: torch.device) -> dict:
    B = len(mol_list)
    pos_arr = np.zeros((B, MAX_ATOMS, 3), dtype=np.float32)
    one_hot_a = np.zeros((B, MAX_ATOMS, IN_NODE_NF), dtype=np.float32)
    atom_mk = np.zeros((B, MAX_ATOMS), dtype=np.float32)

    valid_mol_indices = []
    for idx, mol in enumerate(mol_list):
        try:
            pos, one_hot, mask = _featurize_mol(mol)
            pos_arr[idx] = pos
            one_hot_a[idx] = one_hot
            atom_mk[idx] = mask
            valid_mol_indices.append(idx)
        except (KeyError, ValueError) as e:
            # logging.warning(f"Skipping molecule {idx} due to error: {e}")
            pass
    
    if not valid_mol_indices:
        return None, None
        
    # Filter arrays to only include valid molecules
    pos_arr = pos_arr[valid_mol_indices]
    one_hot_a = one_hot_a[valid_mol_indices]
    atom_mk = atom_mk[valid_mol_indices]
    B = len(valid_mol_indices) # Update batch size

    positions = torch.from_numpy(pos_arr)
    one_hot = torch.from_numpy(one_hot_a)
    atom_mask = torch.from_numpy(atom_mk).unsqueeze(-1)
    
    edge_mask = (atom_mask.unsqueeze(2) * atom_mask.unsqueeze(1))
    diag = ~torch.eye(MAX_ATOMS, dtype=torch.bool).unsqueeze(0)
    edge_mask = (edge_mask * diag.unsqueeze(-1)).view(B * MAX_ATOMS * MAX_ATOMS, 1)
    
    edges = prop_utils.get_adj_matrix(MAX_ATOMS, B, device)

    return {
        "positions": positions.view(B * MAX_ATOMS, 3),
        "one_hot": one_hot.view(B * MAX_ATOMS, IN_NODE_NF),
        "atom_mask": atom_mask.view(B * MAX_ATOMS, 1),
        "edge_mask": edge_mask,
        "edges": edges,
    }, valid_mol_indices

def load_trained_model(ckpt_path: str, args_pickle_path: str, mean: float, mad: float, device: torch.device) -> torch.nn.Module:
    with open(args_pickle_path, 'rb') as f:
        args = pickle.load(f)
    args.device = device

    model = PropEGNN(in_node_nf=IN_NODE_NF, in_edge_nf=0,
                     hidden_nf=args.nf, n_layers=args.n_layers,
                     device=device, coords_weight=1.0,
                     attention=args.attention, node_attr=args.node_attr)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    model.register_buffer('prop_mean', torch.tensor(mean, dtype=torch.float32, device=device))
    model.register_buffer('prop_mad', torch.tensor(mad, dtype=torch.float32, device=device))
    return model

@torch.no_grad()
def predict_mol_list(mol_list: List[Chem.Mol], model: torch.nn.Module, batch_size: int, device: torch.device) -> np.ndarray:
    n_total = len(mol_list)
    results = np.full(n_total, np.nan, dtype=np.float32)
    
    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        chunk = mol_list[start:end]
        
        batch, valid_indices = _mols_to_batch(chunk, device)
        if batch is None:
            continue

        for k in ("positions", "one_hot", "atom_mask", "edge_mask"):
            batch[k] = batch[k].to(device)

        pred_norm = model(h0=batch["one_hot"], x=batch["positions"], edges=batch["edges"],
                          edge_attr=None, node_mask=batch["atom_mask"],
                          edge_mask=batch["edge_mask"], n_nodes=MAX_ATOMS)
        
        pred_real = model.prop_mad * pred_norm + model.prop_mean
        pred_np = pred_real.squeeze(-1).cpu().numpy()

        # Place results back into the correct positions
        original_indices = [start + i for i in valid_indices]
        for i, res in zip(original_indices, pred_np):
            results[i] = res
            
    return results

# =============================================================================
# Main Lightning Module
# =============================================================================
class BFN4MolGenTrain(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.cfg = config
        self.dynamics = bfn4MolEGNN(
            self.cfg.dynamics.in_node_nf,
            self.cfg.dynamics.hidden_nf,
            n_layers=self.cfg.dynamics.n_layers,
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
        self.train_losses = []
        self.save_hyperparameters(logger=False)
        self.atomic_nb = self.cfg.dataset.atomic_nb
        self.remove_h = self.cfg.dataset.remove_h
        self.atom_type_num = len(self.atomic_nb) - self.remove_h

        # --- Load the energy prediction model ---
        logging.info("Loading energy prediction model...")
        ckpt = 'predict/qm9/property_prediction/outputs-0611/exp_class_alpha/best_checkpoint.npy'
        args_pkl = 'predict/qm9/property_prediction/outputs-0611/exp_class_alpha/args.pickle'
        mean = 21.3939
        mad = 6.4255
        
        try:
            self.energy_model = load_trained_model(ckpt, args_pkl, mean, mad, self.device)
            logging.info("Energy prediction model loaded successfully.")
        except FileNotFoundError as e:
            logging.warning(f"Could not load energy prediction model: {e}. Validation energy will not be calculated.")
            self.energy_model = None

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        h, charges, x, edge_index, segment_ids, energy = (
            batch.x,
            batch.charges,
            batch.pos,
            batch.edge_index,
            batch.batch,
            batch.y,
        )
        num_molecules = batch.ptr.shape[0] - 1
        h = charges

        if self.cfg.optimization.difftime:
            t = torch.rand([num_molecules, 1], dtype=x.dtype, device=x.device).index_select(0, segment_ids)
        else:
            t = torch.rand([1, 1], dtype=x.dtype, device=x.device) * torch.ones(
                size=[segment_ids.shape[0], 1], dtype=x.dtype, device=x.device
            )
        
        posloss, charge_loss, assets = self.dynamics.loss_one_step(
            t, x=h, pos=x, edge_index=edge_index, segment_ids=segment_ids, condition=energy
        )
        
        loss = torch.mean(posloss + charge_loss)

        self.log("loss", loss, on_step=True, prog_bar=True, batch_size=self.cfg.optimization.batch_size)
        self.train_losses.append(loss.clone().detach().cpu())
        return loss

    def charge_decode(self, charge):
        anchor = torch.tensor(
            [(2 * k - 1) / max(self.atomic_nb) - 1 for k in self.atomic_nb[self.remove_h:]],
            dtype=torch.float32,
            device=charge.device,
        )
        atom_type = (charge - anchor).abs().argmin(dim=-1)
        one_hot = torch.zeros([charge.shape[0], self.atom_type_num], dtype=torch.float32, device=charge.device)
        one_hot[torch.arange(charge.shape[0]), atom_type] = 1
        return one_hot
        
    def validation_step(self, batch, batch_idx):
        edge_index, segment_ids, energy = (
            batch.edge_index,
            batch.batch,
            batch.y,
        )
        n_nodes = segment_ids.shape[0]

        # Generate molecules
        theta_chain = self.dynamics(
            n_nodes=n_nodes,
            edge_index=edge_index,
            segment_ids=segment_ids,
            condition=energy,
        )
        x, h = theta_chain[-1]
        atom_type = self.charge_decode(h[:, :1])
        
        out_batch = copy.deepcopy(batch)
        out_batch.x, out_batch.pos = atom_type, x
        _slice_dict = {"x": out_batch._slice_dict["zx"], "pos": out_batch._slice_dict["zpos"]}
        _inc_dict = {"x": out_batch._inc_dict["zx"], "pos": out_batch._inc_dict["zpos"]}
        out_batch._inc_dict.update(_inc_dict)
        out_batch._slice_dict.update(_slice_dict)
        out_data_list = out_batch.to_data_list()

        # --- Energy Prediction Integration ---
        if self.energy_model is not None:
            rdkit_mols = []
            for data in out_data_list:
                pos = data.pos
                atom_type_idx = torch.argmax(data.x, dim=1)
                mol = build_molecule(pos, atom_type_idx, self.cfg.dataset.atom_decoder)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                        rdkit_mols.append(mol)
                    except (ValueError, RuntimeError):
                        pass
            print(f"[DEBUG] Number of sanitized molecules: {len(rdkit_mols)}")

            if rdkit_mols:
                predicted_energies = predict_mol_list(rdkit_mols, self.energy_model, self.cfg.evaluation.batch_size, self.device)
                mean_energy = np.nanmean(predicted_energies)
                if not np.isnan(mean_energy):
                    self.log("energy_loss", mean_energy, sync_dist=True)
            else:
                self.log("energy_loss", float('nan'), sync_dist=True)
                
        return out_data_list

    def on_train_epoch_end(self) -> None:
        if len(self.train_losses) == 0:
            epoch_loss = 0
        else:
            epoch_loss = torch.stack([x for x in self.train_losses]).mean()
        self.log("epoch_loss", epoch_loss, batch_size=self.cfg.optimization.batch_size)
        self.train_losses = []

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optimization.lr,
            amsgrad=True,
            weight_decay=float(self.cfg.optimization.weight_decay),
        )
        return optim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        type=str,
        default="debug.yaml",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--logging_level", type=str, default="warning")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sigma1_coord", type=float, default=0.001)
    parser.add_argument("--sigma1_charges", type=float, default=0.15)
    parser.add_argument("--beta1", type=float, default=2.0)
    parser.add_argument("--sample_steps", type=int, default=1000)
    parser.add_argument("--eval_data_num", type=int, default=1000)
    parser.add_argument("--checkpoint_freq", type=int, default=20)
    parser.add_argument("--exp_version", type=str, default=None)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--ckpt_pattern", type=str, default="last*.ckpt")

    _args = parser.parse_args()
    # _args, unknown = parser.parse_known_args()
    cfg = Config(**_args.__dict__)
    if cfg.debug:
        cfg.exp_name = "debug"
        cfg.dynamics.sample_steps = 50
        cfg.optimization.epochs = 2
        cfg.accounting.checkpoint_freq = 1
    
    print(f"The config of this process is:\n{cfg}")
    logging_level = {
        "info": logging.INFO,
        "debug": logging.DEBUG,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }
    logging.set_verbosity(logging_level[cfg.logging_level])
    # create dir if not exist
    os.makedirs(cfg.accounting.wandb_logdir, exist_ok=True)
    wandb_logger = WandbLogger(
        name=cfg.exp_name
        + f'_{datetime.datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d-%H:%M:%S")}',
        project=cfg.project_name,
        offline=cfg.debug or cfg.no_wandb,
        save_dir=cfg.accounting.wandb_logdir,
        version=cfg.accounting.exp_version,
    )  # add wandb parameters
    wandb_logger.log_hyperparams(cfg.todict())
    if not cfg.debug:
        cfg.save2yaml(cfg.accounting.dump_config_path)
    if cfg.dataset.name == "qm9":
        train_loader = QM9Gen(
            datadir=cfg.dataset.datadir,
            batch_size=cfg.optimization.batch_size,
            n_node_histogram=cfg.dataset.n_node_histogram,
            debug=cfg.debug,
            num_workers=cfg.dataset.num_workers,
            split="train" if not cfg.test else "test",
        )
        eval_loader = QM9Gen.initiate_evaluation_dataloader(
            data_num=cfg.evaluation.eval_data_num if not cfg.debug else 50,
            n_node_histogram=cfg.dataset.n_node_histogram,
            batch_size=cfg.evaluation.batch_size,
            num_workers=cfg.dataset.num_workers,
        )
    elif cfg.dataset.name == "compete":
        train_loader = CompeteDataGen(
            datadir=cfg.dataset.datadir,
            batch_size=cfg.optimization.batch_size,
            n_node_histogram=cfg.dataset.n_node_histogram,
            debug=cfg.debug,
            num_workers=cfg.dataset.num_workers,
            split="train" if not cfg.test else "test",
        )
        eval_loader = CompeteDataGen.initiate_evaluation_dataloader(
            data_num=cfg.evaluation.eval_data_num if not cfg.debug else 50,
            n_node_histogram=cfg.dataset.n_node_histogram,
            batch_size=cfg.evaluation.batch_size,
            num_workers=cfg.dataset.num_workers,
            max_n_nodes=60
        )
    elif cfg.dataset.name == "compete_condition":
        train_loader = CompeteDataGen(
            datadir=cfg.dataset.datadir,
            batch_size=cfg.optimization.batch_size,
            n_node_histogram=cfg.dataset.n_node_histogram,
            debug=cfg.debug,
            num_workers=cfg.dataset.num_workers,
            split="train" if not cfg.test else "test",
        )
        eval_loader = CompeteDataGen(
            datadir=cfg.dataset.datadir,
            batch_size=cfg.optimization.batch_size,
            n_node_histogram=cfg.dataset.n_node_histogram,
            debug=cfg.debug,
            num_workers=cfg.dataset.num_workers,
            split="val",
        )
    else:
        raise NotImplementedError

    model = BFN4MolGenTrain(config=cfg)
    # print(model)

    trainer = pl.Trainer(
        limit_test_batches=1,
        default_root_dir=cfg.accounting.logdir,
        max_epochs=cfg.optimization.epochs,
        check_val_every_n_epoch=cfg.accounting.checkpoint_freq,
        devices=1,
        logger=wandb_logger,
        num_sanity_val_steps=0,
        # overfit_batches=10,
        # gradient_clip_val=1.0,
        callbacks=[
            RecoverCallback(
                latest_ckpt=cfg.accounting.checkpoint_path,
                resume=cfg.optimization.resume or cfg.test,
                recover_trigger_loss=cfg.optimization.recover_trigger_loss,
                skip_count_limit=cfg.optimization.skip_count_limit,
            ),
            Gradient_clip(
                maximum_allowed_norm=cfg.optimization.maximum_allowed_norm,
            ),  # time consuming
            NormalizerCallback(normalizer_dict=cfg.dataset.normalizer_dict),
            MolGenValidationCallback(
                dataset=train_loader.ds,
                atom_type_one_hot=True,
                single_bond=cfg.evaluation.single_bond,
            ),
            ModelCheckpoint(
                dirpath=cfg.accounting.checkpoint_dir,
                filename="{epoch}-{energy_loss:2f}",
                every_n_epochs=cfg.accounting.checkpoint_freq,
                save_last=True,
                save_top_k=20,
                mode="max",
                monitor="energy_loss",
            ),
            MolVisualizationCallback(
                atomic_nb=cfg.dataset.atomic_nb,
                remove_h=cfg.dataset.remove_h,
                atom_decoder=cfg.dataset.atom_decoder,
                generated_mol_dir=cfg.accounting.generated_mol_dir,
            ),
            EMACallback(decay=0.9999, ema_device="cuda"),
            # DebugCallback(),
        ],
    )
    # num_sanity_val_steps=2, overfit_batches=10, devices=1
    # trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)
    if not cfg.test:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=eval_loader)
    else:
        trainer.validate(
            model,
            dataloaders=eval_loader,
        )
        # trainer.test(model, dataloaders=train_loader)
    wandb_logger.finalize("success")
    wandb_logger.experiment.finish()
    # trainer.test(model, datamodule=None)
