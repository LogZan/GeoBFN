from typing import Any, Optional, List, Tuple
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch_geometric.data import Data
import numpy as np
import torch
import os
import tqdm
import pickle as pkl
from core.evaluation.utils import convert_atomcloud_to_mol_smiles, save_molist, dump2mol, build_molecule
from core.evaluation.metrics import BasicMolGenMetric
from core.evaluation.visualization import visualize, visualize_chain
import json
import matplotlib
import swanlab as wandb
import copy

# Imports for energy prediction
from rdkit import Chem
from tqdm import tqdm as TQDM_Progress # Use a different alias to avoid conflict
from predict.qm9.property_prediction import prop_utils
from predict.qm9.property_prediction.models_property import EGNN as PropEGNN
from predict.qm9.property_prediction.models_property import Naive, NumNodes
from absl import logging

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==============================================================================
# ==== ENERGY PREDICTION HELPERS (Integrated from predict_qm9_property.py) =====
# ==============================================================================

# ---- 1. General Configuration ------------------------------------------------
MAX_ATOMS = 29
SPECIES = [1, 6, 7, 8, 9, 15, 16, 17, 35]
SPECIES_TO_IDX = {z: i for i, z in enumerate(SPECIES)}
IN_NODE_NF = len(SPECIES)

# ---- 2. Molecule to Tensor ---------------------------------------------------
def _featurize_mol(
    mol: Chem.Mol,
    max_atoms: int = MAX_ATOMS
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    if n_atoms > max_atoms:
        raise ValueError(f"Atom count {n_atoms} > max allowed {max_atoms}")

    pos = np.zeros((max_atoms, 3), dtype=np.float32)
    coord = np.array([conf.GetAtomPosition(i) for i in range(n_atoms)], dtype=np.float32)
    coord -= coord.mean(axis=0, keepdims=True)
    pos[:n_atoms] = coord

    one_hot = np.zeros((max_atoms, IN_NODE_NF), dtype=np.float32)
    for i, atom in enumerate(mol.GetAtoms()):
        z = atom.GetAtomicNum()
        one_hot[i, SPECIES_TO_IDX[z]] = 1.0  # Will KeyError if element is unknown

    atom_mask = np.zeros(max_atoms, dtype=np.float32)
    atom_mask[:n_atoms] = 1.0
    return pos, one_hot, atom_mask

# ---- 3. Mol List to Batch Tensor ---------------------------------------------
def _mols_to_batch(mol_list: List[Chem.Mol], device: torch.device) -> dict:
    B = len(mol_list)
    pos_arr = np.zeros((B, MAX_ATOMS, 3), dtype=np.float32)
    one_hot_a = np.zeros((B, MAX_ATOMS, IN_NODE_NF), dtype=np.float32)
    atom_mk = np.zeros((B, MAX_ATOMS), dtype=np.float32)

    for idx, mol in enumerate(mol_list):
        pos, one_hot, mask = _featurize_mol(mol)
        pos_arr[idx] = pos
        one_hot_a[idx] = one_hot
        atom_mk[idx] = mask

    positions = torch.from_numpy(pos_arr)
    one_hot = torch.from_numpy(one_hot_a)
    atom_mask = torch.from_numpy(atom_mk).unsqueeze(-1)
    atom_mask_f = atom_mask.view(B * MAX_ATOMS, 1)

    edge_mask = (atom_mask.unsqueeze(2) * atom_mask.unsqueeze(1))
    diag = ~torch.eye(MAX_ATOMS, dtype=torch.bool).unsqueeze(0).unsqueeze(-1)
    edge_mask = (edge_mask * diag).view(B * MAX_ATOMS * MAX_ATOMS, 1)

    edges = prop_utils.get_adj_matrix(MAX_ATOMS, B, device)

    return {
        "positions": positions.view(B * MAX_ATOMS, 3),
        "one_hot": one_hot.view(B * MAX_ATOMS, IN_NODE_NF),
        "atom_mask": atom_mask_f,
        "edge_mask": edge_mask,
        "edges": edges,
    }

# ---- 4. Load Trained Model ---------------------------------------------------
def load_trained_model(ckpt_path: str, args_pickle_path: str,
                       mean: float, mad: float, device: torch.device) -> torch.nn.Module:
    import pickle
    with open(args_pickle_path, 'rb') as f:
        args = pickle.load(f)
    args.device = device

    if args.model_name == 'egnn':
        model = PropEGNN(in_node_nf=IN_NODE_NF, in_edge_nf=0,
                         hidden_nf=args.nf, n_layers=args.n_layers,
                         device=device, coords_weight=1.0,
                         attention=args.attention, node_attr=args.node_attr)
    elif args.model_name == 'naive':
        model = Naive(device=device)
    elif args.model_name == 'numnodes':
        model = NumNodes(device=device)
    else:
        raise ValueError(f"Unknown model name {args.model_name}")

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    model.register_buffer('prop_mean', torch.tensor(mean, dtype=torch.float32))
    model.register_buffer('prop_mad', torch.tensor(mad, dtype=torch.float32))
    return model

# ---- 5. Core Interface: Batched & Fault-Tolerant Prediction ------------------
@torch.no_grad()
def predict_mol_list(
    mol_list: List[Chem.Mol],
    model: torch.nn.Module,
    batch_size: int = 512
) -> np.ndarray:
    n_total = len(mol_list)
    results = np.full(n_total, np.nan, dtype=np.float32)
    pbar = TQDM_Progress(total=n_total, desc="Energy Prediction", ncols=90)
    device = next(model.parameters()).device # Infer device from model

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        idxs = list(range(start, end))
        chunk = mol_list[start:end]

        valid_mols, valid_pos = [], []
        for idx_local, mol in enumerate(chunk):
            try:
                _ = _featurize_mol(mol)
                valid_mols.append(mol)
                valid_pos.append(idxs[idx_local])
            except Exception:
                continue

        if not valid_mols:
            pbar.update(len(chunk))
            continue

        try:
            batch = _mols_to_batch(valid_mols, device)
            for k in ("positions", "one_hot", "atom_mask", "edge_mask"):
                batch[k] = batch[k].to(device)

            pred_norm = model(h0=batch["one_hot"],
                              x=batch["positions"],
                              edges=batch["edges"],
                              edge_attr=None,
                              node_mask=batch["atom_mask"],
                              edge_mask=batch["edge_mask"],
                              n_nodes=MAX_ATOMS)

            pred_real = model.prop_mad * pred_norm + model.prop_mean
            pred_np = pred_real.squeeze(-1).cpu().numpy()

            for pos, y in zip(valid_pos, pred_np):
                results[pos] = y

        except Exception as e:
            logging.warning(f"Batch prediction failed with error: {e}")
            pass
        finally:
            if 'batch' in locals():
                del batch
            if 'pred_norm' in locals():
                del pred_norm
            if 'pred_real' in locals():
                del pred_real
            torch.cuda.empty_cache()

        pbar.update(len(chunk))
    pbar.close()
    return results

# ==============================================================================
# ==== Pytorch Lightning Validation Callback ===================================
# ==============================================================================

def compute_or_retrieve_dataset_smiles(
    dataset, atom_decoder, save_path, single_bond=False
):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    if not os.path.exists(save_path):
        all_smiles = []
        with tqdm.tqdm(total=len(dataset)) as pbar:
            print("Computing all smiles")
            for data in dataset:
                mol, smiles = convert_atomcloud_to_mol_smiles(
                    data.pos, data.x, atom_decoder, type_one_hot=True, single_bond=single_bond,
                )
                if smiles is not None:
                    all_smiles.append(smiles)
                pbar.update(1)
        print(f"Saving {len(all_smiles)} smiles to {save_path}")
        with open(save_path, "wb") as f:
            pkl.dump(all_smiles, f)
    else:
        print("Loading all smiles")
        with open(save_path, "rb") as f:
            all_smiles = pkl.load(f)
    if isinstance(all_smiles[0], tuple):
        smiles_set = set([s[1] for s in all_smiles])
    else:
        smiles_set = set([s for s in all_smiles])
    return smiles_set


class MolGenValidationCallback(Callback):
    def __init__(self, dataset, atom_type_one_hot=True, single_bond=False) -> None:
        super().__init__()
        self.dataset = dataset
        self.single_bond = single_bond
        self.type_one_hot = atom_type_one_hot
        self.outputs = []
        self.test_outputs = []
        self.energy_model = None

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        super().setup(trainer, pl_module, stage)
        dataset_all_smiles = compute_or_retrieve_dataset_smiles(
            dataset=self.dataset,
            atom_decoder=pl_module.cfg.dataset.atom_decoder,
            save_path=os.path.join(
                pl_module.cfg.dataset.datadir, "processed", "all_smiles.pkl"
            ),
            single_bond=self.single_bond,
        )
        self.all_smiles = set(dataset_all_smiles)
        self.metric = BasicMolGenMetric(
            atom_decoder=pl_module.cfg.dataset.atom_decoder,
            dataset_smiles_set=self.all_smiles,
            type_one_hot=self.type_one_hot,
            single_bond=self.single_bond,
        )

        # Load energy prediction model
        ckpt = 'predict/qm9/property_prediction/outputs-0611/exp_class_alpha/best_checkpoint.npy'
        args_pkl = 'predict/qm9/property_prediction/outputs-0611/exp_class_alpha/args.pickle'
        mean = 21.3939
        mad = 6.4255
        
        try:
            self.energy_model = load_trained_model(ckpt, args_pkl, mean, mad, pl_module.device)
            logging.info("Energy prediction model loaded successfully.")
        except (FileNotFoundError, ValueError) as e:
            logging.warning(f"Could not load energy prediction model: {e}. Validation energy will not be calculated.")
            self.energy_model = None

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )
        self.outputs.extend(outputs)

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.outputs = []

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        
        if not self.outputs:
            logging.warning("Validation outputs are empty. Skipping evaluation.")
            return

        # --- Energy Prediction Integration ---
        if self.energy_model is not None:
            rdkit_mols = []
            for data in self.outputs:
                pos = data.pos
                atom_type_idx = torch.argmax(data.x, dim=1)
                mol = build_molecule(pos, atom_type_idx, pl_module.cfg.dataset.atom_decoder)
                if mol is not None:
                    try:
                        Chem.SanitizeMol(mol)
                        rdkit_mols.append(mol)
                    except (ValueError, RuntimeError):
                        pass
            
            logging.info(f"Generated {len(rdkit_mols)} sanitizable molecules for energy prediction.")

            if rdkit_mols:
                predicted_energies = predict_mol_list(rdkit_mols, self.energy_model, batch_size=64)
                
                # Extract real energy values from validation outputs that correspond to valid molecules
                real_energies = []
                mol_idx = 0
                for data in self.outputs:
                    pos = data.pos
                    atom_type_idx = torch.argmax(data.x, dim=1)
                    mol = build_molecule(pos, atom_type_idx, pl_module.cfg.dataset.atom_decoder)
                    if mol is not None:
                        try:
                            Chem.SanitizeMol(mol)
                            # This molecule was included in rdkit_mols, so include its real energy
                            real_energies.append(data.y_real.cpu().numpy())
                            mol_idx += 1
                        except (ValueError, RuntimeError):
                            pass
                
                real_energies = np.array(real_energies)
                energy_loss = np.abs(predicted_energies - real_energies)
                mean_energy = np.nanmean(energy_loss)
                
                if not np.isnan(mean_energy):
                    pl_module.log("val/energy", mean_energy, sync_dist=True)
                    logging.info(f"Logged mean energy: {mean_energy:.4f}")
                else:
                    pl_module.log("val/energy", float('nan'), sync_dist=True)
                    logging.warning("Energy prediction resulted in NaN for all valid molecules.")
            else:
                pl_module.log("val/energy", float('nan'), sync_dist=True)
                logging.warning("No valid molecules generated for energy prediction.")
                
        out_metrics = self.metric.evaluate(self.outputs)
        # Log metrics with a 'val/' prefix for better organization
        pl_module.log_dict({f"val/{k}": v for k, v in out_metrics.items()})
        print(json.dumps(out_metrics, indent=4))

class MolVisualizationCallback(Callback):
    # here the call back, we save the molecules and also draw the figures also to the wandb.
    def __init__(self, atomic_nb, remove_h, atom_decoder, generated_mol_dir) -> None:
        super().__init__()
        self.outputs = []
        self.test_outputs = {"in_data": [], "out_data": []}
        self.chain_outputs = []
        self.atomic_nb = atomic_nb
        self.remove_h = remove_h
        self.atom_type_num = len(atomic_nb) - remove_h
        self.generated_mol_dir = generated_mol_dir
        self.atom_decoder = atom_decoder

    # TODO delete this function
    def charge_decode(self, charge):
        """
        charge: [n_nodes, 1]
        """
        anchor = torch.tensor(
            [
                (2 * k - 1) / max(self.atomic_nb) - 1
                for k in self.atomic_nb[self.remove_h :]
            ],
            dtype=torch.float32,
            device=charge.device,
        )
        atom_type = (charge - anchor).abs().argmin(dim=-1)
        one_hot = torch.zeros(
            [charge.shape[0], self.atom_type_num], dtype=torch.float32
        )
        one_hot[torch.arange(charge.shape[0]), atom_type] = 1
        return one_hot

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        super().on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

        self.outputs.extend(outputs)
        if len(self.chain_outputs) == 0:
            _, _, _, edge_index, segment_ids, energy = (
                batch.zx,  # [n_nodes, n_features]
                batch.zpos,  # [n_nodes, 3]
                batch.zcharges,
                batch.edge_index,  # [2, edge_num]
                batch.batch,  # [n_nodes]
                batch.y
            )
            # z_h = (
            #     torch.concat([z_h, z_charges], dim=-1)
            #     if self.cfg.dynamics.include_charges
            #     else z_h
            # )
            # tseqs = torch.linspace(0, 1, 250, dtype=torch.float32, device=z_x.device
            n_nodes = segment_ids.shape[0]
            # TODO don't call model in evaluation callbacks, make sure this stays in the evaluation_step
            theta_chain = pl_module.dynamics(
                n_nodes=n_nodes,
                edge_index=edge_index,
                sample_steps=pl_module.dynamics.sample_steps,
                segment_ids=segment_ids,
                condition=energy,
            )
            for i in tqdm.tqdm(range(len(theta_chain))):
                x, h = theta_chain[i]
                atom_type = self.charge_decode(h[:, :1])
                out_batch = copy.deepcopy(batch)

                out_batch.x, out_batch.pos = (atom_type, x)
                _slice_dict = {
                    "x": out_batch._slice_dict["zx"],
                    "pos": out_batch._slice_dict["zpos"],
                }
                _inc_dict = {
                    "x": out_batch._inc_dict["zx"],
                    "pos": out_batch._inc_dict["zpos"],
                }
                out_batch._inc_dict.update(_inc_dict)
                out_batch._slice_dict.update(_slice_dict)
                out_data_list = out_batch.to_data_list()
                self.chain_outputs.append(
                    out_data_list[0]
                )  # always append the first sampled dtat

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.outputs = []
        self.chain_outputs = []

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        epoch = pl_module.current_epoch

        path = os.path.join(pl_module.cfg.accounting.generated_mol_dir, str(epoch))

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        chain_path = os.path.join(
            pl_module.cfg.accounting.generated_mol_dir, str(epoch), "chain"
        )

        if not os.path.exists(chain_path):
            os.makedirs(chain_path, exist_ok=True)

        if pl_module.cfg.visual.save_mols:
            # we save the figures here.
            save_molist(
                path=path,
                molecule_list=self.outputs,
                index2atom=pl_module.cfg.dataset.atom_decoder,
            )
            if pl_module.cfg.visual.visual_nums > 0:
                images = visualize(
                    path=path,
                    atom_decoder=pl_module.cfg.dataset.atom_decoder,
                    color_dic=pl_module.cfg.dataset.colors_dic,
                    radius_dic=pl_module.cfg.dataset.radius_dic,
                    max_num=pl_module.cfg.visual.visual_nums,
                )
                # table = [[],[]]
                table = []
                for p_ in images:
                    im = plt.imread(p_)
                    table.append(wandb.Image(im))
                    # if len(table[0]) < 5:
                    #     table[0].append(wandb.Image(im))
                    # else:
                    #     table[1].append(wandb.Image(im))
                # pl_module.logger.log_table(key="epoch {}".format(epoch),data=table,columns= ['1','2','3','4','5'])
                # pl_module.logger.log_image(key="epoch {}".format(epoch), images=table)
                # wandb.log()
                # update to wandb
        if pl_module.cfg.visual.visual_chain:
            # we save the chains and visual the gif here.
            # print(len(self.chain_outputs),chain_path)
            save_molist(
                path=chain_path,
                molecule_list=self.chain_outputs,
                index2atom=pl_module.cfg.dataset.atom_decoder,
            )
            # if pl_module.cfg.visual.visual_nums > 0:
            gif_path = visualize_chain(
                path=chain_path,
                atom_decoder=pl_module.cfg.dataset.atom_decoder,
                color_dic=pl_module.cfg.dataset.colors_dic,
                radius_dic=pl_module.cfg.dataset.radius_dic,
                spheres_3d=False,
            )
            gifs = wandb.Video(gif_path)
            columns = ["Generation Path"]
            pl_module.logger.log_table(
                key="epoch_{}".format(epoch), data=[[gifs]], columns=columns
            )

            # table = [[],[]]

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        for key in outputs:
            self.test_outputs[key].extend(outputs[key])

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.test_outputs = {"in_data": [], "out_data": []}
        # create dir if not exist for generated molecules
        if self.generated_mol_dir is not None:
            os.makedirs(self.generated_mol_dir, exist_ok=True)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for key in self.test_outputs:
            for idx, data_list in enumerate(self.test_outputs[key]):
                for step, data in enumerate(data_list):
                    dump2mol(
                        data,
                        os.path.join(
                            self.generated_mol_dir,
                            f"{key}-molid_{idx:03}-step_{step:03}.mol",
                        ),
                        index2atom=self.atom_decoder,
                        get_bond=True,
                    )
