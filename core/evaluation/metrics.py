# We implement the evaluation metric in this file.
from rdkit import Chem
from torch_geometric.data import Data
from core.evaluation.utils import (
    convert_atomcloud_to_mol_smiles,
    build_molecule,
    mol2smiles,
    build_xae_molecule,
    check_stability,
)
from typing import List, Dict, Tuple
from tqdm import tqdm
import numpy as np
from scipy.spatial.distance import jensenshannon

class BasicMolGenMetric(object):
    def __init__(
        self, atom_decoder, dataset_smiles_set, type_one_hot=True, single_bond=False
    ):
        self.atom_decoder = atom_decoder
        self.dataset_smiles_set = dataset_smiles_set
        self.type_one_hot = type_one_hot
        self.single_bond = single_bond
        
        # Reference histograms
        self.element_histogram = np.array([503261, 390055, 87438, 67606, 7875, 1388, 13832, 5878, 1184])
        self.element_histogram = self.element_histogram / np.sum(self.element_histogram)  # Normalize
        
        self.n_node_histogram = np.array([0, 0, 0, 1, 2, 17, 35, 60, 115, 207, 314, 489, 772, 1000, 1331, 1582,
                     1837, 2061, 2281, 2385, 2435, 2502, 2524, 2461, 2404, 2296, 2203, 2024,
                     1904, 1668, 1479, 1265, 1078, 889, 763, 635, 556, 461, 350, 278, 228,
                     174, 159, 130, 91, 84, 84, 58, 51, 29, 25, 15, 20, 6, 5, 4, 9, 6, 5, 4, 3])
        self.n_node_histogram = self.n_node_histogram / np.sum(self.n_node_histogram)  # Normalize

    def compute_stability(self, generated2idx: List[Tuple[Data, int]]):
        n_samples = len(generated2idx)
        molecule_stable = 0
        nr_stable_bonds = 0
        n_atoms = 0
        return_list = []
        for data, idx in tqdm(generated2idx, desc="Computing stability"):
            positions = data.pos
            atom_type = data.x
            stability_results = check_stability(
                positions=positions,
                atom_type=atom_type,
                atom_decoder=self.atom_decoder,
                single_bond=self.single_bond,
            )

            molecule_stable += int(stability_results[0])
            nr_stable_bonds += int(stability_results[1])
            n_atoms += int(stability_results[2])
            if int(stability_results[0]) != 0:
                return_list.append((data, idx))

        # stability
        fraction_mol_stable = molecule_stable / float(n_samples)
        fraction_atm_stable = nr_stable_bonds / float(n_atoms)
        stability_dict = {
            "mol_stable": fraction_mol_stable,
            "atm_stable": fraction_atm_stable,
        }
        return stability_dict, return_list

    def compute_validity(self, generated2idx: List[Tuple[Data, int]]):
        """generated: list of couples (positions, atom_types)"""
        valid = []
        return_list = []
        for graph, idx in tqdm(generated2idx, desc="Computing validity"):
            mol, smiles = convert_atomcloud_to_mol_smiles(
                positions=graph.pos,
                atom_type=graph.x,
                atom_decoder=self.atom_decoder,
                type_one_hot=self.type_one_hot,
                single_bond=self.single_bond,
            )
            if smiles is not None:
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
                largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
                smiles = mol2smiles(largest_mol)
                if smiles is not None:
                    valid.append(smiles)
                    return_list.append((smiles, idx))

        return valid, len(valid) / (len(generated2idx) + 1e-12), return_list

    def compute_uniqueness(self, valid):
        """valid: list of SMILES strings."""
        return list(set(valid)), len(set(valid)) / (len(valid) + 1e-12)

    def compute_novelty(self, unique):
        num_novel = 0
        novel = []
        for smiles in unique:
            if smiles not in self.dataset_smiles_set:
                novel.append(smiles)
                num_novel += 1
        return novel, num_novel / (len(unique) + 1e-12)

    def compute_size_distribution(self, valid_mols):
        """Compute size distribution of generated molecules"""
        size_counts = np.zeros(len(self.n_node_histogram))
        
        for smiles, _ in valid_mols:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                num_atoms = mol.GetNumAtoms()
                if num_atoms < len(size_counts):
                    size_counts[num_atoms] += 1
        
        # Normalize
        if np.sum(size_counts) > 0:
            size_distribution = size_counts / np.sum(size_counts)
        else:
            size_distribution = size_counts
            
        # Calculate similarity with Jensen-Shannon divergence
        # Lower values mean more similar distributions
        js_distance = jensenshannon(size_distribution, self.n_node_histogram)
        size_similarity = 1.0 - min(js_distance, 1.0)  # Convert to similarity score
        
        return size_similarity

    def compute_element_distribution(self, valid_mols):
        """Compute element distribution of generated molecules"""
        element_counts = np.zeros(len(self.atom_decoder))
        
        for smiles, _ in valid_mols:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    if symbol in self.atom_decoder:
                        element_counts[self.atom_decoder.index(symbol)] += 1
        
        # Normalize
        if np.sum(element_counts) > 0:
            element_distribution = element_counts / np.sum(element_counts)
        else:
            element_distribution = element_counts
        
        # Calculate similarity with Jensen-Shannon divergence
        js_distance = jensenshannon(element_distribution, self.element_histogram)
        element_similarity = 1.0 - min(js_distance, 1.0)  # Convert to similarity score
        
        return element_similarity

    def evaluate(self, generated: List[Data]):
        """generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked."""
        generated2idx = [(graph, i) for i, graph in enumerate(generated)]
        stability_dict, return_generated2idx_list = self.compute_stability(
            generated2idx
        )
        valid, validity, _ = self.compute_validity(generated2idx)
        _, _, valid_mols = self.compute_validity(
            return_generated2idx_list
        )
        print(f"Validity over {len(generated)} molecules: {validity * 100 :.2f}%")
        if validity > 0:
            unique, uniqueness = self.compute_uniqueness(valid)
            print(
                f"Uniqueness over {len(valid)} valid molecules: {uniqueness * 100 :.2f}%"
            )

            _, novelty = self.compute_novelty(unique)
            print(
                f"Novelty over {len(unique)} unique valid molecules: {novelty * 100 :.2f}%"
            )
        else:
            novelty = 0.0
            uniqueness = 0.0
            unique = None
        if len(valid_mols) > 0:
            _, stable_valid_uniqueness = self.compute_uniqueness(
                [g for g, i in valid_mols]
            )
            stable_valid_uniqueness = (
                stable_valid_uniqueness
                * len(valid_mols)
                / len(generated)
            )
            
            # Compute distribution similarities
            size_similarity = self.compute_size_distribution(valid_mols)
            element_similarity = self.compute_element_distribution(valid_mols)
            print(f"Size distribution similarity: {size_similarity:.4f}")
            print(f"Element distribution similarity: {element_similarity:.4f}")
        else:
            stable_valid_uniqueness = 0.0
            size_similarity = 0.0
            element_similarity = 0.0

        # Calculate combined score as the average of three metrics
        combined_score = (validity + stable_valid_uniqueness + novelty) / 3

        return {
            "validity": validity,
            "uniqueness": uniqueness,
            "stable_valid_uniqueness": stable_valid_uniqueness,
            "novelty": novelty,
            "combined_score": combined_score,
            "size_similarity": size_similarity,
            "element_similarity": element_similarity,
            **stability_dict,
        }
