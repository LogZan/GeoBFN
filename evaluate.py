import pickle as pkl
import torch
import argparse
from torch_geometric.data import Data
from core.evaluation.metrics import BasicMolGenMetric

def load_molecules(pkl_file):
    """Load molecules from a pkl file."""
    with open(pkl_file, 'rb') as f:
        molecules = pkl.load(f)
    return molecules

def convert_molecules_to_data(molecules, atom_decoder):
    """Convert molecule dictionaries to torch_geometric.data.Data objects."""
    atom_encoder = {atom: i for i, atom in enumerate(atom_decoder)}
    data_list = []
    
    for mol in molecules:
        # Extract data
        natoms = mol['natoms']
        elements = mol['elements']
        coordinates = mol['coordinates']
        
        # Check data consistency
        assert len(elements) == natoms, f"Number of elements ({len(elements)}) doesn't match natoms ({natoms})"
        assert len(coordinates) == natoms, f"Number of coordinates ({len(coordinates)}) doesn't match natoms ({natoms})"
        
        # Convert to one-hot encoding
        x = torch.zeros(natoms, len(atom_decoder))
        valid_mol = True
        for i, element in enumerate(elements):
            if element in atom_encoder:
                x[i, atom_encoder[element]] = 1
            else:
                print(f"Warning: Element {element} not in atom_decoder, skipping molecule")
                valid_mol = False
                break
        
        if not valid_mol:
            continue
            
        # Convert coordinates to tensor
        pos = torch.tensor(coordinates, dtype=torch.float)
        
        # Create Data object
        data = Data(x=x, pos=pos)
        data_list.append(data)
    
    return data_list

def main():
    parser = argparse.ArgumentParser(description='Evaluate molecules from pickle file')
    parser.add_argument('--input', type=str, required=True, help='Path to the input pickle file')
    args = parser.parse_args()
    
    # Initialize parameters as specified
    atom_decoder = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br']
    dataset_smiles_set_path = 'dataset/compete/processed/all_smiles.pkl'
    type_one_hot = True
    single_bond = False
    
    # Load dataset smiles set
    try:
        with open(dataset_smiles_set_path, 'rb') as f:
            dataset_smiles_set = pkl.load(f)
    except FileNotFoundError:
        print(f"Warning: SMILES set file not found at {dataset_smiles_set_path}")
        print("Using empty set for novelty calculation (all molecules will be considered novel)")
        dataset_smiles_set = set()
    
    print(f"Loaded {len(dataset_smiles_set)} reference SMILES")
    
    # Load molecules from pkl file
    print(f"Loading molecules from {args.input}")
    molecules = load_molecules(args.input)
    print(f"Loaded {len(molecules)} molecules")
    
    # Convert to torch_geometric.data.Data format
    data_list = convert_molecules_to_data(molecules, atom_decoder)
    print(f"Converted {len(data_list)} valid molecules")
    
    # Initialize metric
    metric = BasicMolGenMetric(
        atom_decoder=atom_decoder,
        dataset_smiles_set=dataset_smiles_set,
        type_one_hot=type_one_hot,
        single_bond=single_bond
    )
    
    # Evaluate molecules
    print("Evaluating molecules...")
    results = metric.evaluate(data_list)
    
    # Print detailed results
    print("\nEvaluation Results:")
    print(f"Molecular stability: {results['mol_stable'] * 100:.2f}%")
    print(f"Atomic stability: {results['atm_stable'] * 100:.2f}%")
    print(f"Validity: {results['validity'] * 100:.2f}%")
    print(f"Uniqueness: {results['uniqueness'] * 100:.2f}%")
    print(f"Novelty: {results['novelty'] * 100:.2f}%")
    print(f"Stable valid uniqueness: {results['stable_valid_uniqueness'] * 100:.2f}%")
    print(f"Compound score: {results['compound_score'] * 100:.2f}%")

if __name__ == "__main__":
    main()
