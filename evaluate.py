import pickle as pkl
import torch
import argparse
import multiprocessing
from torch_geometric.data import Data
from core.evaluation.metrics import BasicMolGenMetric
from tqdm import tqdm
from tabulate import tabulate
from functools import partial

def load_molecules(pkl_file):
    """Load molecules from a pkl file."""
    with open(pkl_file, 'rb') as f:
        molecules = pkl.load(f)
    return molecules

def process_molecule(mol, atom_decoder, atom_encoder):
    """Process a single molecule for parallel execution."""
    # Extract data
    natoms = mol['natoms']
    elements = mol['elements']
    coordinates = mol['coordinates']
    
    # Check data consistency
    if len(elements) != natoms or len(coordinates) != natoms:
        return None
    
    # Convert to one-hot encoding
    x = torch.zeros(natoms, len(atom_decoder))
    for i, element in enumerate(elements):
        if element in atom_encoder:
            x[i, atom_encoder[element]] = 1
        else:
            return None
    
    # Convert coordinates to tensor
    pos = torch.tensor(coordinates, dtype=torch.float)
    
    # Create Data object
    return Data(x=x, pos=pos)

def convert_molecules_to_data_parallel(molecules, atom_decoder, num_workers=10):
    """Convert molecule dictionaries to Data objects using multiple processes."""
    atom_encoder = {atom: i for i, atom in enumerate(atom_decoder)}
    
    # Create partial function with fixed atom_decoder and atom_encoder
    process_fn = partial(process_molecule, atom_decoder=atom_decoder, atom_encoder=atom_encoder)
    
    # Use multiprocessing to process molecules
    with multiprocessing.Pool(processes=num_workers) as pool:
        data_list = list(tqdm(
            pool.imap(process_fn, molecules),
            total=len(molecules),
            desc="Converting molecules"
        ))
    
    # Filter out None results (invalid molecules)
    data_list = [data for data in data_list if data is not None]
    
    return data_list

def main():
    parser = argparse.ArgumentParser(description='Evaluate molecules from pickle file')
    parser.add_argument('--input', type=str, required=True, help='Path to the input pickle file')
    parser.add_argument('--cores', type=int, default=10, help='Number of CPU cores to use')
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
        print(f"Loaded {len(dataset_smiles_set)} reference SMILES")
    except FileNotFoundError:
        print(f"Warning: SMILES set file not found at {dataset_smiles_set_path}")
        print("Using empty set for novelty calculation (all molecules will be considered novel)")
        dataset_smiles_set = set()
    
    # Load molecules from pkl file
    print(f"Loading molecules from {args.input}")
    molecules = load_molecules(args.input)
    print(f"Loaded {len(molecules)} molecules")
    
    # Convert to torch_geometric.data.Data format using parallel processing
    data_list = convert_molecules_to_data_parallel(
        molecules, 
        atom_decoder, 
        num_workers=args.cores
    )
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
    # We can't directly add tqdm to the evaluate method as it's internal,
    # but the method itself prints progress
    results = metric.evaluate(data_list)
    
    # Format results as a table with 4 decimal precision
    table_data = [
        ["Metric", "Value"],
        ["Molecular stability", f"{results['mol_stable']:.4f}"],
        ["Atomic stability", f"{results['atm_stable']:.4f}"],
        ["Validity", f"{results['validity']:.4f}"],
        ["Uniqueness", f"{results['uniqueness']:.4f}"],
        ["Novelty", f"{results['novelty']:.4f}"],
        ["Stable valid uniqueness", f"{results['stable_valid_uniqueness']:.4f}"],
        ["Compound score", f"{results['compound_score']:.4f}"]
    ]
    
    # Print nicely formatted table
    print("\nEvaluation Results:")
    print(tabulate(table_data, headers="firstrow", tablefmt="grid"))

if __name__ == "__main__":
    main()
