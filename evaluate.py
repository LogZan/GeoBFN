import pickle as pkl
import torch
import argparse
import multiprocessing
import numpy as np
import os
import csv
from collections import Counter
from scipy.spatial.distance import jensenshannon
from torch_geometric.data import Data
from core.evaluation.metrics import BasicMolGenMetric
from tqdm import tqdm
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

def calculate_molecule_size_distribution(molecules):
    """Calculate distribution of molecule sizes (number of atoms)."""
    sizes = [mol['natoms'] for mol in molecules if 'natoms' in mol]
    size_counter = Counter(sizes)
    total = sum(size_counter.values())
    
    # Convert to distribution
    size_dist = {size: count/total for size, count in sorted(size_counter.items())}
    return size_dist

def calculate_element_distribution(molecules, atom_decoder):
    """Calculate distribution of elements across all molecules."""
    all_elements = []
    for mol in molecules:
        if 'elements' in mol:
            all_elements.extend(mol['elements'])
            
    element_counter = Counter(all_elements)
    total = sum(element_counter.values())
    
    # Convert to distribution with all possible atoms from atom_decoder
    element_dist = {element: 0 for element in atom_decoder}
    element_dist.update({element: count/total for element, count in element_counter.items()})
    
    return element_dist

def calculate_distribution_similarity(dist1, dist2):
    """Calculate similarity between two distributions using Jensen-Shannon divergence."""
    # Ensure both distributions have the same keys
    all_keys = sorted(set(list(dist1.keys()) + list(dist2.keys())))
    vec1 = np.array([dist1.get(k, 0) for k in all_keys])
    vec2 = np.array([dist2.get(k, 0) for k in all_keys])
    
    # Normalize if needed
    if np.sum(vec1) > 0:
        vec1 = vec1 / np.sum(vec1)
    if np.sum(vec2) > 0:
        vec2 = vec2 / np.sum(vec2)
    
    # Calculate Jensen-Shannon divergence
    js_divergence = jensenshannon(vec1, vec2)
    
    # Convert to similarity score (1 - divergence)
    if np.isnan(js_divergence):
        return 0.0
    return 1.0 - js_divergence

def format_results_table(results):
    """Format results as a simple text table."""
    lines = ["Evaluation Results:", "="*50]
    for key, value in results.items():
        if isinstance(value, float):
            lines.append(f"{key.replace('_', ' ').title():30s}: {value:.4f}")
        else:
            lines.append(f"{key.replace('_', ' ').title():30s}: {value}")
    return "\n".join(lines)

def save_results_to_csv(results, input_path):
    """Save evaluation results to a CSV file."""
    # Create directory if it doesn't exist
    output_dir = "output/evaluate"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract filename without extension
    base_filename = os.path.basename(input_path)
    output_filename = os.path.splitext(base_filename)[0] + ".csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Write results to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        for key, value in results.items():
            if isinstance(value, float):
                writer.writerow([key, f"{value:.4f}"])
            else:
                writer.writerow([key, value])
    
    return output_path

def format_distribution_table(dist1, dist2, title1="Input", title2="Reference"):
    """Format two distributions side by side for better comparison."""
    # Get all keys from both distributions
    all_keys = sorted(set(list(dist1.keys()) + list(dist2.keys())))
    
    # Calculate maximum key width for alignment
    key_width = max(len(str(k)) for k in all_keys)
    
    # Format header
    lines = [
        f"{title1} vs {title2} Distribution",
        "=" * 60,
        f"{'Key':<{key_width}}  |  {title1:>10}  |  {title2:>10}  |  Difference",
        "-" * 60
    ]
    
    # Format each row
    for key in all_keys:
        val1 = dist1.get(key, 0) * 100  # Convert to percentage
        val2 = dist2.get(key, 0) * 100
        diff = val1 - val2
        
        # Format the row with alignment
        lines.append(f"{str(key):<{key_width}}  |  {val1:>9.2f}%  |  {val2:>9.2f}%  |  {diff:>+8.2f}%")
    
    return "\n".join(lines)

def print_size_distribution_comparison(mol_size_dist, ref_size_dist):
    """Print a comparison of molecule size distributions."""
    print("\nMolecule Size Distribution Comparison (% of molecules):")
    print(format_distribution_table(mol_size_dist, ref_size_dist, 
                                    title1="Input", title2="Reference"))

def print_element_distribution_comparison(element_dist, ref_element_dist):
    """Print a comparison of element distributions."""
    print("\nElement Distribution Comparison (% of all atoms):")
    print(format_distribution_table(element_dist, ref_element_dist,
                                    title1="Input", title2="Reference"))

def group_size_distribution(size_dist, group_size=5):
    """Group molecule sizes into ranges to simplify large distributions."""
    grouped_dist = {}
    
    # Find min and max sizes
    if not size_dist:
        return {}
    
    min_size, max_size = min(size_dist.keys()), max(size_dist.keys())
    
    # Create groups
    for start in range(min_size, max_size + 1, group_size):
        end = start + group_size - 1
        group_key = f"{start}-{end}" if end > start else str(start)
        
        # Sum probabilities for sizes in this range
        group_val = sum(size_dist.get(size, 0) for size in range(start, min(end + 1, max_size + 1)))
        
        if group_val > 0:
            grouped_dist[group_key] = group_val
            
    return grouped_dist

def save_distributions_to_csv(mol_size_dist, ref_size_dist, element_dist, ref_element_dist, input_path):
    """Save distribution data to CSV files."""
    output_dir = "output/evaluate"
    os.makedirs(output_dir, exist_ok=True)
    
    base_filename = os.path.basename(input_path)
    base_name = os.path.splitext(base_filename)[0]
    
    # Save size distribution
    size_path = os.path.join(output_dir, f"{base_name}_size_dist.csv")
    with open(size_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Size', 'Input Distribution', 'Reference Distribution', 'Difference'])
        
        all_sizes = sorted(set(list(mol_size_dist.keys()) + list(ref_size_dist.keys())))
        for size in all_sizes:
            val1 = mol_size_dist.get(size, 0) * 100  # Convert to percentage
            val2 = ref_size_dist.get(size, 0) * 100
            diff = val1 - val2
            writer.writerow([size, f"{val1:.4f}", f"{val2:.4f}", f"{diff:+.4f}"])
    
    # Save element distribution
    elem_path = os.path.join(output_dir, f"{base_name}_element_dist.csv")
    with open(elem_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Element', 'Input Distribution', 'Reference Distribution', 'Difference'])
        
        all_elements = sorted(set(list(element_dist.keys()) + list(ref_element_dist.keys())))
        for elem in all_elements:
            val1 = element_dist.get(elem, 0) * 100  # Convert to percentage
            val2 = ref_element_dist.get(elem, 0) * 100
            diff = val1 - val2
            writer.writerow([elem, f"{val1:.4f}", f"{val2:.4f}", f"{diff:+.4f}"])
    
    return size_path, elem_path

def main():
    parser = argparse.ArgumentParser(description='Evaluate molecules from pickle file')
    parser.add_argument('--input', type=str, default='output/output20250327_1605.pkl', help='Path to the input pickle file')
    parser.add_argument('--reference', type=str, default='dataset/compete/data_all.pkl', 
                        help='Path to the reference pickle file for comparison')
    parser.add_argument('--cores', type=int, default=10, help='Number of CPU cores to use')
    parser.add_argument('--group-sizes', type=int, default=0,
                        help='Group molecule sizes by this number (0 for no grouping)')
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
    
    # Load reference molecules for comparison
    print(f"Loading reference molecules from {args.reference}")
    try:
        reference_molecules = load_molecules(args.reference)
        print(f"Loaded {len(reference_molecules)} reference molecules")
        
        # Calculate distributions
        print("Calculating distributions...")
        mol_size_dist = calculate_molecule_size_distribution(molecules)
        ref_size_dist = calculate_molecule_size_distribution(reference_molecules)
        
        element_dist = calculate_element_distribution(molecules, atom_decoder)
        ref_element_dist = calculate_element_distribution(reference_molecules, atom_decoder)
        
        # Print detailed distribution comparisons
        if args.group_sizes > 0:
            # Group sizes for cleaner presentation if there are many different sizes
            grouped_mol_size_dist = group_size_distribution(mol_size_dist, args.group_sizes)
            grouped_ref_size_dist = group_size_distribution(ref_size_dist, args.group_sizes)
            print_size_distribution_comparison(grouped_mol_size_dist, grouped_ref_size_dist)
            print("\n(Note: Molecule sizes have been grouped for display purposes)")
        else:
            print_size_distribution_comparison(mol_size_dist, ref_size_dist)
            
        print_element_distribution_comparison(element_dist, ref_element_dist)
        
        # Calculate similarities
        size_similarity = calculate_distribution_similarity(mol_size_dist, ref_size_dist)
        element_similarity = calculate_distribution_similarity(element_dist, ref_element_dist)
        
        print(f"\nMolecular size distribution similarity: {size_similarity:.4f}")
        print(f"Element distribution similarity: {element_similarity:.4f}")
        
        # Save distribution data to CSV
        size_path, elem_path = save_distributions_to_csv(
            mol_size_dist, ref_size_dist, 
            element_dist, ref_element_dist,
            args.input
        )
        print(f"Size distribution saved to: {size_path}")
        print(f"Element distribution saved to: {elem_path}")
        
    except Exception as e:
        print(f"Error processing reference molecules: {e}")
        size_similarity = None
        element_similarity = None
    
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
    results = metric.evaluate(data_list)
    
    # Add distribution similarities to results
    if size_similarity is not None:
        results['size_distribution_similarity'] = size_similarity
    if element_similarity is not None:
        results['element_distribution_similarity'] = element_similarity
    
    # Format and print results
    formatted_results = format_results_table(results)
    print("\n" + formatted_results)
    
    # Save results to CSV
    output_path = save_results_to_csv(results, args.input)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
