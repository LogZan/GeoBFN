import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split

charge_map = {
    'H': 1,
    'C': 6,
    'N': 7,
    'O': 8,
    'F': 9,
    'P': 15,
    'S': 16,
    'Cl': 17,
    'Br': 35,
    # Add other elements if needed
}
DEFAULT_CHARGE = 0 # Charge for elements not in the map
ELEMENT_PADDING_VALUE = None # Value to use for padding element lists
CHARGE_PADDING_VALUE = 0    # Value to use for padding charge lists
COORD_PADDING_VALUE = 0.0     # Value to use for padding coordinate arrays

def process_molecular_data(data_list):
    """
    Processes a list of molecular data dictionaries into a single dictionary
    with structured, padded arrays and charge information.

    Args:
        data_list: A list of dictionaries, where each dictionary has keys
                'natoms' (int), 'elements' (list of str), and
                'coordinates' (list of list of float).

    Returns:
        A single dictionary containing:
        - 'natoms': List of atom counts for each molecule.
        - 'elements': List of lists, padded with ELEMENT_PADDING_VALUE.
                    Shape: (num_molecules, max_atoms)
        - 'coordinates': NumPy array of coordinates, padded with COORD_PADDING_VALUE.
                        Shape: (num_molecules, max_atoms, 3)
        - 'charge': List of lists of charges, padded with CHARGE_PADDING_VALUE.
                    Shape: (num_molecules, max_atoms)
    """
    num_molecules = len(data_list)

    if num_molecules == 0:
        return {
            'natoms': [],
            'elements': [],
            'coordinates': np.empty((0, 0, 3), dtype=float), # Shape (0, 0, 3)
            'charge': []
        }

    # 1. Collect natoms and find max_atoms
    natoms_list = [item.get('natoms', 0) for item in data_list]
    max_atoms = max(natoms_list) if natoms_list else 0

    # 2. Initialize padded structures
    # Use lists for elements and charges first, easier to build
    padded_elements = [[ELEMENT_PADDING_VALUE] * max_atoms for _ in range(num_molecules)]
    padded_charges = [[CHARGE_PADDING_VALUE] * max_atoms for _ in range(num_molecules)]
    # Use NumPy array initialized with padding value for coordinates
    padded_coords = np.full((num_molecules, max_atoms, 3), fill_value=COORD_PADDING_VALUE, dtype=float)

    # 3. Populate padded structures
    for i, item_dict in enumerate(data_list):
        current_natoms = item_dict.get('natoms', 0)
        current_elements = item_dict.get('elements', [])
        current_coords = item_dict.get('coordinates', [])

        if current_natoms > 0: # Ensure there are atoms to process
            # Pad elements
            padded_elements[i][:current_natoms] = current_elements

            # Calculate and pad charges
            charges = [charge_map.get(el, DEFAULT_CHARGE) for el in current_elements]
            padded_charges[i][:current_natoms] = charges

            # Pad coordinates
            coords_array = np.array(current_coords, dtype=float)
            # Ensure coords_array is not empty and has the right dimension before assigning
            if coords_array.size > 0 and coords_array.shape == (current_natoms, 3):
                padded_coords[i, :current_natoms, :] = coords_array
            elif current_natoms > 0 : # Handle cases with natoms > 0 but empty/malformed coords
                print(f"Warning: Molecule {i} has {current_natoms} atoms but coordinates are missing or malformed. Padding with zeros.")

    # 4. Assemble final dictionary
    processed_data = {
        'natoms': natoms_list,
        'elements': padded_elements,
        'coordinates': padded_coords,
        'charge': padded_charges
    }

    return processed_data

def split_and_save_data(processed_data, args):
    """
    Split the processed data into train, validation and test sets and save them.
    """
    # Calculate indices for splitting
    indices = np.arange(len(processed_data['natoms']))
    
    # First split: separate train set
    train_idx, temp_idx = train_test_split(
        indices, 
        train_size=args.train_ratio,
        random_state=args.seed
    )
    
    # Second split: separate validation and test sets
    val_size = args.val_ratio / (1 - args.train_ratio)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        random_state=args.seed
    )
    
    # Create directory if it doesn't exist
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Function to split data by indices
    def split_data(data, idx):
        return {
            'natoms': [data['natoms'][i] for i in idx],
            # 'elements': [data['elements'][i] for i in idx],
            'coordinates': data['coordinates'][idx],
            'charges': [data['charge'][i] for i in idx]
        }
    
    # Split and save data
    splits = {
        'train': split_data(processed_data, train_idx),
        'valid': split_data(processed_data, val_idx),
        'test': split_data(processed_data, test_idx)
    }
    
    print("Saving splits with sizes:")
    for split_name, split_data in splits.items():
        print(f"{split_name}: {len(split_data['natoms'])} molecules")
        save_path = os.path.join(args.data_dir, f'{split_name}.npz')
        np.savez_compressed(save_path, **split_data)
        print(f"Saved to {save_path}")

def parse_args():
    parser = argparse.ArgumentParser(description='Process and split molecular data')
    parser.add_argument('--data_dir', type=str, default='dataset/compete',
                       help='Directory containing the data files')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for splitting')
    parser.add_argument('--train_ratio', type=float, default=0.98,
                       help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.01,
                       help='Ratio of validation data')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load data
    data_path = os.path.join(args.data_dir, 'data_all.pkl')
    print(f"Loading data from {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Process data with progress bar
    print("Processing molecular data...")
    data_list = tqdm(data, desc="Processing molecules")
    processed_data = process_molecular_data(data_list)
    
    # Split and save data
    print("Splitting and saving data...")
    split_and_save_data(processed_data, args)