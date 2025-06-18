import torch
import os
import argparse
from collections import OrderedDict

def fix_checkpoint(ckpt_path, output_path=None):
    """
    Fix checkpoint file to be compatible with refactored code structure.
    
    Args:
        ckpt_path: Path to the original checkpoint file
        output_path: Path for the fixed checkpoint (optional, defaults to adding '_fixed' suffix)
    """
    if output_path is None:
        base_path, ext = os.path.splitext(ckpt_path)
        output_path = f"{base_path}_fixed{ext}"
    
    print(f"Loading checkpoint from: {ckpt_path}")
    
    # Load the checkpoint
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return False
    
    print("Checkpoint loaded successfully")
    print(f"Original checkpoint keys: {list(checkpoint.keys())}")
    
    # The main issue is likely that the energy_model was part of the main model
    # but now it's moved to the validation callback. We need to remove any
    # energy_model related state_dict entries from the main model.
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        new_state_dict = OrderedDict()
        
        # Filter out energy_model related parameters
        energy_model_keys = []
        for key in state_dict.keys():
            if 'energy_model' in key:
                energy_model_keys.append(key)
                print(f"Removing energy_model key: {key}")
            else:
                new_state_dict[key] = state_dict[key]
        
        if energy_model_keys:
            print(f"Removed {len(energy_model_keys)} energy_model related parameters")
            checkpoint['state_dict'] = new_state_dict
        else:
            print("No energy_model parameters found in checkpoint")
    
    # Remove any energy_model related optimizer states if they exist
    if 'optimizer_states' in checkpoint:
        # This is more complex as optimizer states are indexed by parameter groups
        # For now, we'll keep the existing optimizer state as it should still work
        print("Keeping existing optimizer states")
    
    # Update any hyperparameters that might reference energy_model
    if 'hyper_parameters' in checkpoint:
        hyper_params = checkpoint['hyper_parameters']
        # Remove any energy_model related hyperparameters
        energy_hp_keys = [k for k in hyper_params.keys() if 'energy' in k.lower()]
        for key in energy_hp_keys:
            print(f"Checking hyperparameter: {key}")
            # Only remove if it's clearly energy_model related, not energy loss related
            if 'energy_model' in key.lower():
                print(f"Removing energy_model hyperparameter: {key}")
                del hyper_params[key]
    
    # Ensure the checkpoint structure is clean
    print(f"Final state_dict keys count: {len(checkpoint['state_dict'])}")
    
    # Save the fixed checkpoint
    try:
        torch.save(checkpoint, output_path)
        print(f"Fixed checkpoint saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving fixed checkpoint: {e}")
        return False

def validate_checkpoint(ckpt_path):
    """
    Validate that the checkpoint can be loaded properly.
    """
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print(f"Checkpoint validation successful")
        print(f"Checkpoint contains keys: {list(checkpoint.keys())}")
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            print(f"State dict contains {len(state_dict)} parameters")
            
            # Check for any remaining energy_model references
            energy_keys = [k for k in state_dict.keys() if 'energy_model' in k]
            if energy_keys:
                print(f"Warning: Found {len(energy_keys)} energy_model keys still present:")
                for key in energy_keys[:5]:  # Show first 5
                    print(f"  - {key}")
            else:
                print("No energy_model references found in state_dict")
        
        return True
    except Exception as e:
        print(f"Checkpoint validation failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix checkpoint file after code refactoring")
    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="logs/zengchuanlong_geobfn/compete_round2_energy_normalized_val/checkpoints/epoch=299-energy_loss=nan-v1.ckpt",
        help="Path to the checkpoint file to fix"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Output path for fixed checkpoint (optional)"
    )
    parser.add_argument(
        "--validate_only", 
        action="store_true",
        help="Only validate the checkpoint without fixing"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ckpt_path):
        print(f"Error: Checkpoint file not found: {args.ckpt_path}")
        exit(1)
    
    if args.validate_only:
        print("Validating checkpoint...")
        success = validate_checkpoint(args.ckpt_path)
        if success:
            print("Checkpoint is valid")
        else:
            print("Checkpoint validation failed")
            exit(1)
    else:
        print("Fixing checkpoint...")
        success = fix_checkpoint(args.ckpt_path, args.output_path)
        
        if success:
            # Validate the fixed checkpoint
            output_path = args.output_path
            if output_path is None:
                base_path, ext = os.path.splitext(args.ckpt_path)
                output_path = f"{base_path}_fixed{ext}"
            
            print("\nValidating fixed checkpoint...")
            validate_success = validate_checkpoint(output_path)
            
            if validate_success:
                print(f"\nCheckpoint successfully fixed and validated!")
                print(f"Original: {args.ckpt_path}")
                print(f"Fixed: {output_path}")
                print(f"\nYou can now use the fixed checkpoint: {output_path}")
            else:
                print("Fixed checkpoint validation failed")
                exit(1)
        else:
            print("Failed to fix checkpoint")
            exit(1)
