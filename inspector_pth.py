import torch
import sys

def inspect_pth(file_path):
    try:
        # Load the .pth file
        state_dict = torch.load(file_path, map_location=torch.device('cpu'),weights_only=True)
        
        print(f"Successfully loaded {file_path}")
        print(f"Type of loaded object: {type(state_dict)}")
        
        if isinstance(state_dict, dict):
            print("\nKeys in the state_dict:")
            for key in state_dict.keys():
                print(f"- {key}")
                if isinstance(state_dict[key], torch.Tensor):
                    print(f"  Shape: {state_dict[key].shape}")
                    print(f"  Data type: {state_dict[key].dtype}")
        else:
            print("The file does not contain a dictionary (state_dict).")
            print("Contents:", state_dict)
        
    except Exception as e:
        print(f"Error loading the file: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_pth.py <path_to_pth_file>")
    else:
        file_path = sys.argv[1]
        inspect_pth(file_path)