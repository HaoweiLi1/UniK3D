import torch
import numpy as np
import cv2
import os
from PIL import Image
from unik3d.models import UniK3D
from unik3d.utils.camera import Pinhole, OPENCV, Fisheye624, MEI, Spherical

def normalize_depth_for_visualization(depth_map):
    """
    Normalize depth map to 0-255 for visualization and saving as PNG.
    
    Args:
        depth_map: Numpy array of depth values
    
    Returns:
        Normalized depth map as uint16 (0-65535)
    """
    # Ensure no NaN or inf values
    depth_map = np.nan_to_num(depth_map)
    
    # Get min and max, avoiding division by zero
    min_val = np.min(depth_map)
    max_val = np.max(depth_map)
    
    if max_val - min_val < 1e-8:
        # Return zeros if the depth map is flat (avoid division by zero)
        return np.zeros_like(depth_map, dtype=np.uint16)
    
    # Normalize to 0-65535 for 16-bit PNG
    normalized = ((depth_map - min_val) / (max_val - min_val) * 65535).astype(np.uint16)
    
    return normalized

def predict_single_image(rgb_path, output_path, config_file="configs/eval/vitl.json", camera_path=None):
    """
    Use UniK3D to predict depth from a single RGB image and save the result.
    
    Args:
        rgb_path: Path to the input RGB image
        output_path: Path to save the depth prediction
        config_file: Path to the model configuration file
        camera_path: Optional path to camera parameters JSON file
    """
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize UniK3D model
    print("Initializing UniK3D model...")
    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(device)
    model.eval()
    
    # Load RGB image
    print(f"Loading RGB image: {rgb_path}")
    image = Image.open(rgb_path)
    rgb = np.array(image)
    
    # Check if the image is grayscale (2D) or RGB (3D)
    if len(rgb.shape) == 2:
        # For grayscale images, add a channel dimension
        rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).float().to(device)
        # rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).repeat(3, 1, 1).float().to(device)
    elif len(rgb.shape) == 3:
        # For RGB images, permute as before
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)
        
    else:
        raise ValueError(f"Unexpected image shape: {rgb.shape}")

    # # Convert to tensor and permute dimensions to match model expectations (C, H, W)
    # rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)
    
    # Load camera if specified
    camera = None
    if camera_path:
        print(f"Loading camera parameters from: {camera_path}")
        import json
        
        with open(camera_path, 'r') as f:
            camera_dict = json.load(f)
        
        # Handle the specific format of your JSON camera parameters
        if "camera" in camera_dict:
            cam_params = camera_dict["camera"]
            # Create pinhole camera parameters [fx, fy, cx, cy]
            params = torch.tensor([
                float(cam_params["fx"]), 
                float(cam_params["fy"]), 
                float(cam_params["cx"]), 
                float(cam_params["cy"])
            ]).to(device)
            camera = Pinhole(params=params).to(device)
        elif "params" in camera_dict and "name" in camera_dict:
            # Handle the original expected format
            params = torch.tensor(camera_dict["params"]).to(device)
            name = camera_dict["name"]
            camera = eval(name)(params=params).to(device)
        else:
            print("Warning: Camera JSON format not recognized. Using default camera.")
    
    # Get UniK3D depth prediction
    print("Predicting depth with UniK3D...")
    with torch.no_grad():
        # Run inference
        predictions = model.infer(rgb_tensor, camera=camera)
        
        # Get predicted depth
        pred_depth_map = predictions["depth"]
    
    # Convert to numpy and properly flatten dimensions
    # Force shape to be 2D for PIL
    pred_depth = pred_depth_map.cpu().numpy()
    if len(pred_depth.shape) == 4:  # (1, 1, H, W)
        pred_depth = pred_depth[0, 0]
    
    print(f"Processed predicted depth shape: {pred_depth.shape}")
    
    # Normalize depth for saving as PNG
    normalized_depth = normalize_depth_for_visualization(pred_depth)
    
    # Save the depth map
    print(f"Saving depth prediction to {output_path}")
    depth_img = Image.fromarray(normalized_depth)
    depth_img.save(output_path)
    
    print("Depth prediction complete!")
    return pred_depth

def process_image_with_model(model, rgb_path, output_path, device, camera_path=None):
    """
    Process a single image using an already initialized model.
    Used by predict_folder to avoid loading the model for each image.
    """
    # Load RGB image
    image = Image.open(rgb_path)
    rgb = np.array(image)
    # rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)
    # Check if the image is grayscale (2D) or RGB (3D)
    if len(rgb.shape) == 2:
        # For grayscale images, add a channel dimension
        rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).float().to(device)
        # rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).repeat(3, 1, 1).float().to(device)
    elif len(rgb.shape) == 3:
        # For RGB images, permute as before
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().to(device)
    else:
        raise ValueError(f"Unexpected image shape: {rgb.shape}")
    
    # Load camera if specified
    camera = None
    if camera_path:
        import json
        
        with open(camera_path, 'r') as f:
            camera_dict = json.load(f)
        
        # Handle the specific format of your JSON camera parameters
        if "camera" in camera_dict:
            cam_params = camera_dict["camera"]
            # Create pinhole camera parameters [fx, fy, cx, cy]
            params = torch.tensor([
                float(cam_params["fx"]), 
                float(cam_params["fy"]), 
                float(cam_params["cx"]), 
                float(cam_params["cy"])
            ]).to(device)
            camera = Pinhole(params=params).to(device)
        elif "params" in camera_dict and "name" in camera_dict:
            # Handle the original expected format
            params = torch.tensor(camera_dict["params"]).to(device)
            name = camera_dict["name"]
            camera = eval(name)(params=params).to(device)
        else:
            print("Warning: Camera JSON format not recognized. Using default camera.")
    
    # Get UniK3D depth prediction
    with torch.no_grad():
        # Run inference
        predictions = model.infer(rgb_tensor, camera=camera)
        
        # Get predicted depth
        pred_depth_map = predictions["depth"]
    
    # Convert to numpy and properly flatten dimensions
    pred_depth = pred_depth_map.cpu().numpy()
    if len(pred_depth.shape) == 4:  # (1, 1, H, W)
        pred_depth = pred_depth[0, 0]
    
    # Normalize depth for saving as PNG
    normalized_depth = normalize_depth_for_visualization(pred_depth)
    
    # Save the depth map
    depth_img = Image.fromarray(normalized_depth)
    depth_img.save(output_path)
    
    return pred_depth

def predict_folder(rgb_folder, output_folder='depth_predictions', config_file="configs/eval/vitl.json", camera_path=None):
    """
    Use UniK3D to predict depth for all images in a folder.
    
    Args:
        rgb_folder: Path to folder containing RGB images
        output_folder: Path to save depth predictions
        config_file: Path to the model configuration file
        camera_path: Optional path to camera parameters JSON file
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize UniK3D model
    print("Initializing UniK3D model...")
    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(device)
    model.eval()
    
    # Get list of RGB images
    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Process each image
    for i, rgb_file in enumerate(rgb_files):
        print(f"\nProcessing image {i+1}/{len(rgb_files)}: {rgb_file}")
        
        # Construct RGB path
        rgb_path = os.path.join(rgb_folder, rgb_file)
        
        # Construct output path
        output_filename = os.path.splitext(rgb_file)[0] + '_depth.png'
        output_path = os.path.join(output_folder, output_filename)
        
        # Process single image
        try:
            # We'll use the same model instance to avoid loading it for each image
            process_image_with_model(model, rgb_path, output_path, device, camera_path)
            print(f"Saved depth prediction to {output_path}")
        except Exception as e:
            print(f"Error processing {rgb_file}: {str(e)}")
            continue
    
    print(f"All depth predictions saved to {output_folder}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict depth using UniK3D")
    parser.add_argument("--mode", type=str, choices=["single", "folder"], default="single",
                        help="Prediction mode: single image or folder")
    parser.add_argument("--rgb_path", type=str, help="Path to RGB image (for single mode)")
    parser.add_argument("--output_path", type=str, default="predicted_depth.png", 
                        help="Path to save depth prediction (for single mode)")
    parser.add_argument("--rgb_folder", type=str, help="Path to folder with RGB images (for folder mode)")
    parser.add_argument("--output_folder", type=str, default="depth_predictions", 
                        help="Path to save depth predictions (for folder mode)")
    parser.add_argument("--config_file", type=str, default="configs/eval/vitl.json",
                        help="Path to model configuration file")
    parser.add_argument("--camera_path", type=str, default=None,
                        help="Optional path to camera parameters JSON file")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        if not args.rgb_path:
            print("Error: rgb_path is required for single mode")
            parser.print_help()
            exit(1)
        
        predict_single_image(args.rgb_path, args.output_path, args.config_file, args.camera_path)
    else:  # folder mode
        if not args.rgb_folder:
            print("Error: rgb_folder is required for folder mode")
            parser.print_help()
            exit(1)
        
        predict_folder(args.rgb_folder, args.output_folder, args.config_file, args.camera_path)