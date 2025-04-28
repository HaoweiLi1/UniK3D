import numpy as np
import cv2
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

def align_depth_scale(pred_depth, gt_depth, valid_mask):
    """
    Simple min-max normalization to align one depth map to another's range.
    
    Args:
        pred_depth: First depth map
        gt_depth: Second depth map (used as reference)
        valid_mask: Mask of valid depth pixels (excluding zeros/missing values)
    
    Returns:
        Scaled depth map
    """
    if valid_mask.sum() == 0:
        return pred_depth.copy()
    
    # Get valid depths only for calculations
    valid_pred = pred_depth[valid_mask]
    valid_gt = gt_depth[valid_mask]
    
    # Get min and max of reference
    gt_min = np.min(valid_gt)
    gt_max = np.max(valid_gt)
    gt_range = gt_max - gt_min
    
    # Get min and max of prediction
    pred_min = np.min(valid_pred)
    pred_max = np.max(valid_pred)
    pred_range = pred_max - pred_min
    
    # Avoid division by zero
    if pred_range < 1e-8:
        return np.ones_like(pred_depth) * gt_min
    
    # Normalize prediction to [0,1] and then scale to reference range
    normalized_pred = (pred_depth - pred_min) / pred_range
    aligned_pred = normalized_pred * gt_range + gt_min
    
    return aligned_pred

def calculate_metrics(pred_depth, gt_depth, valid_mask):
    """
    Calculate depth evaluation metrics.
    
    Args:
        pred_depth: First depth map
        gt_depth: Second depth map (reference)
        valid_mask: Mask of valid depth pixels (excluding zeros/missing values)
    
    Returns:
        Dictionary of metrics
    """
    if valid_mask.sum() == 0:
        return {'rmse': None, 'mae': None, 'abs_rel': None}
    
    # Make sure we're only using valid depth pixels
    valid_pred = pred_depth[valid_mask]
    valid_gt = gt_depth[valid_mask]
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((valid_pred - valid_gt) ** 2))
    
    # Mean Absolute Error
    mae = np.mean(np.abs(valid_pred - valid_gt))
    
    # Absolute Relative Error
    abs_rel = np.mean(np.abs(valid_pred - valid_gt) / np.maximum(valid_gt, 1e-8))
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'abs_rel': float(abs_rel)
    }

def load_depth_map(depth_path):
    """
    Load a depth map from file and convert to float32.
    
    Args:
        depth_path: Path to the depth map image
    
    Returns:
        Depth map as numpy array (float32)
    """
    # Load depth map
    depth_img = Image.open(depth_path)
    depth = np.array(depth_img)
    
    # Convert to float if needed
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    
    # Normalize 16-bit depth maps if needed
    if depth.max() > 1.0:
        depth = depth / 65535.0
    
    return depth

def load_image(image_path):
    """
    Load an RGB image from file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as numpy array
    """
    img = Image.open(image_path)
    return np.array(img)

def visualize_depth_comparison(rgb_path, depth_low_res_path, depth_high_res_path, output_path):
    """
    Create a visualization comparing different resolution depth maps without alignment.
    
    Args:
        rgb_path: Path to the original RGB image
        depth_low_res_path: Path to the low-resolution depth map
        depth_high_res_path: Path to the high-resolution depth map
        output_path: Path to save the visualization
    
    Returns:
        Dictionary of metrics
    """
    # Load RGB image and depth maps
    rgb_img = load_image(rgb_path)
    depth_low_res = load_depth_map(depth_low_res_path)
    depth_high_res = load_depth_map(depth_high_res_path)
    
    # Resize low-res depth to match high-res if needed
    if depth_low_res.shape != depth_high_res.shape:
        depth_low_res = cv2.resize(depth_low_res, 
                                  (depth_high_res.shape[1], depth_high_res.shape[0]), 
                                  interpolation=cv2.INTER_LINEAR)
    
    # Create valid mask - exclude zeros which represent missing values
    valid_mask = (depth_high_res > 0) & (depth_low_res > 0)
    
    # Calculate metrics (no alignment)
    metrics = calculate_metrics(depth_low_res, depth_high_res, valid_mask)
    
    # Calculate absolute difference map directly
    diff_map = np.abs(depth_low_res - depth_high_res)
    
    # Set min and max values for consistent visualization
    valid_min = np.min(depth_high_res[valid_mask]) if valid_mask.sum() > 0 else 0
    valid_max = np.max(depth_high_res[valid_mask]) if valid_mask.sum() > 0 else 1
    
    # Create figure with subplots (2x3)
    plt.figure(figsize=(18, 10))
    
    # Original RGB
    plt.subplot(2, 3, 1)
    plt.imshow(rgb_img)
    plt.title('Original RGB')
    plt.axis('off')
    
    # Low-resolution depth
    plt.subplot(2, 3, 2)
    im = plt.imshow(depth_low_res, cmap='Spectral_r')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'Low Resolution Depth\nShape: {depth_low_res.shape}')
    plt.axis('off')
    
    # Placeholder (empty)
    plt.subplot(2, 3, 3)
    plt.axis('off')
    
    # High-resolution depth
    plt.subplot(2, 3, 5)
    im = plt.imshow(depth_high_res, cmap='Spectral_r')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'High Resolution Depth\nShape: {depth_high_res.shape}')
    plt.axis('off')
    
    # Difference map
    plt.subplot(2, 3, 6)
    # For better visualization, we can use a heatmap for the differences
    max_diff = np.max(diff_map[valid_mask]) if valid_mask.sum() > 0 else 1
    im = plt.imshow(diff_map, cmap='hot', vmin=0, vmax=max_diff)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Absolute Difference')
    plt.axis('off')
    
    # Add overall title with metrics
    plt.suptitle(f"Depth Map Resolution Comparison\n"
                f"RMSE={metrics['rmse']:.3f}, MAE={metrics['mae']:.3f}, AbsRel={metrics['abs_rel']:.3f}",
                fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved visualization to {output_path}")
    
    return metrics

def compare_unik3d_depths(rgb_path, depth_low_res_path, depth_high_res_path, output_path):
    """
    Compare depth maps generated by UniK3D at different resolutions.
    
    Args:
        rgb_path: Path to the original RGB image
        depth_low_res_path: Path to the low-resolution depth map
        depth_high_res_path: Path to the high-resolution depth map
        output_path: Path to save the visualization
    """
    print(f"Comparing depth maps for image: {os.path.basename(rgb_path)}")
    
    try:
        # Create visualization and get metrics
        metrics = visualize_depth_comparison(
            rgb_path, depth_low_res_path, depth_high_res_path, output_path
        )
        
        # Print metrics
        print("\nMetrics between low-res and high-res depths:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  Abs Rel: {metrics['abs_rel']:.4f}")
        
        return metrics
        
    except Exception as e:
        print(f"Error processing depth maps: {str(e)}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare UniK3D depth maps at different resolutions")
    parser.add_argument("--rgb", type=str, required=True, 
                        help="Path to original RGB image")
    parser.add_argument("--low_res", type=str, required=True, 
                        help="Path to low-resolution depth map")
    parser.add_argument("--high_res", type=str, required=True, 
                        help="Path to high-resolution depth map")
    parser.add_argument("--output", type=str, default="depth_comparison.png", 
                        help="Path to save comparison visualization")
    
    args = parser.parse_args()
    
    compare_unik3d_depths(args.rgb, args.low_res, args.high_res, args.output)