# import numpy as np
# import cv2
# import os
# import json
# from PIL import Image
# import matplotlib.pyplot as plt

# def align_depth_scale(pred_depth, gt_depth, valid_mask):
#     """
#     Simple min-max normalization to align predicted depth to ground truth depth range.
    
#     Args:
#         pred_depth: Predicted depth map
#         gt_depth: Ground truth depth map
#         valid_mask: Mask of valid depth pixels (excluding zeros/missing values)
    
#     Returns:
#         Scaled predicted depth map
#     """
#     if valid_mask.sum() == 0:
#         return pred_depth.copy()
    
#     # Get valid depths only for calculations
#     valid_pred = pred_depth[valid_mask]
#     valid_gt = gt_depth[valid_mask]
    
#     # Get min and max of ground truth
#     gt_min = np.min(valid_gt)
#     gt_max = np.max(valid_gt)
#     gt_range = gt_max - gt_min
    
#     # Get min and max of prediction
#     pred_min = np.min(valid_pred)
#     pred_max = np.max(valid_pred)
#     pred_range = pred_max - pred_min
    
#     # Avoid division by zero
#     if pred_range < 1e-8:
#         return np.ones_like(pred_depth) * gt_min
    
#     # Normalize prediction to [0,1] and then scale to ground truth range
#     normalized_pred = (pred_depth - pred_min) / pred_range
#     aligned_pred = normalized_pred * gt_range + gt_min
    
#     return aligned_pred

# def calculate_metrics(pred_depth, gt_depth, valid_mask):
#     """
#     Calculate depth evaluation metrics.
    
#     Args:
#         pred_depth: Predicted depth map
#         gt_depth: Ground truth depth map
#         valid_mask: Mask of valid depth pixels (excluding zeros/missing values)
    
#     Returns:
#         Dictionary of metrics
#     """
#     if valid_mask.sum() == 0:
#         return {'rmse': None, 'mae': None, 'abs_rel': None}
    
#     # Make sure we're only using valid depth pixels (non-zero in ground truth)
#     valid_pred = pred_depth[valid_mask]
#     valid_gt = gt_depth[valid_mask]
    
#     # Root Mean Square Error
#     rmse = np.sqrt(np.mean((valid_pred - valid_gt) ** 2))
    
#     # Mean Absolute Error
#     mae = np.mean(np.abs(valid_pred - valid_gt))
    
#     # Absolute Relative Error
#     abs_rel = np.mean(np.abs(valid_pred - valid_gt) / np.maximum(valid_gt, 1e-8))
    
#     return {
#         'rmse': float(rmse),
#         'mae': float(mae),
#         'abs_rel': float(abs_rel)
#     }

# def load_image(image_path):
#     """
#     Load an image from file.
    
#     Args:
#         image_path: Path to the image file
    
#     Returns:
#         Image as numpy array
#     """
#     img = Image.open(image_path)
#     return np.array(img)

# def load_depth_map(depth_path):
#     """
#     Load a depth map from file and convert to float32.
    
#     Args:
#         depth_path: Path to the depth map image
    
#     Returns:
#         Depth map as numpy array (float32)
#     """
#     # Load depth map
#     gt_depth_img = Image.open(depth_path)
#     gt_depth = np.array(gt_depth_img)
    
#     # Convert to float if needed
#     if gt_depth.dtype != np.float32:
#         gt_depth = gt_depth.astype(np.float32)
    
#     # Normalize ground truth depth if needed (adjust based on your depth format)
#     # This is a common normalization, but might need to be adjusted for specific data
#     if gt_depth.max() > 1.0:
#         gt_depth = gt_depth / 65535.0  # For 16-bit depth maps
    
#     return gt_depth

# def visualize_depth_comparison(rgb_path, seg_mask_path, pred_depth_rgb_path, pred_depth_seg_path, 
#                              gt_depth_path, output_path):
#     """
#     Create a visualization comparing different depth maps in a 2x4 grid.
    
#     Args:
#         rgb_path: Path to the original RGB image
#         seg_mask_path: Path to the segmentation mask image
#         pred_depth_rgb_path: Path to the predicted depth from RGB
#         pred_depth_seg_path: Path to the predicted depth from segmentation
#         gt_depth_path: Path to the ground truth depth map
#         output_path: Path to save the visualization
    
#     Returns:
#         Dictionary of metrics for both prediction methods
#     """
#     # Load all images
#     rgb_img = load_image(rgb_path)
#     seg_mask_img = load_image(seg_mask_path)
    
#     # Load depth maps
#     pred_depth_rgb = load_depth_map(pred_depth_rgb_path)
#     pred_depth_seg = load_depth_map(pred_depth_seg_path)
#     gt_depth = load_depth_map(gt_depth_path)
    
#     # Resize predictions to match ground truth if needed
#     if pred_depth_rgb.shape != gt_depth.shape:
#         pred_depth_rgb = cv2.resize(pred_depth_rgb, (gt_depth.shape[1], gt_depth.shape[0]), 
#                                    interpolation=cv2.INTER_LINEAR)
    
#     if pred_depth_seg.shape != gt_depth.shape:
#         pred_depth_seg = cv2.resize(pred_depth_seg, (gt_depth.shape[1], gt_depth.shape[0]), 
#                                    interpolation=cv2.INTER_LINEAR)
    
#     # Create valid mask - exclude zeros which represent missing values
#     valid_mask = (gt_depth > 0)  # Exclude missing depth pixels (zeros)
    
#     # Calculate metrics for RGB-based prediction
#     metrics_rgb_before = calculate_metrics(pred_depth_rgb, gt_depth, valid_mask)
#     pred_depth_rgb_aligned = align_depth_scale(pred_depth_rgb, gt_depth, valid_mask)
#     metrics_rgb_after = calculate_metrics(pred_depth_rgb_aligned, gt_depth, valid_mask)
    
#     # Calculate metrics for segmentation-based prediction
#     metrics_seg_before = calculate_metrics(pred_depth_seg, gt_depth, valid_mask)
#     pred_depth_seg_aligned = align_depth_scale(pred_depth_seg, gt_depth, valid_mask)
#     metrics_seg_after = calculate_metrics(pred_depth_seg_aligned, gt_depth, valid_mask)
    
#     # Create figure with subplots (2 rows, 4 columns)
#     plt.figure(figsize=(20, 10))
    
#     # Get the min/max values of GT for consistent colorbar
#     gt_valid_min = np.min(gt_depth[valid_mask]) if valid_mask.sum() > 0 else 0
#     gt_valid_max = np.max(gt_depth[valid_mask]) if valid_mask.sum() > 0 else 1
    
#     # Top row - RGB method
#     # RGB input
#     plt.subplot(2, 4, 1)
#     plt.imshow(rgb_img)
#     plt.title('RGB')
#     plt.axis('off')
    
#     # Predicted Depth via RGB before alignment
#     plt.subplot(2, 4, 2)
#     im = plt.imshow(pred_depth_rgb, cmap='Spectral_r')
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.title('Depth via RGB')
#     plt.axis('off')
    
#     # Predicted Depth via RGB after alignment
#     plt.subplot(2, 4, 3)
#     im = plt.imshow(pred_depth_rgb_aligned, cmap='Spectral_r', vmin=gt_valid_min, vmax=gt_valid_max)
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.title('Depth via RGB\n(scale-aligned)')
#     plt.axis('off')
    
#     # Ground Truth Depth (top row)
#     plt.subplot(2, 4, 4)
#     im = plt.imshow(gt_depth, cmap='Spectral_r', vmin=gt_valid_min, vmax=gt_valid_max)
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.title('GT Depth')
#     plt.axis('off')
    
#     # Bottom row - Segmentation method
#     # Segmentation mask
#     plt.subplot(2, 4, 5)
#     plt.imshow(seg_mask_img)
#     plt.title('Seg Mask')
#     plt.axis('off')
    
#     # Predicted Depth via Segmentation before alignment
#     plt.subplot(2, 4, 6)
#     im = plt.imshow(pred_depth_seg, cmap='Spectral_r')
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.title('Depth via Seg\nMask')
#     plt.axis('off')
    
#     # Predicted Depth via Segmentation after alignment
#     plt.subplot(2, 4, 7)
#     im = plt.imshow(pred_depth_seg_aligned, cmap='Spectral_r', vmin=gt_valid_min, vmax=gt_valid_max)
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.title('Depth via Seg\nMask (scale-aligned)')
#     plt.axis('off')
    
#     # Ground Truth Depth (bottom row)
#     plt.subplot(2, 4, 8)
#     im = plt.imshow(gt_depth, cmap='Spectral_r', vmin=gt_valid_min, vmax=gt_valid_max)
#     plt.colorbar(im, fraction=0.046, pad=0.04)
#     plt.title('GT Depth')
#     plt.axis('off')
    
#     # Add overall title with metrics
#     plt.suptitle(f"Depth Quantitative Metrics After Align\nRGB: RMSE={metrics_rgb_after['rmse']:.3f}, AbsRel={metrics_rgb_after['abs_rel']:.3f}\nSeg: RMSE={metrics_seg_after['rmse']:.3f}, AbsRel={metrics_seg_after['abs_rel']:.3f}", 
#                 fontsize=16)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.95])
#     plt.savefig(output_path)
#     plt.close()
    
#     # Combine metrics
#     metrics = {
#         'rgb': {
#             'before_align': metrics_rgb_before,
#             'after_align': metrics_rgb_after
#         },
#         'segmentation': {
#             'before_align': metrics_seg_before,
#             'after_align': metrics_seg_after
#         }
#     }
    
#     return metrics

# def evaluate_single_set(rgb_path, seg_mask_path, pred_depth_rgb_path, pred_depth_seg_path, 
#                        gt_depth_path, output_path='result.png'):
#     """
#     Evaluate a single set of images and depth maps.
    
#     Args:
#         rgb_path: Path to the RGB image
#         seg_mask_path: Path to the segmentation mask image
#         pred_depth_rgb_path: Path to the predicted depth from RGB
#         pred_depth_seg_path: Path to the predicted depth from segmentation
#         gt_depth_path: Path to the ground truth depth map
#         output_path: Path to save visualization result
    
#     Returns:
#         Dictionary containing evaluation metrics
#     """
#     print(f"Evaluating depth maps for: {os.path.basename(rgb_path)}")
    
#     try:
#         # Create visualization and get metrics
#         metrics = visualize_depth_comparison(
#             rgb_path, seg_mask_path, pred_depth_rgb_path, pred_depth_seg_path, 
#             gt_depth_path, output_path
#         )
        
#         # Print metrics
#         print("\nMetrics for RGB-based Depth:")
#         print("  Before Alignment:")
#         print(f"    RMSE: {metrics['rgb']['before_align']['rmse']:.4f}")
#         print(f"    MAE: {metrics['rgb']['before_align']['mae']:.4f}")
#         print(f"    Abs Rel: {metrics['rgb']['before_align']['abs_rel']:.4f}")
#         print("  After Alignment:")
#         print(f"    RMSE: {metrics['rgb']['after_align']['rmse']:.4f}")
#         print(f"    MAE: {metrics['rgb']['after_align']['mae']:.4f}")
#         print(f"    Abs Rel: {metrics['rgb']['after_align']['abs_rel']:.4f}")
        
#         print("\nMetrics for Segmentation-based Depth:")
#         print("  Before Alignment:")
#         print(f"    RMSE: {metrics['segmentation']['before_align']['rmse']:.4f}")
#         print(f"    MAE: {metrics['segmentation']['before_align']['mae']:.4f}")
#         print(f"    Abs Rel: {metrics['segmentation']['before_align']['abs_rel']:.4f}")
#         print("  After Alignment:")
#         print(f"    RMSE: {metrics['segmentation']['after_align']['rmse']:.4f}")
#         print(f"    MAE: {metrics['segmentation']['after_align']['mae']:.4f}")
#         print(f"    Abs Rel: {metrics['segmentation']['after_align']['abs_rel']:.4f}")
        
#         print(f"Saved visualization to {output_path}")
#         return metrics
        
#     except Exception as e:
#         print(f"Error processing images: {str(e)}")
#         return None

# def evaluate_folder(rgb_folder, seg_mask_folder, pred_depth_rgb_folder, pred_depth_seg_folder, 
#                    gt_depth_folder, output_folder='results'):
#     """
#     Evaluate depth predictions against ground truth depth maps for all images in folders.
    
#     Args:
#         rgb_folder: Path to folder containing RGB images
#         seg_mask_folder: Path to folder containing segmentation masks
#         pred_depth_rgb_folder: Path to folder containing predicted depths from RGB
#         pred_depth_seg_folder: Path to folder containing predicted depths from segmentation
#         gt_depth_folder: Path to folder containing ground truth depth maps
#         output_folder: Path to save evaluation results
#     """
#     # Create output folder if it doesn't exist
#     os.makedirs(output_folder, exist_ok=True)
    
#     # Get list of RGB images
#     rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
#     # Dictionary to store all metrics
#     all_metrics = {}
    
#     # Process each image
#     for i, rgb_file in enumerate(rgb_files):
#         print(f"\nProcessing image {i+1}/{len(rgb_files)}: {rgb_file}")
        
#         # Construct RGB path
#         rgb_path = os.path.join(rgb_folder, rgb_file)
        
#         # Construct segmentation mask path
#         seg_mask_file = rgb_file.replace('frame', 'instance_color').replace('.jpg', '.png').replace('.jpeg', '.png')
#         seg_mask_path = os.path.join(seg_mask_folder, seg_mask_file)
        
#         # Construct predicted depth paths
#         pred_depth_rgb_file = os.path.splitext(rgb_file)[0] + '_depth.png'
#         pred_depth_rgb_path = os.path.join(pred_depth_rgb_folder, pred_depth_rgb_file)
        
#         pred_depth_seg_file = seg_mask_file.replace('.png', '_depth.png')
#         pred_depth_seg_path = os.path.join(pred_depth_seg_folder, pred_depth_seg_file)
        
#         # Construct ground truth depth path
#         gt_depth_file = rgb_file.replace('frame', 'depth').replace('.jpg', '.png').replace('.jpeg', '.png')
#         gt_depth_path = os.path.join(gt_depth_folder, gt_depth_file)
        
#         # Check if all files exist
#         files_to_check = [
#             (rgb_path, "RGB image"),
#             (seg_mask_path, "segmentation mask"),
#             (pred_depth_rgb_path, "predicted depth from RGB"),
#             (pred_depth_seg_path, "predicted depth from segmentation"),
#             (gt_depth_path, "ground truth depth")
#         ]
        
#         missing_files = False
#         for file_path, file_type in files_to_check:
#             if not os.path.exists(file_path):
#                 print(f"Warning: Missing {file_type} file: {file_path}")
#                 missing_files = True
                
#         if missing_files:
#             print(f"Skipping evaluation for {rgb_file} due to missing files.")
#             continue
        
#         # Output path for this image
#         output_path = os.path.join(output_folder, f"comparison_{i:03d}.png")
        
#         # Evaluate single set
#         metrics = evaluate_single_set(
#             rgb_path, seg_mask_path, pred_depth_rgb_path, pred_depth_seg_path, 
#             gt_depth_path, output_path
#         )
        
#         if metrics:
#             # Store metrics for this image
#             all_metrics[rgb_file] = metrics
    
#     # Calculate average metrics if we have results
#     if all_metrics:
#         avg_rgb_before = {
#             'rmse': np.mean([m['rgb']['before_align']['rmse'] for m in all_metrics.values() if m['rgb']['before_align']['rmse'] is not None]),
#             'mae': np.mean([m['rgb']['before_align']['mae'] for m in all_metrics.values() if m['rgb']['before_align']['mae'] is not None]),
#             'abs_rel': np.mean([m['rgb']['before_align']['abs_rel'] for m in all_metrics.values() if m['rgb']['before_align']['abs_rel'] is not None])
#         }
        
#         avg_rgb_after = {
#             'rmse': np.mean([m['rgb']['after_align']['rmse'] for m in all_metrics.values() if m['rgb']['after_align']['rmse'] is not None]),
#             'mae': np.mean([m['rgb']['after_align']['mae'] for m in all_metrics.values() if m['rgb']['after_align']['mae'] is not None]),
#             'abs_rel': np.mean([m['rgb']['after_align']['abs_rel'] for m in all_metrics.values() if m['rgb']['after_align']['abs_rel'] is not None])
#         }
        
#         avg_seg_before = {
#             'rmse': np.mean([m['segmentation']['before_align']['rmse'] for m in all_metrics.values() if m['segmentation']['before_align']['rmse'] is not None]),
#             'mae': np.mean([m['segmentation']['before_align']['mae'] for m in all_metrics.values() if m['segmentation']['before_align']['mae'] is not None]),
#             'abs_rel': np.mean([m['segmentation']['before_align']['abs_rel'] for m in all_metrics.values() if m['segmentation']['before_align']['abs_rel'] is not None])
#         }
        
#         avg_seg_after = {
#             'rmse': np.mean([m['segmentation']['after_align']['rmse'] for m in all_metrics.values() if m['segmentation']['after_align']['rmse'] is not None]),
#             'mae': np.mean([m['segmentation']['after_align']['mae'] for m in all_metrics.values() if m['segmentation']['after_align']['mae'] is not None]),
#             'abs_rel': np.mean([m['segmentation']['after_align']['abs_rel'] for m in all_metrics.values() if m['segmentation']['after_align']['abs_rel'] is not None])
#         }
        
#         # Add average metrics to the dictionary
#         all_metrics['average'] = {
#             'rgb': {
#                 'before_align': avg_rgb_before,
#                 'after_align': avg_rgb_after
#             },
#             'segmentation': {
#                 'before_align': avg_seg_before,
#                 'after_align': avg_seg_after
#             }
#         }
        
#         # Print average metrics
#         print("\nAverage Metrics for RGB-based Depth:")
#         print("  Before Alignment:")
#         print(f"    RMSE: {avg_rgb_before['rmse']:.4f}")
#         print(f"    MAE: {avg_rgb_before['mae']:.4f}")
#         print(f"    Abs Rel: {avg_rgb_before['abs_rel']:.4f}")
#         print("  After Alignment:")
#         print(f"    RMSE: {avg_rgb_after['rmse']:.4f}")
#         print(f"    MAE: {avg_rgb_after['mae']:.4f}")
#         print(f"    Abs Rel: {avg_rgb_after['abs_rel']:.4f}")
        
#         print("\nAverage Metrics for Segmentation-based Depth:")
#         print("  Before Alignment:")
#         print(f"    RMSE: {avg_seg_before['rmse']:.4f}")
#         print(f"    MAE: {avg_seg_before['mae']:.4f}")
#         print(f"    Abs Rel: {avg_seg_before['abs_rel']:.4f}")
#         print("  After Alignment:")
#         print(f"    RMSE: {avg_seg_after['rmse']:.4f}")
#         print(f"    MAE: {avg_seg_after['mae']:.4f}")
#         print(f"    Abs Rel: {avg_seg_after['abs_rel']:.4f}")
    
#     # Save all metrics as JSON
#     metrics_path = os.path.join(output_folder, "all_metrics.json")
#     with open(metrics_path, 'w') as f:
#         json.dump(all_metrics, f, indent=4)
#     print(f"Saved all metrics to {metrics_path}")

# # Example usage
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Evaluate depth predictions against ground truth")
#     parser.add_argument("--mode", type=str, choices=["single", "folder"], default="single",
#                         help="Evaluation mode: single image set or folder")
    
#     # Single mode arguments
#     parser.add_argument("--rgb_path", type=str, help="Path to RGB image (for single mode)")
#     parser.add_argument("--seg_mask_path", type=str, help="Path to segmentation mask (for single mode)")
#     parser.add_argument("--pred_depth_rgb_path", type=str, help="Path to predicted depth from RGB (for single mode)")
#     parser.add_argument("--pred_depth_seg_path", type=str, help="Path to predicted depth from segmentation (for single mode)")
#     parser.add_argument("--gt_depth_path", type=str, help="Path to ground truth depth image (for single mode)")
#     parser.add_argument("--output_path", type=str, default="depth_comparison.png", 
#                         help="Path to save output visualization (for single mode)")
    
#     # Folder mode arguments
#     parser.add_argument("--rgb_folder", type=str, help="Path to folder with RGB images (for folder mode)")
#     parser.add_argument("--seg_mask_folder", type=str, help="Path to folder with segmentation masks (for folder mode)")
#     parser.add_argument("--pred_depth_rgb_folder", type=str, help="Path to folder with predicted depths from RGB (for folder mode)")
#     parser.add_argument("--pred_depth_seg_folder", type=str, help="Path to folder with predicted depths from segmentation (for folder mode)")
#     parser.add_argument("--gt_depth_folder", type=str, help="Path to folder with ground truth depth maps (for folder mode)")
#     parser.add_argument("--output_folder", type=str, default="evaluation_results", 
#                         help="Path to save output visualizations (for folder mode)")
    
#     args = parser.parse_args()
    
#     if args.mode == "single":
#         required_args = ["rgb_path", "seg_mask_path", "pred_depth_rgb_path", "pred_depth_seg_path", "gt_depth_path"]
#         missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        
#         if missing_args:
#             print(f"Error: Missing required arguments for single mode: {', '.join(missing_args)}")
#             parser.print_help()
#             exit(1)
        
#         evaluate_single_set(
#             args.rgb_path, args.seg_mask_path, args.pred_depth_rgb_path, 
#             args.pred_depth_seg_path, args.gt_depth_path, args.output_path
#         )
#     else:  # folder mode
#         required_args = ["rgb_folder", "seg_mask_folder", "pred_depth_rgb_folder", 
#                         "pred_depth_seg_folder", "gt_depth_folder"]
#         missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        
#         if missing_args:
#             print(f"Error: Missing required arguments for folder mode: {', '.join(missing_args)}")
#             parser.print_help()
#             exit(1)
        
#         evaluate_folder(
#             args.rgb_folder, args.seg_mask_folder, args.pred_depth_rgb_folder,
#             args.pred_depth_seg_folder, args.gt_depth_folder, args.output_folder
#         )

import numpy as np
import cv2
import os
import json
from PIL import Image
import matplotlib.pyplot as plt

def align_depth_scale(pred_depth, gt_depth, valid_mask):
    """
    Simple min-max normalization to align predicted depth to ground truth depth range.
    
    Args:
        pred_depth: Predicted depth map
        gt_depth: Ground truth depth map
        valid_mask: Mask of valid depth pixels (excluding zeros/missing values)
    
    Returns:
        Scaled predicted depth map
    """
    if valid_mask.sum() == 0:
        return pred_depth.copy()
    
    # Get valid depths only for calculations
    valid_pred = pred_depth[valid_mask]
    valid_gt = gt_depth[valid_mask]
    
    # Get min and max of ground truth
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
    
    # Normalize prediction to [0,1] and then scale to ground truth range
    normalized_pred = (pred_depth - pred_min) / pred_range
    aligned_pred = normalized_pred * gt_range + gt_min
    
    return aligned_pred

def calculate_metrics(pred_depth, gt_depth, valid_mask):
    """
    Calculate depth evaluation metrics.
    
    Args:
        pred_depth: Predicted depth map
        gt_depth: Ground truth depth map
        valid_mask: Mask of valid depth pixels (excluding zeros/missing values)
    
    Returns:
        Dictionary of metrics
    """
    if valid_mask.sum() == 0:
        return {'rmse': None, 'mae': None, 'abs_rel': None}
    
    # Make sure we're only using valid depth pixels (non-zero in ground truth)
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

def load_image(image_path):
    """
    Load an image from file.
    
    Args:
        image_path: Path to the image file
    
    Returns:
        Image as numpy array
    """
    img = Image.open(image_path)
    return np.array(img)

def load_depth_map(depth_path):
    """
    Load a depth map from file and convert to float32.
    
    Args:
        depth_path: Path to the depth map image
    
    Returns:
        Depth map as numpy array (float32)
    """
    # Load depth map
    gt_depth_img = Image.open(depth_path)
    gt_depth = np.array(gt_depth_img)
    
    # Convert to float if needed
    if gt_depth.dtype != np.float32:
        gt_depth = gt_depth.astype(np.float32)
    
    # Normalize ground truth depth if needed (adjust based on your depth format)
    # This is a common normalization, but might need to be adjusted for specific data
    if gt_depth.max() > 1.0:
        gt_depth = gt_depth / 65535.0  # For 16-bit depth maps
    
    return gt_depth

def visualize_depth_comparison(rgb_path, seg_mask_path, inst_id_path, 
                             pred_depth_rgb_path, pred_depth_seg_path, pred_depth_inst_path,
                             gt_depth_path, output_path):
    """
    Create a visualization comparing different depth maps in a 3x4 grid.
    
    Args:
        rgb_path: Path to the original RGB image
        seg_mask_path: Path to the segmentation mask image
        inst_id_path: Path to the instance ID image
        pred_depth_rgb_path: Path to the predicted depth from RGB
        pred_depth_seg_path: Path to the predicted depth from segmentation
        pred_depth_inst_path: Path to the predicted depth from instance IDs
        gt_depth_path: Path to the ground truth depth map
        output_path: Path to save the visualization
    
    Returns:
        Dictionary of metrics for all prediction methods
    """
    # Load all images
    rgb_img = load_image(rgb_path)
    seg_mask_img = load_image(seg_mask_path)
    inst_id_img = load_image(inst_id_path)
    
    # Load depth maps
    pred_depth_rgb = load_depth_map(pred_depth_rgb_path)
    pred_depth_seg = load_depth_map(pred_depth_seg_path)
    pred_depth_inst = load_depth_map(pred_depth_inst_path)
    gt_depth = load_depth_map(gt_depth_path)
    
    # Resize predictions to match ground truth if needed
    if pred_depth_rgb.shape != gt_depth.shape:
        pred_depth_rgb = cv2.resize(pred_depth_rgb, (gt_depth.shape[1], gt_depth.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
    
    if pred_depth_seg.shape != gt_depth.shape:
        pred_depth_seg = cv2.resize(pred_depth_seg, (gt_depth.shape[1], gt_depth.shape[0]), 
                                   interpolation=cv2.INTER_LINEAR)
    
    if pred_depth_inst.shape != gt_depth.shape:
        pred_depth_inst = cv2.resize(pred_depth_inst, (gt_depth.shape[1], gt_depth.shape[0]), 
                                    interpolation=cv2.INTER_LINEAR)
    
    # Create valid mask - exclude zeros which represent missing values
    valid_mask = (gt_depth > 0)  # Exclude missing depth pixels (zeros)
    
    # Calculate metrics for RGB-based prediction
    metrics_rgb_before = calculate_metrics(pred_depth_rgb, gt_depth, valid_mask)
    pred_depth_rgb_aligned = align_depth_scale(pred_depth_rgb, gt_depth, valid_mask)
    metrics_rgb_after = calculate_metrics(pred_depth_rgb_aligned, gt_depth, valid_mask)
    
    # Calculate metrics for segmentation-based prediction
    metrics_seg_before = calculate_metrics(pred_depth_seg, gt_depth, valid_mask)
    pred_depth_seg_aligned = align_depth_scale(pred_depth_seg, gt_depth, valid_mask)
    metrics_seg_after = calculate_metrics(pred_depth_seg_aligned, gt_depth, valid_mask)
    
    # Calculate metrics for instance ID-based prediction
    metrics_inst_before = calculate_metrics(pred_depth_inst, gt_depth, valid_mask)
    pred_depth_inst_aligned = align_depth_scale(pred_depth_inst, gt_depth, valid_mask)
    metrics_inst_after = calculate_metrics(pred_depth_inst_aligned, gt_depth, valid_mask)
    
    # Create figure with subplots (3 rows, 4 columns)
    plt.figure(figsize=(20, 15))
    
    # Get the min/max values of GT for consistent colorbar
    gt_valid_min = np.min(gt_depth[valid_mask]) if valid_mask.sum() > 0 else 0
    gt_valid_max = np.max(gt_depth[valid_mask]) if valid_mask.sum() > 0 else 1
    
    # Top row - RGB method
    # RGB input
    plt.subplot(3, 4, 1)
    plt.imshow(rgb_img)
    plt.title('RGB')
    plt.axis('off')
    
    # Predicted Depth via RGB before alignment
    plt.subplot(3, 4, 2)
    im = plt.imshow(pred_depth_rgb, cmap='Spectral_r')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Depth via RGB')
    plt.axis('off')
    
    # Predicted Depth via RGB after alignment
    plt.subplot(3, 4, 3)
    im = plt.imshow(pred_depth_rgb_aligned, cmap='Spectral_r', vmin=gt_valid_min, vmax=gt_valid_max)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Depth via RGB\n(scale-aligned)')
    plt.axis('off')
    
    # Ground Truth Depth (top row)
    plt.subplot(3, 4, 4)
    im = plt.imshow(gt_depth, cmap='Spectral_r', vmin=gt_valid_min, vmax=gt_valid_max)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('GT Depth')
    plt.axis('off')
    
    # Middle row - Segmentation method
    # Segmentation mask
    plt.subplot(3, 4, 5)
    plt.imshow(seg_mask_img)
    plt.title('Seg Mask')
    plt.axis('off')
    
    # Predicted Depth via Segmentation before alignment
    plt.subplot(3, 4, 6)
    im = plt.imshow(pred_depth_seg, cmap='Spectral_r')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Depth via Seg\nMask')
    plt.axis('off')
    
    # Predicted Depth via Segmentation after alignment
    plt.subplot(3, 4, 7)
    im = plt.imshow(pred_depth_seg_aligned, cmap='Spectral_r', vmin=gt_valid_min, vmax=gt_valid_max)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Depth via Seg\nMask (scale-aligned)')
    plt.axis('off')
    
    # Ground Truth Depth (middle row)
    plt.subplot(3, 4, 8)
    im = plt.imshow(gt_depth, cmap='Spectral_r', vmin=gt_valid_min, vmax=gt_valid_max)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('GT Depth')
    plt.axis('off')
    
    # Bottom row - Instance ID method
    # Instance ID image
    plt.subplot(3, 4, 9)
    plt.imshow(inst_id_img, cmap='gray')
    plt.title('Instance ID')
    plt.axis('off')
    
    # Predicted Depth via Instance ID before alignment
    plt.subplot(3, 4, 10)
    im = plt.imshow(pred_depth_inst, cmap='Spectral_r')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Depth via Inst\nID')
    plt.axis('off')
    
    # Predicted Depth via Instance ID after alignment
    plt.subplot(3, 4, 11)
    im = plt.imshow(pred_depth_inst_aligned, cmap='Spectral_r', vmin=gt_valid_min, vmax=gt_valid_max)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('Depth via Inst\nID (scale-aligned)')
    plt.axis('off')
    
    # Ground Truth Depth (bottom row)
    plt.subplot(3, 4, 12)
    im = plt.imshow(gt_depth, cmap='Spectral_r', vmin=gt_valid_min, vmax=gt_valid_max)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title('GT Depth')
    plt.axis('off')
    
    # Add overall title with metrics
    plt.suptitle(f"Depth Quantitative Metrics After Align\n"
                f"RGB: RMSE={metrics_rgb_after['rmse']:.3f}, AbsRel={metrics_rgb_after['abs_rel']:.3f}\n"
                f"Seg: RMSE={metrics_seg_after['rmse']:.3f}, AbsRel={metrics_seg_after['abs_rel']:.3f}\n"
                f"Inst: RMSE={metrics_inst_after['rmse']:.3f}, AbsRel={metrics_inst_after['abs_rel']:.3f}",
                fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path)
    plt.close()
    
    # Combine metrics
    metrics = {
        'rgb': {
            'before_align': metrics_rgb_before,
            'after_align': metrics_rgb_after
        },
        'segmentation': {
            'before_align': metrics_seg_before,
            'after_align': metrics_seg_after
        },
        'instance': {
            'before_align': metrics_inst_before,
            'after_align': metrics_inst_after
        }
    }
    
    return metrics

def evaluate_single_set(rgb_path, seg_mask_path, inst_id_path,
                       pred_depth_rgb_path, pred_depth_seg_path, pred_depth_inst_path,
                       gt_depth_path, output_path='result.png'):
    """
    Evaluate a single set of images and depth maps.
    
    Args:
        rgb_path: Path to the RGB image
        seg_mask_path: Path to the segmentation mask image
        inst_id_path: Path to the instance ID image
        pred_depth_rgb_path: Path to the predicted depth from RGB
        pred_depth_seg_path: Path to the predicted depth from segmentation
        pred_depth_inst_path: Path to the predicted depth from instance IDs
        gt_depth_path: Path to the ground truth depth map
        output_path: Path to save visualization result
    
    Returns:
        Dictionary containing evaluation metrics
    """
    print(f"Evaluating depth maps for: {os.path.basename(rgb_path)}")
    
    try:
        # Create visualization and get metrics
        metrics = visualize_depth_comparison(
            rgb_path, seg_mask_path, inst_id_path,
            pred_depth_rgb_path, pred_depth_seg_path, pred_depth_inst_path,
            gt_depth_path, output_path
        )
        
        # Print metrics
        print("\nMetrics for RGB-based Depth:")
        print("  Before Alignment:")
        print(f"    RMSE: {metrics['rgb']['before_align']['rmse']:.4f}")
        print(f"    MAE: {metrics['rgb']['before_align']['mae']:.4f}")
        print(f"    Abs Rel: {metrics['rgb']['before_align']['abs_rel']:.4f}")
        print("  After Alignment:")
        print(f"    RMSE: {metrics['rgb']['after_align']['rmse']:.4f}")
        print(f"    MAE: {metrics['rgb']['after_align']['mae']:.4f}")
        print(f"    Abs Rel: {metrics['rgb']['after_align']['abs_rel']:.4f}")
        
        print("\nMetrics for Segmentation-based Depth:")
        print("  Before Alignment:")
        print(f"    RMSE: {metrics['segmentation']['before_align']['rmse']:.4f}")
        print(f"    MAE: {metrics['segmentation']['before_align']['mae']:.4f}")
        print(f"    Abs Rel: {metrics['segmentation']['before_align']['abs_rel']:.4f}")
        print("  After Alignment:")
        print(f"    RMSE: {metrics['segmentation']['after_align']['rmse']:.4f}")
        print(f"    MAE: {metrics['segmentation']['after_align']['mae']:.4f}")
        print(f"    Abs Rel: {metrics['segmentation']['after_align']['abs_rel']:.4f}")
        
        print("\nMetrics for Instance ID-based Depth:")
        print("  Before Alignment:")
        print(f"    RMSE: {metrics['instance']['before_align']['rmse']:.4f}")
        print(f"    MAE: {metrics['instance']['before_align']['mae']:.4f}")
        print(f"    Abs Rel: {metrics['instance']['before_align']['abs_rel']:.4f}")
        print("  After Alignment:")
        print(f"    RMSE: {metrics['instance']['after_align']['rmse']:.4f}")
        print(f"    MAE: {metrics['instance']['after_align']['mae']:.4f}")
        print(f"    Abs Rel: {metrics['instance']['after_align']['abs_rel']:.4f}")
        
        print(f"Saved visualization to {output_path}")
        return metrics
        
    except Exception as e:
        print(f"Error processing images: {str(e)}")
        return None

def evaluate_folder(rgb_folder, seg_mask_folder, inst_id_folder,
                   pred_depth_rgb_folder, pred_depth_seg_folder, pred_depth_inst_folder,
                   gt_depth_folder, output_folder='results'):
    """
    Evaluate depth predictions against ground truth depth maps for all images in folders.
    
    Args:
        rgb_folder: Path to folder containing RGB images
        seg_mask_folder: Path to folder containing segmentation masks
        inst_id_folder: Path to folder containing instance ID images
        pred_depth_rgb_folder: Path to folder containing predicted depths from RGB
        pred_depth_seg_folder: Path to folder containing predicted depths from segmentation
        pred_depth_inst_folder: Path to folder containing predicted depths from instance IDs
        gt_depth_folder: Path to folder containing ground truth depth maps
        output_folder: Path to save evaluation results
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get list of RGB images
    rgb_files = sorted([f for f in os.listdir(rgb_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Dictionary to store all metrics
    all_metrics = {}
    
    # Process each image
    for i, rgb_file in enumerate(rgb_files):
        print(f"\nProcessing image {i+1}/{len(rgb_files)}: {rgb_file}")
        
        # Construct RGB path
        rgb_path = os.path.join(rgb_folder, rgb_file)
        
        # Construct segmentation mask path
        seg_mask_file = rgb_file.replace('frame', 'instance_color').replace('.jpg', '.png').replace('.jpeg', '.png')
        seg_mask_path = os.path.join(seg_mask_folder, seg_mask_file)
        
        # Construct instance ID path
        inst_id_file = rgb_file.replace('frame', 'instance_id').replace('.jpg', '.png').replace('.jpeg', '.png')
        inst_id_path = os.path.join(inst_id_folder, inst_id_file)
        
        # Construct predicted depth paths
        pred_depth_rgb_file = os.path.splitext(rgb_file)[0] + '_depth.png'
        pred_depth_rgb_path = os.path.join(pred_depth_rgb_folder, pred_depth_rgb_file)
        
        pred_depth_seg_file = seg_mask_file.replace('.png', '_depth.png')
        pred_depth_seg_path = os.path.join(pred_depth_seg_folder, pred_depth_seg_file)
        
        pred_depth_inst_file = inst_id_file.replace('.png', '_depth.png')
        pred_depth_inst_path = os.path.join(pred_depth_inst_folder, pred_depth_inst_file)
        
        # Construct ground truth depth path
        gt_depth_file = rgb_file.replace('frame', 'depth').replace('.jpg', '.png').replace('.jpeg', '.png')
        gt_depth_path = os.path.join(gt_depth_folder, gt_depth_file)
        
        # Check if all files exist
        files_to_check = [
            (rgb_path, "RGB image"),
            (seg_mask_path, "segmentation mask"),
            (inst_id_path, "instance ID image"),
            (pred_depth_rgb_path, "predicted depth from RGB"),
            (pred_depth_seg_path, "predicted depth from segmentation"),
            (pred_depth_inst_path, "predicted depth from instance ID"),
            (gt_depth_path, "ground truth depth")
        ]
        
        missing_files = False
        for file_path, file_type in files_to_check:
            if not os.path.exists(file_path):
                print(f"Warning: Missing {file_type} file: {file_path}")
                missing_files = True
                
        if missing_files:
            print(f"Skipping evaluation for {rgb_file} due to missing files.")
            continue
        
        # Output path for this image
        output_path = os.path.join(output_folder, f"comparison_{i:03d}.png")
        
        # Evaluate single set
        metrics = evaluate_single_set(
            rgb_path, seg_mask_path, inst_id_path,
            pred_depth_rgb_path, pred_depth_seg_path, pred_depth_inst_path,
            gt_depth_path, output_path
        )
        
        if metrics:
            # Store metrics for this image
            all_metrics[rgb_file] = metrics
    
    # Calculate average metrics if we have results
    if all_metrics:
        avg_rgb_before = {
            'rmse': np.mean([m['rgb']['before_align']['rmse'] for m in all_metrics.values() if m['rgb']['before_align']['rmse'] is not None]),
            'mae': np.mean([m['rgb']['before_align']['mae'] for m in all_metrics.values() if m['rgb']['before_align']['mae'] is not None]),
            'abs_rel': np.mean([m['rgb']['before_align']['abs_rel'] for m in all_metrics.values() if m['rgb']['before_align']['abs_rel'] is not None])
        }
        
        avg_rgb_after = {
            'rmse': np.mean([m['rgb']['after_align']['rmse'] for m in all_metrics.values() if m['rgb']['after_align']['rmse'] is not None]),
            'mae': np.mean([m['rgb']['after_align']['mae'] for m in all_metrics.values() if m['rgb']['after_align']['mae'] is not None]),
            'abs_rel': np.mean([m['rgb']['after_align']['abs_rel'] for m in all_metrics.values() if m['rgb']['after_align']['abs_rel'] is not None])
        }
        
        avg_seg_before = {
            'rmse': np.mean([m['segmentation']['before_align']['rmse'] for m in all_metrics.values() if m['segmentation']['before_align']['rmse'] is not None]),
            'mae': np.mean([m['segmentation']['before_align']['mae'] for m in all_metrics.values() if m['segmentation']['before_align']['mae'] is not None]),
            'abs_rel': np.mean([m['segmentation']['before_align']['abs_rel'] for m in all_metrics.values() if m['segmentation']['before_align']['abs_rel'] is not None])
        }
        
        avg_seg_after = {
            'rmse': np.mean([m['segmentation']['after_align']['rmse'] for m in all_metrics.values() if m['segmentation']['after_align']['rmse'] is not None]),
            'mae': np.mean([m['segmentation']['after_align']['mae'] for m in all_metrics.values() if m['segmentation']['after_align']['mae'] is not None]),
            'abs_rel': np.mean([m['segmentation']['after_align']['abs_rel'] for m in all_metrics.values() if m['segmentation']['after_align']['abs_rel'] is not None])
        }
        
        avg_inst_before = {
            'rmse': np.mean([m['instance']['before_align']['rmse'] for m in all_metrics.values() if m['instance']['before_align']['rmse'] is not None]),
            'mae': np.mean([m['instance']['before_align']['mae'] for m in all_metrics.values() if m['instance']['before_align']['mae'] is not None]),
            'abs_rel': np.mean([m['instance']['before_align']['abs_rel'] for m in all_metrics.values() if m['instance']['before_align']['abs_rel'] is not None])
        }
        
        avg_inst_after = {
            'rmse': np.mean([m['instance']['after_align']['rmse'] for m in all_metrics.values() if m['instance']['after_align']['rmse'] is not None]),
            'mae': np.mean([m['instance']['after_align']['mae'] for m in all_metrics.values() if m['instance']['after_align']['mae'] is not None]),
            'abs_rel': np.mean([m['instance']['after_align']['abs_rel'] for m in all_metrics.values() if m['instance']['after_align']['abs_rel'] is not None])
        }
        
        # Add average metrics to the dictionary
        all_metrics['average'] = {
            'rgb': {
                'before_align': avg_rgb_before,
                'after_align': avg_rgb_after
            },
            'segmentation': {
                'before_align': avg_seg_before,
                'after_align': avg_seg_after
            },
            'instance': {
                'before_align': avg_inst_before,
                'after_align': avg_inst_after
            }
        }
        
        # Print average metrics
        print("\nAverage Metrics for RGB-based Depth:")
        print("  Before Alignment:")
        print(f"    RMSE: {avg_rgb_before['rmse']:.4f}")
        print(f"    MAE: {avg_rgb_before['mae']:.4f}")
        print(f"    Abs Rel: {avg_rgb_before['abs_rel']:.4f}")
        print("  After Alignment:")
        print(f"    RMSE: {avg_rgb_after['rmse']:.4f}")
        print(f"    MAE: {avg_rgb_after['mae']:.4f}")
        print(f"    Abs Rel: {avg_rgb_after['abs_rel']:.4f}")
        
        print("\nAverage Metrics for Segmentation-based Depth:")
        print("  Before Alignment:")
        print(f"    RMSE: {avg_seg_before['rmse']:.4f}")
        print(f"    MAE: {avg_seg_before['mae']:.4f}")
        print(f"    Abs Rel: {avg_seg_before['abs_rel']:.4f}")
        print("  After Alignment:")
        print(f"    RMSE: {avg_seg_after['rmse']:.4f}")
        print(f"    MAE: {avg_seg_after['mae']:.4f}")
        print(f"    Abs Rel: {avg_seg_after['abs_rel']:.4f}")
        
        print("\nAverage Metrics for Instance ID-based Depth:")
        print("  Before Alignment:")
        print(f"    RMSE: {avg_inst_before['rmse']:.4f}")
        print(f"    MAE: {avg_inst_before['mae']:.4f}")
        print(f"    Abs Rel: {avg_inst_before['abs_rel']:.4f}")
        print("  After Alignment:")
        print(f"    RMSE: {avg_inst_after['rmse']:.4f}")
        print(f"    MAE: {avg_inst_after['mae']:.4f}")
        print(f"    Abs Rel: {avg_inst_after['abs_rel']:.4f}")
    
    # Save all metrics as JSON
    metrics_path = os.path.join(output_folder, "all_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"Saved all metrics to {metrics_path}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate depth predictions against ground truth")
    parser.add_argument("--mode", type=str, choices=["single", "folder"], default="single",
                        help="Evaluation mode: single image set or folder")
    
    # Single mode arguments
    parser.add_argument("--rgb_path", type=str, help="Path to RGB image (for single mode)")
    parser.add_argument("--seg_mask_path", type=str, help="Path to segmentation mask (for single mode)")
    parser.add_argument("--inst_id_path", type=str, help="Path to instance ID image (for single mode)")
    parser.add_argument("--pred_depth_rgb_path", type=str, help="Path to predicted depth from RGB (for single mode)")
    parser.add_argument("--pred_depth_seg_path", type=str, help="Path to predicted depth from segmentation (for single mode)")
    parser.add_argument("--pred_depth_inst_path", type=str, help="Path to predicted depth from instance ID (for single mode)")
    parser.add_argument("--gt_depth_path", type=str, help="Path to ground truth depth image (for single mode)")
    parser.add_argument("--output_path", type=str, default="depth_comparison.png", 
                        help="Path to save output visualization (for single mode)")
    
    # Folder mode arguments
    parser.add_argument("--rgb_folder", type=str, help="Path to folder with RGB images (for folder mode)")
    parser.add_argument("--seg_mask_folder", type=str, help="Path to folder with segmentation masks (for folder mode)")
    parser.add_argument("--inst_id_folder", type=str, help="Path to folder with instance ID images (for folder mode)")
    parser.add_argument("--pred_depth_rgb_folder", type=str, help="Path to folder with predicted depths from RGB (for folder mode)")
    parser.add_argument("--pred_depth_seg_folder", type=str, help="Path to folder with predicted depths from segmentation (for folder mode)")
    parser.add_argument("--pred_depth_inst_folder", type=str, help="Path to folder with predicted depths from instance ID (for folder mode)")
    parser.add_argument("--gt_depth_folder", type=str, help="Path to folder with ground truth depth maps (for folder mode)")
    parser.add_argument("--output_folder", type=str, default="evaluation_results", 
                        help="Path to save output visualizations (for folder mode)")
    
    args = parser.parse_args()
    
    if args.mode == "single":
        required_args = ["rgb_path", "seg_mask_path", "inst_id_path", 
                        "pred_depth_rgb_path", "pred_depth_seg_path", "pred_depth_inst_path", 
                        "gt_depth_path"]
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        
        if missing_args:
            print(f"Error: Missing required arguments for single mode: {', '.join(missing_args)}")
            parser.print_help()
            exit(1)
        
        evaluate_single_set(
            args.rgb_path, args.seg_mask_path, args.inst_id_path,
            args.pred_depth_rgb_path, args.pred_depth_seg_path, args.pred_depth_inst_path, 
            args.gt_depth_path, args.output_path
        )
    else:  # folder mode
        required_args = ["rgb_folder", "seg_mask_folder", "inst_id_folder",
                        "pred_depth_rgb_folder", "pred_depth_seg_folder", "pred_depth_inst_folder", 
                        "gt_depth_folder"]
        missing_args = [arg for arg in required_args if getattr(args, arg) is None]
        
        if missing_args:
            print(f"Error: Missing required arguments for folder mode: {', '.join(missing_args)}")
            parser.print_help()
            exit(1)
        
        evaluate_folder(
            args.rgb_folder, args.seg_mask_folder, args.inst_id_folder,
            args.pred_depth_rgb_folder, args.pred_depth_seg_folder, args.pred_depth_inst_folder,
            args.gt_depth_folder, args.output_folder
        )