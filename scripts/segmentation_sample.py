import argparse
import os
import nibabel as nib
# from visdom import Visdom
import sys
import random
sys.path.append(".")
sys.path.append("..")
import numpy as np
import time
import torch as th
import torch.distributed as dist
from guided_diffusion import dist_util, logger
from guided_diffusion.bratsloader import BRATSDataset
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from guided_diffusion.dataset import Dataset
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from torchvision import utils
import datetime
import cv2

def makedirs(path):
	if not os.path.isdir(path):
		os.makedirs(path)

def main():
    # Set batch size according to VRAM
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    
    # Set up DataLoader using batch size from args
    ds = Dataset(args.image_dir, args.heatmap_dir, args.image_size, cond_exits=True)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())

    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    # Loop over batches from DataLoader
    for batch_idx, (heat_batch, cell_batch) in enumerate(dl):
        if batch_idx >= args.max_batches: # End condition
             break
        
        # Set seeds for reproducibility
        seed = 10 + batch_idx
        th.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # List to manage locations for each image in the batch
        locs_batch = [[] for _ in range(args.batch_size)]
        
        logger.log(f"Processing batch {batch_idx}...")

        # Create initial input (j=0)
        # This step needs to be performed individually for each image in the batch
        current_img1_list = []
        current_img2_list = []
        
        # Apply initial patches individually to each image in the batch
        for i in range(args.batch_size):
            cnt1 = batch_idx * args.batch_size + i
            folderName1 = os.path.join(args.output_dir, 'paste', f'{cnt1:0>4}')
            makedirs(folderName1)
            
            target_heat = heat_batch[i].detach().cpu().squeeze().numpy()
            
            locs_batch[i] = find_gaussian_peaks(target_heat, threshold=0.5)
            img2, locs_batch[i] = copy_patch(cell_batch[i], heat_batch[i], target_heat, crop_size=(args.crop_size, args.crop_size), min_distance=args.min_distance_initial, loc=locs_batch[i], cnt=0)
            cv2.imwrite(f"{folderName1}/{0:0>5}.png", img2*255)

            # current_img1_list.append(th.from_numpy(img1.astype(np.float32)))
            current_img2_list.append(th.from_numpy(img2.astype(np.float32)))

        # Reconstruct tensors from list (batching)
        img1_batch = cell_batch # Shape: [B, 1, H, W]
        img2_batch = th.stack(current_img2_list).unsqueeze(1) # Shape: [B, 1, H, W]

        # Iterative generation loop
        for j in range(args.num_steps):
            logger.log(f"  Batch {batch_idx}, Step {j}...")
            
            # For j>0, update inputs using the model's output from the previous step
            if j > 0:
                prev_out_batch = sample # Output from previous step
                
                next_img1_list = []
                next_img2_list = []

                # Deconstruct batch and process per image
                for i in range(args.batch_size):
                    cnt1 = batch_idx * args.batch_size + i
                    folderName1 = os.path.join(args.output_dir, 'paste', f'{cnt1:0>4}')

                    # img2 carries over from the previous step
                    prev_img2_np = img2_batch[i].cpu().detach().squeeze().numpy()

                    # Apply copy_patch individually
                    img2, locs_batch[i] = copy_patch(cell_batch[i], heat_batch[i], prev_img2_np, crop_size=(args.crop_size, args.crop_size), min_distance=args.min_distance_step, loc=locs_batch[i], cnt=j)
                    cv2.imwrite(f"{folderName1}/{j:0>5}.png", img2*255)

                    # next_img1_list.append(th.from_numpy(img1.astype(np.float32)))
                    next_img2_list.append(th.from_numpy(img2.astype(np.float32)))
                
                # Reassemble results back into batch
                img1_batch = prev_out_batch
                img2_batch = th.stack(next_img2_list).unsqueeze(1)

            # Create inputs for model
            time_tensor = th.tensor([100 - 1], device=dist_util.dev())
            img = th.cat((img2_batch.to(dist_util.dev()), diffusion.q_sample(img1_batch.to(dist_util.dev()), time_tensor)), dim=1)
            
            model_kwargs = {}
            sample_fn = (
                diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
            )
            
            # Model processes batch at once
            sample, _, _ = sample_fn(
                model,
                (args.batch_size, 1, args.image_size, args.image_size), img,
                time=100,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            
            # Save results (deconstruct batch and save)
            for i in range(args.batch_size):
                cnt1 = batch_idx * args.batch_size + i
                folderName = os.path.join(args.output_dir, 'pre', f'{cnt1:0>4}')
                makedirs(folderName)
                
                out_img = sample[i].cpu() # Get the i-th image
                nomalized_output_img = np.asarray(out_img)
                output_image = nomalized_output_img.transpose(1, 2, 0) * 255
                cv2.imwrite(f"{folderName}/{j:0>5}.png", output_image)
        # End of iterative steps loop
    # End of batch loop
    logger.log("All batches processed.")

def create_argparser():
	defaults = dict(
		data_dir="./data/testing",
		image_dir="../dataset/below400/image",
		heatmap_dir="../dataset/below400/heatmap",
		output_dir="./results_output",
		max_batches=125,
		num_steps=40,
		crop_size=20,
		min_distance_initial=5,
		min_distance_step=20,
		clip_denoised=True,
		num_samples=1,
		batch_size=1,
		use_ddim=False,
		model_path="",
		num_ensemble=1      #number of samples in the ensemble
	)
	defaults.update(model_and_diffusion_defaults())
	parser = argparse.ArgumentParser()
	add_dict_to_argparser(parser, defaults)
	return parser

import cv2
import numpy as np
import random
from scipy.ndimage import maximum_filter

def find_gaussian_peaks(image, threshold=0.7, neighborhood_size=15):
	"""
	Detect multiple Gaussian peaks (bright points).
	:param image: Input image BGR
	:param threshold: Luminance threshold for peaks (0-255)
	:param neighborhood_size: Local area size for peak recognition
	:return: List of peak coordinates [(x1, y1), (x2, y2), ...]
	"""

	# Detect local maximums
	local_max = maximum_filter(image, size=neighborhood_size) == image

	# Extract only brightness above threshold
	peaks = np.where((image >= threshold) & local_max)

	# Create list of (x, y) coordinates
	peak_coords = list(zip(peaks[1], peaks[0]))
	return peak_coords

def crop_around_peak(image, center, crop_size):
	"""Crop the image around a Gaussian peak"""
	x, y = center[0]
	w, h = crop_size

	x1 = max(x - w // 2, 0)
	y1 = max(y - h // 2, 0)
	x2 = min(x1 + w, image.shape[1])
	y2 = min(y1 + h, image.shape[0])

	cropped = image[y1:y2, x1:x2].copy()
	return cropped

def is_far_enough_from_all(point, other_points, min_dist):
	"""Check if the point is a certain distance away from all specified points"""
	return all(np.linalg.norm(np.array(point) - np.array(p)) >= min_dist for p in other_points)

def get_random_paste_position(image_shape, patch_shape, avoid_points, min_distance):
	"""Return coordinates within the image range at a distance from multiple avoid points"""
	h_img, w_img = image_shape[:2]
	h_patch, w_patch = patch_shape[:2]

	max_attempts = 100
	for _ in range(max_attempts):
		x = random.randint(0, w_img - w_patch)
		y = random.randint(0, h_img - h_patch)
		center = (x + w_patch // 2, y + h_patch // 2)
		if is_far_enough_from_all(center, avoid_points, min_distance):
			return (x, y)
	return (x, y)  # Return the result of the last attempt

def copy_patch(source_cell, source_heat, target_heat, crop_size=(40, 40), min_distance=100, loc=[], cnt=0):
	
	# 0. Load images

	if (not target_heat.max() == 0):
		target_heat = target_heat / target_heat.max()

	# 2. Detect all peaks in the target paste image
	peaks_dst = find_gaussian_peaks(target_heat, threshold=0.5)

	# 3. Get a random paste position
	paste_x, paste_y = get_random_paste_position(target_heat.shape, (40,40), peaks_dst, min_distance)

		# 4. Paste processing
	# h_patch, w_patch = patch_cell.shape[:2]
	# target_cell[paste_y:paste_y + h_patch, paste_x:paste_x + w_patch] = patch_cell
	loc.append((paste_x, paste_y))
	target_heat = create_heatmap(drawn_points=loc)

	return target_heat, loc

def create_heatmap(interval=100, image_size=(256, 256), padding_size=36, dot_radius=6, dot_color=255, min_distance=10, drawn_points=[]):
	count = 0
	padded_size = (image_size[0] + 2 * padding_size, image_size[1] + 2 * padding_size)

	img_final = np.zeros((image_size[0], image_size[1], 1), dtype=np.float32)
	
	for loc in drawn_points:
		image1 = np.zeros((image_size[0], image_size[1], 1), dtype=np.float32)
		cv2.circle(image1, (loc[0], loc[1]), dot_radius, dot_color, -1)
	
		image1 = cv2.GaussianBlur(image1, (15, 15), 5)
		img_final = cv2.max(img_final, image1)

	img_final = img_final / img_final.max()
	
	return img_final

if __name__ == "__main__":
    main()