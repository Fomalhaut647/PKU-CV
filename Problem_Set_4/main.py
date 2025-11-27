import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import trimesh
import multiprocessing as mp
from tqdm import tqdm
from typing import Tuple
import time


def normalize_disparity_map(disparity_map):
    """Normalize disparity map for visualization
    disparity should be larger than zero
    """
    return np.maximum(disparity_map, 0.0) / (disparity_map.max() + 1e-10)


def visualize_disparity_map(disparity_map, gt_map, save_path=None):
    """Visualize or save disparity map and compare with ground truth"""
    # Normalize disparity maps
    disparity_map = normalize_disparity_map(disparity_map)
    gt_map = normalize_disparity_map(gt_map)
    # Visualize or save to file
    if save_path is None:
        concat_map = np.concatenate([disparity_map, gt_map], axis=1)
        plt.imshow(concat_map, "gray")
        plt.show()
    else:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        concat_map = np.concatenate([disparity_map, gt_map], axis=1)
        plt.imsave(save_path, concat_map, cmap="gray")


def task1_compute_disparity_map_simple(
    ref_img: np.ndarray,  # shape (H, W)
    sec_img: np.ndarray,  # shape (H, W)
    window_size: int,
    disparity_range: Tuple[int, int],  # (min_disparity, max_disparity)
    matching_function: str,  # can be 'SSD', 'SAD', 'normalized_correlation'
):
    """Assume image planes are parallel to each other
    Compute disparity map using simple stereo system following the steps:
    1. For each row, scan all pixels in that row
    2. Generate a window for each pixel in ref_img
    3. Search for a disparity (d) within (min_disparity, max_disparity) in sec_img
    4. Select the best disparity that minimize window difference between ref_img[row, col] and sec_img[row, col - d]
    """

    h, w = ref_img.shape
    min_disp, max_disp = disparity_range

    # Initialize disparity map and best cost map
    # For SSD/SAD, we want to minimize cost (init with infinity)
    # For NCC, we want to maximize correlation (init with -1)
    disparity_map = np.zeros((h, w), dtype=np.float32)

    if matching_function in ["SSD", "SAD"]:
        best_cost = np.full((h, w), np.inf, dtype=np.float32)
    elif matching_function == "normalized_correlation":
        best_cost = np.full((h, w), -1.0, dtype=np.float32)
    else:
        raise ValueError(f"Unknown matching function: {matching_function}")

    # Kernel for summing up costs within the window
    kernel = np.ones((window_size, window_size), np.float32)

    # Pre-computation for Normalized Correlation to speed up
    if matching_function == "normalized_correlation":
        # Compute local sum of squares for ref_img once
        ref_sq = ref_img.astype(np.float32) ** 2
        ref_sq_sum = cv2.filter2D(ref_sq, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        # Fix: Clamp negative values due to floating point precision errors
        ref_sq_sum = np.maximum(ref_sq_sum, 0)
        ref_sq_sum = np.sqrt(ref_sq_sum)  # Standard deviation part 1

    print(f"Computing disparity ({matching_function})...")

    # Iterate over the disparity range
    for d in tqdm(range(min_disp, max_disp)):
        # Create a shifted version of the secondary image
        # Pixels that shift out of bounds need handling.
        # We simulate ref_img(x, y) matching with sec_img(x - d, y)

        # Translation matrix for shifting: x -> x + d (since sec_img is naturally to the right, we shift it right to align)
        # Wait, standard stereo: x_left = x, x_right = x - d.
        # So we want to compare Ref(x) with Sec(x-d).
        # To align Sec(x-d) to Ref(x), we need to shift Sec image to the *right* by d pixels.
        M = np.float32([[1, 0, d], [0, 1, 0]])
        sec_shifted = cv2.warpAffine(
            sec_img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # Mask to ignore boundary effects where shifted image has no data
        valid_mask = np.ones((h, w), dtype=np.float32)
        valid_mask = cv2.warpAffine(
            valid_mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        ref_f = ref_img.astype(np.float32)
        sec_f = sec_shifted.astype(np.float32)

        if matching_function == "SSD":
            # Cost = Sum((I1 - I2)^2)
            diff = ref_f - sec_f
            sq_diff = diff**2
            # Sum over window using box filter
            current_cost = cv2.filter2D(
                sq_diff, -1, kernel, borderType=cv2.BORDER_CONSTANT
            )

            # Update best disparity
            # We only update where valid_mask is true to avoid boundary artifacts
            mask = (current_cost < best_cost) & (valid_mask > 0)
            best_cost[mask] = current_cost[mask]
            disparity_map[mask] = d

        elif matching_function == "SAD":
            # Cost = Sum(|I1 - I2|)
            abs_diff = np.abs(ref_f - sec_f)
            current_cost = cv2.filter2D(
                abs_diff, -1, kernel, borderType=cv2.BORDER_CONSTANT
            )

            mask = (current_cost < best_cost) & (valid_mask > 0)
            best_cost[mask] = current_cost[mask]
            disparity_map[mask] = d

        elif matching_function == "normalized_correlation":
            # NCC = Sum(I1 * I2) / sqrt(Sum(I1^2) * Sum(I2^2))
            # Note: This is a simplified local NCC. Zero-mean NCC (ZNCC) subtracts mean first.
            # The prompt asks for "normalized_correlation", usually implying standard NCC.

            numerator = cv2.filter2D(
                ref_f * sec_f, -1, kernel, borderType=cv2.BORDER_CONSTANT
            )

            # Denominator
            sec_sq = sec_f**2
            sec_sq_sum = cv2.filter2D(
                sec_sq, -1, kernel, borderType=cv2.BORDER_CONSTANT
            )
            # Fix: Clamp negative values due to floating point precision errors
            sec_sq_sum = np.maximum(sec_sq_sum, 0)
            sec_sq_sum = np.sqrt(sec_sq_sum)

            denominator = ref_sq_sum * sec_sq_sum

            # Avoid division by zero
            denominator[denominator < 1e-5] = 1e-5

            current_score = numerator / denominator

            # Maximize correlation
            mask = (current_score > best_cost) & (valid_mask > 0)
            best_cost[mask] = current_score[mask]
            disparity_map[mask] = d

    return disparity_map


def task1_simple_disparity(ref_img, sec_img, gt_map, img_name="tsukuba"):
    """Compute disparity maps for different settings"""
    window_sizes = [3, 9, 15]  # Try different window sizes
    disparity_range = (0, 16)  # Determine appropriate disparity range
    matching_functions = [
        "SSD",
        "SAD",
        "normalized_correlation",
    ]  # Try different matching functions

    disparity_maps = []

    # Generate disparity maps for different settings
    for window_size in window_sizes:
        for matching_function in matching_functions:
            print(
                f"Computing disparity map for window_size={window_size}, disparity_range={disparity_range}, matching_function={matching_function}"
            )

            start_time = time.time()
            disparity_map = task1_compute_disparity_map_simple(
                ref_img, sec_img, window_size, disparity_range, matching_function
            )
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.4f} seconds")

            disparity_maps.append(
                (disparity_map, window_size, matching_function, disparity_range)
            )
            dmin, dmax = disparity_range
            visualize_disparity_map(
                disparity_map,
                gt_map,
                save_path=f"output/task1_{img_name}_w{window_size}_d{dmin}-{dmax}_{matching_function}.png",
            )
    return disparity_maps


def task2_compute_depth_map(disparity_map, baseline, focal_length):
    """Compute depth map by z = fB / (x - x')
    Note that a disparity less or equal to zero should be ignored (set to zero)
    """
    depth_map = ...
    return depth_map


def task2_visualize_pointcloud(
    ref_img: np.ndarray,  # shape (H, W, 3)
    disparity_map: np.ndarray,  # shape (H, W)
    save_path: str = "output/task2_tsukuba.ply",
):
    """Visualize 3D pointcloud from disparity map following the steps:
    1. Calculate depth map from disparity
    2. Set pointcloud's XY as image's XY and and pointcloud's Z as depth
    3. Set pointcloud's color as ref_img's color
    4. Save pointcloud to ply files for visualizationh. We recommend to open ply file with MeshLab
    5. Adjust the baseline and focal_length for better performance
    6. You may need to cut some outliers for better performance
    """
    baseline = 10
    focal_length = 10
    depth_map = task2_compute_depth_map(disparity_map, baseline, focal_length)

    # Points
    points = ...

    # Colors
    colors = ...

    # Save pointcloud to ply file
    pointcloud = trimesh.PointCloud(points, colors)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pointcloud.export(save_path, file_type="ply")


def task3_compute_disparity_map_dp(ref_img, sec_img):
    """Conduct stereo matching with dynamic programming"""

    disparity_map_dp = ...

    return disparity_map_dp


def main(tasks):

    # Read images and ground truth disparity maps
    moebius_img1 = cv2.imread("data/moebius1.png")
    moebius_img1_gray = cv2.cvtColor(moebius_img1, cv2.COLOR_BGR2GRAY)
    moebius_img2 = cv2.imread("data/moebius2.png")
    moebius_img2_gray = cv2.cvtColor(moebius_img2, cv2.COLOR_BGR2GRAY)
    moebius_gt = cv2.imread("data/moebius_gt.png", cv2.IMREAD_GRAYSCALE)

    tsukuba_img1 = cv2.imread("data/tsukuba1.jpg")
    tsukuba_img1_gray = cv2.cvtColor(tsukuba_img1, cv2.COLOR_BGR2GRAY)
    tsukuba_img2 = cv2.imread("data/tsukuba2.jpg")
    tsukuba_img2_gray = cv2.cvtColor(tsukuba_img2, cv2.COLOR_BGR2GRAY)
    tsukuba_gt = cv2.imread("data/tsukuba_gt.jpg", cv2.IMREAD_GRAYSCALE)

    # Task 0: Visualize cv2 Results
    if "0" in tasks:
        # Compute disparity maps using cv2
        stereo = cv2.StereoBM.create(numDisparities=64, blockSize=15)
        moebius_disparity_cv2 = stereo.compute(moebius_img1_gray, moebius_img2_gray)
        visualize_disparity_map(
            moebius_disparity_cv2, moebius_gt, save_path="output/task0_1.png"
        )
        tsukuba_disparity_cv2 = stereo.compute(tsukuba_img1_gray, tsukuba_img2_gray)
        visualize_disparity_map(
            tsukuba_disparity_cv2, tsukuba_gt, save_path="output/task0_2.png"
        )

        if "2" in tasks:
            print("Running task2 with cv2 results ...")
            task2_visualize_pointcloud(
                tsukuba_img1,
                tsukuba_disparity_cv2,
                save_path="output/task2_tsukuba_cv2.ply",
            )

    ######################################################################
    # Note. Running on moebius may take a long time with your own code   #
    # In this homework, you are allowed only to deal with tsukuba images #
    ######################################################################

    # Task 1: Simple Disparity Algorithm
    if "1" in tasks:
        print("Running task1 ...")
        disparity_maps = task1_simple_disparity(
            tsukuba_img1_gray, tsukuba_img2_gray, tsukuba_gt, img_name="tsukuba"
        )

        #####################################################
        # If you want to run on moebius images,             #
        # parallelizing with multiprocessing is recommended #
        #####################################################
        # task1_simple_disparity(moebius_img1_gray, moebius_img2_gray, moebius_gt, img_name='moebius')

        if "2" in tasks:
            print("Running task2 with disparity maps from task1 ...")
            for (
                disparity_map,
                window_size,
                matching_function,
                disparity_range,
            ) in disparity_maps:
                dmin, dmax = disparity_range
                task2_visualize_pointcloud(
                    tsukuba_img1,
                    disparity_map,
                    save_path=f"output/task2_tsukuba_{window_size}_{dmin}_{dmax}_{matching_function}.ply",
                )

    # Task 3: Non-local constraints
    if "3" in tasks:
        print("----------------- Task 3 -----------------")
        tsukuba_disparity_dp = task3_compute_disparity_map_dp(
            tsukuba_img1_gray, tsukuba_img2_gray
        )
        visualize_disparity_map(
            tsukuba_disparity_dp, tsukuba_gt, save_path="output/task3_tsukuba.png"
        )

        if "2" in tasks:
            print("Running task2 with disparity maps from task3 ...")
            task2_visualize_pointcloud(
                tsukuba_img1,
                tsukuba_disparity_dp,
                save_path="output/task2_tsukuba_dp.ply",
            )


if __name__ == "__main__":
    # Set tasks to run
    parser = argparse.ArgumentParser(description="Homework 4")
    parser.add_argument("--tasks", type=str, default="0123")
    args = parser.parse_args()

    main(args.tasks)
