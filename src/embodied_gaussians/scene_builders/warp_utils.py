# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

import torch
import warp as wp
import numpy as np


def convert_matrix_to_transform(matrix: np.ndarray) -> wp.transformf:
    quat = wp.quat_from_matrix(matrix[:3, :3])  
    pos = matrix[:3, 3]
    return wp.transformf(*pos, *quat)

def find_distant_query_points(
    max_distance: float, query_xyz: np.ndarray, target_xyz: np.ndarray
) -> np.ndarray:    
    query_xyz = np.asarray(query_xyz, dtype=np.float32)
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    grid = wp.HashGrid(128, 128, 128)
    num_queries = query_xyz.shape[0]
    mask = torch.zeros((num_queries,), dtype=torch.int32, device="cuda")
    # mask_warp = wp.from_torch(mask)
    query_xyz_warp = wp.from_numpy(query_xyz, dtype=wp.vec3f)
    target_xyz_warp = wp.from_numpy(target_xyz, dtype=wp.vec3f)
    grid.build(target_xyz_warp, max_distance)
    wp.launch(
        kernel=find_distant_query_points_kernel,
        dim=num_queries,  # type: ignore
        inputs=[
            grid.id,
            max_distance,
            query_xyz_warp,
            target_xyz_warp,
            mask
            # mask_warp,
        ],
    )
    mask = mask.cpu().numpy()
    return mask == 1

@wp.kernel
def find_distant_query_points_kernel(
    grid: wp.uint64,
    max_distance: float,
    query_xyz: wp.array(dtype=wp.vec3), # type: ignore
    target_xyz: wp.array(dtype=wp.vec3), # type: ignore
    mask: wp.array(dtype=wp.int32), # type: ignore  
):
    tid = wp.tid()
    gx = query_xyz[tid]
    query = wp.hash_grid_query(grid, gx, wp.float32(max_distance))
    index = int(0)
    best_distance = wp.float32(max_distance)
    best_index = int(-1)
    while wp.hash_grid_query_next(query, index):
        n = gx - target_xyz[index]
        d = wp.length(n)
        if d < best_distance:
            best_distance = d
            best_index = index

    if best_index == -1:
        mask[tid] = 1
