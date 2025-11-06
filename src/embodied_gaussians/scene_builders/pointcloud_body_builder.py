# Copyright (c) 2025 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

from dataclasses import dataclass, field
from collections import namedtuple
import logging

import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
import torch
import warp as wp

from gsplat.rendering import rasterization

from embodied_gaussians.scene_builders.domain import (
    Gaussians,
    Body,
    GaussianLearningRates,
    GaussianActivations,
    MaskedPosedImageAndDepth,
)

from .simple_visualizer import ellipsoid_meshes

logger = logging.getLogger(__name__)


@dataclass
class PointCloudBodyBuilderSettings:
    max_depth: float = 2.0  # Maximum depth to consider
    training_iterations: int = 4000  # Number of iterations to optimize the particles
    training_learning_rates: GaussianLearningRates = field(
        default_factory=lambda: GaussianLearningRates()
    )
    opacity_threshold: float = 0.5  # Opacity threshold for optimization
    min_scale: tuple[float, float, float] = (0.01, 0.01, 0.01)
    max_scale: tuple[float, float, float] = (0.03, 0.03, 0.03)
    max_depth: float = 2.0
    """
    If true, the gaussians will be disks. If false, the gaussians will be ellipsoids. This used to make sure the ground is flat.
    """


class PointCloudBodyBuilder:
    @staticmethod
    def build(
        name: str,
        settings: PointCloudBodyBuilderSettings,
        points: np.ndarray,
        datapoints: list[MaskedPosedImageAndDepth],
        visualize: bool = False,
    ) -> Body:
        wp.init()

        # visualize points
        if visualize:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([pcd])

        # # ================ Step 1: Train Gaussians =================
        gaussians = PointCloudBodyBuilder._train_gaussians(
            initial_points=points,
            num_iterations=settings.training_iterations,
            learning_rates=settings.training_learning_rates,
            datapoints=datapoints,
            min_scale=settings.min_scale,
            max_scale=settings.max_scale,
            visualize=visualize,
            max_depth=settings.max_depth,
        )

        if visualize:
            o3d.visualization.draw_geometries(
                [
                    o3d.geometry.TriangleMesh.create_coordinate_frame(0.1),
                    *ellipsoid_meshes(gaussians),
                ]
            )

        body = Body(
            name=name,
            X_WB=np.eye(4).tolist(),
            particles=None,
            gaussians=gaussians,
        )

        return body

    @staticmethod
    def _train_gaussians(
        initial_points: np.ndarray,
        num_iterations: int,
        learning_rates: GaussianLearningRates,
        datapoints: list[MaskedPosedImageAndDepth],
        min_scale: tuple[float, float, float],
        max_scale: tuple[float, float, float],
        visualize: bool = False,
        max_depth: float = 2.0,
    ):
        assert initial_points.shape[1] == 3
        params = PointCloudBodyBuilder._create_initial_gaussian_state(initial_points)
        # with torch.no_grad():
        #     params["means"] += torch.randn_like(params["means"]) * 0.1
        initial_gaussians = Gaussians(
            means=params["means"].detach().cpu().numpy(),
            quats=GaussianActivations.quat(params["quats"]).detach().cpu().numpy(),
            scales=GaussianActivations.scale(params["scales"]).detach().cpu().numpy(),
            opacities=GaussianActivations.opacity(params["opacities"])
            .detach()
            .cpu()
            .numpy(),
            colors=GaussianActivations.color(params["colors"]).detach().cpu().numpy(),
        )

        # This takes a long time to run because it's a lot of gaussians
        # if visualize:
        #     o3d.visualization.draw_geometries(
        #         [
        #             o3d.geometry.TriangleMesh.create_coordinate_frame(0.1),
        #             *ellipsoid_meshes(initial_gaussians),
        #         ]
        #     )

        datapoints = [datapoints[0]]  # , datapoints[1]]#, datapoints[2]]
        gt_data = PointCloudBodyBuilder._get_rasterization_groundtruth(
            datapoints, max_depth=max_depth
        )
        if visualize:
            # concat all depth
            depths = []
            for i in range(len(gt_data.depths)):
                depth = gt_data.depths[i].detach().cpu().numpy()
                depth_mask = gt_data.depth_masks[i].detach().cpu().numpy()
                depth = depth * depth_mask
                depths.append(depth)
            groundtruth_depth = np.concatenate(depths, axis=1)

            groundtruth = gt_data.images.detach().cpu().numpy()
            groundtruth = np.concatenate([i for i in groundtruth], axis=1)

        optimizers = PointCloudBodyBuilder._create_optimizers_for_params(
            params,
            {
                "means": learning_rates.means,
                "opacities": learning_rates.opacities,
                "colors": learning_rates.colors,
                "quats": learning_rates.quats,
                "scales": learning_rates.scales,
            },
        )
        inv_min_scale = GaussianActivations.inv_scale(
            torch.tensor(min_scale).cuda()
        )
        inv_max_scale = GaussianActivations.inv_scale(
            torch.tensor(max_scale).cuda()
        )
        backgrounds = torch.rand((num_iterations, 3)).float().cuda()
        num_images = gt_data.images.shape[0]

        for i in tqdm(range(num_iterations)):
            background = backgrounds[i % num_iterations]
            gt_data.images[gt_data.masks == 0, :] = background

            render_colors, render_alphas, info = rasterization(
                means=params["means"],
                quats=GaussianActivations.quat(params["quats"]),
                scales=GaussianActivations.scale(params["scales"]),
                colors=GaussianActivations.color(params["colors"]),
                opacities=GaussianActivations.opacity(params["opacities"]),
                viewmats=gt_data.X_CWs,
                Ks=gt_data.Ks,
                width=gt_data.width,
                height=gt_data.height,
                camera_model="pinhole",
                render_mode="RGB+D",
                backgrounds=background.reshape(1, 3).repeat(num_images, 1),
            )

            w_photmetric = 1.0
            loss = w_photmetric * torch.nn.functional.mse_loss(
                render_colors[..., :3], gt_data.images
            )
            # for j in range(len(gt_data.depth_masks)):
            #     depth_mask = gt_data.depth_masks[j]
            #     valid_depth_pixels = gt_data.valid_depth_pixels[j]
            #     depth_loss = torch.nn.functional.mse_loss(render_colors[j, ..., -1][depth_mask], gt_data.depths[j][depth_mask])/valid_depth_pixels
            #     loss += (1-w_photmetric)*depth_loss
            params.zero_grad()
            loss.backward()

            for optimizer in optimizers.values():
                optimizer.step()

            params["scales"].detach().clamp_(inv_min_scale, inv_max_scale)

            if visualize and i % 100 == 0:
                # print(float(loss))
                num_images = gt_data.images.shape[0]
                rgb = render_colors[..., :3].detach().cpu().numpy()
                depth = render_colors[..., -1].detach().cpu().numpy()
                aspect = rgb.shape[1] / rgb.shape[2]
                rgb = np.concatenate([i for i in rgb], axis=1)
                rgb = np.concatenate([rgb, groundtruth], axis=0)
                depth = np.concatenate([i for i in depth], axis=1)
                depth = np.concatenate([depth, groundtruth_depth], axis=0)
                width = 1920
                height = int(width * aspect)
                rgb = cv2.resize(rgb, (width, height))
                depth = cv2.resize(depth, (width, height))
                cv2.imshow("Colors", rgb)
                cv2.imshow("Depth", depth)
                cv2.waitKey(1)

        if visualize:
            cv2.destroyAllWindows()

        return Gaussians(
            means=params["means"].detach().cpu().numpy(),
            quats=GaussianActivations.quat(params["quats"]).detach().cpu().numpy(),
            scales=GaussianActivations.scale(params["scales"]).detach().cpu().numpy(),
            opacities=GaussianActivations.opacity(params["opacities"])
            .detach()
            .cpu()
            .numpy(),
            colors=GaussianActivations.color(params["colors"]).detach().cpu().numpy(),
        )

    @staticmethod
    def _get_rasterization_groundtruth(
        datapoints: list[MaskedPosedImageAndDepth], max_depth: float
    ):
        X_CWs = []
        Ks = []
        gts = []
        depth_gts = []
        width, height = datapoints[0].image.shape[1], datapoints[0].image.shape[0]
        depth_masks = []
        valid_depth_pixels = []
        masks = []
        for datapoint in datapoints:
            assert datapoint.image.shape[1] == width, (
                "All images must have the same width"
            )
            assert datapoint.image.shape[0] == height, (
                "All images must have the same height"
            )

            if datapoint.mask is None:
                continue

            X_WC = datapoint.X_WC @ np.array(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0.0, 0.0, 0.0, 1.0]]
            )  # rotate areound x axis to make it in opencv standard
            X_CW = np.linalg.inv(X_WC)
            X_CW = torch.from_numpy(X_CW).float().cuda()
            X_CWs.append(X_CW)
            K = torch.from_numpy(datapoint.K).float().cuda()
            Ks.append(K)
            # image = torch.from_numpy(datapoint.mask).float().cuda().unsqueeze(-1).repeat(1, 1, 3)
            mask = torch.from_numpy(datapoint.mask).cuda()
            masks.append(mask)

            image = torch.from_numpy(datapoint.image).float().cuda() / 255.0
            image[datapoint.mask == 0, :] = 0.0
            depth = (
                torch.from_numpy(datapoint.depth).float().cuda() * datapoint.depth_scale
            )
            depth[datapoint.mask == 0] = 0.0
            depth_mask = (depth > 0).__and__(depth < max_depth)
            valid_depth_pixels.append(int(depth_mask.sum()))
            depth_masks.append(depth_mask)
            gts.append(image)
            depth_gts.append(depth)

        gts = torch.stack(gts)
        depth_gts = torch.stack(depth_gts)
        X_CWs = torch.stack(X_CWs)
        Ks = torch.stack(Ks)
        masks = torch.stack(masks)

        return namedtuple(
            "GroundTruth",
            [
                "images",
                "depths",
                "X_CWs",
                "Ks",
                "depth_masks",
                "masks",
                "width",
                "height",
                "valid_depth_pixels",
            ],
        )(
            gts,
            depth_gts,
            X_CWs,
            Ks,
            depth_masks,
            masks,
            width,
            height,
            valid_depth_pixels,
        )

    @staticmethod
    def _create_initial_gaussian_state(means_: np.ndarray):
        num_points = means_.shape[0]

        sq_dist, _ = PointCloudBodyBuilder._o3d_knn(means_, 3)
        mean3_sq_dist = sq_dist.mean(-1).clip(min=1e-6)
        scales = np.sqrt(mean3_sq_dist)
        scales = np.tile(scales[..., None], (1, 3))
        scales[:, 2] = 0.001  # make disks
        scales = torch.from_numpy(scales).float().cuda()
        scales = GaussianActivations.inv_scale(scales)

        quats = torch.zeros(
            (num_points, 4), dtype=torch.float32
        ).cuda()  # (n, 4) w x y z
        quats[:, 0] = 1.0

        means = torch.from_numpy(means_).float().cuda()

        opacities = torch.zeros((num_points,), dtype=torch.float32).cuda()

        colors = torch.zeros((num_points, 3), dtype=torch.float32).cuda()

        means.requires_grad = True
        opacities.requires_grad = True
        colors.requires_grad = True
        quats.requires_grad = True
        scales.requires_grad = True

        params = torch.nn.ParameterDict(
            {
                "means": means,
                "quats": quats,
                "scales": scales,
                "opacities": opacities,
                "colors": colors,
            }
        )

        return params

    @staticmethod
    def _o3d_knn(pts: np.ndarray, num_knn: int):
        indices = []
        sq_dists = []
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        for p in pcd.points:
            [_, i, d] = pcd_tree.search_knn_vector_3d(p, num_knn + 1)
            indices.append(i[1:])
            sq_dists.append(d[1:])
        return np.array(sq_dists), np.array(indices)

    @staticmethod
    def _create_optimizers_for_params(
        params: torch.nn.ParameterDict, learning_rates: dict[str, float]
    ) -> dict[str, torch.optim.Optimizer]:
        optimizers = {}
        for name, learning_rate in learning_rates.items():
            assert name in params, f"Name {name} not in params"
            optimizers[name] = torch.optim.Adam([params[name]], lr=learning_rate)
        return optimizers
