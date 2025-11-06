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
from scipy.spatial.transform import Rotation as R

from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy

from embodied_gaussians.scene_builders.domain import (
    Particles,
    Gaussians,
    Body,
    Ground,
    GaussianLearningRates,
    GaussianActivations,
    MaskedPosedImageAndDepth
)
from embodied_gaussians.scene_builders.warp_utils import find_distant_query_points

from .simple_visualizer import ellipsoid_meshes, sphere_meshes

logger = logging.getLogger(__name__)



@dataclass
class SimpleBodyBuilderSettings:
    ground: Ground = field(default_factory=Ground)
    particle_radius: float = (
        0.006  # Radius of particles in the resulting body. Choose a radius that represents the minimum geometric feature you want to capture
    )

    training_iterations: int = 1000  # Number of iterations to optimize the particles
    training_learning_rates: GaussianLearningRates = field(default_factory=lambda: GaussianLearningRates(
        means=0.0001,
    ))
    opacity_threshold: float = 0.5  # Opacity threshold for optimization

    voxel_size: float = (
        0.007  # Method creates a pointcloud from the depth images then downsamples it using this voxel size
    )
    outlier_radius: float = (
        0.01  # Method removes points that do not have enough neighbors within this radius
    )
    outlier_nb_points: int = (
        20  # Method removes points that do not have atleast this many neighbors in the radius
    )
    max_depth: float = 2.0
    cohesion_distance: float = 0.002


class SimpleBodyBuilder:

    @staticmethod
    def build(
        name: str,
        settings: SimpleBodyBuilderSettings,
        datapoints: list[MaskedPosedImageAndDepth],
        visualize: bool = False,
    ) -> Body:
        wp.init()

        # Reference: Physically Embodied Gaussian Splatting
        # https://openreview.net/pdf?id=AEq0onGrN2 (Figure 2)

        # ================= Step 1: Merge all datapoints into a single pointcloud =================
        pc = SimpleBodyBuilder._merge_into_pointcloud(datapoints, settings.max_depth)
        if pc is None:
            return None

        # ================ Step 2: Get the bounding box of the pointcloud =================
        obb = SimpleBodyBuilder._filter_and_get_bounding_box(
            pc, settings.outlier_radius, settings.outlier_nb_points
        )
        if obb is None:
            return None
        obb.color = (1, 0, 0)

        # ================ Step 3: Fill the bounding box with spheres =================
        sphere_means = SimpleBodyBuilder._fill_bounding_box_with_spheres(
            obb, settings.particle_radius
        )

        # ================ Step 4: Prune points not in masks =================
        mask = SimpleBodyBuilder._prune_points_not_in_masks(sphere_means, datapoints)
        sphere_means = sphere_means[mask]
        if sphere_means.shape[0] == 0:
            logger.warning("No points left after pruning")
            return None

        # ================ Step 5: Prune points below ground =================
        if settings.ground is not None:
            sphere_means = SimpleBodyBuilder._prune_points_below_ground(
                sphere_means, settings.ground
            )

        if sphere_means.shape[0] == 0:
            logger.warning("No points left after pruning points below ground")
            return None

        if visualize:
            o3d.visualization.draw_geometries(
                [
                    pc,
                    obb,
                    *sphere_meshes(
                        sphere_means, settings.particle_radius
                    ),
                ]
            )

        # ================ Step 6: Optimize particles =================
        particles = SimpleBodyBuilder._optimize_particles(
            initial_points=sphere_means,
            radius=settings.particle_radius,
            num_iterations=settings.training_iterations,
            learning_rates=settings.training_learning_rates,
            opacity_threshold=settings.opacity_threshold,
            ground=settings.ground,
            datapoints=datapoints,
            max_depth=settings.max_depth,
            cohesion_distance=settings.cohesion_distance,
            visualize=visualize,
        )

        if len(particles.means) == 0:
            logger.warning("No points left after particle optimization. Something went wrong.")
            return None

        if visualize:
            o3d.visualization.draw_geometries(
                [
                    pc,
                    obb,
                    *sphere_meshes(
                        particles.means, settings.particle_radius, particles.colors
                    ),
                ]
            )

        # ================ Step 7: Grow Gaussians =================
        gaussians = SimpleBodyBuilder._grow_gaussians(
            initial_points=sphere_means,
            radius=settings.particle_radius,
            num_iterations=settings.training_iterations,
            learning_rates=settings.training_learning_rates,
            datapoints=datapoints,
            min_scale=0.5*settings.particle_radius,
            max_scale=2.0*settings.particle_radius,
            max_depth=settings.max_depth,
            visualize=visualize,
        )
        mask = find_distant_query_points(
            settings.particle_radius * 2.3, gaussians.means, particles.means
        )
        gaussians = gaussians.mask(~mask)


        # ================ Step 8: Convert to body frame =================
        X_WB = SimpleBodyBuilder._convert_to_body_frame(gaussians, particles)

        if visualize:
            o3d.visualization.draw_geometries(
                [
                    o3d.geometry.TriangleMesh.create_coordinate_frame(0.1),
                    *sphere_meshes(particles.means, settings.particle_radius, particles.colors),
                    *ellipsoid_meshes(gaussians),
                ]
            )
        
        body = Body(
            name=name,
            X_WB=X_WB.tolist(),
            particles=particles,
            gaussians=gaussians,
        )

        return body


    @staticmethod
    def _filter_and_get_bounding_box(
        pc: o3d.geometry.PointCloud, outlier_radius: float, outlier_nb_points: float
    ) -> o3d.geometry.OrientedBoundingBox | None:
        cl, ind = pc.remove_radius_outlier(
            nb_points=outlier_nb_points, radius=outlier_radius
        )
        pc = pc.select_by_index(ind)
        if len(pc.points) == 0:
            logger.warning(
                "Could not find bounding box because no points left after outlier removal"
            )
            return None

        obb = pc.get_minimal_oriented_bounding_box()
        return obb

    @staticmethod
    def _merge_into_pointcloud(
        datapoints: list[MaskedPosedImageAndDepth], 
        max_depth: float
    ) -> o3d.geometry.PointCloud | None:
        all_pointclouds = []
        for datapoint in datapoints:

            if datapoint.mask is not None:
                datapoint.depth[datapoint.mask == 0] = 0.0

            w = datapoint.depth.shape[1]
            h = datapoint.depth.shape[0]
            fl_x, fl_y = datapoint.K[0, 0], datapoint.K[1, 1]
            cx, cy = datapoint.K[0, 2], datapoint.K[1, 2]
            intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fl_x, fl_y, cx, cy)
            depth_image = o3d.geometry.Image(datapoint.depth)
            if datapoint.image is not None:
                color_image = o3d.geometry.Image(datapoint.image)
                rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_image, depth_image, convert_rgb_to_intensity=False
                )
                pointcloud = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, intrinsics
                )
            else:
                pointcloud = o3d.geometry.PointCloud.create_from_depth_image(
                    depth_image,
                    intrinsics,
                    depth_scale=1.0 / datapoint.depth_scale,
                    depth_trunc=max_depth,
                )
            X_WC = datapoint.X_WC @ np.array(
                [[1, 0, 0, 0.0], [0, -1, 0, 0.0], [0, 0, -1, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )  # rotate areound x axis to make it in opencv standard
            pointcloud.transform(X_WC)
            all_pointclouds.append(pointcloud)

        final_pointcloud = o3d.geometry.PointCloud()
        for p in all_pointclouds:
            final_pointcloud += p

        if len(final_pointcloud.points) == 0:
            logger.warning("The pointcloud is empty")
            return None

        return final_pointcloud

    @staticmethod
    def _fill_bounding_box_with_spheres(
        obb: o3d.geometry.OrientedBoundingBox, radius: float
    ) -> np.ndarray:
        """
        Returns an array of shape (n, 3) where n is the number of points in the bounding box
        """
        X_WO = np.eye(4, dtype=np.float32)
        X_WO[:3, :3] = obb.R
        X_WO[:3, 3] = obb.get_center()
        extent = obb.extent - 2 * radius
        extent = np.ceil(extent / radius) * radius
        d = radius * 2.0
        n_x = int(extent[0] / d)
        n_y = int(extent[1] / d)
        n_z = int(extent[2] / d)
        pts = np.stack(
            np.meshgrid(
                np.linspace(-extent[0] / 2, extent[0] / 2, n_x + 1),
                np.linspace(-extent[1] / 2, extent[1] / 2, n_y + 1),
                np.linspace(-extent[2] / 2, extent[2] / 2, n_z + 2),
                indexing="ij",
            ),
            axis=-1,
        ).reshape(-1, 3)

        pts = (X_WO[:3, :3] @ pts.T + X_WO[:3, 3:4]).T

        return pts

    @staticmethod
    def _prune_points_not_in_masks(
        points: np.ndarray, datapoints: list[MaskedPosedImageAndDepth]
    ):
        assert points.shape[1] == 3

        final_mask = np.zeros((points.shape[0],), dtype=bool)

        for datapoint in datapoints:
            if datapoint.mask is None:
                logger.warning("Mask is None for datapoint")
                continue
            remove_mask = np.zeros((points.shape[0],), dtype=bool)
            width, height = datapoint.mask.shape[1], datapoint.mask.shape[0]
            uv, valid_mask = mask = SimpleBodyBuilder._project_points(
                points, datapoint.K, datapoint.X_WC, width, height
            )
            uv = uv[valid_mask]
            inds = np.where(valid_mask)[0]

            mask = datapoint.mask == 0  # 1 is object, 0 is background, 2 is occlusion
            inds_to_remove = inds[mask[uv[:, 1], uv[:, 0]]]
            remove_mask[inds_to_remove] = True
            final_mask |= remove_mask

        return ~final_mask

    @staticmethod
    def _prune_points_below_ground(xyz: np.ndarray, ground: Ground) -> np.ndarray:
        a, b, c, d = ground.plane
        mask = (a * xyz[:, 0] + b * xyz[:, 1] + c * xyz[:, 2] + d) > 0
        xyz = xyz[mask]
        return xyz

    @staticmethod
    def _optimize_particles(
        initial_points: np.ndarray,
        radius: float,
        num_iterations: int,
        learning_rates: GaussianLearningRates,
        ground: Ground,
        opacity_threshold: float,
        datapoints: list[MaskedPosedImageAndDepth],
        max_depth: float,
        cohesion_distance: float = 0.001,
        visualize: bool = False,
    ) -> Particles:
        assert initial_points.shape[1] == 3

        ground = torch.tensor(ground.plane).float().cuda()
        params = SimpleBodyBuilder._create_initial_gaussian_state(
            initial_points, radius
        )

        gt_data = SimpleBodyBuilder._get_rasterization_groundtruth(datapoints, max_depth)
        optimizers = SimpleBodyBuilder._create_optimizers_for_params(
            params,
            {
                "means": learning_rates.means,
                "opacities": learning_rates.opacities,
                "colors": learning_rates.colors,
            },
        )

        num_images = gt_data.images.shape[0]
        backgrounds = torch.rand((num_iterations, 3)).float().cuda()
        for i in tqdm(range(num_iterations)):

            # ==================================================================
            # Step 1: Gaussian Splatting Step
            # ==================================================================

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

            loss = torch.nn.functional.mse_loss(render_colors[..., :3], gt_data.images)
            # for j in range(len(gt_data.depth_masks)):
            #     depth_mask = gt_data.depth_masks[j]
            #     depth_loss = torch.nn.functional.mse_loss(render_colors[j, ..., -1][depth_mask], gt_data.depths[j][depth_mask])
            #     loss += 0.01 * depth_loss
            params.zero_grad()
            loss.backward()

            for optimizer in optimizers.values():
                optimizer.step()

            if visualize and i % 100 == 0:
                depth = render_colors[..., -1]
                rgb = render_colors[..., :3].detach().cpu().numpy()
                for cam in range(rgb.shape[0]):
                    cv2.imshow(f"color_{cam}", rgb[cam])
                    cv2.imshow(f"groundtruth_{cam}", gt_data.images[cam].detach().cpu().numpy())
                    break
                cv2.waitKey(1)

            # ==================================================================
            # Step 2: Solve Collisions
            # ==================================================================

            SimpleBodyBuilder._solve_collisions_jacobi(
                params["means"].detach(),
                GaussianActivations.scale(params["scales"].detach()[..., 0]),
                ground,
                num_iterations=8,
                relaxation=0.2,
                cohesian_distance=cohesion_distance,
            )

        if visualize:
            cv2.destroyAllWindows()

        mask = GaussianActivations.opacity(params["opacities"]) > opacity_threshold

        return Particles(
            means=params["means"][mask].detach().cpu().numpy(),
            quats=GaussianActivations.quat(params["quats"][mask]).detach().cpu().numpy(),
            radii=GaussianActivations.scale(params["scales"][mask]).detach().cpu().numpy()[..., 0],
            colors=GaussianActivations.color(params["colors"][mask]).detach().cpu().numpy(),
        )

    @staticmethod
    def _grow_gaussians(
        initial_points: np.ndarray,
        radius: float,
        num_iterations: int,
        learning_rates: GaussianLearningRates,
        datapoints: list[MaskedPosedImageAndDepth],
        min_scale: float,
        max_scale: float,
        max_depth: float,
        visualize: bool = False,
    ):

        assert initial_points.shape[1] == 3
        params = SimpleBodyBuilder._create_initial_gaussian_state(
            initial_points, radius
        )

        gt_data = SimpleBodyBuilder._get_rasterization_groundtruth(datapoints, max_depth=max_depth)
        optimizers = SimpleBodyBuilder._create_optimizers_for_params(
            params,
            {
                "means": learning_rates.means,
                "opacities": learning_rates.opacities,
                "colors": learning_rates.colors,
                "quats": learning_rates.quats,
                "scales": learning_rates.scales,
            },
        )
        inv_min_scale = GaussianActivations.inv_scale(torch.tensor([min_scale, min_scale, min_scale]).cuda())
        inv_max_scale = GaussianActivations.inv_scale(torch.tensor([max_scale, max_scale, max_scale]).cuda())
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

            loss = torch.nn.functional.mse_loss(render_colors[..., :3], gt_data.images)
            params.zero_grad()
            loss.backward()

            for optimizer in optimizers.values():
                optimizer.step()
            
            params["scales"].detach().clamp_(inv_min_scale, inv_max_scale)

            if visualize and i % 100 == 0:
                depth = render_colors[..., -1]
                rgb = render_colors[..., :3].detach().cpu().numpy()
                for cam in range(rgb.shape[0]):
                    cv2.imshow(f"color_{cam}", rgb[cam])
                    break
                cv2.waitKey(1)

        if visualize:
            cv2.destroyAllWindows()

        
        return Gaussians(
            means=params["means"].detach().cpu().numpy(),
            quats=GaussianActivations.quat(params["quats"]).detach().cpu().numpy(),
            scales=GaussianActivations.scale(params["scales"]).detach().cpu().numpy(),
            opacities=GaussianActivations.opacity(params["opacities"]).detach().cpu().numpy(),
            colors=GaussianActivations.color(params["colors"]).detach().cpu().numpy(),
        )
    
    @staticmethod
    def _convert_to_body_frame(gaussians: Gaussians, particles: Particles):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(particles.means)
        obb: o3d.geometry.OrientedBoundingBox = pc.get_minimal_oriented_bounding_box()
        X_WB = np.eye(4, dtype=np.float32)
        X_WB[:3, :3] = obb.R
        X_WB[:3, 3] = obb.get_center()
        X_BW = np.linalg.inv(X_WB)

        particles_means = np.asarray(particles.means)
        gaussians_means = np.asarray(gaussians.means)

        particles_means = (X_BW[:3, :3] @ particles_means.T + X_BW[:3, 3:4]).T
        gaussians_means = (X_BW[:3, :3] @ gaussians_means.T + X_BW[:3, 3:4]).T
        particles.means = particles_means.tolist()
        gaussians.means = gaussians_means.tolist()
        for i, quat in enumerate(gaussians.quats):
            quat = R.from_quat(quat, scalar_first=True).as_matrix()
            quat = X_BW[:3, :3] @ quat
            quat = R.from_matrix(quat).as_quat(scalar_first=True)
            gaussians.quats[i] = quat.tolist()
        return X_WB

    @staticmethod
    def _project_points(
        points: np.ndarray, K: np.ndarray, X_WC: np.ndarray, width: int, height: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        points: (n, 3)
        X_WC expected in blender standard
        returns: (n, 2) in pixel coordinates and (n,) boolean mask where True means the point is in front of the camera and in the image
        """
        X_WC = X_WC @ np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0.0, 0.0, 0.0, 1.0]]
        )  # rotate areound x axis to make it in opencv standard
        X_CW = np.linalg.inv(X_WC)
        p_C = X_CW[:3, :3] @ points.T + X_CW[:3, 3:4]
        p_C = K @ p_C
        p_C /= p_C[2:3, :]
        p_C = p_C.astype(np.int32)
        mask = (
            (p_C[0, :] >= 0)
            & (p_C[0, :] < width)
            & (p_C[1, :] >= 0)
            & (p_C[1, :] < height)
            & (p_C[2, :] > 0)
        )
        return p_C[:2, :].T, mask

    @staticmethod
    def _solve_collisions_jacobi(
        xyz: torch.Tensor,
        radii: torch.Tensor,
        ground: torch.Tensor,
        num_iterations: int = 8,
        relaxation: float = 0.2,
        cohesian_distance: float = 0.001,
    ):
        grid = wp.HashGrid(128, 128, 128)
        max_radius = radii.max()
        deltas = torch.empty_like(xyz)
        with torch.no_grad():
            xyz_warp = wp.from_torch(xyz, dtype=wp.vec3f)
            for _ in range(num_iterations):
                grid.build(xyz_warp, max_radius)
                deltas.zero_()
                wp.launch(
                    kernel=solve_particle_particle_collisions,
                    dim=xyz.shape[0],
                    inputs=[
                        grid.id,
                        xyz_warp,
                        radii,
                        ground,
                        max_radius,
                        relaxation,
                        cohesian_distance,
                    ],
                    outputs=[
                        deltas,
                    ],
                )
                xyz.add_(deltas)


    @staticmethod
    def _get_rasterization_groundtruth(datapoints: list[MaskedPosedImageAndDepth], max_depth: float):
        X_CWs = []
        Ks = []
        gts = []
        depth_gts = []
        width, height = datapoints[0].image.shape[1], datapoints[0].image.shape[0]
        depth_masks = []
        masks = []
        for datapoint in datapoints:
            assert (
                datapoint.image.shape[1] == width
            ), "All images must have the same width"
            assert (
                datapoint.image.shape[0] == height
            ), "All images must have the same height"

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
            depth_mask = (depth > 0).__and__(depth < max_depth)
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
            ["images", "depths", "X_CWs", "Ks", "depth_masks", "masks", "width", "height"],
        )(gts, depth_gts, X_CWs, Ks, depth_masks, masks, width, height)

    @staticmethod
    def _create_initial_gaussian_state(means_: np.ndarray, radius: float):
        num_points = means_.shape[0]

        quats = torch.zeros(
            (num_points, 4), dtype=torch.float32
        ).cuda()  # (n, 4) w x y z
        quats[:, 0] = 1.0

        scales = torch.ones((num_points, 3), dtype=torch.float32).cuda() * radius
        scales = GaussianActivations.inv_scale(scales)

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
    def _create_optimizers_for_params(
        params: torch.nn.ParameterDict, learning_rates: dict[str, float]
    ) -> dict[str, torch.optim.Optimizer]:
        optimizers = {}
        for name, learning_rate in learning_rates.items():
            assert name in params, f"Name {name} not in params"
            optimizers[name] = torch.optim.Adam([params[name]], lr=learning_rate)
        return optimizers
    



@wp.kernel
def solve_particle_particle_collisions(
    grid: wp.uint64,
    particle_x: wp.array(dtype=wp.vec3),  # type: ignore
    particle_radius: wp.array(dtype=float),  # type: ignore
    ground: wp.array(dtype=float),  # type: ignore
    max_radius: float,
    relaxation: float,
    k_cohesion: float,
    # outputs
    deltas: wp.array(dtype=wp.vec3),  # type: ignore
):
    tid = wp.tid()

    # order threads by cell
    i = wp.hash_grid_point_id(grid, tid)
    if i == -1:
        # hash grid has not been built yet
        return

    x = particle_x[i]
    radius = particle_radius[i]

    # particle contact
    query = wp.hash_grid_query(grid, x, radius + max_radius + k_cohesion)
    index = int(0)

    delta = wp.vec3(0.0)
    w1 = 1.0

    while wp.hash_grid_query_next(query, index):
        # compute distance to point
        n = x - particle_x[index]
        d = wp.length(n) + 1e-20
        w2 = 1.0
        err = d - radius - particle_radius[index]
        denom = w1 + w2
        if err <= k_cohesion:
            n = n / d
            lambda_n = err
            delta_n = n * lambda_n
            delta += (-delta_n) / denom * w1

    # ground
    n = wp.vec3(ground[0], ground[1], ground[2])
    c = wp.min(wp.dot(n, x) + ground[3] - particle_radius[tid], 0.0)
    if c <= 0.0:
        lambda_n = c
        delta_n = n * lambda_n
        delta += -delta_n

    wp.atomic_add(deltas, i, delta * relaxation)
