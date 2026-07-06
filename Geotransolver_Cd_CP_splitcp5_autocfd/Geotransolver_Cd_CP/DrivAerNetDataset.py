#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 20:54:56 2023

@author: Mohamed Elrefaie, mohamed.elrefaie@mit.edu mohamed.elrefaie@tum.de

This module is part of the research presented in the paper":
"DrivAerNet: A Parametric Car Dataset for Data-driven Aerodynamic Design and Graph-Based Drag Prediction".

The module defines a PyTorch Dataset for loading and transforming 3D car models from the DrivAerNet dataset
stored as STL files.
It includes functionality to subsample or pad the vertices of the models to a fixed number of points as well as
visualization methods for the DrivAerNet dataset.
"""
import os
import logging
import hashlib
import torch
import numpy as np
import pandas as pd
import trimesh
from torch.utils.data import Dataset
import pyvista as pv
import seaborn as sns
from typing import Callable, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataAugmentation:
    """
    Class encapsulating various data augmentation techniques for point clouds.
    """
    @staticmethod
    def translate_pointcloud(
        pointcloud: torch.Tensor,
        scale_range: Tuple[float, float] = (0.98, 1.02),
        translation_range: Tuple[float, float] = (-0.02, 0.02),
    ) -> torch.Tensor:
        """
        Applies mild isotropic scaling and translation to pointcloud.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            scale_range: A tuple specifying isotropic scaling range.
            translation_range: A tuple specifying xyz translation range.

        Returns:
            Augmented point cloud as a torch.Tensor.
        """
        # For Cd regression, avoid strong anisotropic deformation (x/y/z independent scaling),
        # which can break geometry-label consistency.
        s = float(np.random.uniform(low=scale_range[0], high=scale_range[1]))
        t = np.random.uniform(low=translation_range[0], high=translation_range[1], size=[3])
        augmented_pointcloud = (pointcloud * s + t).astype("float32")
        return torch.tensor(augmented_pointcloud, dtype=torch.float32)

    @staticmethod
    def jitter_pointcloud(pointcloud: torch.Tensor, sigma: float = 0.01, clip: float = 0.02) -> torch.Tensor:
        """
        Adds Gaussian noise to the pointcloud.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            sigma: Standard deviation of the Gaussian noise.
            clip: Maximum absolute value for noise.

        Returns:
            Jittered point cloud as a torch.Tensor.
        """
        # Add Gaussian noise and clip to the specified range
        N, C = pointcloud.shape
        jittered_pointcloud = pointcloud + torch.clamp(sigma * torch.randn(N, C), -clip, clip)
        return jittered_pointcloud

    @staticmethod
    def drop_points(pointcloud: torch.Tensor, drop_rate: float = 0.1) -> torch.Tensor:
        """
        Randomly removes points from the point cloud based on the drop rate.

        Args:
            pointcloud: The input point cloud as a torch.Tensor.
            drop_rate: The percentage of points to be randomly dropped.

        Returns:
            The point cloud with points dropped as a torch.Tensor.
        """
        # Calculate the number of points to drop
        num_drop = int(drop_rate * pointcloud.size(0))
        # Generate random indices for points to drop
        drop_indices = np.random.choice(pointcloud.size(0), num_drop, replace=False)
        # Drop the points
        keep_indices = np.setdiff1d(np.arange(pointcloud.size(0)), drop_indices)
        dropped_pointcloud = pointcloud[keep_indices, :]
        return dropped_pointcloud

class DrivAerNetDataset(Dataset):
    """
    PyTorch Dataset class for the DrivAerNet dataset, handling loading, transforming, and augmenting 3D car models.
    """
    def __init__(
        self,
        root_dir: str,
        csv_file: str,
        num_points: int,
        transform: Optional[Callable] = None,
        apply_augmentations: bool = True,
        normalize: bool = True,
        design_column: str = "Design",
        target_column: str = "Average Cd",
        file_suffix: str = ".stl",
        normalize_target: bool = False,
        target_mean: Optional[float] = None,
        target_std: Optional[float] = None,
        global_descriptor_mean: Optional[torch.Tensor] = None,
        global_descriptor_std: Optional[torch.Tensor] = None,
        deterministic_sampling: bool = False,
        deterministic_seed_base: int = 42,
        enable_point_cache: bool = False,
        point_cache_dir: str = "./cache/pointclouds",
        point_cache_version: str = "v1",
        enable_mesh_cache: bool = True,
        mesh_cache_dir: str = "./cache/meshes",
        mesh_cache_version: str = "v1",
    ):

        """
        Initializes the DrivAerNetDataset instance.

        Args:
            root_dir: Directory containing the STL files for 3D car models.
            csv_file: Path to the CSV file with metadata for the models.
            num_points: Fixed number of points to sample from each 3D model.
            transform: Optional transform function to apply to each sample.
        """
        self.root_dir = root_dir
        # Attempt to load the metadata CSV file and log errors if unsuccessful
        try:
            self.data_frame = pd.read_csv(csv_file)
        except Exception as e:
            logging.error(f"Failed to load CSV file: {csv_file}. Error: {e}")
            raise

        self.transform = transform  # Transformation function to be applied to each sample
        self.num_points = num_points  # Number of points each sample should have
        self.augmentation = DataAugmentation()  # Instantiate the DataAugmentation class
        self.apply_augmentations = apply_augmentations
        self.normalize = normalize
        self.design_column = design_column
        self.target_column = target_column
        self.file_suffix = file_suffix
        self.normalize_target = normalize_target
        self.target_mean = target_mean
        self.target_std = target_std
        self.global_descriptor_mean = global_descriptor_mean
        self.global_descriptor_std = global_descriptor_std
        self.deterministic_sampling = deterministic_sampling
        self.deterministic_seed_base = deterministic_seed_base
        self.enable_point_cache = enable_point_cache
        self.point_cache_dir = point_cache_dir
        self.point_cache_version = point_cache_version
        self.enable_mesh_cache = enable_mesh_cache
        self.mesh_cache_dir = mesh_cache_dir
        self.mesh_cache_version = mesh_cache_version

        # 容错: 自动清理列名两端空白，避免CSV头里出现隐藏空格导致读取失败
        self.data_frame.columns = [str(c).strip() for c in self.data_frame.columns]

        if self.design_column not in self.data_frame.columns:
            raise KeyError(
                f"Design列不存在: '{self.design_column}'. 可用列: {list(self.data_frame.columns)}"
            )
        if self.target_column not in self.data_frame.columns:
            raise KeyError(
                f"目标列不存在: '{self.target_column}'. 可用列: {list(self.data_frame.columns)}"
            )
        if self.enable_point_cache:
            os.makedirs(self.point_cache_dir, exist_ok=True)
            if not self.deterministic_sampling:
                logging.warning(
                    "点云缓存已启用且 deterministic_sampling=False。"
                    "缓存命中后将复用固定采样点云，以换取训练加速。"
                )
        if self.enable_mesh_cache:
            os.makedirs(self.mesh_cache_dir, exist_ok=True)

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.data_frame)

    def normalize_pointcloud(self, data: torch.Tensor) -> torch.Tensor:
        """
        Center point cloud without per-sample scaling.

        Args:
            data: Input data as a torch.Tensor.

        Returns:
            Centered data as a torch.Tensor.
        """
        return data - data.mean(dim=0, keepdim=True)

    def _sample_or_pad_vertices(
        self, vertices: torch.Tensor, num_points: int, rng: Optional[np.random.Generator] = None
    ) -> torch.Tensor:
        """
        Subsamples or pads the vertices of the model to a fixed number of points.

        Args:
            vertices: The vertices of the 3D model as a torch.Tensor.
            num_points: The desired number of points for the model.

        Returns:
            The vertices standardized to the specified number of points.
        """
        num_vertices = vertices.size(0)
        if rng is None:
            rng = np.random.default_rng()
        # Subsample the vertices if there are more than the desired number
        if num_vertices > num_points:
            indices = rng.choice(num_vertices, num_points, replace=False)
            vertices = vertices[indices]
        # Pad with zeros if there are fewer vertices than desired
        elif num_vertices < num_points:
            padding = torch.zeros((num_points - num_vertices, 3), dtype=torch.float32)
            vertices = torch.cat((vertices, padding), dim=0)
        return vertices

    def _sample_surface_points(
        self, mesh: trimesh.Trimesh, num_points: int, rng: Optional[np.random.Generator] = None
    ) -> torch.Tensor:
        """
        Uniformly samples points on mesh surface by triangle area.

        Falls back to vertex sampling/padding if surface sampling is unavailable.
        """
        if rng is None:
            rng = np.random.default_rng()

        try:
            triangles = np.asarray(mesh.triangles, dtype=np.float32)  # (F, 3, 3)
            area_faces = np.asarray(mesh.area_faces, dtype=np.float64)  # (F,)
            if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
                raise ValueError(f"Unexpected triangles shape: {triangles.shape}")
            area_sum = float(area_faces.sum())
            if area_sum <= 0.0:
                raise ValueError("Mesh total face area is non-positive")

            face_probs = area_faces / area_sum
            face_indices = rng.choice(len(area_faces), size=num_points, p=face_probs)
            tri = triangles[face_indices]  # (N, 3, 3)

            # Uniform sampling inside triangle by barycentric coordinates.
            u = rng.random(num_points)
            v = rng.random(num_points)
            sqrt_u = np.sqrt(u)
            w0 = 1.0 - sqrt_u
            w1 = sqrt_u * (1.0 - v)
            w2 = sqrt_u * v

            sampled_points = (
                w0[:, None] * tri[:, 0, :]
                + w1[:, None] * tri[:, 1, :]
                + w2[:, None] * tri[:, 2, :]
            ).astype(np.float32)
            return torch.from_numpy(sampled_points)
        except Exception as e:
            logging.warning(f"Surface sampling failed, fallback to vertex sampling. Error: {e}")

        # Fallback path for degenerate/invalid mesh cases.
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
        vertices = self._sample_or_pad_vertices(vertices, num_points, rng=rng)
        return vertices

    @staticmethod
    def compute_global_geometry_descriptors(full_vertices: torch.Tensor) -> torch.Tensor:
        mins = full_vertices.min(dim=0).values
        maxs = full_vertices.max(dim=0).values
        spans = (maxs - mins).clamp_min(1e-6)

        length = spans[0:1]
        width = spans[1:2]
        height = spans[2:3]
        frontal_area = width * height
        planform_area = length * width
        volume_proxy = length * width * height
        length_to_height = length / height.clamp_min(1e-6)
        width_to_height = width / height.clamp_min(1e-6)
        length_to_width = length / width.clamp_min(1e-6)
        stds = full_vertices.std(dim=0, unbiased=False)
        raw_z_min = mins[2:3]
        raw_z_max = maxs[2:3]
        raw_z_span = spans[2:3]

        return torch.cat(
            [
                length,
                width,
                height,
                frontal_area,
                planform_area,
                volume_proxy,
                length_to_height,
                width_to_height,
                length_to_width,
                stds,
                raw_z_min,
                raw_z_max,
                raw_z_span,
            ],
            dim=0,
        ).to(dtype=torch.float32)

    def __getitem__(self, idx: int, apply_augmentations: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample and its corresponding label from the dataset, with an option to apply augmentations.

        Args:
            idx (int): Index of the sample to retrieve.
            apply_augmentations (bool, optional): Whether to apply data augmentations. Defaults to True.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                The sample point cloud, standardized global geometry descriptors,
                and its label (Cd value).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Extract the relevant row from the DataFrame using the provided index
        row = self.data_frame.iloc[idx]
        design_id = str(row[self.design_column]).strip()
        cd_value = row[self.target_column]
        geometry_path = os.path.join(self.root_dir, f"{design_id}{self.file_suffix}")

        rng = None
        if self.deterministic_sampling:
            seed = int(self.deterministic_seed_base + int(idx))
            rng = np.random.default_rng(seed)

        vertices = None
        global_geometry_descriptors = None
        if self.enable_point_cache:
            vertices, global_geometry_descriptors = self._load_cached_sample(design_id)

        if vertices is None or global_geometry_descriptors is None:
            mesh_payload = None
            if self.enable_mesh_cache:
                mesh_payload = self._load_cached_mesh_payload(design_id)

            if mesh_payload is None:
                # Load the STL file and handle errors
                try:
                    mesh = trimesh.load(geometry_path, force='mesh')
                except Exception as e:
                    logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
                    raise
                mesh_payload = {
                    "vertices": np.asarray(mesh.vertices, dtype=np.float32),
                    "triangles": np.asarray(mesh.triangles, dtype=np.float32),
                    "area_faces": np.asarray(mesh.area_faces, dtype=np.float64),
                }
                if self.enable_mesh_cache:
                    self._save_cached_mesh_payload(design_id, mesh_payload)

            full_vertices = torch.from_numpy(mesh_payload["vertices"].astype(np.float32))
            global_geometry_descriptors = self.compute_global_geometry_descriptors(full_vertices)
            vertices = self._sample_surface_points_from_arrays(
                triangles=mesh_payload["triangles"],
                area_faces=mesh_payload["area_faces"],
                num_points=self.num_points,
                rng=rng,
                fallback_vertices=full_vertices,
            )
            if self.enable_point_cache:
                self._save_cached_sample(design_id, vertices, global_geometry_descriptors)

        if self.global_descriptor_mean is not None and self.global_descriptor_std is not None:
            global_geometry_descriptors = (
                global_geometry_descriptors - self.global_descriptor_mean
            ) / (self.global_descriptor_std + 1e-8)

        # Apply data augmentations if enabled
        if apply_augmentations is None:
            apply_augmentations = self.apply_augmentations
        if apply_augmentations:
            # Keep only mild jitter for Cd regression.
            # Translation is canceled by later centering normalization.
            vertices = self.augmentation.jitter_pointcloud(vertices, sigma=0.003, clip=0.01)

        # Apply optional transformations
        if self.transform:
            vertices = self.transform(vertices)

        # Normalize the features of the point cloud
        if self.normalize:
            vertices = self.normalize_pointcloud(vertices)

        cd_value = float(cd_value)
        if self.normalize_target:
            if self.target_mean is None or self.target_std is None:
                raise ValueError("normalize_target=True 时必须提供 target_mean 和 target_std")
            cd_value = (cd_value - self.target_mean) / (self.target_std + 1e-8)

        cd_value = torch.tensor(cd_value, dtype=torch.float32).view(-1)
        return vertices, global_geometry_descriptors, cd_value

    def _cache_file_path(self, design_id: str) -> str:
        key = (
            f"{self.point_cache_version}|{design_id}|{self.num_points}|"
            f"{self.file_suffix}|{int(self.normalize)}"
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return os.path.join(self.point_cache_dir, f"{digest}.npz")

    def _mesh_cache_file_path(self, design_id: str) -> str:
        key = (
            f"{self.mesh_cache_version}|{design_id}|{self.file_suffix}"
        )
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return os.path.join(self.mesh_cache_dir, f"{digest}.npz")

    def _load_cached_sample(
        self, design_id: str
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        cache_path = self._cache_file_path(design_id)
        if not os.path.exists(cache_path):
            return None, None
        try:
            payload = np.load(cache_path)
            vertices = torch.from_numpy(payload["vertices"].astype(np.float32))
            global_desc = torch.from_numpy(payload["global_desc"].astype(np.float32))
            return vertices, global_desc
        except Exception as e:
            logging.warning(f"读取点云缓存失败，回退到STL加载: {cache_path}, error={e}")
            return None, None

    def _save_cached_sample(
        self, design_id: str, vertices: torch.Tensor, global_desc: torch.Tensor
    ) -> None:
        cache_path = self._cache_file_path(design_id)
        if os.path.exists(cache_path):
            return
        try:
            np.savez_compressed(
                cache_path,
                vertices=vertices.detach().cpu().numpy().astype(np.float32),
                global_desc=global_desc.detach().cpu().numpy().astype(np.float32),
            )
        except Exception as e:
            logging.warning(f"写入点云缓存失败: {cache_path}, error={e}")

    def _load_cached_mesh_payload(self, design_id: str):
        cache_path = self._mesh_cache_file_path(design_id)
        if not os.path.exists(cache_path):
            return None
        try:
            payload = np.load(cache_path)
            return {
                "vertices": payload["vertices"].astype(np.float32),
                "triangles": payload["triangles"].astype(np.float32),
                "area_faces": payload["area_faces"].astype(np.float64),
            }
        except Exception as e:
            logging.warning(f"读取mesh缓存失败，回退到STL加载: {cache_path}, error={e}")
            return None

    def _save_cached_mesh_payload(self, design_id: str, mesh_payload) -> None:
        cache_path = self._mesh_cache_file_path(design_id)
        if os.path.exists(cache_path):
            return
        try:
            np.savez_compressed(
                cache_path,
                vertices=np.asarray(mesh_payload["vertices"], dtype=np.float32),
                triangles=np.asarray(mesh_payload["triangles"], dtype=np.float32),
                area_faces=np.asarray(mesh_payload["area_faces"], dtype=np.float64),
            )
        except Exception as e:
            logging.warning(f"写入mesh缓存失败: {cache_path}, error={e}")

    def _sample_surface_points_from_arrays(
        self,
        triangles: np.ndarray,
        area_faces: np.ndarray,
        num_points: int,
        rng: Optional[np.random.Generator] = None,
        fallback_vertices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if rng is None:
            rng = np.random.default_rng()
        try:
            triangles = np.asarray(triangles, dtype=np.float32)
            area_faces = np.asarray(area_faces, dtype=np.float64)
            if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
                raise ValueError(f"Unexpected triangles shape: {triangles.shape}")
            area_sum = float(area_faces.sum())
            if area_sum <= 0.0:
                raise ValueError("Mesh total face area is non-positive")
            face_probs = area_faces / area_sum
            face_indices = rng.choice(len(area_faces), size=num_points, p=face_probs)
            tri = triangles[face_indices]
            u = rng.random(num_points)
            v = rng.random(num_points)
            sqrt_u = np.sqrt(u)
            w0 = 1.0 - sqrt_u
            w1 = sqrt_u * (1.0 - v)
            w2 = sqrt_u * v
            sampled_points = (
                w0[:, None] * tri[:, 0, :]
                + w1[:, None] * tri[:, 1, :]
                + w2[:, None] * tri[:, 2, :]
            ).astype(np.float32)
            return torch.from_numpy(sampled_points)
        except Exception as e:
            logging.warning(f"Mesh-array surface sampling failed, fallback to vertices. Error: {e}")
            if fallback_vertices is None:
                raise
            return self._sample_or_pad_vertices(fallback_vertices, num_points, rng=rng)

    # Visualization methods for the DrivAerNetDataset class

    def visualize_mesh(self, idx):
        """
        Visualize the STL mesh for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function loads the mesh from the STL file corresponding to the design ID at the given index,
        wraps it using PyVista for visualization, and then sets up a PyVista plotter to display the mesh.
        """
        # Retrieve design ID and construct the file path for the STL file
        row = self.data_frame.iloc[idx]
        design_id = row['Design']
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")

        # Attempt to load the mesh from the STL file and handle potential errors
        try:
            mesh = trimesh.load(geometry_path, force='mesh')
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise

        # Convert the trimesh mesh to a PyVista mesh for visualization
        pv_mesh = pv.wrap(mesh)

        # Set up the PyVista plotter
        plotter = pv.Plotter()
        plotter.add_mesh(pv_mesh, color='lightgrey', show_edges=True)
        plotter.add_axes()

        # Define a specific camera position for a consistent viewing angle
        camera_position = [(-11.073024242161921, -5.621499358347753, 5.862225824910342),
                           (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
                           (0.34000174095454166, 0.10379556639001211, 0.9346792479485448)]
        plotter.camera_position = camera_position

        # Display the plotter window with the mesh
        plotter.show()

    def visualize_mesh_withNode(self, idx):
        """
        Visualizes the mesh for a specific design from the dataset with nodes highlighted.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function loads the mesh from the STL file and highlights the nodes (vertices) of the mesh using spheres.
        It uses seaborn to obtain visually distinct colors for the mesh and nodes.
        """
        # Retrieve design ID and construct the file path for the STL file
        row = self.data_frame.iloc[idx]
        design_id = row['Design']
        geometry_path = os.path.join(self.root_dir, f"{design_id}.stl")

        # Attempt to load the mesh from the STL file and handle potential errors
        try:
            mesh = trimesh.load(geometry_path, force='mesh')
            pv_mesh = pv.wrap(mesh)
        except Exception as e:
            logging.error(f"Failed to load STL file: {geometry_path}. Error: {e}")
            raise

        # Set up the PyVista plotter
        plotter = pv.Plotter()
        sns_blue = sns.color_palette("colorblind")[0]  # Using seaborn to get a visually distinct blue color

        # Add the mesh to the plotter with light grey color and black edges
        plotter.add_mesh(pv_mesh, color='lightgrey', show_edges=True, edge_color='black')

        # Highlight nodes (vertices) of the mesh as blue spheres
        nodes = pv_mesh.points
        plotter.add_points(nodes, color=sns_blue, point_size=10, render_points_as_spheres=True)

        # Add axes for orientation and display the plotter window
        plotter.add_axes()
        plotter.show()

    def visualize_point_cloud(self, idx):
        """
        Visualizes the point cloud for a specific design from the dataset.

        Args:
            idx (int): Index of the design to visualize in the dataset.

        This function retrieves the vertices for the specified design, converts them into a point cloud,
        and uses the z-coordinate for color mapping. PyVista's Eye-Dome Lighting is enabled for improved depth perception.
        """
        # Retrieve vertices and corresponding CD value for the specified index
        vertices, _ = self.__getitem__(idx)
        vertices = vertices.numpy()

        # Convert vertices to a PyVista PolyData object for visualization
        point_cloud = pv.PolyData(vertices)
        colors = vertices[:, 2]  # Using the z-coordinate for color mapping
        point_cloud["colors"] = colors  # Add the colors to the point cloud

        # Set up the PyVista plotter
        plotter = pv.Plotter()

        # Add the point cloud to the plotter with color mapping based on the z-coordinate
        plotter.add_points(point_cloud, scalars="colors", cmap="Blues", point_size=3, render_points_as_spheres=True)

        # Enable Eye-Dome Lighting for better depth perception
        plotter.enable_eye_dome_lighting()

        # Add axes for orientation and display the plotter window
        plotter.add_axes()
        camera_position = [(-11.073024242161921, -5.621499358347753, 5.862225824910342),
                           (1.458462064391673, 0.002314306982062475, 0.6792134746589196),
                           (0.34000174095454166, 0.10379556639001211, 0.9346792479485448)]

        # Set the camera position
        plotter.camera_position = camera_position

        plotter.show()
    def visualize_augmentations(self, idx):
        """
        Visualizes various augmentations applied to the point cloud of a specific design in the dataset.

        Args:
            idx (int): Index of the sample in the dataset to be visualized.

        This function retrieves the original point cloud for the specified design and then applies a series of augmentations,
        including translation, jittering, and point dropping. Each version of the point cloud (original and augmented) is then
        visualized in a 2x2 grid using PyVista to illustrate the effects of these augmentations.
        """
        # Retrieve the original point cloud without applying any augmentations
        vertices, _ = self.__getitem__(idx, apply_augmentations=False)
        original_pc = pv.PolyData(vertices.numpy())

        # Apply translation augmentation to the original point cloud
        translated_pc = self.augmentation.translate_pointcloud(vertices.numpy())
        # Apply jitter augmentation to the translated point cloud
        jittered_pc = self.augmentation.jitter_pointcloud(translated_pc)
        # Apply point dropping augmentation to the jittered point cloud
        dropped_pc = self.augmentation.drop_points(jittered_pc)

        # Initialize a PyVista plotter with a 2x2 grid for displaying the point clouds
        plotter = pv.Plotter(shape=(2, 2))

        # Display the original point cloud in the top left corner of the grid
        plotter.subplot(0, 0)  # Select the first subplot
        plotter.add_text("Original Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(original_pc, color='black', point_size=3)  # Add the original point cloud to the plot

        # Display the translated point cloud in the top right corner of the grid
        plotter.subplot(0, 1)  # Select the second subplot
        plotter.add_text("Translated Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(pv.PolyData(translated_pc.numpy()), color='lightblue', point_size=3)  # Add the translated point cloud to the plot

        # Display the jittered point cloud in the bottom left corner of the grid
        plotter.subplot(1, 0)  # Select the third subplot
        plotter.add_text("Jittered Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(pv.PolyData(jittered_pc.numpy()), color='lightgreen', point_size=3)  # Add the jittered point cloud to the plot

        # Display the dropped point cloud in the bottom right corner of the grid
        plotter.subplot(1, 1)  # Select the fourth subplot
        plotter.add_text("Dropped Point Cloud", font_size=10)  # Add descriptive text
        plotter.add_mesh(pv.PolyData(dropped_pc.numpy()), color='salmon', point_size=3)  # Add the dropped point cloud to the plot

        # Display the plot with all point clouds
        plotter.show()


# Example usage
#if __name__ == '__main__':
    #dataset = DrivAerNetDataset(root_dir='../DrivAerNet_STLs_Combined',
    #                                 csv_file='../AeroCoefficients_DrivAerNet_FilteredCorrected.csv',
    #                                 num_points=500000)

    #dataset.visualize_mesh_withNode(300)  # Visualize the mesh of the first sample

    #dataset.visualize_point_cloud(300)  # Visualize the point cloud of the first sample