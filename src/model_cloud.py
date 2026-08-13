"""CAD model as a point cloud, prepared once and reused for every instance."""

from dataclasses import dataclass, field

import cv2
import numpy as np
import open3d as o3d


@dataclass
class ModelCloud:
    """The CAD surface sampled at two densities.

    ``coarse`` drives global registration (FPFH features want a uniform,
    moderately dense cloud); ``fine`` drives ICP refinement, where more points
    give a better-conditioned point-to-plane solve. ``mesh_path`` lets the
    final polish stage query the actual triangle surface.
    """

    coarse: o3d.geometry.PointCloud
    fine: o3d.geometry.PointCloud
    fpfh: o3d.pipelines.registration.Feature
    voxel: float
    mesh_path: str
    plate_axis: np.ndarray = field(default=None)     # unit, model frame
    plate_span: tuple = field(default=None)          # (min, max) along axis
    hole_centres: np.ndarray = field(default=None)   # (k, 3) mid-plate
    hole_radii: np.ndarray = field(default=None)     # (k,) metres


def load_model_cloud(ply_path: str, voxel: float = 0.003,
                     fine_points: int = 20000) -> ModelCloud:
    """Sample the CAD surface and precompute FPFH features.

    Poisson-disk sampling interpolates the mesh's vertex normals, so both
    clouds carry correct outward normals without re-estimation.

    Args:
        ply_path: The CAD model, metres, in the object frame.
        voxel: Downsampling voxel for the registration cloud, metres.
        fine_points: Poisson-disk sample count for the ICP cloud.
    """
    mesh = o3d.io.read_triangle_mesh(ply_path)
    mesh.compute_vertex_normals()

    fine = mesh.sample_points_poisson_disk(fine_points)
    coarse = fine.voxel_down_sample(voxel)

    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        coarse,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5.0, max_nn=100))
    model = ModelCloud(coarse=coarse, fine=fine, fpfh=fpfh, voxel=voxel,
                       mesh_path=ply_path)
    _find_holes(model)
    return model


def _find_holes(model: ModelCloud, resolution: float = 0.0005) -> None:
    """Locate the part's through-holes in the model frame.

    The part is a plate; looking straight down its thinnest axis, the
    through-holes appear as enclosed background regions of the rasterised
    silhouette. Their centres are strong image features later (a hole
    belongs to exactly one instance), so their model-frame positions are
    computed once here.
    """
    pts = np.asarray(model.fine.points)
    centred = pts - pts.mean(axis=0)
    _, vectors = np.linalg.eigh(centred.T @ centred)
    axis = vectors[:, 0]                     # thinnest direction: plate normal
    b1, b2 = vectors[:, 2], vectors[:, 1]    # in-plate basis

    u, v = pts @ b1, pts @ b2
    u0, v0 = u.min(), v.min()
    cols = np.rint((u - u0) / resolution).astype(np.int64)
    rows = np.rint((v - v0) / resolution).astype(np.int64)
    sil = np.zeros((rows.max() + 1, cols.max() + 1), np.uint8)
    sil[rows, cols] = 1
    sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    contours, hierarchy = cv2.findContours(sil, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_NONE)
    centres, radii = [], []
    span = pts @ axis
    mid = 0.5 * (span.min() + span.max())
    for contour, info in zip(contours, hierarchy[0] if hierarchy is not None else []):
        if info[3] == -1:
            continue
        area = cv2.contourArea(contour)
        if area < 40:                        # sub-millimetre pits are noise
            continue
        m = cv2.moments(contour)
        cu = m["m10"] / m["m00"] * resolution + u0
        cv_ = m["m01"] / m["m00"] * resolution + v0
        centres.append(b1 * cu + b2 * cv_ + axis * mid)
        radii.append(np.sqrt(area / np.pi) * resolution)

    model.plate_axis = axis
    model.plate_span = (float(span.min()), float(span.max()))
    model.hole_centres = np.array(centres) if centres else np.empty((0, 3))
    model.hole_radii = np.array(radii)
