from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from PIL import ImageColor
from pyquaternion import Quaternion

from navsim.common.dataclasses import Annotations, Camera, Lidar
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
from navsim.visualization.config import AGENT_CONFIG
from navsim.visualization.lidar import filter_lidar_pc, get_lidar_pc_color


def add_camera_ax(ax: plt.Axes, camera: Camera) -> plt.Axes:
    """
    Adds camera image to matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :return: ax object with image
    """
    ax.imshow(camera.image)
    return ax


def add_lidar_to_camera_ax(ax: plt.Axes, camera: Camera, lidar: Lidar) -> plt.Axes:
    """
    Adds camera image with lidar point cloud on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param lidar: navsim lidar dataclass
    :return: ax object with image
    """

    image, lidar_pc = camera.image.copy(), lidar.lidar_pc.copy()
    image_height, image_width = image.shape[:2]

    lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar_pc))

    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
        lidar_pc,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=(image_height, image_width),
    )

    for (x, y), color in zip(pc_in_cam[pc_in_fov_mask], lidar_pc_colors[pc_in_fov_mask]):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    ax.imshow(image)
    return ax


def add_annotations_to_camera_ax(ax: plt.Axes, camera: Camera, annotations: Annotations) -> plt.Axes:
    """
    Adds camera image with bounding boxes on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param annotations: navsim annotations dataclass
    :return: ax object with image
    """

    box_labels = annotations.names
    boxes = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions, box_dimensions, box_heading = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.DIMENSION],
        boxes[:, BoundingBoxIndex.HEADING],
    )
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    # Then draw project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, box_labels = box_corners[valid_corners], box_labels[valid_corners]
    image = _plot_rect_3d_on_img(camera.image.copy(), box_corners, box_labels)

    ax.imshow(image)
    return ax


def _transform_annotations_to_camera(
    boxes: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Helper function to transform bounding boxes into camera frame
    TODO: Refactor
    :param boxes: array representation of bounding boxes
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :return: bounding boxes in camera coordinates
    """

    locs, rots = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.HEADING :],
    )
    dims_cam = boxes[
        :, [BoundingBoxIndex.LENGTH, BoundingBoxIndex.HEIGHT, BoundingBoxIndex.WIDTH]
    ]  # l, w, h -> l, h, w

    rots_cam = np.zeros_like(rots)
    for idx, rot in enumerate(rots):
        rot = Quaternion(axis=[0, 0, 1], radians=rot)
        rot = Quaternion(matrix=sensor2lidar_rotation).inverse * rot
        rots_cam[idx] = -rot.yaw_pitch_roll[0]

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    locs_cam = np.concatenate([locs, np.ones_like(locs)[:, :1]], -1)  # -1, 4
    locs_cam = lidar2cam_rt.T @ locs_cam.T
    locs_cam = locs_cam.T
    locs_cam = locs_cam[:, :-1]
    return np.concatenate([locs_cam, dims_cam, rots_cam], -1)


def _rotation_3d_in_axis(points: npt.NDArray[np.float32], angles: npt.NDArray[np.float32], axis: int = 0):
    """
    Rotate 3D points by angles according to axis.
    TODO: Refactor
    :param points: array of points
    :param angles: array of angles
    :param axis: axis to perform rotation, defaults to 0
    :raises value: _description_
    :raises ValueError: if axis invalid
    :return: rotated points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, zeros, -rot_sin]),
                np.stack([zeros, ones, zeros]),
                np.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, -rot_sin, zeros]),
                np.stack([rot_sin, rot_cos, zeros]),
                np.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                np.stack([zeros, rot_cos, -rot_sin]),
                np.stack([zeros, rot_sin, rot_cos]),
                np.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def _plot_rect_3d_on_img(
    image: npt.NDArray[np.float32],
    box_corners: npt.NDArray[np.float32],
    box_labels: List[str],
    thickness: int = 3,
) -> npt.NDArray[np.uint8]:
    """
    Plot the boundary lines of 3D rectangular on 2D images.
    TODO: refactor
    :param image:  The numpy array of image.
    :param box_corners: Coordinates of the corners of 3D, shape of [N, 8, 2].
    :param box_labels: labels of boxes for coloring
    :param thickness: pixel width of liens, defaults to 3
    :return: image with 3D bounding boxes
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    for i in range(len(box_corners)):
        layer = tracked_object_types[box_labels[i]]
        color = ImageColor.getcolor(AGENT_CONFIG[layer]["fill_color"], "RGB")
        corners = box_corners[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(
                image,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return image.astype(np.uint8)


def _transform_points_to_image(
    points: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    image_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param points: points in camera frame
    :param intrinsic: camera intrinsics
    :param image_shape: shape of image in pixel
    :param eps: lower threshold of points, defaults to 1e-3
    :return: points in pixel coordinates, mask of values in frame
    """
    points = points[:, :3]

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

    pc_img = np.concatenate([points, np.ones_like(points)[:, :1]], -1)
    pc_img = viewpad @ pc_img.T
    pc_img = pc_img.T

    cur_pc_in_fov = pc_img[:, 2] > eps
    pc_img = pc_img[..., 0:2] / np.maximum(pc_img[..., 2:3], np.ones_like(pc_img[..., 2:3]) * eps)
    if image_shape is not None:
        img_h, img_w = image_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (pc_img[:, 0] < (img_w - 1))
            & (pc_img[:, 0] > 0)
            & (pc_img[:, 1] < (img_h - 1))
            & (pc_img[:, 1] > 0)
        )
    return pc_img, cur_pc_in_fov


def _count_lidar_points_in_boxes(
    lidar_pc: npt.NDArray[np.float32], boxes: npt.NDArray[np.float32]
) -> npt.NDArray[np.int32]:
    """Count lidar returns that fall inside each 3D bounding box (ego frame)."""

    points = lidar_pc[LidarIndex.POSITION, :].T  # (N, 3)

    centers = boxes[:, BoundingBoxIndex.POSITION]
    dims = boxes[:, BoundingBoxIndex.DIMENSION]
    headings = boxes[:, BoundingBoxIndex.HEADING]

    cos_h = np.cos(-headings)
    sin_h = np.sin(-headings)
    rot_mats = np.stack(
        [
            np.stack([cos_h, -sin_h, np.zeros_like(cos_h)], axis=-1),
            np.stack([sin_h, cos_h, np.zeros_like(cos_h)], axis=-1),
            np.tile(np.array([0.0, 0.0, 1.0]), (len(cos_h), 1)),
        ],
        axis=1,
    )  # (B, 3, 3)

    rel_points = points[None, :, :] - centers[:, None, :]  # (B, N, 3)
    aligned = np.einsum("bij,bnj->bni", rot_mats, rel_points)

    half_dims = dims / 2.0
    inside = np.all(np.abs(aligned) <= (half_dims[:, None, :] + 1e-6), axis=-1)
    return inside.sum(axis=1).astype(np.int32)

def render_dynamic_object_mask(
    annotations: Annotations,
    camera: Camera,
    lidar: Optional[Lidar] = None,
    speed_threshold: float = 1.0,
    min_lidar_points: int = 1,
) -> Tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32]]:
    """Rasterize a per-pixel mask of moving objects in the camera view.

    This helper mirrors the Waymo snippet that thresholds on per-box speed, filters
    out boxes without lidar returns, and uses projected 3D boxes to fill a mask.
    It intentionally omits the occlusion-based filtering step because lidar
    availability already provides an equivalent visibility check.

    Args:
        annotations: Frame annotations (boxes, labels, velocities) in ego frame.
        camera: Camera dataclass with RGB image, intrinsics, and sensor2lidar extrinsics.
        lidar: Optional lidar dataclass used to count points inside each box; boxes
            with fewer than ``min_lidar_points`` returns are dropped before mask
            rasterization (mirroring Waymo's lidar-availability filtering).
        speed_threshold: Minimum speed (m/s) for an object to be considered dynamic.
        min_lidar_points: Minimum lidar returns required to keep a box. Ignored if
            ``lidar`` is ``None`` or contains no point cloud.

    Returns:
        mask: ``uint8`` image aligned with ``camera.image`` where dynamic pixels are 255
            and static/empty pixels are 0.
        visible_fraction: Per-box visibility indicators aligned with ``annotations``
            (1 for boxes that met speed/points thresholds and projected into the image,
            0 otherwise).
    """

    speeds = np.linalg.norm(annotations["gt_velocity_3d"][:, :2], axis=1)
    moving_mask = speeds > speed_threshold

    # 用 lidar 点数做过滤（boxes 在 ego frame）
    if lidar is not None and len(annotations["gt_boxes"]) > 0:
        lidar_counts = _count_lidar_points_in_boxes(lidar, annotations["gt_boxes"])
        moving_mask &= lidar_counts >= min_lidar_points

    image_height, image_width = 1080, 1920
    if not moving_mask.any():
        return (
            np.zeros((image_height, image_width), dtype=np.uint8),
            np.zeros_like(speeds, dtype=np.float32),
        )

    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])

    # 选出运动 box（仍在 ego frame）
    moving_boxes = annotations["gt_boxes"][moving_mask]
    moving_speeds = speeds[moving_mask]

    # ===== 关键：先把 box 从 ego -> camera 坐标系 =====
    moving_boxes_cam = _transform_annotations_to_camera(
        moving_boxes,
        camera['sensor2lidar_rotation'],
        camera['sensor2lidar_translation'],
    )

    # 在 camera frame 里构造 8 个角点
    corners = (
        moving_boxes_cam[:, BoundingBoxIndex.DIMENSION].reshape([-1, 1, 3])
        * corners_norm.reshape([1, 8, 3])
    )
    corners = _rotation_3d_in_axis(
        corners, moving_boxes_cam[:, BoundingBoxIndex.HEADING], axis=1
    )
    corners += moving_boxes_cam[:, BoundingBoxIndex.POSITION].reshape(-1, 1, 3)

    # 投影到图像（points 已经在 camera frame，逻辑同 add_annotations_to_camera_ax）
    projected_corners, corners_pc_in_fov = _transform_points_to_image(
        corners.reshape(-1, 3),
        camera['cam_intrinsic'],
        image_shape=(image_height, image_width),
        eps=1e-3,
    )
    projected_corners = projected_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)

    moving_visible = corners_pc_in_fov.any(axis=1)
    visible_fraction = np.zeros_like(speeds, dtype=np.float32)
    visible_fraction[moving_mask] = moving_visible.astype(np.float32)

    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for box_idx, (polygon_2d, in_fov) in enumerate(
        zip(projected_corners, corners_pc_in_fov)
    ):
        if not in_fov.all():
            continue
        polygon = cv2.convexHull(polygon_2d.astype(np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], int(np.clip(moving_speeds[box_idx], 0, 255)))

    mask = np.clip((mask > 0).astype(np.uint8) * 255, 0, 255)

    return mask, visible_fraction


# OPENCV2DATASET = np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])

OPENCV2DATASET = np.eye(4)

DATASET2OPENCV = np.linalg.inv(OPENCV2DATASET).astype(np.float32)  # 纯旋转的话也等于 OPENCV2DATASET.T


def _transform_pcs_to_images(
    lidar_pc: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    img_shape: Optional[Tuple[int, int]] = None,
    tar_shape = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param lidar_pc: lidar point cloud
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :param intrinsic: camera intrinsics
    :param img_shape: image shape in pixels, defaults to None
    :param eps: threshold for lidar pc height, defaults to 1e-3
    :return: lidar pc in pixel coordinates, mask of values in frame
    """
    pc_xyz = lidar_pc[LidarIndex.POSITION, :].T

    # lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    # lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    # lidar2cam_rt = np.eye(4)
    # lidar2cam_rt[:3, :3] = lidar2cam_r.T
    # lidar2cam_rt[3, :3] = -lidar2cam_t

    # --- cam(ds) -> lidar (given) ---
    R_c2l = sensor2lidar_rotation.astype(np.float32)          # (3,3)
    t_c2l = sensor2lidar_translation.reshape(3, 1).astype(np.float32)  # (3,1)

    # --- lidar -> cam(ds) ---
    R_l2c = R_c2l.T
    t_l2c = -R_l2c @ t_c2l

    T_l2c_ds = np.eye(4, dtype=np.float32)
    T_l2c_ds[:3, :3] = R_l2c
    T_l2c_ds[:3,  3] = t_l2c[:, 0]

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    # lidar2img_rt = viewpad @ lidar2cam_rt.T

    # --- key line: cam(ds) axes -> cam(opencv) axes, BEFORE K ---
    T_l2c_cv = DATASET2OPENCV @ T_l2c_ds

    lidar2img = viewpad @ T_l2c_cv

    # cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], -1)
    # cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T
    # cur_pc_cam = cur_pc_cam.T

    pts_h = np.concatenate([pc_xyz, np.ones((pc_xyz.shape[0], 1), np.float32)], axis=1)  # (N,4)
    cur_pc_cam = (lidar2img @ pts_h.T).T  # (N,4)

    # cur_pc_in_fov = cur_pc_cam[:, 2] > eps
    depth = cur_pc_cam[:, 2]
    cur_pc_in_fov = depth > eps
    cur_pc_cam = cur_pc_cam[..., 0:2] / np.maximum(cur_pc_cam[..., 2:3], np.ones_like(cur_pc_cam[..., 2:3]) * eps)


    if tar_shape is not None:
        if img_shape is None:
            raise ValueError("img_shape must be provided when tar_shape is set (to compute scaling).")
        img_h, img_w = img_shape
        tar_h, tar_w = tar_shape

        sx = tar_w / float(img_w)
        sy = tar_h / float(img_h)
        cur_pc_pix = cur_pc_cam.copy()
        cur_pc_pix[:, 0] *= sx
        cur_pc_pix[:, 1] *= sy

        # FOV check in target space
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_pix[:, 0] < (tar_w - 1))
            & (cur_pc_pix[:, 0] > 0)
            & (cur_pc_pix[:, 1] < (tar_h - 1))
            & (cur_pc_pix[:, 1] > 0)
        )
        return cur_pc_pix, cur_pc_in_fov, depth


    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )
    return cur_pc_cam, cur_pc_in_fov, depth
