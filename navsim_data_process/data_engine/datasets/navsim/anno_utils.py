# navsim_annotation_utils.py
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum
from enum import Enum
import numpy.typing as npt

# ============== NavSim 官方枚举定义 ==============
class BoundingBoxIndex(Enum):
    """Index for bounding box array elements"""
    X = 0
    Y = 1
    Z = 2
    LENGTH = 3
    WIDTH = 4
    HEIGHT = 5
    HEADING = 6
    
    # Slice indices
    POSITION = slice(0, 3)
    DIMENSION = slice(3, 6)


class LidarIndex(Enum):
    """Index for lidar point cloud array elements"""
    X = 0
    Y = 1
    Z = 2
    INTENSITY = 3
    RING = 4
    ID = 5
    
    POSITION = slice(0, 3)


# ============== 类别映射 ==============
# NavSim 官方的类别映射 (从 navsim_scenario_utils)
TRACKED_OBJECT_TYPES = {
    'vehicle': 'VEHICLE',
    'pedestrian': 'PEDESTRIAN', 
    'bicycle': 'BICYCLE',
    'traffic_cone': 'TRAFFIC_CONE',
    'barrier': 'BARRIER',
    'czone_sign': 'CZONE_SIGN',
    'generic_object': 'GENERIC_OBJECT',
}

# 需要关注的动态类别
DYNAMIC_CLASSES = {'vehicle', 'pedestrian', 'bicycle'}

# 映射到CARLA格式
NAVSIM_TO_CARLA_TYPE = {
    'vehicle': 'car',
    'pedestrian': 'walker',
    'bicycle': 'bicycle',
    'generic_object': 'static',
    'traffic_cone': 'traffic_cone',
    'barrier': 'barrier',
}


@dataclass
class BoundingBox3D:
    """3D边界框数据类"""
    center: np.ndarray      # (x, y, z)
    size: np.ndarray        # (length, width, height)
    yaw: float              # heading (rad)
    velocity: np.ndarray    # (vx, vy, vz)
    label: str
    instance_token: Optional[str] = None
    track_token: Optional[str] = None


def parse_navsim_annotations(
    ann: Dict,
    filter_classes: Optional[List[str]] = None,
    max_distance: float = 50.0,
    min_distance: float = 1.5,
) -> List[BoundingBox3D]:
    """
    解析NavSim的annotations数据
    
    Args:
        ann: NavSim annotations 字典，包含:
            - gt_boxes: (N, 7) [x, y, z, length, width, height, heading]
            - gt_names: (N,) 类别名称
            - gt_velocity_3d: (N, 3) [vx, vy, vz]
            - instance_tokens: list
            - track_tokens: list
        filter_classes: 要保留的类别列表，None表示保留所有
        max_distance: 最大距离过滤
        min_distance: 最小距离过滤 (排除自车)
    
    Returns:
        List[BoundingBox3D]
    """
    boxes = []
    
    gt_boxes = np.asarray(ann.get('gt_boxes', []))
    gt_names = np.asarray(ann.get('gt_names', []))
    gt_velocity_3d = np.asarray(ann.get('gt_velocity_3d', np.zeros((len(gt_boxes), 3))))
    instance_tokens = ann.get('instance_tokens', [None] * len(gt_boxes))
    track_tokens = ann.get('track_tokens', [None] * len(gt_boxes))
    
    if len(gt_boxes) == 0:
        return boxes

    min_dis= 1e6
    
    for i in range(len(gt_boxes)):
        box = gt_boxes[i]
        label = str(gt_names[i]) if i < len(gt_names) else 'generic_object'
        
        # 类别过滤
        if filter_classes is not None and label not in filter_classes:
            continue
        
        # 位置信息
        center = np.array([
            box[BoundingBoxIndex.X.value],
            box[BoundingBoxIndex.Y.value], 
            box[BoundingBoxIndex.Z.value]
        ], dtype=np.float32)

        # if center[0] < 0:
        #     continue
        
        # 距离过滤
        dist = np.sqrt(center[0]**2 + center[1]**2)
        if dist > max_distance or dist < min_distance:
            continue

        # if dist < min_dis:
        #     min_dis = dist
        # else:
        #     continue
        
        # 尺寸信息
        size = np.array([
            box[BoundingBoxIndex.LENGTH.value],
            box[BoundingBoxIndex.WIDTH.value],
            box[BoundingBoxIndex.HEIGHT.value]
        ], dtype=np.float32)
        
        # 朝向
        yaw = float(box[BoundingBoxIndex.HEADING.value])
        
        # 速度
        vel = gt_velocity_3d[i] if i < len(gt_velocity_3d) else np.zeros(3)
        
        bbox = BoundingBox3D(
            center=center,
            size=size,
            yaw=yaw,
            velocity=np.asarray(vel, dtype=np.float32),
            label=label,
            instance_token=instance_tokens[i] if i < len(instance_tokens) else None,
            track_token=track_tokens[i] if i < len(track_tokens) else None,
        )
        boxes.append(bbox)
    
    return boxes


def count_lidar_points_in_boxes(
    lidar_pc: npt.NDArray[np.float32],
    boxes: npt.NDArray[np.float32]
) -> npt.NDArray[np.int32]:
    """
    统计每个box内的lidar点数 (官方函数)
    
    Args:
        lidar_pc: (6, N) or (N, 6) lidar点云
        boxes: (M, 7) 边界框
    
    Returns:
        (M,) 每个box内的点数
    """
    # 确保lidar_pc是 (6, N) 格式
    if lidar_pc.shape[0] != 6 and lidar_pc.shape[1] == 6:
        lidar_pc = lidar_pc.T
    
    points = lidar_pc[LidarIndex.POSITION.value, :].T  # (N, 3)
    
    centers = boxes[:, BoundingBoxIndex.POSITION.value]
    dims = boxes[:, BoundingBoxIndex.DIMENSION.value]
    headings = boxes[:, BoundingBoxIndex.HEADING.value]
    
    cos_h = np.cos(-headings)
    sin_h = np.sin(-headings)
    rot_mats = np.stack([
        np.stack([cos_h, -sin_h, np.zeros_like(cos_h)], axis=-1),
        np.stack([sin_h, cos_h, np.zeros_like(cos_h)], axis=-1),
        np.tile(np.array([0.0, 0.0, 1.0]), (len(cos_h), 1)),
    ], axis=1)  # (B, 3, 3)
    
    rel_points = points[None, :, :] - centers[:, None, :]  # (B, N, 3)
    aligned = np.einsum("bij,bnj->bni", rot_mats, rel_points)
    
    half_dims = dims / 2.0
    inside = np.all(np.abs(aligned) <= (half_dims[:, None, :] + 1e-6), axis=-1)
    
    return inside.sum(axis=1).astype(np.int32)


def get_dynamic_objects(
    ann: Dict,
    lidar_pc: Optional[npt.NDArray[np.float32]] = None,
    speed_threshold: float = 1.0,
    min_lidar_points: int = 1,
) -> Tuple[List[BoundingBox3D], npt.NDArray[np.bool_]]:
    """
    获取动态物体 (基于速度和lidar点数过滤)
    
    Args:
        ann: annotations字典
        lidar_pc: lidar点云，用于过滤
        speed_threshold: 速度阈值 (m/s)
        min_lidar_points: 最小lidar点数
    
    Returns:
        dynamic_boxes: 动态物体列表
        moving_mask: 运动mask
    """
    gt_boxes = np.asarray(ann.get('gt_boxes', []))
    gt_velocity_3d = np.asarray(ann.get('gt_velocity_3d', np.zeros((len(gt_boxes), 3))))
    
    if len(gt_boxes) == 0:
        return [], np.array([], dtype=bool)
    
    # 速度过滤
    speeds = np.linalg.norm(gt_velocity_3d[:, :2], axis=1)
    moving_mask = speeds > speed_threshold
    
    # Lidar点数过滤
    if lidar_pc is not None and len(gt_boxes) > 0:
        lidar_counts = count_lidar_points_in_boxes(lidar_pc, gt_boxes)
        moving_mask &= lidar_counts >= min_lidar_points
    
    # 解析动态物体
    dynamic_boxes = []
    gt_names = np.asarray(ann.get('gt_names', []))
    instance_tokens = ann.get('instance_tokens', [])
    track_tokens = ann.get('track_tokens', [])
    
    for i in np.where(moving_mask)[0]:
        box = gt_boxes[i]
        bbox = BoundingBox3D(
            center=np.array([box[0], box[1], box[2]], dtype=np.float32),
            size=np.array([box[3], box[4], box[5]], dtype=np.float32),
            yaw=float(box[6]),
            velocity=gt_velocity_3d[i].astype(np.float32),
            label=str(gt_names[i]) if i < len(gt_names) else 'vehicle',
            instance_token=instance_tokens[i] if i < len(instance_tokens) else None,
            track_token=track_tokens[i] if i < len(track_tokens) else None,
        )
        dynamic_boxes.append(bbox)
    
    return dynamic_boxes, moving_mask


def transform_annotations_to_camera(
    boxes: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    将边界框从ego/lidar坐标系转换到相机坐标系 (官方函数)
    
    Args:
        boxes: (N, 7) [x, y, z, l, w, h, yaw]
        sensor2lidar_rotation: (3, 3) 相机到lidar的旋转
        sensor2lidar_translation: (3,) 相机到lidar的平移
    
    Returns:
        boxes_cam: (N, 7) 相机坐标系下的边界框
    """
    from pyquaternion import Quaternion
    
    locs = boxes[:, BoundingBoxIndex.POSITION.value]
    rots = boxes[:, BoundingBoxIndex.HEADING:]
    # 注意维度顺序转换: l, w, h -> l, h, w (相机坐标系)
    dims_cam = boxes[:, [BoundingBoxIndex.LENGTH, BoundingBoxIndex.HEIGHT, BoundingBoxIndex.WIDTH]]
    
    rots_cam = np.zeros_like(rots)
    for idx, rot in enumerate(rots):
        rot_quat = Quaternion(axis=[0, 0, 1], radians=rot[0])
        rot_quat = Quaternion(matrix=sensor2lidar_rotation).inverse * rot_quat
        rots_cam[idx] = -rot_quat.yaw_pitch_roll[0]
    
    # lidar -> camera 变换
    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t
    
    locs_cam = np.concatenate([locs, np.ones_like(locs)[:, :1]], axis=-1)
    locs_cam = lidar2cam_rt.T @ locs_cam.T
    locs_cam = locs_cam.T[:, :-1]
    
    return np.concatenate([locs_cam, dims_cam, rots_cam], axis=-1)


def rotation_3d_in_axis(
    points: npt.NDArray[np.float32],
    angles: npt.NDArray[np.float32],
    axis: int = 1
) -> npt.NDArray[np.float32]:
    """3D旋转 (官方函数)"""
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    
    if axis == 1:  # Y轴旋转
        rot_mat_T = np.stack([
            np.stack([rot_cos, zeros, -rot_sin]),
            np.stack([zeros, ones, zeros]),
            np.stack([rot_sin, zeros, rot_cos]),
        ])
    elif axis == 2 or axis == -1:  # Z轴旋转
        rot_mat_T = np.stack([
            np.stack([rot_cos, -rot_sin, zeros]),
            np.stack([rot_sin, rot_cos, zeros]),
            np.stack([zeros, zeros, ones]),
        ])
    elif axis == 0:  # X轴旋转
        rot_mat_T = np.stack([
            np.stack([zeros, rot_cos, -rot_sin]),
            np.stack([zeros, rot_sin, rot_cos]),
            np.stack([ones, zeros, zeros]),
        ])
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")
    
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def project_boxes_to_image(
    boxes: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    image_shape: Tuple[int, int],
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    将3D边界框投影到图像
    
    Args:
        boxes: (N, 7) 相机坐标系下的边界框
        intrinsic: (3, 3) 相机内参
        image_shape: (H, W)
    
    Returns:
        corners_2d: (N, 8, 2) 投影后的角点
        valid_mask: (N,) 有效mask
    """
    # 构建8个角点
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    
    box_positions = boxes[:, BoundingBoxIndex.POSITION.value]
    box_dimensions = boxes[:, BoundingBoxIndex.DIMENSION.value]
    box_headings = boxes[:, BoundingBoxIndex.HEADING]
    
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    corners = rotation_3d_in_axis(corners, box_headings, axis=1)
    corners += box_positions.reshape(-1, 1, 3)
    
    # 投影到图像
    corners_flat = corners.reshape(-1, 3)
    
    viewpad = np.eye(4)
    viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
    
    pts_h = np.concatenate([corners_flat, np.ones((len(corners_flat), 1))], axis=-1)
    pts_img = viewpad @ pts_h.T
    pts_img = pts_img.T
    
    # 深度检查
    depth_valid = pts_img[:, 2] > 1e-3
    pts_2d = pts_img[:, :2] / np.maximum(pts_img[:, 2:3], 1e-3)
    
    # FOV检查
    img_h, img_w = image_shape
    in_fov = (
        depth_valid &
        (pts_2d[:, 0] > 0) & (pts_2d[:, 0] < img_w - 1) &
        (pts_2d[:, 1] > 0) & (pts_2d[:, 1] < img_h - 1)
    )
    
    corners_2d = pts_2d.reshape(-1, 8, 2)
    in_fov = in_fov.reshape(-1, 8)
    valid_mask = in_fov.any(axis=1)
    
    return corners_2d, valid_mask


# ============== 转换为训练数据格式 ==============

def annotations_to_carla_actors_data(boxes: List[BoundingBox3D]) -> Dict:
    """转换为CARLA格式的actors_data"""
    actors_data = {}
    
    for i, box in enumerate(boxes):
        carla_type = NAVSIM_TO_CARLA_TYPE.get(box.label.lower(), 'static')
        speed = np.linalg.norm(box.velocity[:2])
        
        actors_data[i] = {
            'loc': box.center.tolist(),
            'ori': [0, 0, float(box.yaw)],
            'box': box.size.tolist(),
            'vel': box.velocity.tolist(),
            'speed': float(speed),
            'tpe': carla_type,
            'class': box.label,
            'instance_token': box.instance_token,
            'track_token': box.track_token,
        }
    
    return actors_data


def generate_detection_grid(
    actors_data: Dict,
    grid_size: Tuple[int, int] = (50, 50),
    x_range: Tuple[float, float] = (-25, 25),  # 左右 (y axis)
    y_range: Tuple[float, float] = (-10, 40),  # 后前 (x axis)
) -> np.ndarray:
    """
    NavSim: x forward, y left
    grid_x -> left/right (y)
    grid_y -> back/forward (x)
    """
    H, W = grid_size
    det_data = np.zeros((H, W, 8), dtype=np.float32)

    # rename by meaning (to avoid confusion)
    lr_range = x_range   # y axis
    fb_range = y_range   # x axis

    cell_w = (lr_range[1] - lr_range[0]) / W   # left/right per col
    cell_h = (fb_range[1] - fb_range[0]) / H   # forward/back per row

    for actor in actors_data.values():
        x, y = float(actor["loc"][0]), float(actor["loc"][1])  # x forward, y left

        # correct range check
        if not (fb_range[0] <= x <= fb_range[1] and lr_range[0] <= y <= lr_range[1]):
            continue

        # col index from y (left/right), row index from x (forward/back)
        grid_x = int((y - lr_range[0]) / cell_w)
        grid_y = int((x - fb_range[0]) / cell_h)
        grid_x = int(np.clip(grid_x, 0, W - 1))
        grid_y = int(np.clip(grid_y, 0, H - 1))

        # normalized offset inside the cell in [0,1)
        offset_x = (y - (lr_range[0] + grid_x * cell_w)) / cell_w
        offset_y = (x - (fb_range[0] + grid_y * cell_h)) / cell_h

        yaw = float(actor["ori"][2])
        length, width = float(actor["box"][0]), float(actor["box"][1])
        speed = float(actor.get("speed", 0.0))

        if det_data[grid_y, grid_x, 0] == 0:
            det_data[grid_y, grid_x] = [
                1.0,
                offset_x, offset_y,
                length / 4.5,
                width / 2.0,
                np.sin(yaw),
                np.cos(yaw),
                speed / 8.0,
            ]

    return det_data.reshape(-1, 8)



def generate_bev_occupancy(
    actors_data: Dict,
    canvas_size: Tuple[int, int] = (256, 256),
    x_range: Tuple[float, float] = (-32, 32),
    y_range: Tuple[float, float] = (-32, 32),
) -> np.ndarray:
    """
    生成BEV占用图
    
    Returns:
        occupancy: (H, W) uint8 图像
    """
    H, W = canvas_size
    occupancy = np.zeros((H, W), dtype=np.uint8)
    
    meters_per_pixel_x = (x_range[1] - x_range[0]) / W
    meters_per_pixel_y = (y_range[1] - y_range[0]) / H
    
    for actor in actors_data.values():
        x, y = actor['loc'][0], actor['loc'][1]
        
        if not (x_range[0] <= y <= x_range[1] and y_range[0] <= x <= y_range[1]):
            continue
        
        # 像素坐标
        px = int((y - x_range[0]) / meters_per_pixel_x)
        py = int((x - y_range[0]) / meters_per_pixel_y)
        
        # 绘制旋转矩形
        length = actor['box'][0] / meters_per_pixel_y
        width = actor['box'][1] / meters_per_pixel_x
        yaw = actor['ori'][2]
        
        # 角点
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        half_l, half_w = length / 2, width / 2
        
        corners = np.array([
            [-half_l, -half_w],
            [half_l, -half_w],
            [half_l, half_w],
            [-half_l, half_w]
        ])
        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        corners = (R @ corners.T).T + np.array([py, px])
        corners = corners.astype(np.int32)
        
        cv2.fillPoly(occupancy, [corners], 255)
    
    return occupancy


def _grid_params_from_ranges(
    grid_size: Tuple[int, int],
    lr_range: Tuple[float, float],   # left/right in meters  (NavSim y)
    fb_range: Tuple[float, float],   # forward/back in meters (NavSim x)
):
    """
    Returns:
      H,W, cell_lr, cell_fb, lr0, fb0
    """
    H, W = grid_size
    lr0, lr1 = lr_range
    fb0, fb1 = fb_range
    cell_lr = (lr1 - lr0) / W  # meters per column
    cell_fb = (fb1 - fb0) / H  # meters per row
    return H, W, cell_lr, cell_fb, lr0, fb0


def _xy_to_grid(
    x_forward: float,
    y_left: float,
    grid_size: Tuple[int, int],
    lr_range: Tuple[float, float],
    fb_range: Tuple[float, float],
) -> Tuple[Optional[int], Optional[int]]:
    """
    NavSim: x forward, y left
    grid:  row -> forward/back (x), col -> left/right (y)

    Returns:
      (row, col) or (None,None) if outside
    """
    H, W, cell_lr, cell_fb, lr0, fb0 = _grid_params_from_ranges(grid_size, lr_range, fb_range)

    if not (fb_range[0] <= x_forward <= fb_range[1] and lr_range[0] <= y_left <= lr_range[1]):
        return None, None

    col = int((y_left - lr0) / cell_lr)
    row = int((x_forward - fb0) / cell_fb)
    col = int(np.clip(col, 0, W - 1))
    row = int(np.clip(row, 0, H - 1))
    return row, col


def rasterize_bev_boxes(
    boxes: List[BoundingBox3D],
    grid_size: Tuple[int, int] = (50, 50),
    lr_range: Tuple[float, float] = (-25.0, 25.0),  # y axis (left/right)
    fb_range: Tuple[float, float] = (0.0, 50.0),    # x axis (forward)
    classes: Optional[List[str]] = None,
    return_multichannel: bool = False,
) -> np.ndarray:
    """
    Rasterize boxes into a strict BEV grid.

    - NavSim coordinates: x forward, y left
    - Grid definition:
        rows (H)  correspond to forward x in fb_range
        cols (W)  correspond to left/right y in lr_range

    Args:
      boxes: List[BoundingBox3D] (center: [x,y,z], size: [l,w,h], yaw)
      return_multichannel:
        False -> return (H,W) uint8 {0,255}
        True  -> return (C,H,W) uint8 {0,255} with channels for classes

    Notes:
      - yaw is heading in ego frame (around z).
      - We only rasterize footprint in XY plane using (length,width).
    """
    H, W, cell_lr, cell_fb, lr0, fb0 = _grid_params_from_ranges(grid_size, lr_range, fb_range)

    if return_multichannel:
        if classes is None:
            classes = ["vehicle", "pedestrian", "bicycle"]
        C = len(classes)
        out = np.zeros((C, H, W), dtype=np.uint8)
        class_to_c = {c: i for i, c in enumerate(classes)}
    else:
        out = np.zeros((H, W), dtype=np.uint8)

    # helper: metric (x,y) -> pixel (row,col) in float
    def metric_to_rc(x_fwd: float, y_left: float) -> Tuple[float, float]:
        row = (x_fwd - fb0) / cell_fb
        col = (y_left - lr0) / cell_lr
        row = (H - 1) - row
        return row, col

    for b in boxes:
        # center in NavSim ego
        x = float(b.center[0])  # forward
        y = float(b.center[1])  # left

        # quick reject by center
        if not (fb_range[0] <= x <= fb_range[1] and lr_range[0] <= y <= lr_range[1]):
            continue

        length = float(b.size[0])
        width  = float(b.size[1])
        yaw    = - float(b.yaw)

        # corners in metric space (x forward, y left)
        hl = 0.5 * length
        hw = 0.5 * width

        # define rectangle in local box frame (x forward, y left)
        # order: (x,y)
        corners_local = np.array([
            [ hl,  hw],
            [ hl, -hw],
            [-hl, -hw],
            [-hl,  hw],
        ], dtype=np.float32)

        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s],
                      [s,  c]], dtype=np.float32)

        corners_metric = corners_local @ R.T
        corners_metric[:, 0] += x
        corners_metric[:, 1] += y

        # map to grid coords (row,col) in float, then to int points for cv2
        pts_rc = np.array([metric_to_rc(px, py) for px, py in corners_metric], dtype=np.float32)
        # cv2 wants (x,y) = (col,row)
        pts_xy = np.stack([pts_rc[:, 1], pts_rc[:, 0]], axis=1)

        # clip / ignore polygons fully outside grid
        if np.all((pts_xy[:, 0] < 0) | (pts_xy[:, 0] >= W) | (pts_xy[:, 1] < 0) | (pts_xy[:, 1] >= H)):
            continue

        poly = np.round(pts_xy).astype(np.int32)  # (4,2)

        if return_multichannel:
            lbl = str(b.label).lower()
            if lbl not in class_to_c:
                continue
            cidx = class_to_c[lbl]
            cv2.fillPoly(out[cidx], [poly], 255)
        else:
            cv2.fillPoly(out, [poly], 255)

    return out


# ============== 完整处理函数 ==============

def process_frame_annotations(
    ann: Dict,
    camera_info: Optional[Dict] = None,
    lidar_pc: Optional[npt.NDArray] = None,
) -> Dict[str, Any]:
    """
    处理单帧annotations的完整函数
    
    Args:
        ann: NavSim annotations字典
        camera_info: 相机信息 (用于投影)
        lidar_pc: lidar点云 (用于过滤)
    
    Returns:
        处理后的数据字典
    """
    # 1. 解析所有物体
    # all_boxes = parse_navsim_annotations(
    #     ann,
    #     filter_classes=None,
    #     max_distance=80.0,
    #     min_distance=0.1,
    # )
    
    # 2. 只保留动态类别用于检测
    dynamic_boxes = parse_navsim_annotations(
        ann,
        filter_classes=list(DYNAMIC_CLASSES),
        max_distance=50.0,
        min_distance=0.1,
    )
    
    # # 3. 转换为CARLA格式
    # actors_data = annotations_to_carla_actors_data(dynamic_boxes)
    
    # # 4. 生成检测网格: already in ego
    # det_data = generate_detection_grid(
    #     actors_data,
    #     grid_size=(50, 50),
    #     x_range=(-25, 25),
    #     y_range=(0, 50),
    # )

    bev_raster = rasterize_bev_boxes(
        dynamic_boxes,
        grid_size=(50, 50),
        lr_range=(-25.0, 25.0),  # y left/right
        fb_range=(0.0, 50.0),    # x forward
        return_multichannel=False,    # 想二值就 False
        classes=["vehicle", "pedestrian", "bicycle"],
    )
    
    # 5. 生成BEV占用图
    # bev_occupancy = generate_bev_occupancy(
    #     actors_data,
    #     canvas_size=(256, 256),
    #     y_range=(-25, 25),
    #     x_range=(0, 50),
    # )
    
    # 6. 统计信息
    gt_names = np.asarray(ann.get('gt_names', []))
    unique, counts = np.unique(gt_names, return_counts=True)
    class_counts = dict(zip(unique, counts))
    
    result = {
        # 'actors_data': actors_data,
        # 'det_data': det_data,
        "bev_raster": bev_raster
        # 'bev_occupancy': bev_occupancy,
        # 'all_boxes': all_boxes,
        # 'dynamic_boxes': dynamic_boxes,
        # 'num_total': len(all_boxes),
        # 'num_dynamic': len(dynamic_boxes),
        # 'num_vehicles': class_counts.get('vehicle', 0),
        # 'num_pedestrians': class_counts.get('pedestrian', 0),
        # 'num_bicycles': class_counts.get('bicycle', 0),
        # 'class_counts': class_counts,
    }
    
    # 7. 如果提供相机信息，计算投影
    if camera_info is not None and len(dynamic_boxes) > 0:
        gt_boxes = np.asarray(ann['gt_boxes'])
        
        # 转换到相机坐标系
        boxes_cam = transform_annotations_to_camera(
            gt_boxes,
            camera_info['sensor2lidar_rotation'],
            camera_info['sensor2lidar_translation'],
        )
        
        # 投影到图像
        corners_2d, valid_mask = project_boxes_to_image(
            boxes_cam,
            camera_info['intrinsics'],
            image_shape=(1080, 1920),
        )
        
        result['corners_2d'] = corners_2d
        result['projected_valid'] = valid_mask
    
    return result