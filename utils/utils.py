import math
import random

import numpy as np
from scipy.interpolate import splev, splprep


def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def interpolate_path(path, num_points=None, resolution=0.05):
    """
    使用 B-样条对路径进行平滑和插值
    :param path: 原始路径 [[x, y], ...] (N, 2)
    :param num_points: 期望的点数。如果为None，根据 resolution 自动计算
    :param resolution: 如果 num_points 为 None，每隔多少米插一个点
    :return: 插值后的路径 [[x, y, theta], ...] (M, 3)
    """
    path = np.array(path)
    if len(path) < 3:  # 点太少无法做 B-Spline
        return path

    # 去重：如果相邻点完全一样，插值会报错
    # 检查相邻点的距离，保留距离 > 1e-4 的点
    dist_check = np.linalg.norm(np.diff(path, axis=0), axis=1)
    mask = np.concatenate(([True], dist_check > 1e-4))
    path = path[mask]

    # 1. 计算插值参数
    # k=3 表示三次样条，s=0 表示不过滤噪声（强制经过所有控制点），或者设小一点的值允许平滑
    tck, u = splprep([path[:, 0], path[:, 1]], k=3, s=0.05)

    # 2. 决定插多少个点
    if num_points is None:
        # 计算总路径长度
        total_dist = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
        num_points = int(total_dist / resolution)

    # 生成新的参数序列 (0 到 1)
    u_new = np.linspace(0, 1, num_points)

    # 3. 计算新坐标
    x_new, y_new = splev(u_new, tck)

    # 4. 计算切线角度 (Yaw)
    # splev 的 der=1 参数可以直接算出导数 (速度向量)
    # dx, dy = splev(u_new, tck, der=1)
    # yaw_new = np.arctan2(dy, dx)

    new_path = np.column_stack((x_new, y_new))
    full_path = calculate_path_yaw(new_path)
    return full_path


def calculate_path_yaw(path):
    """
    为路径计算航向角 (Yaw)
    :param path: A* 输出的路径，形状为 (N, 2) -> [[x, y], ...]
    :return: 包含角度的路径，形状为 (N, 3) -> [[x, y, yaw], ...]
    """
    if path is None or len(path) < 2:
        return path

    n_points = len(path)
    # 初始化一个 (N, 3) 的数组
    path_with_yaw = np.zeros((n_points, 3))

    # 复制 x, y 坐标
    path_with_yaw[:, :2] = path

    # 遍历除了最后一个点之外的所有点
    for i in range(n_points - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]

        # 使用 atan2 计算角度，范围是 [-pi, pi]
        yaw = math.atan2(dy, dx)
        path_with_yaw[i][2] = yaw

    # --- 处理最后一个点 ---
    # 最后一个点没有“下一个点”，通常的做法是：
    # 1. 如果有明确的目标姿态 goal_theta，直接赋值
    # 2. 或者，直接沿用倒数第二个点的角度（假设直线进入终点）
    path_with_yaw[-1][2] = path_with_yaw[-2][2]

    return path_with_yaw


def get_path_length(path):
    le = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = math.hypot(dx, dy)
        le += d

    return le


def get_target_point(path, targetL):
    le = 0
    ti = 0
    lastPairLen = 0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        d = math.hypot(dx, dy)
        le += d
        if le >= targetL:
            ti = i - 1
            lastPairLen = d
            break

    partRatio = (le - targetL) / lastPairLen

    x = path[ti][0] + (path[ti + 1][0] - path[ti][0]) * partRatio
    y = path[ti][1] + (path[ti + 1][1] - path[ti][1]) * partRatio

    return [x, y, ti]


def path_smoothing(env, path, max_iter):
    """
    Smooths a given path by iteratively replacing segments with shortcut connections,
    while ensuring the new segments are collision-free.

    The algorithm randomly picks two points along the original path and attempts to
    connect them with a straight line. If the line does not collide with any obstacles
    (considering the robot's radius), the intermediate path points between them are
    replaced with the direct connection.

    Args:
        path (List[List[float]]): The original path as a list of [x, y] coordinates.
        max_iter (int): Number of iterations for smoothing attempts.
        obstacle_list (List[Tuple[float, float, float]]): List of obstacles represented as
            (x, y, radius).
        robot_radius (float, optional): Radius of the robot, used to inflate obstacle size
            during collision checking. Defaults to 0.0.

    Returns:
        List[List[float]]: The smoothed path as a list of [x, y] coordinates.

    Example:
        >>> smoothed = path_smoothing(path, 1000, obstacle_list, robot_radius=0.5)
    """
    le = get_path_length(path)

    for i in range(max_iter):
        # Sample two points
        pickPoints = [random.uniform(0, le), random.uniform(0, le)]
        pickPoints.sort()
        first = get_target_point(path, pickPoints[0])
        second = get_target_point(path, pickPoints[1])

        if first[2] <= 0 or second[2] <= 0:
            continue

        if (second[2] + 1) > len(path):
            continue

        if second[2] == first[2]:
            continue
        # === 关键修改开始 ===
        # 我们要利用 env.check_collision(node, ...)
        # 这个函数检查的是 node.parent -> node 这一段路
        # A. 将起点坐标包装成 Node
        node_from = env.Node(first[0], first[1])
        # B. 将终点坐标包装成 Node
        node_to = env.Node(second[0], second[1])
        # C. 建立父子关系：让 "终点" 的父亲是 "起点"
        # 这样 check_collision 就会检测从 first 到 second 的连线
        node_to.parent = node_from
        # D. 调用 env 的碰撞检测
        # 注意：RRTStarMujoco 中 obstacle_list 可以传 []，robot_radius 也可以传 0 (因为用物理引擎)
        # 如果是普通的 RRT，保持原样传入即可
        if not env.check_collision(node_to, env.obstacle_list, env.robot_radius):
            continue  # 如果碰撞了，跳过这次尝试
        # 4. 剪接路径 (Shortcut)
        newPath = []
        newPath.extend(path[: first[2] + 1])  # 保留前半段
        newPath.append([first[0], first[1]])  # 插入捷径起点
        newPath.append([second[0], second[1]])  # 插入捷径终点
        newPath.extend(path[second[2] + 1 :])  # 保留后半段

        path = newPath
        le = get_path_length(path)  # 更新路径长度

    return path
