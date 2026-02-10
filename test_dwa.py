"""
An example showing how to use A* for global planning and DWA for local navigation in MuJoCo.
"""

import math

import mujoco
import numpy as np

# 引入你的环境类
from env.mujoco_visualization import MujocoViewer
from env.robot import Robot

# 引入规划算法
from planning.a_star import graph_search
from planning.DWA import DWAConfig as DWAConfig
from planning.DWA import DWAPlanner
from utils.gridmap_2d import GridMap


def get_local_goal(current_pos, global_path, lookahead_dist=1.5):
    """
    在全局路径上找一个“前方”的点作为 DWA 的局部目标
    """
    # 1. 找最近点索引
    dists = np.linalg.norm(global_path[:, 0:2] - current_pos[0:2], axis=1)
    min_idx = np.argmin(dists)

    # 2. 往前找 lookahead_dist 远的点
    for i in range(min_idx, len(global_path)):
        dist = np.linalg.norm(global_path[i, 0:2] - global_path[min_idx, 0:2])
        if dist > lookahead_dist:
            return global_path[i, 0:2]

    # 3. 如果找不到（快到终点了），就返回全局终点
    return global_path[-1, 0:2]


if __name__ == "__main__":
    # 1. 初始化 MuJoCo
    model = mujoco.MjModel.from_xml_path("env/assets/rrt_connect_scene.xml")
    data = mujoco.MjData(model)
    robot = Robot(model, data, robot_body_name="pusher1", actuator_names=["forward", "turn"], max_v=1.0, max_w=1.0)
    mjv = MujocoViewer(model, data)

    # 2. 建立栅格地图 & 获取障碍物列表
    grid_map = GridMap(model=model, data=data, resolution=0.05, width=10.0, height=10.0, robot_radius=0.15, margin=0.1)

    # 3. 全局规划 (A*/RRT) (不变)
    start = [-4.0, 2.0]
    goal = [4.0, 0]
    raw_path = graph_search(start=start[0:2], goal=goal[0:2], gridmap=grid_map)
    global_path = np.array(raw_path)
    print("Global path planned.")

    # 可视化全局路径
    mjv.draw_traj(
        np.column_stack((global_path[:, 0], global_path[:, 1], np.zeros(len(global_path)) + 0.05)),
        size=0.02,
        rgba=[0, 0, 1, 1],
    )

    # 4. 初始化 DWA
    dwa_config = DWAConfig(robot)
    dwa_planner = DWAPlanner(dwa_config)

    # 设置初始状态
    # 注意：robot.set_state 只能设置位置和偏航，速度需要额外设或者让它慢慢动起来
    robot.set_state(np.array([start[0], start[1], 0.03]), 0.0)
    mujoco.mj_forward(model, data)

    # 5. 主循环
    step = 0

    # DWA 的控制频率通常比 MuJoCo 低
    dwa_dt = dwa_config.dt  # 0.1s
    sim_dt = model.opt.timestep  # 0.002s
    steps_per_dwa = int(dwa_dt / sim_dt)  # 50

    while mjv.is_running():
        # --- A. 感知 Robot 状态 ---
        # Robot 类需要一个 get_odom 方法返回 [x, y, yaw, v, w]
        # 如果没有，我们需要手动拼凑
        pos = robot.get_state()  # [x, y, z] (假设 get_state 返回这个)
        # 获取 yaw (这里假设 robot 内部维护了 yaw，或者从四元数算)
        # 简单起见，我们假设 robot.get_state() 返回 [x, y, yaw] 或者我们能获取到
        # 这里仅作演示，假设 get_state 返回 [x, y, z] 和 get_yaw 返回 yaw
        # 如果你的 Robot 类不同，请相应调整

        # 临时获取真实状态（为了演示）
        x_curr = data.qpos[0]
        y_curr = data.qpos[1]

        # 获取 Yaw (从四元数)
        q = data.qpos[3:7]
        yaw_curr = math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

        # 获取速度 (世界坐标系 -> 机器人坐标系)
        vx_world = data.qvel[0]
        vy_world = data.qvel[1]
        w_curr = data.qvel[5]
        v_curr = vx_world * math.cos(yaw_curr) + vy_world * math.sin(yaw_curr)

        state_vector = np.array([x_curr, y_curr, yaw_curr, v_curr, w_curr])

        # --- B. 寻找局部目标 ---
        local_goal = get_local_goal(state_vector, global_path, lookahead_dist=1.5)

        # --- C. DWA 规划 ---
        # 如果有动态障碍物，这里应该更新 obstacle_array
        best_u, best_traj = dwa_planner.plan(local_goal, grid_map)

        v_cmd, w_cmd = best_u

        # --- D. 执行控制 ---
        # 保持这个控制量运行一段时间 (dwa_dt)
        for _ in range(steps_per_dwa):
            robot.set_ctrl(v_cmd, w_cmd)
            mujoco.mj_step(model, data)

        # --- E. 判断到达 ---
        dist_to_global_goal = np.linalg.norm(np.array([x_curr, y_curr]) - np.array(goal))
        if dist_to_global_goal < 0.2:
            print("Arrived at Goal!")
            robot.set_ctrl(0, 0)
            break

        mjv.render()

    print("Finished.")
