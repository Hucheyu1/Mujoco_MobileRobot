"""
An example showing how to use RRT-Connect to plan a path and visualize it in mujoco.
"""

import time

import mujoco
import numpy as np

from control.mpc_casadi import MPCController
from env.mujoco_visualization import MujocoViewer
from env.robot import Robot
from planning.RRT.rrt_star import RRTStarGridMap, RRTStarMujoco
from utils.gridmap_2d import GridMap
from utils.utils import interpolate_path, path_smoothing, wrap_pi

if __name__ == "__main__":
    # load mujoco model
    model = mujoco.MjModel.from_xml_path("env/assets/rrt_connect_scene.xml")
    data = mujoco.MjData(model)
    robot = Robot(model, data, robot_body_name="pusher1", actuator_names=["forward", "turn"], max_v=1.0, max_w=1.0)
    # viewer
    mjv = MujocoViewer(model, data)

    rand_area = [-5, 5]
    play_area = [-5, 5, -5, 5]  # [xmin, xmax, ymin, ymax]
    print("Planning path with RRT*...")
    start = [-4.0, 2.0]
    goal = [4.0, 0]
    USE_MAP = False
    if USE_MAP:
        grid_map = GridMap(
            model=model, data=data, resolution=0.05, width=10.0, height=10.0, robot_radius=0.15, margin=0.1
        )
        grid_map.show_map_real()
        rrt_star = RRTStarGridMap(
            start=start,
            goal=goal,
            grid_map=grid_map,  # 传入机器人实例
            rand_area=rand_area,
            expand_dis=0.1,  # 树枝生长的步长，不宜太大
            path_resolution=0.05,  # 碰撞检测密度 (越小越准但越慢)
            goal_sample_rate=5,  # 采样目标点的概率，增大可加快收敛
            max_iter=5000,  # 迭代次数越多，路径越优
            play_area=play_area,  # 允许活动区域
            connect_circle_dist=1.0,  # 邻域半径
            robot_radius=0,  # 机器人半径
            search_until_max_iter=False,
        )
    else:
        rrt_star = RRTStarMujoco(
            start=start,
            goal=goal,
            robot=robot,  # 传入机器人实例
            rand_area=rand_area,
            expand_dis=0.1,  # 树枝生长的步长，不宜太大
            path_resolution=0.05,  # 碰撞检测密度 (越小越准但越慢)
            goal_sample_rate=5,  # 采样目标点的概率，增大可加快收敛
            max_iter=5000,  # 迭代次数越多，路径越优
            play_area=play_area,  # 允许活动区域
            connect_circle_dist=1.0,  # 邻域半径
            robot_radius=0.25,  # 机器人半径
            robot_z=0.05,  # 机器人悬浮高度
            search_until_max_iter=False,
        )
    # 3. 执行规划
    path = rrt_star.planning(animation=False)
    path = np.array(path)[::-1, :]  # 反转路径
    path = path_smoothing(rrt_star, path, max_iter=500)
    path = interpolate_path(path, num_points=1000)
    print("Planned path:", len(path))  # (167, 3)
    start_xyz = np.array([start[0], start[1], 0.03])
    goal_xyz = np.array([goal[0], goal[1], 0.03])
    path_xyz = np.zeros((len(path), 3)) + 0.03
    path_xyz[:, 0:2] = np.array(path)[:, 0:2]
    mjv.draw_point(start_xyz, size=0.05, rgba=np.array([1, 0, 0, 1]))
    mjv.draw_point(goal_xyz, size=0.05, rgba=np.array([0, 1, 0, 1]))
    mjv.draw_traj(path_xyz, size=0.02, rgba=np.array([0.1, 0.5, 1, 1]))

    # mpc
    start_x, start_y = path[0][0], path[0][1]
    start_theta = path[0][2]
    robot.set_state(np.array([start_x, start_y, 0.03]), start_theta)
    mujoco.mj_forward(model, data)
    mjv.render()

    x0 = robot.get_state()
    u0 = robot.get_ctrl()
    # init controller
    controller = MPCController(x0, u0)
    # MuJoCo 仿真器内部的当前物理时间 (Simulation Time)
    step = 0
    # MPC 的控制周期通常较长 (例如 dt=0.05s, 即 20Hz)
    # MuJoCo 的物理仿真步长很短 (通常 0.002s, 即 500Hz)
    k_step = int(round(controller.dt / model.opt.timestep))
    # loop
    y_ref = np.zeros((controller.horizon, 5))  # x, y, theta, v, w
    y_refN = np.zeros((5,))  # x, y, theta, v, w
    path_len = len(path)
    while mjv.is_running():
        # --- A. 构建参考轨迹 (Reference Generation) ---
        # 关键修改：处理 path 索引越界问题
        # 检查是否到达终点附近
        if step >= path_len - 1:
            print("Goal Reached!")
            break

        for j in range(controller.horizon):
            # 计算参考点索引
            ref_idx = step + j + 1

            # 如果索引超出了路径长度，就一直取最后一个点（终点）
            if ref_idx >= path_len:
                ref_idx = path_len - 1

            traj_state = path[ref_idx]
            xr, yr, thr = traj_state[0], traj_state[1], traj_state[2]
            # [重点] 角度解缠
            thr_wrapped = x0[2] + wrap_pi(thr - x0[2])
            y_ref[j] = [xr, yr, thr_wrapped, 0, 0]

        # 终端参考状态 (Terminal Cost)
        final_idx = step + controller.horizon
        if final_idx >= path_len:
            final_idx = path_len - 1

        traj_state_N = path[final_idx]
        xN, yN, thN = traj_state_N[0], traj_state_N[1], traj_state_N[2]
        thN_wrapped = x0[2] + wrap_pi(thN - x0[2])
        y_refN = np.array([xN, yN, thN_wrapped, 0, 0])

        # --- B. MPC 求解 ---
        u0 = controller.step(x0, y_ref, y_refN)
        v, w = u0
        v = np.clip(v, -robot.max_v, robot.max_v)
        w = np.clip(w, -robot.max_w, robot.max_w)
        # 记录调试信息
        print(f"Step: {step}, Pos: {x0[:2]}, Ctrl: {v:.2f}, {w:.2f}")
        # --- C. 执行控制 ---
        for _ in range(k_step):
            robot.set_ctrl(v, w)
            mujoco.mj_step(model, data)

        # --- D. 状态更新 ---
        x0 = robot.get_state()

        # 简单策略：每一帧 MPC 往前推移一个 Path 点
        # 注意：这假设了 MPC 的 dt (0.05s) 足够让机器人走完网格的一个分辨率 (0.05m => 1m/s)
        # 如果分辨率很小或者 dt 很大，可能需要调整这个逻辑
        if step < path_len - 1:
            step += 1
        # 显示
        mjv.render()
        time.sleep(0.01)
        # 打印调试信息
        # print(f"Step: {step}/{path_len}, Pos: {x0[:2]}")
