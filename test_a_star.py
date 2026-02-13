"""
An example showing how to use RRT-Connect to plan a path and visualize it in mujoco.
"""

# from m0.planning.rrt_connect import RRTConnect
import os
import time

import mujoco
import numpy as np

from control.mpc_casadi import MPCController
from env.mujoco_visualization import MujocoViewer
from env.robot import Robot
from planning.a_star import graph_search
from utils.gridmap_2d import GridMap
from utils.utils import interpolate_path, wrap_pi

if __name__ == "__main__":
    # load mujoco model
    model = mujoco.MjModel.from_xml_path("env/assets/rrt_connect_scene.xml")
    data = mujoco.MjData(model)
    robot = Robot(model, data, robot_body_name="pusher1", actuator_names=["forward", "turn"], max_v=1.0, max_w=1.0)
    # viewer
    mjv = MujocoViewer(model, data)
    # build 2D gridmap from mujoco model
    grid_map = GridMap(model=model, data=data, resolution=0.05, width=10.0, height=10.0, robot_radius=0.15, margin=0.1)
    # plan a path using A-star
    start = [-4.0, 2.0]
    goal = [4.0, 0]

    path = graph_search(start=start[0:2], goal=goal[0:2], gridmap=grid_map)
    path = interpolate_path(path, num_points=1000)
    print("Planned path:", path)  # (1000, 3)

    filename = os.path.join("media", "grid_map.png")
    grid_map.show_map_real(filename=filename)
    filename = os.path.join("media", "grid_map_with_path.png")
    grid_map.show_map_real(path=path, filename=filename)
    # 开始录制
    # 设置输出路径
    output_dir = "media/videos"
    os.makedirs(output_dir, exist_ok=True)
    video_filename = os.path.join(output_dir, "A_Star_MPC.mp4")
    mjv.start_recording(output_path=video_filename, framerate=30)
    # 添加帧率控制变量
    frame_count = 0
    last_capture_time = time.time()
    capture_interval = 1.0 / 30  # 30 FPS对应的时间间隔
    # visualize the path in mujoco
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
    mjv.render(capture_frame=True)

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
        current_time = time.time()
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
        # print(f"Step: {step}, Pos: {x0[:2]}, Ctrl: {v:.2f}, {w:.2f}")
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
        # --- E. 渲染和录制 ---
        # 控制录制频率，避免过度录制
        if current_time - last_capture_time >= capture_interval:
            mjv.render(capture_frame=True)
            last_capture_time = current_time
            frame_count += 1
        time.sleep(0.01)
        # 打印调试信息
        # print(f"Step: {step}/{path_len}, Pos: {x0[:2]}")
    # 停止录制并保存视频
    print(f"仿真完成，共录制 {frame_count} 帧")
    mjv.stop_recording()
    print(f"视频已保存到: {video_filename}")
    print("程序结束")
