"""
Example script showing how to use mpc to control a differential robot follow a time-parametrized path
"""

import time

import mujoco
import numpy as np

from control.mpc_casadi import MPCController
from control.TrajectoryGenerator import TrajectoryGenerator
from env.mujoco_visualization import MujocoViewer
from env.robot import Robot
from utils.utils import wrap_pi

if __name__ == "__main__":
    # 1. 加载 MuJoCo 物理场景
    model = mujoco.MjModel.from_xml_path("env/assets/rl_scene.xml")
    data = mujoco.MjData(model)
    mjv = MujocoViewer(model, data)

    robot1 = Robot(model, data, robot_body_name="pusher1", actuator_names=["forward", "turn"], max_v=1.0, max_w=1.0)

    traj = TrajectoryGenerator("figure8", params={"a": 3.0}, duration=60, repeat=True)
    start_ref = traj.update(0)
    start_x, start_y = start_ref["x"]
    start_theta = start_ref["theta"]
    robot1.set_state(np.array([start_x, start_y, 0.03]), start_theta)
    mujoco.mj_forward(model, data)
    mjv.render()
    x0 = robot1.get_state()
    u0 = robot1.get_ctrl()
    # init controller
    controller = MPCController(x0, u0)
    ts = np.linspace(0, traj.duration, 1000)
    traj_points = np.array([traj.update(t)["x"] for t in ts])
    # 将二维坐标 (x, y) 转换为三维坐标 (x, y, z)，以便可视化工具可以在 3D 空间中画出它们
    traj_points = np.hstack((traj_points, 0.03 * np.ones((traj_points.shape[0], 1))))
    mjv.draw_traj(traj_points)

    # MuJoCo 仿真器内部的当前物理时间 (Simulation Time)
    t = data.time
    step = 0

    # MPC 的控制周期通常较长 (例如 dt=0.05s, 即 20Hz)
    # MuJoCo 的物理仿真步长很短 (通常 0.002s, 即 500Hz)
    k_step = int(round(controller.dt / model.opt.timestep))

    # loop
    y_ref = np.zeros((controller.horizon, 5))  # x, y, theta, v, w
    y_refN = np.zeros((5,))  # x, y, theta, v, w

    while mjv.is_running():
        # --- A. 构建参考轨迹 (Reference Generation) ---
        for j in range(controller.horizon):
            # 计算未来第 j 步的时间
            tj = t + j * controller.dt
            traj_state = traj.update(tj)
            # [重点] 角度解缠 (Angle Unwrapping/Wrapping)
            # 机器人的当前角度可能是 360 度，参考角度是 1 度。
            # 直接相减误差是 -359 度，控制器会发疯。
            # wrap_pi 确保误差在 [-pi, pi] 之间，让控制器知道只差了 1 度。
            xr, yr = traj_state["x"]
            thr = traj_state["theta"]
            # angle wrapper
            thr_wrapped = x0[2] + wrap_pi(thr - x0[2])
            y_ref[j] = [xr, yr, thr_wrapped, 0, 0]

        traj_state = traj.update(t + controller.horizon * controller.dt)
        xN, yN = traj_state["x"]
        thN = traj_state["theta"]
        thN_wrapped = x0[2] + wrap_pi(thN - x0[2])
        # only need state at the end of horizon
        y_refN = np.array([xN, yN, thN_wrapped, 0, 0])

        # mpc step
        u0 = controller.step(x0, y_ref, y_refN)

        # send ctrl to robot to k_step times
        for _ in range(k_step):
            robot1.set_ctrl(u0[0], u0[1])
            mujoco.mj_step(model, data)

        # update time
        t = data.time

        # update current state of robot
        x0 = robot1.get_state()

        # mujoco show
        mjv.render()

        # slow down
        time.sleep(0.01)

        step += 1
