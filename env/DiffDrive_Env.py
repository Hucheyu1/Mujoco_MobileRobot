import gymnasium as gym
import mujoco
import numpy as np

from .mujoco_visualization import MujocoViewer
from .robot import Robot


class DiffDriveEnv(gym.Env):
    """
    精简版差分驱动机器人环境
    专注于：物理仿真、状态获取、动作执行。
    移除了：RL 训练专用的复杂奖励计算变量 (prev_u, prev_dist 等)。
    """

    def __init__(self, xml_path: str, dt: float = 0.05, render=False):
        # 1. 加载物理引擎
        # --- MuJoCo 初始化 ---
        try:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"MuJoCo XML 文件未找到: {xml_path}")
        self.data = mujoco.MjData(self.model)

        # 2. 机器人接口 (用于设置电机和获取位置)
        self.robot = Robot(
            self.model,
            self.data,
            robot_body_name="pusher1",
            actuator_names=["forward", "turn"],
            max_v=1.0,
            max_w=1.0,
        )

        self.frame_skip = max(1, int(np.round(dt / self.model.opt.timestep)))  # 25
        self.dt = self.model.opt.timestep * self.frame_skip
        print(f"环境控制步长(dt): {self.dt:.4f}s (执行 {self.frame_skip} 个物理步骤)")
        # 可视化接口
        self.viewer = None
        if render:
            self.viewer = MujocoViewer(self.model, self.data)

    def reset(self, state=None):
        """
        重置环境：随机放置机器人，以确保数据覆盖不同的 x, y, theta
        """
        mujoco.mj_resetData(self.model, self.data)
        if self.viewer is not None:
            self.viewer.reset()

        # 随机生成状态 (范围根据你的地图大小调整)
        if state is None:
            x = np.random.uniform(-2.0, 2.0)
            y = np.random.uniform(-2.0, 2.0)
            z = 0.03  # 固定高度
            theta = np.random.uniform(-np.pi, np.pi)
        else:
            x, y, z, theta = state

        # 设置物理状态
        self.robot.set_state(np.array([x, y, z]), theta)
        mujoco.mj_forward(self.model, self.data)

        # 返回当前真实状态 [x, y, theta]
        return self._get_state()

    def step(self, action):
        """
        核心物理步进：输入动作 -> 物理计算 -> 输出新状态
        """
        # 动作截断 [-1, 1]
        # v, w = np.clip(action, -1.0, 1.0)
        v, w = action
        v = np.clip(v, -1.0, 1.0)
        w = np.clip(w, -0.5, 0.5)
        # 模拟 N 步 (frame_skip)
        for _ in range(self.frame_skip):
            self.robot.set_ctrl(v, w)
            mujoco.mj_step(self.model, self.data)

        # 渲染 (如果开启)
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.render()

        # 直接返回真实物理状态 [x, y, theta]
        return self._get_state()

    def _get_state(self):
        """提取绝对物理坐标"""
        pos = self.robot.get_pos()  # [x, y, z]
        yaw = self.robot.get_yaw()  # theta
        return np.array([pos[0], pos[1], yaw])

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
