import math

import numpy as np


class DWAConfig:
    def __init__(self, robot=None):
        """
        初始化 DWA 参数
        :param robot: Robot 类实例 (可选)。如果传入，自动同步最大速度限制。
        """
        # === 1. 机器人物理限制 (从 Robot 类同步) ===
        self.robot = robot
        if robot is not None:
            self.max_speed = robot.max_v  # [m/s]
            self.max_yaw_rate = robot.max_w  # [rad/s]
        else:
            self.max_speed = 1.0
            self.max_yaw_rate = 40.0 * math.pi / 180.0

        self.min_speed = -0.2  # [m/s] (允许倒车速度)
        self.max_accel = 5.0  # [m/s^2] (根据物理仿真调整，设大一点以提高响应)
        self.max_delta_yaw_rate = 2.0 * math.pi  # [rad/s^2]

        # === 2. 采样分辨率 ===
        self.v_resolution = 0.1  # [m/s] 0.02
        self.yaw_rate_resolution = 5 * math.pi / 180.0  # 0.1

        # === 3. 预测参数 ===
        self.dt = 0.05  # [s] 预测步长 (比物理仿真步长大)
        self.predict_time = 2  # [s] 向前预测总时间

        # === 4. 代价权重 (核心调优参数) ===
        self.to_goal_cost_gain = 0.15  # 目标导向权重
        self.speed_cost_gain = 3.0  # 速度优先权重
        self.obstacle_cost_gain = 2.0  # 避障权重

        # === 5. 碰撞参数 ===
        # 注意：这里主要用于 GridMap 未覆盖的动态障碍物。
        # 对于 GridMap，我们直接查网格，网格已经包含了膨胀。
        self.robot_radius = 0.2


class DWAPlanner:
    def __init__(self, config):
        self.config = config
        self.robot = config.robot

    def get_dwa_state(self):
        """
        [Helper] 从 Robot 类实例中提取 DWA 所需的 5 维状态向量
        Robot 提供: get_state() -> [x, y, yaw]
        Robot 提供: get_v() -> [vx, vy, vz, wx, wy, wz] (世界坐标系)

        DWA 需要: [x, y, yaw, v_body, w_body]
        """
        # 1. 位置和偏航
        pos_state = self.robot.get_state()  # [x, y, yaw]
        x, y, yaw = pos_state[0], pos_state[1], pos_state[2]

        # 2. 速度 (Robot 返回的是世界坐标系下的 6D 速度)
        vel_world = self.robot.get_v()
        vx_world = vel_world[0]
        vy_world = vel_world[1]
        w_world = vel_world[5]  # Z轴角速度

        # 3. 将世界坐标系速度投影到机器人局部坐标系 (Body Frame)
        # v = vx * cos(yaw) + vy * sin(yaw)
        v_body = vx_world * math.cos(yaw) + vy_world * math.sin(yaw)
        w_body = w_world

        return np.array([x, y, yaw, v_body, w_body])

    def plan(self, goal, grid_map):
        """
        主规划函数
        :param robot: Robot 类实例
        :param goal: 局部目标 [gx, gy]
        :param grid_map: GridMap 类实例
        :return: best_u [v, w], best_trajectory
        """
        # 1. 获取状态
        x = self.get_dwa_state()

        # 2. 计算动态窗口
        dw = self.calc_dynamic_window(x)

        # 3. 搜索最佳控制
        u, traj = self.calc_control_and_trajectory(x, dw, goal, grid_map)

        return u, traj

    def calc_dynamic_window(self, x):
        # x: [x, y, yaw, v, w]

        # 机器人物理极限
        Vs = [self.config.min_speed, self.config.max_speed, -self.config.max_yaw_rate, self.config.max_yaw_rate]

        # 运动学极限 (下一时刻能达到的速度)
        Vd = [
            x[3] - self.config.max_accel * self.config.dt,
            x[3] + self.config.max_accel * self.config.dt,
            x[4] - self.config.max_delta_yaw_rate * self.config.dt,
            x[4] + self.config.max_delta_yaw_rate * self.config.dt,
        ]

        # 交集
        return [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]), max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    def predict_trajectory(self, x_init, v, w):
        """
        使用 RK4 (Runge-Kutta 4th order) 进行高精度轨迹预测
        """
        x = np.array(x_init)
        # 优化：使用 list 存储比 np.vstack 在循环中更快
        traj = [x]
        dt = self.config.dt
        time = 0

        # 提取当前状态变量，减少数组索引开销
        curr_x = x[0]
        curr_y = x[1]
        curr_theta = x[2]

        while time <= self.config.predict_time:
            # === RK4 核心步骤 ===
            # k1: 当前时刻的导数
            k1_dx = v * math.cos(curr_theta)
            k1_dy = v * math.sin(curr_theta)
            k1_dtheta = w

            # k2: 预测 dt/2 后的导数 (使用 k1)
            mid_theta_2 = curr_theta + k1_dtheta * 0.5 * dt
            k2_dx = v * math.cos(mid_theta_2)
            k2_dy = v * math.sin(mid_theta_2)
            k2_dtheta = w

            # k3: 再次预测 dt/2 后的导数 (使用 k2)
            mid_theta_3 = curr_theta + k2_dtheta * 0.5 * dt
            k3_dx = v * math.cos(mid_theta_3)
            k3_dy = v * math.sin(mid_theta_3)
            k3_dtheta = w

            # k4: 预测 dt 后的导数 (使用 k3)
            end_theta_4 = curr_theta + k3_dtheta * dt
            k4_dx = v * math.cos(end_theta_4)
            k4_dy = v * math.sin(end_theta_4)
            k4_dtheta = w

            # === 加权更新 ===
            curr_x += (dt / 6.0) * (k1_dx + 2 * k2_dx + 2 * k3_dx + k4_dx)
            curr_y += (dt / 6.0) * (k1_dy + 2 * k2_dy + 2 * k3_dy + k4_dy)
            curr_theta += (dt / 6.0) * (k1_dtheta + 2 * k2_dtheta + 2 * k3_dtheta + k4_dtheta)

            # 更新时间
            time += dt

            # 组装新状态 [x, y, yaw, v, w]
            # 注意：在 DWA 预测窗口内，假设 v 和 w 是常数
            new_state = np.array([curr_x, curr_y, curr_theta, v, w])
            traj.append(new_state)

        return np.array(traj)

    def calc_control_and_trajectory(self, x, dw, goal, grid_map):
        x_init = x[:]
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        best_traj = np.array([x])

        # 暴力搜索速度空间
        for v in np.arange(dw[0], dw[1], self.config.v_resolution):
            for w in np.arange(dw[2], dw[3], self.config.yaw_rate_resolution):
                traj = self.predict_trajectory(x_init, v, w)

                # 计算代价
                to_goal_cost = self.config.to_goal_cost_gain * self.calc_to_goal_cost(traj, goal)
                speed_cost = self.config.speed_cost_gain * (self.config.max_speed - traj[-1, 3])
                ob_cost = self.config.obstacle_cost_gain * self.calc_obstacle_cost(traj, grid_map)

                final_cost = to_goal_cost + speed_cost + ob_cost

                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, w]
                    best_traj = traj

        return best_u, best_traj

    def calc_obstacle_cost(self, trajectory, grid_map):
        """
        利用 GridMap 进行快速碰撞检测
        """
        step_size = 2  # 采样间隔，减少计算量

        for i in range(0, len(trajectory), step_size):
            x_pos = trajectory[i, 0]
            y_pos = trajectory[i, 1]

            # 转换坐标并检查
            try:
                row, col = grid_map.coor_to_index([x_pos, y_pos])
                if not grid_map.is_valid_index((row, col)):
                    return float("Inf")  # 出界视为碰撞
                if grid_map.is_occupied_index((row, col)):
                    return float("Inf")  # 撞墙
            except:  # noqa: E722
                return float("Inf")

        return 0.0

    def calc_to_goal_cost(self, traj, goal):
        dx = goal[0] - traj[-1, 0]
        dy = goal[1] - traj[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - traj[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        return cost
