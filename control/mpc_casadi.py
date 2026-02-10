import casadi as ca
import numpy as np


class MPCController:
    def __init__(
        self,
        X0,
        U0,
        horizon=12,
        dt=0.05,
        Q=np.diag([20, 20, 2]),
        R=np.diag([0.1, 0.1]),
        max_v=1.0,
        max_w=1.0,
        can_backward=False,
    ):
        """
        基于 CasADi (IPOPT) 的 MPC 控制器
        保持了与原 Acados 版本一致的接口
        """
        self.horizon = horizon
        self.dt = dt
        self.Q = Q
        self.R = R
        self.max_v = max_v
        self.max_w = max_w
        self.can_backward = can_backward
        # 确保 X0 是列向量 (3, 1)
        X0 = np.array(X0).reshape(3, 1)
        # 内部存储改为 (特征维度, 时间维度) 以匹配 CasADi
        self.u_prev = np.zeros((2, self.horizon))  # (2, horizon)
        self.x_prev = np.tile(X0, (1, self.horizon + 1))  # (3, horizon + 1)

        # --- 构建优化问题 (只构建一次) ---
        self._setup_opti()

    def _setup_opti(self):
        # 1. 创建 Opti 实例
        self.opti = ca.Opti()

        # 2. 定义决策变量
        # 状态变量 X: [x, y, theta], 维度 (3, N+1)
        self.var_X = self.opti.variable(3, self.horizon + 1)
        # 控制变量 U: [v, w], 维度 (2, N)
        self.var_U = self.opti.variable(2, self.horizon)

        # 3. 定义外部参数 (Parameters) 每次控制循环中会变的已知数
        # 这里的参数在 solve() 之前通过 set_value 赋值，不需要重新生成问题
        # 当前时刻的初始状态
        self.p_x0 = self.opti.parameter(3)
        # 参考轨迹: [x, y, theta, v, w] * N_horizon
        self.p_yref = self.opti.parameter(5, self.horizon)
        # 终端参考: [x, y, theta, v, w] (虽然通常只需要前3个状态)
        self.p_yrefN = self.opti.parameter(5)

        # 4. 定义系统动力学 (Runge-Kutta 4)
        # 连续动力学方程 f(x, u)
        x_sym = ca.MX.sym("x", 3)
        u_sym = ca.MX.sym("u", 2)
        # 运动学模型: [v*cos(th), v*sin(th), w]
        x_dot = ca.vertcat(u_sym[0] * ca.cos(x_sym[2]), u_sym[0] * ca.sin(x_sym[2]), u_sym[1])
        # 参数1：函数的名字 (字符串) 参数2：输入变量列表 (Inputs) 参数3：输出表达式列表 (Outputs)
        self.f_dyn = ca.Function("f_dyn", [x_sym, u_sym], [x_dot])

        # 5. 构建 Cost Function 和 约束
        obj = 0

        # 初始状态约束
        self.opti.subject_to(self.var_X[:, 0] == self.p_x0)

        for k in range(self.horizon):
            # --- 动力学约束 (RK4 离散化) ---
            x_k = self.var_X[:, k]
            u_k = self.var_U[:, k]

            k1 = self.f_dyn(x_k, u_k)
            k2 = self.f_dyn(x_k + self.dt / 2 * k1, u_k)
            k3 = self.f_dyn(x_k + self.dt / 2 * k2, u_k)
            k4 = self.f_dyn(x_k + self.dt * k3, u_k)
            x_next = x_k + (self.dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            # 要求第 k+1 步的状态必须等于第 k 步状态经过动力学积分后的结果。这保证了生成的轨迹符合物理定律
            self.opti.subject_to(self.var_X[:, k + 1] == x_next)

            # --- 代价函数 accumulation ---
            # 状态误差 (x, y, theta)
            e_state = self.var_X[:, k] - self.p_yref[:3, k]
            # 控制误差 (v, w)
            e_ctrl = self.var_U[:, k] - self.p_yref[3:, k]

            # Cost = e_state.T * Q * e_state + e_ctrl.T * R * e_ctrl
            # 使用 CasADi 的矩阵乘法
            obj += ca.mtimes([e_state.T, self.Q, e_state]) + ca.mtimes([e_ctrl.T, self.R, e_ctrl])

            # --- 控制量约束 ---
            # 线速度 v
            if not self.can_backward:
                self.opti.subject_to(self.opti.bounded(0, u_k[0], self.max_v))
            else:
                self.opti.subject_to(self.opti.bounded(-self.max_v, u_k[0], self.max_v))

            # 角速度 w
            self.opti.subject_to(self.opti.bounded(-self.max_w, u_k[1], self.max_w))

        # 终端代价 (Terminal Cost)
        e_state_N = self.var_X[:, self.horizon] - self.p_yrefN[:3]
        obj += ca.mtimes([e_state_N.T, self.Q, e_state_N])  # 终端权重通常与运行权重Q一致

        self.opti.minimize(obj)

        # 6. 配置求解器 (IPOPT)
        # print_level=0: 静默模式
        # sb='yes': 屏蔽 IPOPT header
        # max_iter: 限制最大迭代次数以保证实时性
        opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes", "ipopt.max_iter": 100, "ipopt.tol": 1e-4}
        self.opti.solver("ipopt", opts)

    def step(self, x0, y_ref, y_refN):
        """
        求解 MPC
        x0: 当前状态 [x, y, theta]
        y_ref: 参考轨迹 (N_horizon, 5) -> [x, y, theta, v, w]
        y_refN: 终端参考 (5, )
        """
        # 1. 设置参数 (当前状态和参考轨迹)
        # 使用 set_value 将当前的真实状态 x0 和参考轨迹 y_ref 填入优化器
        self.opti.set_value(self.p_x0, x0)
        self.opti.set_value(self.p_yref, y_ref.T)  # 转置以匹配 (5, N)
        self.opti.set_value(self.p_yrefN, y_refN)

        # 2. 热启动 (Warm Start) - 设置初值猜测
        # 将上一次的解平移作为本次的猜测值
        self.opti.set_initial(self.var_X, self.x_prev)
        self.opti.set_initial(self.var_U, self.u_prev)

        # 3. 求解
        try:
            sol = self.opti.solve()
            # 获取最优控制序列
            u_opt = sol.value(self.var_U)
            x_opt = sol.value(self.var_X)

            # 取第一个控制量
            u0 = u_opt[:, 0]  # shape (2,)

            # 更新用于下一次 Warm Start 的缓存
            # 这里的逻辑是：去掉第一个，并在最后复制最后一个
            self.u_prev[:, :-1] = u_opt[:, 1:]
            self.u_prev[:, -1] = u_opt[:, -1]

            self.x_prev[:, :-1] = x_opt[:, 1:]
            self.x_prev[:, -1] = x_opt[:, -1]

            status = "Optimization_Success"

        except Exception as e:
            # 如果求解失败 (例如不可行)，打印警告并执行安全策略 (如刹车或保持上一帧控制)
            print(f"[MPC Warning]: Solver failed! {e}")
            # 简单的故障保护：使用上一帧的第二个控制量（即原本计划的这一帧控制量）
            u0 = self.u_prev[0]
            status = "Optimization_Failed"

        return u0[0], u0[1]
