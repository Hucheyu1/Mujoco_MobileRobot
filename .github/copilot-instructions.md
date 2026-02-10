## 快速目标

为 AI 编码/补全代理提供本仓库的关键上下文：架构要点、运行/调试步骤、项目特有约定和常见陷阱。

## 一句话概览

这是一个基于 MuJoCo 的差分驱动机器人研究项目，目录按职责划分：`env/`（仿真环境和可视化）、`planning/`（路径规划算法）、`control/`（控制器实现，含 CasADi MPC）、`utils/`（网格、插值等工具）。测试/示例以脚本形式存在（例如 `test_a_star.py`），常以直接运行脚本来演示模块集成。

## 关键文件（快速索引）
- `env/DiffDrive_Env.py` — 物理环境封装；机器人状态为 `[x, y, theta]`，动作为 `[v, w]`；注意 `frame_skip` 与 `model.opt.timestep` 的关系。
- `env/robot.py` — 机器人 API（`set_state`, `set_ctrl`, `get_state/get_pos/get_yaw`）。
- `control/mpc_casadi.py` — CasADi+IPOPT 实现的 MPC：决策变量 `var_X (3, N+1)`, `var_U (2, N)`；参考轨迹在外部以 `(N,5)` 构造，但 `set_value` 处需要转置为 `(5, N)`（见 `step()`）。
- `planning/a_star.py` — 在网格地图上的 A* 搜索；接口依赖 `GridMap`（查看 `utils/gridmap_2d.py`）：必须支持 `coor_to_index`, `index_to_coor`, `is_valid_index`, `is_occupied_index`。
- `utils/gridmap_2d.py`, `utils/utils.py` — 路径插值与角度规范化（`wrap_pi`），用于在 `test_a_star.py` 中生成 MPC 的参考轨迹。
- `test_a_star.py` / `test_dwa.py` / `test_mpc.py` / `test_rrt.py` — 可直接作为运行示例（不是 pytest 单元测试）：通常使用 `python test_a_star.py` 在 MuJoCo 可视化下演示端到端流程。

## 环境与依赖（可复制步骤）
- 本项目使用 `uv` 进行依赖管理。
- 推荐在项目根目录运行：

```powershell
uv sync
```

- 依赖中包含 `mujoco`（需要在系统上安装 MuJoCo 并设置许可证/环境变量）和 `casadi`（IPOPT solver 在 `control/mpc_casadi.py` 中被调用）。请参阅 MuJoCo 官方安装指南；XML 场景存放在 `env/assets/`，脚本通常以项目根为工作目录去加载这些 XML 文件（相对路径）。

## 运行与调试要点
- 快速示例（在项目根运行）：

```powershell
uv run test_a_star.py
```

- 如果出现 CasADi/solver 失败，`MPCController.step()` 会捕获异常并回退使用 `u_prev`（查看 `control/mpc_casadi.py`），这通常用于安全降级 MPC。
- MuJoCo 可视化要求本地有图形后端；在远程/无头机器上请配置合适的 OSMesa/egl 或在本机运行演示脚本。

## 代码风格与约定（项目特有）
- 形状/维度很重要：许多函数期望列向量或特定维度（例如 MPC 内部使用 `(3, N+1)` / `(2, N)`），AI 代理在生成代码时请谨慎维度转换（`reshape`, `T`）。
- `pyproject.toml` 包含 `ruff` 配置并对 `__init__.py` 文件允许 `F401`（导入但未使用）。不要随意删除 `__init__` 中的导出，仅在未使用时保留注释说明。
- 轨迹表示：路径数组通常为 (N, 2) 或 (N, 3)，其中第三列为 heading；示例见 `planning/a_star.py` 与 `test_a_star.py`。

## 常见改动示例（AI 代理可直接修改/补全）
- 当新增控制器接口时，确保：
  - `step(x0, y_ref, y_refN)` 中 `y_ref` 为 `(N,5)`，但 `opti.set_value(self.p_yref, y_ref.T)`（参见 `MPCController.step()`）。
  - Warm-start 的 `u_prev` 与 `x_prev` 维度保持一致并在求解成功后正确滚动更新（参见 `self.u_prev` 更新逻辑）。
- 当修改计划器-控制器集成（例如 `test_a_star.py`）时，请保证 `wrap_pi` 在插值后用于角度解缠，避免角度突变。

## 编辑/测试/格式化
- 格式化/静态检查：项目使用 `ruff`（配置在 `pyproject.toml`）。在改动后可以运行：

```powershell
ruff check --fix
```

- 运行示例脚本而非 pytest：示例 `test_*.py` 是运行脚本，设计上用于演示集成（可直接执行），因此当你修补集成逻辑时优先运行这些脚本来验证端到端行为。

## 常见陷阱 / 注意事项
- 1) 相对路径：许多脚本按“项目根”为工作目录来加载 `env/assets/*.xml`；在 IDE / CI 中运行时请使用 repository root 或调整路径。
- 2) MuJoCo 依赖：需要正确安装并设置许可证；否则 `mujoco.MjModel.from_xml_path` 会报错。
- 3) 数组形状错误经常导致 CasADi 报错，优先检查 `.shape` 而非仅检查内容。

---

如果这些说明中有不完整或需要展开的地方，请告诉我想要补充的细节（例如常用脚本参数、CI 命令或常见本地安装问题），我会立刻迭代更新。 
