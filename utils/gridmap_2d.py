"""
2D grid map class, used for path planning and obstacle avoidance.
"""

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np


class GridMap:
    def __init__(self, model, data, resolution, width, height, robot_radius, margin, min_x=None, min_y=None):
        """
        :param model: MuJoCo MjModel object
        :param data: MuJoCo MjData object
        :param resolution: grid resolution
        :param width: width of the gridmap (in meters)
        :param height: height of the gridmap (in meters)
        :param robot_radius: radius of the robot (in meters)
        :param margin: safety margin around obstacles (in meters)
        """
        self.model = model
        self.data = data
        self.resolution = resolution
        self.width = width
        self.height = height
        # 计算网格的维度
        self.grid_width = int(width / self.resolution)
        self.grid_height = int(height / self.resolution)
        # === 关键修改：定义地图左下角的物理坐标 ===
        # 假设 (0,0) 在地图中心
        self.min_x = min_x if min_x is not None else -width / 2.0
        self.min_y = min_y if min_y is not None else -height / 2.0

        self.robot_radius = robot_radius
        self.margin = margin
        self.inflation_radius = robot_radius + margin
        # 初始化网格 (y, x) -> (row, col)
        # 注意：通常图像处理中 grid[row][col] 对应 y 和 x
        self.grid = np.zeros((self.grid_height, self.grid_width))

        self.create_grid()

    def create_grid(self):
        for i in range(self.model.ngeom):
            # 获取几何体的名字
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            # 过滤：只处理名字以 "ob_" 开头的障碍物，或者是墙壁
            # 如果不加这个，机器人的轮子也会被当成障碍物
            if name is None or "ob_" not in name:
                continue

            geom_type = self.model.geom_type[i]
            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                self._add_box(i)
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                self._add_sphere(i)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                self._add_cylinder(i)
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                self._add_capsule(i)
            else:
                pass

    def _add_box(self, geom_id):
        # box center
        center = self.data.geom_xpos[geom_id]
        lx, ly, lz = self.model.geom_size[geom_id]
        R = self.data.geom_xmat[geom_id].reshape(3, 3)

        # 4 cornors in box coordinate
        local_pts = np.array(
            [
                [-lx - self.inflation_radius, -ly - self.inflation_radius],
                [lx + self.inflation_radius, -ly - self.inflation_radius],
                [lx + self.inflation_radius, ly + self.inflation_radius],
                [-lx - self.inflation_radius, ly + self.inflation_radius],
            ]
        )

        # to world coordinate
        world_pts = np.dot(local_pts, R[:2, :2].T) + center[:2]

        # mark grid inside obs
        for i in range(self.grid_width):
            for j in range(self.grid_height):
                # === 修正开始 ===
                # 加上 self.min_x 和 self.min_y，将网格索引转换为正确的世界坐标
                x = (i + 0.5) * self.resolution + self.min_x
                y = (j + 0.5) * self.resolution + self.min_y

                if self._point_in_polygon(np.array([x, y]), world_pts):
                    self.grid[j, i] = 1

    def _add_sphere(self, geom_id):
        center = self.data.geom_xpos[geom_id]
        radius = self.model.geom_size[geom_id][0]

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                x = (i + 0.5) * self.resolution + self.min_x
                y = (j + 0.5) * self.resolution + self.min_y
                dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                if dist <= radius + self.inflation_radius:
                    self.grid[j, i] = 1

    def _add_cylinder(self, geom_id):
        center = self.data.geom_xpos[geom_id]
        radius = self.model.geom_size[geom_id][0]
        height = self.model.geom_size[geom_id][1]

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                x = (i + 0.5) * self.resolution + self.min_x
                y = (j + 0.5) * self.resolution + self.min_y
                dist = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                if dist <= radius + self.inflation_radius:
                    self.grid[j, i] = 1

    def _point_in_polygon(self, point, polygon):
        """
        Ray-casting algorithm to determine if point is in polygon
        """
        n = len(polygon)
        inside = False
        x, y = point
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def coor_to_index(self, coor):
        x, y = coor[0], coor[1]

        # 修正：去掉 + self.resolution / 2
        # 使用 int() 进行向下取整，这才是标准的 "Floor" 操作
        col = int((x - self.min_x) / self.resolution)
        row = int((y - self.min_y) / self.resolution)

        return row, col

    def index_to_coor(self, ind):
        row, col = ind[0], ind[1]

        # 保持不变：这里加半个分辨率是为了取中心点
        x = col * self.resolution + self.min_x + self.resolution / 2.0
        y = row * self.resolution + self.min_y + self.resolution / 2.0

        return x, y

    def is_valid_index(self, index):
        row, col = index
        return 0 <= row < self.grid_height and 0 <= col < self.grid_width

    def is_occupied_index(self, index):
        row, col = index
        return self.grid[row, col] == 1

    def show_map(self):
        plt.imshow(self.grid, cmap="gray")
        plt.title("2D Grid Map with Obstacles")
        plt.show()

    def show_map_real(self, path=None):
        """
        使用 Matplotlib 显示地图
        :param path: 可选，[(x1,y1), (x2,y2), ...] 路径点列表
        """
        plt.figure(figsize=(8, 8))

        # imshow 的 origin='lower' 让 (0,0) 在左下角，符合常规坐标系
        # extent 设置坐标轴刻度为真实物理尺寸
        plt.imshow(
            self.grid,
            cmap="Greys",
            origin="lower",
            extent=[self.min_x, self.min_x + self.width, self.min_y, self.min_y + self.height],
        )

        if path is not None:
            path = np.array(path)
            plt.plot(path[:, 0], path[:, 1], "r.-", label="Path")
            plt.legend()

        plt.title("Grid Map Visualization")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.grid(which="both", color="gray", linestyle=":", linewidth=0.5)
        plt.show()

    def draw_path(self, model, data, path, radius=0.01, color=[1, 0, 0]):
        for i, (x, y) in enumerate(path):
            # 在路径点处添加一个小球
            geom_name = f"path_point_{i}"

            # 创建一个小球（sphere）来表示路径点
            # x, y 表示坐标，radius 表示半径，z 坐标可以设置为路径点的高度（默认为 0）
            geom = mujoco.MjsGeom(name=geom_name, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius])

            # 设置路径点的位置（我们假设路径在 2D 平面内，z 坐标为 0）
            geom.pos = np.array([x, y, 0.0])

            # 将该小球添加到模型的 worldbody 中
            model.worldbody.append(geom)

            # 设置颜色
            model.geom_rgba.append(color)

        return model, data
