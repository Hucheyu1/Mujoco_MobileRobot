"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

"""

import math
import pathlib
import sys

import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parent.parent))

import mujoco
import numpy as np
from RRT.rrt import RRT

show_animation = True


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(
        self,
        start,
        goal,
        obstacle_list,
        rand_area,
        expand_dis=30.0,
        path_resolution=1.0,
        goal_sample_rate=20,
        max_iter=300,
        connect_circle_dist=50.0,
        search_until_max_iter=False,
        play_area=None,
        robot_radius=0.0,
    ):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        connect_circle_dist: 核心参数。当生成一个新节点时，算法会搜索这个半径范围内的所有邻居节点，尝试优化连接。
        search_until_max_iter=False: 如果设为 True，即使找到了路径，程序也会继续运行直到跑完 max_iter 次迭代，以便不断优化路径让它变得更直、更短


        """
        super().__init__(
            start,
            goal,
            obstacle_list,
            rand_area,
            expand_dis,
            path_resolution,
            goal_sample_rate,
            max_iter,
            play_area,
            robot_radius=robot_radius,
        )
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter
        self.node_list = []

    def planning(self, animation=True):
        """
        rrt star path planning

        animation: flag for animation on or off .
        """
        # 初始化树，放入起点
        self.node_list = [self.start]
        for i in tqdm(range(self.max_iter)):
            # print("Iter:", i, ", number of nodes:", len(self.node_list))
            # --- 步骤 1: 标准 RRT 扩展过程 ---
            rnd = self.get_random_node()
            # 找最近的树节点
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)
            # --- 步骤 2: 计算新节点的初始代价 ---
            near_node = self.node_list[nearest_ind]
            # 新节点代价 = 父节点代价 + 两者间距离
            new_node.cost = near_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)
            # --- 步骤 3: 碰撞检测 ---
            if self.check_collision(new_node, self.obstacle_list, self.robot_radius):
                # --- 步骤 4: 寻找附近的邻居 (Search Ball) ---
                # 找出以 new_node 为圆心，connect_circle_dist 为半径内的所有现有节点
                near_inds = self.find_near_nodes(new_node)
                # --- 步骤 5: 选择最优父节点 (Choose Parent) ---
                # RRT 直接连接 nearest_node，但 RRT* 会问：
                # “在这些邻居里，有没有谁做我的父亲，能让我离起点的总距离更短？”
                node_with_updated_parent = self.choose_parent(new_node, near_inds)
                if node_with_updated_parent:
                    # --- 步骤 6: 重连 (Rewire) ---
                    # 既然新节点加入树了，RRT* 会反过来问邻居：
                    # “邻居们，如果你把父节点换成我（经过我走），会不会比你们原来的路更近？”
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if animation:
                self.draw_graph(rnd)

            if (not self.search_until_max_iter) and new_node:  # if reaches goal
                last_index = self.search_best_goal_node()
                if last_index is not None:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

        return None

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node

            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list, self.robot_radius):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacle_list, self.robot_radius):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        safe_goal_costs = [
            self.node_list[i].cost + self.calc_dist_to_goal(self.node_list[i].x, self.node_list[i].y)
            for i in safe_goal_inds
        ]

        min_cost = min(safe_goal_costs)
        for i, cost in zip(safe_goal_inds, safe_goal_costs):
            if cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt(math.log(nnode) / nnode)
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, "expand_dis"):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x) ** 2 + (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
        For each node in near_inds, this will check if it is cheaper to
        arrive to them from new_node.
        In such a case, this will re-assign the parent of the nodes in
        near_inds to new_node.
        Parameters:
        ----------
            new_node, Node
                Node randomly added which can be joined to the tree

            near_inds, list of uints
                A list of indices of the self.new_node which contains
                nodes within a circle of a given radius.
        Remark: parent is designated in choose_parent.

        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list, self.robot_radius)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                for node in self.node_list:
                    if node.parent == self.node_list[i]:
                        node.parent = edge_node
                self.node_list[i] = edge_node
                self.propagate_cost_to_leaves(self.node_list[i])

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


class RRTStarMujoco(RRTStar):
    def __init__(
        self,
        start,
        goal,
        robot,
        rand_area,
        expand_dis=1.0,
        path_resolution=0.05,
        goal_sample_rate=5,
        max_iter=500,
        play_area=None,
        connect_circle_dist=1.0,
        robot_radius=0.15,
        robot_z=0.03,
        search_until_max_iter=True,
    ):
        """
        :param robot: 封装好的 Robot 类实例 (包含 set_state 和 is_collid 方法)
        :param path_resolution: 碰撞检测的步长 (插值密度)
        :param robot_z: 碰撞检测时机器人的高度 (防止与地面误触)
        """
        # obstacle_list 设为空，因为我们直接查物理引擎
        super().__init__(
            start,
            goal,
            obstacle_list=[],
            rand_area=rand_area,
            expand_dis=expand_dis,
            path_resolution=path_resolution,
            goal_sample_rate=goal_sample_rate,
            max_iter=max_iter,
            play_area=play_area,
            connect_circle_dist=connect_circle_dist,
            robot_radius=robot_radius,
            search_until_max_iter=search_until_max_iter,
        )

        self.robot = robot
        self.z = robot_z

    def check_collision(self, node, obstacleList, robot_radius):
        """
        重写碰撞检测：使用 MuJoCo 物理引擎
        注意：必须检查从 node.parent 到 node 的整条路径，防止穿墙
        """
        if node is None:
            return False

        # 1. 检查是否在采样范围内 (边界检查)
        if not (self.min_rand <= node.x <= self.max_rand and self.min_rand <= node.y <= self.max_rand):
            return False

        # 如果没有父节点，只检查当前点
        if node.parent is None:
            return self._is_state_valid(node.x, node.y)

        # 2. 路径插值检查 (防止穿墙)
        # 计算父节点到当前节点的距离和角度
        dx = node.x - node.parent.x
        dy = node.y - node.parent.y
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)

        # 根据 resolution 计算需要检查多少个点
        n_steps = int(dist / self.path_resolution)

        # 循环检测路径上的每一个点
        for i in range(n_steps + 1):
            # 计算插值点坐标
            x_check = node.parent.x + i * self.path_resolution * math.cos(angle)
            y_check = node.parent.y + i * self.path_resolution * math.sin(angle)

            if not self._is_state_valid(x_check, y_check):
                return False  # 只要有一点碰撞，整个连线无效

        # 最后再检查一下终点（虽然循环通常会覆盖，但保险起见）
        if not self._is_state_valid(node.x, node.y):
            return False

        return True  # 全程无碰撞

    def _is_state_valid(self, x, y):
        """
        单一状态的物理查询 helper 函数
        """
        # 设置机器人状态
        # 注意：这里我们只规划了 x, y。对于 theta，如果机器人是圆的，设为 0 即可。
        # 如果机器人是非圆形的，可能需要把 theta 也放入 Node 中规划。
        self.robot.set_state(np.array([x, y, self.z]), 0.0)

        # 必须调用 mj_forward 更新几何体位置
        mujoco.mj_forward(self.robot.model, self.robot.data)

        # 检查是否发生碰撞
        # is_collid() 返回 True 表示撞了，所以 valid 返回 False
        return not self.robot.is_collid()


class RRTStarGridMap(RRTStar):
    def __init__(
        self,
        start,
        goal,
        grid_map,
        rand_area,
        expand_dis=1.0,
        path_resolution=0.05,
        goal_sample_rate=5,
        max_iter=500,
        play_area=None,
        connect_circle_dist=1.0,
        robot_radius=0.0,
        search_until_max_iter=True,
    ):
        """
        grid_map: 传入你的 GridMap 对象
        """
        # 我们不再需要 obstacle_list，传个空列表 [] 进去
        super().__init__(
            start,
            goal,
            obstacle_list=[],
            rand_area=rand_area,
            expand_dis=expand_dis,
            path_resolution=path_resolution,
            goal_sample_rate=goal_sample_rate,
            max_iter=max_iter,
            play_area=play_area,
            connect_circle_dist=connect_circle_dist,
            robot_radius=robot_radius,
            search_until_max_iter=search_until_max_iter,
        )
        self.grid_map = grid_map

    def check_collision(self, node, obstacleList, robot_radius):
        """
        重写碰撞检测逻辑：
        检查节点及其父节点之间的连线是否穿过 GridMap 的障碍物
        """
        if node is None:
            return False

        # 1. 检查节点本身是否在地图外
        if not self._is_inside_map(node.x, node.y):
            return False

        # 2. 检查节点本身是否在障碍物内
        if self._is_collided(node.x, node.y):
            return False

        # 3. 检查连线 (Path Collision)
        # 如果这个节点有父节点，我们需要检查连线是否碰撞
        if node.parent is None:
            return True

        dx = node.x - node.parent.x
        dy = node.y - node.parent.y
        dist = math.hypot(dx, dy)
        angle = math.atan2(dy, dx)

        # 沿连线进行采样检查
        # 步长取 resolution 的一半以防漏掉障碍
        step = self.grid_map.resolution / 2.0
        n_steps = int(dist / step)

        for i in range(n_steps + 1):
            check_x = node.parent.x + step * i * math.cos(angle)
            check_y = node.parent.y + step * i * math.sin(angle)

            if self._is_collided(check_x, check_y):
                return False  # 碰撞了

        return True  # 安全

    def _is_inside_map(self, x, y):
        # 检查是否超出 GridMap 的物理边界
        return (
            self.grid_map.min_x <= x <= self.grid_map.min_x + self.grid_map.width
            and self.grid_map.min_y <= y <= self.grid_map.min_y + self.grid_map.height
        )

    def _is_collided(self, x, y):
        """
        查询 GridMap
        """
        # 将物理坐标转为索引
        try:
            row, col = self.grid_map.coor_to_index([x, y])
        except:  # noqa: E722
            return True  # 转换失败通常意味着出界

        # 检查索引有效性
        if not self.grid_map.is_valid_index((row, col)):
            return True  # 出界视为碰撞

        # 检查占用 (GridMap 中 1 代表障碍)
        if self.grid_map.is_occupied_index((row, col)):
            return True

        return False

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect("key_release_event", lambda event: [exit(0) if event.key == "escape" else None])

        # === 1. 绘制栅格地图背景 ===
        if hasattr(self, "grid_map"):
            # origin='lower' 让 (0,0) 在左下角
            # extent 确保显示的图片坐标与你的物理坐标 (米) 对齐
            plt.imshow(
                self.grid_map.grid,
                cmap="Greys",
                origin="lower",
                extent=[
                    self.grid_map.min_x,
                    self.grid_map.min_x + self.grid_map.width,
                    self.grid_map.min_y,
                    self.grid_map.min_y + self.grid_map.height,
                ],
            )

        # === 2. 绘制随机采样点 ===
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
            # 如果需要显示机器人在该点的大小，可以保留画圆
            if self.robot_radius > 0.0:
                self.plot_circle(rnd.x, rnd.y, self.robot_radius, "-r")

        # === 3. 绘制 RRT 树 (绿线) ===
        for node in self.node_list:
            if node.parent:
                # 只有当路径不为空时才画
                if node.path_x is not None and len(node.path_x) > 0:
                    plt.plot(node.path_x, node.path_y, "-g")
                else:
                    # 如果没有存储详细路径，直接画直线连接父子节点
                    plt.plot([node.x, node.parent.x], [node.y, node.parent.y], "-g")

        # === 4. 移除原本画圆圈障碍物的代码 ===
        # for (ox, oy, size) in self.obstacle_list:
        #     self.plot_circle(ox, oy, size)

        # === 5. 绘制起点和终点 ===
        plt.plot(self.start.x, self.start.y, "xr", markersize=10, label="Start")
        plt.plot(self.end.x, self.end.y, "xb", markersize=10, label="Goal")

        # === 6. 设置坐标轴 ===
        # 这里的范围应该跟 GridMap 一致
        if hasattr(self, "grid_map"):
            plt.axis(
                [
                    self.grid_map.min_x,
                    self.grid_map.min_x + self.grid_map.width,
                    self.grid_map.min_y,
                    self.grid_map.min_y + self.grid_map.height,
                ]
            )
        else:
            plt.axis("equal")
            plt.axis([self.min_rand, self.max_rand, self.min_rand, self.max_rand])

        # plt.grid(True)
        plt.pause(0.01)


def main():
    print("Start " + __file__)

    # ====Search Path with RRT====
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
        (6, 12, 1),
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    rrt_star = RRTStar(
        start=[0, 0], goal=[6, 10], rand_area=[-2, 15], obstacle_list=obstacle_list, expand_dis=1, robot_radius=0.8
    )
    path = rrt_star.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt_star.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], "r--")
            plt.grid(True)
            plt.show()


if __name__ == "__main__":
    main()
