import os
from datetime import datetime

import mediapy as media
import mujoco
import mujoco.viewer
import numpy as np


class MujocoViewer:
    def __init__(self, mujoco_model, mujoco_data):
        """
        mujoco_model: model in mujoco you want to show
        mujoco_data: data of the corresponding model
        """
        self.model = mujoco_model
        self.data = mujoco_data

        if self.model is None or self.data is None:
            raise ValueError("[MujocoViewer]: model or data cannot be None")

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # self.max_geoms 记录了这个上限，self.ngeo 用来记录当前画了多少个。
        self.ngeo = 0
        self.max_geoms = len(self.viewer.user_scn.geoms)  # 100000
        self.persistent_points = []  # 用于存储持久化的点信息（位置、大小、颜色），以便在视频录制中使用
        # mediapy视频录制相关属性
        self.renderer = None
        self.frames = []  # 存储捕获的帧
        self.framerate = 30  # 默认帧率
        self.recording = False
        self.output_path = None

    def start_recording(self, output_path=None, framerate=30):
        """
        开始录制视频
        :param output_path: 输出文件路径
        :param framerate: 帧率
        """
        if self.recording:
            print("[MujocoViewer]: 已经在录制中")
            return
        # 设置渲染器（如果还没有设置）
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=720, width=1280)
            self.max_scene_geoms = len(self.renderer.scene.geoms)  # 10000
        # 重置帧列表
        self.frames = []
        self.framerate = framerate
        self.recording = True
        # 设置输出路径
        if output_path is not None:
            self.output_path = output_path
        elif self.output_path is None:
            # 生成默认输出路径
            output_dir = "media/videos"
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_path = os.path.join(output_dir, f"simulation_{timestamp}.mp4")
        print(f"[MujocoViewer]: 开始录制视频: {self.output_path}, FPS: {framerate}")

    def stop_recording(self, save_video=True):
        """
        停止录制视频
        :param save_video: 是否保存视频文件
        """
        if not self.recording:
            print("[MujocoViewer]: 没有正在进行的录制")
            return

        self.recording = False

        if save_video and self.frames and self.output_path:
            try:
                # 确保目录存在
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

                # 使用mediapy保存视频
                media.write_video(self.output_path, self.frames, fps=self.framerate)
                print(f"[MujocoViewer]: 视频已保存: {self.output_path}")
                print(f"[MujocoViewer]: 总帧数: {len(self.frames)}, 时长: {len(self.frames) / self.framerate:.2f}秒")

            except Exception as e:
                print(f"[MujocoViewer]: 保存视频失败: {e}")

        # 清空帧列表（除非用户想保留）
        # self.frames = []

    def modify_scene(self):
        for pt, size, rgba in self.persistent_points:
            if self.renderer.scene.ngeom >= self.max_scene_geoms:
                print(
                    f"[MujocoViewer]: Exceeded max_scene_geoms={self.max_scene_geoms}, cannot add more persistent points."
                )
                break
            mujoco.mjv_initGeom(
                self.renderer.scene.geoms[self.renderer.scene.ngeom],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([size, 0, 0]),  # 转换为numpy数组更安全
                pos=np.array([pt[0], pt[1], pt[2]]),
                mat=np.eye(3).flatten(),
                rgba=rgba.astype(np.float32),  # renderer通常需要float32
            )
            self.renderer.scene.ngeom += 1

    def capture_frame(self):
        """捕获当前帧"""
        if not self.recording or self.renderer is None:
            return False

        try:
            # 更新场景并渲染
            self.renderer.update_scene(self.data, camera="top_down")
            self.modify_scene()  # 确保持久化的点也被渲染到当前帧
            pixels = self.renderer.render()
            self.frames.append(pixels)

            # 每100帧打印一次进度
            if len(self.frames) % 100 == 0:
                print(f"[MujocoViewer]: 已捕获 {len(self.frames)} 帧")

            return True

        except Exception as e:
            print(f"[MujocoViewer]: 捕获帧失败: {e}")
            return False

    def show_video(self):
        """显示录制的视频"""
        if self.frames:
            media.show_video(self.frames, fps=self.framerate)
        else:
            print("[MujocoViewer]: 没有可显示的帧")

    def save_video(self, output_path=None, framerate=None):
        """保存视频到指定路径"""
        if not self.frames:
            print("[MujocoViewer]: 没有帧可保存")
            return False

        if output_path is None:
            output_path = self.output_path

        if framerate is None:
            framerate = self.framerate

        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            media.write_video(output_path, self.frames, fps=framerate)
            print(f"[MujocoViewer]: 视频已保存: {output_path}")
            return True
        except Exception as e:
            print(f"[MujocoViewer]: 保存视频失败: {e}")
            return False

    def clear_frames(self):
        """清空帧缓存"""
        self.frames = []
        print("[MujocoViewer]: 帧缓存已清空")

    def draw_point(self, pt, size=0.05, rgba=np.array([1, 0, 0, 1])):
        """
        pt: [x, y, z]
        size: size of marker
        rgba: color [r, g, b, alpha]
        功能：在 3D 空间的 pt 坐标处画一个红色的（默认）小球。
        """
        if self.viewer.is_running():
            if self.ngeo >= self.max_geoms:
                print(f"[MujocoViewer]: Exceeded max_geoms={self.max_geoms}, resetting to 0.")
                self.reset()

            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[self.ngeo],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[size, 0, 0],
                pos=np.array([pt[0], pt[1], pt[2]]),
                mat=np.eye(3).flatten(),
                rgba=rgba,
            )
            self.ngeo += 1
            self.viewer.user_scn.ngeom = self.ngeo
        else:
            raise RuntimeError("[MujocoViewer]: Viewer window is closed")

        # 添加到持久化列表（用于视频录制）
        self.persistent_points.append((np.array(pt, dtype=np.float32), size, rgba.astype(np.float32)))

    def draw_line_segment(self, pt_from, pt_to, width=0.001, rgba=np.array([1, 0, 0, 1])):
        """
        pt_from: 1 end point of line segment, [x,y,z]
        pt_to: the other end point, [x,y,z]
        width: width of the line
        rgba: color
        功能：在两点之间画一条线
        """
        if self.viewer.is_running():
            if self.ngeo >= self.max_geoms:
                print(f"[MujocoViewer]: Exceeded max_geoms={self.max_geoms}, resetting to 0.")
                self.reset()

            mujoco.mjv_connector(
                self.viewer.user_scn.geoms[self.ngeo],
                type=mujoco.mjtGeom.mjGEOM_LINE,
                width=width,
                from_=np.array([pt_from[0], pt_from[1], pt_from[2]]),
                to=np.array([pt_to[0], pt_to[1], pt_to[2]]),
            )

            self.viewer.user_scn.geoms[self.ngeo].rgba = rgba
            self.ngeo += 1
            self.viewer.user_scn.ngeom += self.ngeo
        else:
            raise RuntimeError("[MujocoViewer]: Viewer window is closed")

    def draw_traj(self, traj, size=0.05, rgba=np.array([1, 0, 0, 1])):
        """
        traj: Nx3 array
        """

        if self.viewer.is_running():
            for pt in traj:
                self.draw_point(pt, size=size, rgba=rgba)
        else:
            raise RuntimeError("[MujocoViewer]: Viewer window is closed")

    def render(self, capture_frame=False):
        """
        show in mujoco
        :param capture_frame: 是否捕获当前帧用于视频录制
        """
        if self.viewer.is_running():
            self.viewer.sync()

            # 如果需要捕获帧
            if capture_frame and self.recording:
                self.capture_frame()
        else:
            raise RuntimeError("Viewer window is closed")

    def close(self):
        """
        close it
        """
        # 如果还在录制，停止并保存
        if self.recording:
            self.stop_recording()
        self.viewer.close()

    def reset(self):
        """
        need to reset if the current number of geom > max
        """
        self.viewer.user_scn.ngeom = 0
        self.ngeo = 0

    # def reset(self, ngeom):
    #     """
    #     instead of directly setting to 0, we can start to overwrite from specific position in the list

    #     """
    #     self.viewer.user_scn.ngeom = ngeom
    #     self.ngeo = ngeom

    def is_running(self):
        return self.viewer.is_running()
