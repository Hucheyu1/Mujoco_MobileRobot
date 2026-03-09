"""
An example script showing how to train and test a RL policy for a differential robot to reach a goal using stable-baselines3
"""

import time

import mujoco
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.ppo.policies import MlpPolicy

from env.goal_env import DiffDrivePointGoalEnv

if __name__ == "__main__":
    train = False
    xml_path = "env/assets/rl_scene.xml"

    # ---------- train ----------
    def make_env():
        return DiffDrivePointGoalEnv(xml_path=xml_path, dt=0.05, max_duration=20, seed=0, render=True)

    if train:
        SEED = 2
        n_envs = 1
        vec_env = make_vec_env(make_env, n_envs=n_envs, seed=SEED)

        model = PPO(
            MlpPolicy,  # 使用多层感知机（全连接神经网络）策略
            vec_env,  # 训练环境
            verbose=1,  # 打印训练进度日志
            n_steps=1024,  # 每次更新前收集的步数（经验池大小）
            batch_size=64,  # 每次梯度下降的小批量大小
            n_epochs=10,  # 每次更新时重复优化的次数
            gamma=0.995,  # 折扣因子，越接近 1 越看重长期奖励
            gae_lambda=0.95,  # 广义优势估计参数，用于权衡偏差与方差
            ent_coef=0.0,
            device="cuda",
            tensorboard_log="./learning/tensorboard",
        )
        # 在 PPO 中，1 次迭代通常等于 n_steps * n_envs 步。
        # 如果你设置 n_steps=1024，n_envs=1，那么每 1024 * 5 = 5120 步，控制台才会跳出一次数据。
        model.learn(total_timesteps=1_500_000, log_interval=10)
        model.save("./learning/model/model.zip")

    # ---------- test ----------
    else:
        # load model
        model = PPO.load("./learning/model/model.zip")
        # load env
        test_env = make_env()

        #### draw the goal position in mujoco viewer
        test_env.viewer.draw_point(np.array([test_env.goal[0], test_env.goal[1], 0.03]))

        obs = test_env.reset()[0]
        test_env.render()

        ## run several episodes
        n_episodes = 10

        for ep in range(n_episodes):
            done = False
            ep_rew = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, rew, terminated, truncated, info = test_env.step(action)
                time.sleep(0.003)
                ep_rew += float(rew)
                done = bool(terminated or truncated)

                if done:
                    for i in range(200):  # stop at the goal for seconds
                        test_env.data.ctrl[0] = 0.0
                        test_env.data.ctrl[1] = 0.0
                        mujoco.mj_step(test_env.model, test_env.data)
                        test_env.render()
                        time.sleep(0.001)

            print(f"Episode {ep + 1}: Return={ep_rew}, Success={info.get('success', False)}")
            obs = test_env.reset()[0]
