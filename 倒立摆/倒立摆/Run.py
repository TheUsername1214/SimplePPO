from utils.Env.Pendulumn_Env import Pendulumn_Env
from utils.PPO.Actor_Critic_Discrete import Actor_Critic
from utils.Config.Config import *

maximum_step = PPO_Config.PPOParam.maximum_step
episode = PPO_Config.PPOParam.episode
train = Env_Config.EnvParam.train
AC = Actor_Critic(PPO_Config, Env_Config)
env = Pendulumn_Env(Env_Config, Robot_Config, PPO_Config)
import torch
import matplotlib.pyplot as plt
import numpy as np

for epi in range(episode):
    print(f"===================episode: {epi}===================")
    angle = []
    for step in range(maximum_step):
        """获取当前状态"""
        state = env.get_current_observations()

        """做动作"""
        action_index, torque = AC.sample_action(state)

        if not train:
            angle = state[0, 0].item()
            tor = torque[0, 0].item()
            if tor == 1:
                plt.plot(np.array([0, -np.sin(angle)]),
                         np.array([0, np.cos(angle)]), "r-")
            elif tor == 0:
                plt.plot(np.array([0, -np.sin(angle)]),
                         np.array([0, np.cos(angle)]), "g-")
            elif tor == -1:
                plt.plot(np.array([0, -np.sin(angle)]),
                         np.array([0, np.cos(angle)]), "b-")

            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.title(f"torque = {tor}N/m")
            plt.show(block=False)
            plt.pause(0.01)
            plt.clf()

        """更新环境"""
        env.step(torque=torque)

        """获取下一个状态"""
        next_state = env.get_next_observations()

        """计算奖励 判断是否结束"""
        reward, over = env.compute_reward()

        """存储经验"""
        if train:
            AC.store_experience(state,
                                action_index,
                                next_state,
                                reward,
                                over,
                                step)

        """重置挂掉的机器人"""
        env.reset(torch.nonzero(over.flatten()).flatten())
    env.reset_all()

    """每个回合结束后训练一次"""
    if train:
        AC.update()
        env.print_reward_sum()
