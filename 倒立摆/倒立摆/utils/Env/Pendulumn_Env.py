import torch


class Pendulumn_Env:
    def __init__(self, Env_Config, Robot_Config, PPO_Config):
        """初始化环境变量"""
        self.g = 9.8
        self.l = 1.0
        self.m = 0.2
        self.dt = Env_Config.EnvParam.dt
        self.sub_step = Env_Config.EnvParam.sub_step
        self.device = Env_Config.EnvParam.device
        self.device = Env_Config.EnvParam.device
        self.train = Env_Config.EnvParam.train
        self.agents_num = (Env_Config.EnvParam.agents_num - Env_Config.EnvParam.agents_num_in_play) * self.train + \
                          Env_Config.EnvParam.agents_num_in_play

        """初始化机器人变量"""
        self.actuator_num = Robot_Config.ActuatorParam.actuator_num

        """初始化机器人位姿参数"""
        self.initial_angle = Robot_Config.InitialState.initial_angle
        self.initial_angular_vel = Robot_Config.InitialState.initial_angular_vel
        self.tht = self.initial_angle * (2 * torch.rand(device=self.device, size=(self.agents_num, 1)) - 1)
        self.tht_d = self.initial_angular_vel * (2 * torch.rand(device=self.device, size=(self.agents_num, 1)) - 1)

        """初始化额外机器人参数"""
        self.action = torch.zeros((self.agents_num, self.actuator_num), device=self.device)  # 动作

        """奖励和"""
        self.max_step = PPO_Config.PPOParam.maximum_step
        self.torque_reward_sum = 0
        self.termination_reward_sum = 0

    """-------------------以上均为初始化代码-----------------------"""
    """-------------------以上均为初始化代码-----------------------"""
    """-------------------以上均为初始化代码-----------------------"""

    """-------------------以下均为环境运行代码-----------------------"""
    """-------------------以下均为环境运行代码-----------------------"""
    """-------------------以下均为环境运行代码-----------------------"""

    def reset(self, index):
        # 从随机位置开始
        self.tht[index] = self.initial_angle * (2 * torch.rand(device=self.device, size=(len(index), 1)) - 1)
        self.tht_d[index] = self.initial_angular_vel * (2 * torch.rand(device=self.device, size=(len(index), 1)) - 1)

    def reset_all(self):
        self.tht = self.initial_angle * (2 * torch.rand(device=self.device, size=(self.agents_num, 1)) - 1)
        self.tht_d = self.initial_angular_vel * (2 * torch.rand(device=self.device, size=(self.agents_num, 1)) - 1)

    """机器人状态更新"""

    def step(self, torque):
        real_dt = self.dt / self.sub_step
        self.action = torque
        for i in range(self.sub_step):
            # 动力学更新
            tht_dd = (self.m * self.g * self.l * torch.sin(self.tht) + torque) / (self.m * self.l ** 2)

            self.tht_d += tht_dd * real_dt
            self.tht += self.tht_d * real_dt
            # 规范化角度到 [-pi, pi]
            self.tht = (self.tht + torch.pi) % (2 * torch.pi) - torch.pi

    def get_current_observations(self):
        current_state = torch.concatenate((self.tht, self.tht_d), dim=1)
        self.current_angle = self.tht
        self.current_angular_vel = self.tht_d
        # #——————————————————————获取额外机器人状态结束————————————————————————————————##

        return current_state

    def get_next_observations(self):
        next_state = torch.concatenate((self.tht, self.tht_d), dim=1)
        self.next_angle = self.tht
        self.next_angular_vel = self.tht_d
        return next_state

    """-------------------以上均为环境运行代码-----------------------"""
    """-------------------以上均为环境运行代码-----------------------"""
    """-------------------以上均为环境运行代码-----------------------"""

    """-------------------以下均为奖励计算代码-----------------------"""
    """-------------------以下均为奖励计算代码-----------------------"""
    """-------------------以下均为奖励计算代码-----------------------"""

    """惩罚关节用力"""

    def effort_penalty_reward(self):
        joint_effort_reward = -0.05 * torch.abs(self.action.sum(dim=1, keepdim=True))
        return joint_effort_reward

    def termination_reward(self):
        over1 = torch.abs(self.next_angular_vel) < 0.1
        over2 = torch.abs(self.next_angle) < 0.02
        over3 = ((self.next_angle * self.current_angle) < 0) & (torch.abs(self.next_angle) < 1)
        self.over = over1 & (over2 | over3)
        reward = 300 * self.over.float()

        return reward

    def compute_reward(self):
        reward = 0

        reward += 1 * self.effort_penalty_reward()
        reward += 1 * self.termination_reward()
        self.torque_reward_sum += self.effort_penalty_reward().mean().item() / self.max_step
        self.termination_reward_sum += self.termination_reward().mean().item() / self.max_step

        return reward, self.over.float()

    def print_reward_sum(self):
        print(f"torque_reward_sum: {self.torque_reward_sum:.4f}")
        print(f"termination_reward_sum: {self.termination_reward_sum:.4f}")
        self.torque_reward_sum = 0
        self.termination_reward_sum = 0
