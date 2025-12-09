# ============================================================================
# 多智能体随机动作测试脚本
# 功能：在 Unreal 环境中放置多个智能体，每个智能体执行随机动作
# 用途：用于测试和演示多智能体环境，评估环境性能
# ============================================================================

import os
import argparse
import gym_unrealcv
import gym
from gym import wrappers
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE

class RandomAgent(object):
    """
    随机动作智能体类
    
    这是一个最简单的智能体实现，不进行任何学习，纯粹随机选择动作。
    策略特点：
    - 每 keep_steps 步随机采样一个新动作
    - 其他时间保持当前动作不变（避免动作抖动）
    - 支持离散和连续动作空间
    """
    def __init__(self, action_space):
        """
        初始化随机智能体
        
        Args:
            action_space: Gym 动作空间对象（Discrete 或 Box）
        """
        self.action_space = action_space  # 保存动作空间引用
        self.count_steps = 0  # 当前动作已执行的步数
        self.action = self.action_space.sample()  # 初始化时立即采样第一个随机动作

    def act(self, observation, keep_steps=10):
        """
        根据观察选择动作（实际上不使用观察，纯随机策略）
        
        Args:
            observation: 当前观察（此策略中不使用）
            keep_steps: 保持同一动作的步数，默认10步
            
        Returns:
            动作值（离散动作返回整数，连续动作返回数组）
        """
        self.count_steps += 1  # 增加步数计数
        
        # 如果超过保持步数，采样新动作
        if self.count_steps > keep_steps:
            self.action = self.action_space.sample()  # 从动作空间中随机采样新动作
            self.count_steps = 0  # 重置计数器
        else:
            # 否则返回当前动作（保持不变）
            return self.action
        
        return self.action

    def reset(self):
        """
        重置智能体状态（在每个 episode 开始时调用）
        """
        self.action = self.action_space.sample()  # 采样新的随机动作
        self.count_steps = 0  # 重置步数计数器

# UnrealEnv 环境变量会在 base_env.py 中自动设置，无需在此硬编码
if __name__ == '__main__':
    # ========================================================================
    # 1. 解析命令行参数
    # ========================================================================
    parser = argparse.ArgumentParser(description='多智能体随机动作测试脚本')
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-MiddleEast-ContinuousColor-v0',
                        help='选择要运行的环境ID')
    parser.add_argument("-r", '--render', dest='render', action='store_true', 
                        help='使用 cv2 显示环境画面')
    parser.add_argument("-s", '--seed', dest='seed', default=0, 
                        help='随机种子，用于可重复性')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=10, 
                        help='时间膨胀系数，用于加速模拟（-1表示禁用）')
    parser.add_argument("-n", '--nav-agent', dest='nav_agent', action='store_true', 
                        help='使用导航智能体控制智能体移动')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, 
                        help='目标丢失N步后提前结束episode（-1表示禁用）')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', 
                        help='启用自动监控显示')

    args = parser.parse_args()

    # ========================================================================
    # 2. 创建和配置环境
    # ========================================================================
    # 创建基础环境
    env = gym.make(args.env_id)
    
    # 配置 Unreal 引擎渲染设置（非离屏模式，分辨率240x240）
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(240, 240))
    
    # 设置智能体类型为 'player'（只使用玩家类型的智能体，忽略其他类型如drone）
    env.unwrapped.agents_category = ['player']

    # 应用时间膨胀包装器（加速模拟，默认10倍速度）
    if int(args.time_dilation) > 0:  # -1 表示不使用时间膨胀
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    
    # 应用提前结束包装器（目标丢失N步后结束）
    if int(args.early_done) > 0:  # -1 表示不提前结束
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    
    # 应用监控显示包装器（可选）
    if args.monitor:
        env = monitor.DisplayWrapper(env)

    # 应用随机种群包装器（设置智能体数量为2-2个，不随机目标）
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    
    # 应用导航智能体包装器（可选，使用导航系统控制智能体）
    if args.nav_agent:
        env = agents.NavAgents(env, mask_agent=False)

    # ========================================================================
    # 3. 初始化训练循环变量
    # ========================================================================
    episode_count = 100  # 运行100个episode
    rewards = 0
    done = False
    Total_rewards = 0  # 累计所有episode的奖励

    # 设置随机种子（确保可重复性）
    env.seed(int(args.seed))
    print("seed:", args.seed)

    # ========================================================================
    # 4. 主训练循环
    # ========================================================================
    for eps in range(1, episode_count):
        print("eps:" + str(eps))
        # 重置环境，获取初始观察
        obs = env.reset()
        
        # 获取智能体数量（从动作空间数量推断）
        agents_num = len(env.action_space)
        print("agents_num:" + str(agents_num))
        
        # 为每个智能体创建一个 RandomAgent 实例
        agents = [RandomAgent(env.action_space[i]) for i in range(agents_num)]
        
        # 初始化episode统计变量
        count_step = 0  # 当前episode的步数
        t0 = time.time()  # 记录episode开始时间
        agents_num = len(obs)  # 从观察数量确认智能体数量
        C_rewards = np.zeros(agents_num)  # 当前episode各智能体的累计奖励
        
        # Episode 内部循环
        while True:
            # 每个智能体根据观察选择动作（RandomAgent实际上不使用观察）
            actions = [agents[i].act(obs[i]) for i in range(agents_num)]
            
            print("start env.step")
            # 执行动作，获取新的观察、奖励、完成标志和信息
            obs, rewards, done, info = env.step(actions)
            print(obs)
            print(rewards)

            print("done")
            print(done)
            print("info")
            print(info)
            print("end env step")

            # 累计奖励
            C_rewards += rewards
            count_step += 1
            
            # 如果启用渲染，显示环境画面
            if args.render:
                img = env.render(mode='rgb_array')  # 获取RGB图像
                # img = img[..., ::-1]  # 如果需要BGR转RGB，取消注释
                cv2.imshow('show', img)  # 显示图像
                cv2.waitKey(1)  # 等待1ms（保持窗口响应）
            
            # 如果episode结束，打印统计信息并跳出循环
            if done:
                fps = count_step / (time.time() - t0)  # 计算帧率（步数/秒）
                Total_rewards += C_rewards[0]  # 累计第一个智能体的奖励
                # 打印：帧率、当前episode奖励、平均奖励
                print('Fps:' + str(fps), 'R:' + str(C_rewards), 
                      'R_ave:' + str(Total_rewards / eps))
                break

    # ========================================================================
    # 5. 清理和结束
    # ========================================================================
    print('Finished')
    env.close()  # 关闭环境，释放资源



