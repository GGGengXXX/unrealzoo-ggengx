# ============================================================================
# 多相机录制脚本
# 功能：选择主体（动物或人类），设置动作，配置多个不同角度的摄像头，录制指定时长的视频
# ============================================================================

import os
import argparse
import json
import gym_unrealcv
import gym
from gym import spaces
from gym.wrappers import TimeLimit
import cv2
import time
import numpy as np
from datetime import datetime
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE
from gym_unrealcv.envs.utils import misc
from gym_unrealcv.envs.tracking.baseline import PoseTracker
import sys

class ActionAgent(object):
    """动作控制智能体"""
    def __init__(self, action_space, action_mode='random', agent_index=0, is_tracker=False):
        """
        初始化动作智能体
        
        Args:
            action_space: 动作空间
            action_mode: 动作模式
                - 'random': 随机动作
                - 'forward': 持续向前
                - 'circle': 圆形移动
                - 'idle': 静止不动
                - 'track': 追踪模式（追踪目标智能体）
            agent_index: 智能体索引
            is_tracker: 是否为追踪者（仅在track模式下使用）
        """
        self.action_space = action_space
        self.action_mode = action_mode
        self.agent_index = agent_index
        self.is_tracker = is_tracker
        self.count_steps = 0
        self.action = self._get_initial_action()
        self.tracking_info = None  # 存储追踪调试信息
        
        # 如果是追踪模式且是追踪者，创建PoseTracker
        if self.action_mode == 'track' and self.is_tracker:
            try:
                # 处理Tuple类型的动作空间（Mixed模式）
                # Tuple包含 (move_space, turn_space, animation_space)
                # 对于追踪，我们只需要move_space（连续动作）
                actual_action_space = action_space
                
                # 检查是否是gym.spaces.Tuple类型
                if hasattr(spaces, 'Tuple') and isinstance(action_space, spaces.Tuple):
                    # 如果是Tuple空间，使用spaces属性访问子空间
                    if hasattr(action_space, 'spaces') and len(action_space.spaces) > 0:
                        actual_action_space = action_space.spaces[0]  # 第一个是move_space
                        print(f"  检测到Mixed动作空间(Tuple)，使用move_space进行追踪")
                    elif hasattr(action_space, '__getitem__'):
                        # 如果支持索引访问，尝试使用索引
                        try:
                            actual_action_space = action_space[0]
                            print(f"  检测到Mixed动作空间(索引访问)，使用move_space进行追踪")
                        except (TypeError, IndexError):
                            pass
                # 检查是否是普通的tuple类型（不是gym.spaces.Tuple）
                elif isinstance(action_space, tuple) and len(action_space) > 0:
                    # 如果是普通tuple，第一个元素应该是move_space
                    actual_action_space = action_space[0]
                    print(f"  检测到tuple类型动作空间，使用第一个元素进行追踪")
                
                self.pose_tracker = PoseTracker(actual_action_space, expected_distance=250, expected_angle=0)
            except (ImportError, AttributeError, TypeError, IndexError) as e:
                print(f"警告：无法创建PoseTracker: {e}，追踪模式可能无法正常工作")
                import traceback
                traceback.print_exc()
                self.pose_tracker = None
        else:
            self.pose_tracker = None
    
    def _get_initial_action(self):
        """获取初始动作"""
        if self.action_mode == 'idle':
            # 静止：返回零动作或最小动作
            if hasattr(self.action_space, 'low'):
                return np.zeros_like(self.action_space.low)
            else:
                return 0
        elif self.action_mode == 'forward':
            # 向前：对于连续动作空间，设置前进速度
            if hasattr(self.action_space, 'low'):
                action = np.zeros_like(self.action_space.low)
                if len(action) >= 2:
                    action[1] = 1.0  # 假设第二个维度是前进速度
                return action
            else:
                return 1  # 假设动作1是前进
        else:
            return self.action_space.sample()
    
    def act(self, observation=None, keep_steps=10, agent_poses=None, target_index=None):
        """
        生成动作
        
        Args:
            observation: 观察值（可选）
            keep_steps: 保持动作的步数（用于某些模式）
            agent_poses: 所有智能体的位置列表（用于追踪模式）
            target_index: 目标智能体索引（用于追踪模式）
        
        Returns:
            动作值
        """
        self.count_steps += 1
        
        if self.action_mode == 'random':
            if self.count_steps > keep_steps:
                self.action = self.action_space.sample()
                self.count_steps = 0
        
        elif self.action_mode == 'forward':
            # 持续向前，每keep_steps步可能微调方向
            if self.count_steps > keep_steps:
                self.action = self._get_initial_action()
                self.count_steps = 0
        
        elif self.action_mode == 'circle':
            # 圆形移动：前进+旋转
            if hasattr(self.action_space, 'low'):
                action = np.zeros_like(self.action_space.low)
                if len(action) >= 2:
                    action[1] = 0.8  # 前进速度
                    # 旋转角度随时间变化
                    angle = (self.count_steps % 360) * np.pi / 180.0
                    if len(action) >= 1:
                        action[0] = 0.3 * np.sin(angle)  # 左右转向
                self.action = action
            else:
                # 离散动作空间：交替前进和转向
                if self.count_steps % 20 < 10:
                    self.action = 1  # 前进
                else:
                    self.action = 2  # 右转
        
        elif self.action_mode == 'idle':
            # 静止不动
            self.action = self._get_initial_action()
        
        elif self.action_mode == 'track':
            # 追踪模式
            if self.is_tracker and self.pose_tracker is not None and agent_poses is not None:
                # 追踪者：使用PoseTracker追踪目标
                if target_index is not None and target_index < len(agent_poses):
                    tracker_pose = agent_poses[self.agent_index]
                    target_pose = agent_poses[target_index]
                    
                    if tracker_pose is not None and target_pose is not None and len(tracker_pose) >= 6 and len(target_pose) >= 6:
                        # 计算追踪信息（用于调试）
                        distance_2d = np.linalg.norm(np.array(tracker_pose[:2]) - np.array(target_pose[:2]))
                        direction_angle = misc.get_direction(tracker_pose, target_pose)
                        expected_distance = self.pose_tracker.expected_distance
                        distance_diff = distance_2d - expected_distance
                        
                        # 使用PoseTracker计算追踪动作（返回 [angle, velocity]）
                        move_action = self.pose_tracker.act(tracker_pose, target_pose)
                        
                        # 如果动作空间是Tuple类型（Mixed模式），需要构造完整的动作
                        is_tuple_space = False
                        if hasattr(spaces, 'Tuple') and isinstance(self.action_space, spaces.Tuple):
                            is_tuple_space = True
                        elif isinstance(self.action_space, tuple):
                            is_tuple_space = True
                        
                        if is_tuple_space:
                            # Tuple格式: (move_space, turn_space, animation_space)
                            # move_action 是 [angle, velocity]
                            # turn_action 和 animation_action 使用默认值（0 = 不动作）
                            self.action = (move_action, 0, 0)
                        else:
                            # 连续动作空间，直接使用move_action
                            self.action = move_action
                        
                        # 存储追踪信息用于调试输出
                        self.tracking_info = {
                            'distance': distance_2d,
                            'direction_angle': direction_angle,
                            'expected_distance': expected_distance,
                            'distance_diff': distance_diff,
                            'action_angle': move_action[0] if isinstance(move_action, (list, np.ndarray)) and len(move_action) > 0 else 0,
                            'action_velocity': move_action[1] if isinstance(move_action, (list, np.ndarray)) and len(move_action) > 1 else 0,
                            'tracker_pos': tracker_pose[:3],
                            'target_pos': target_pose[:3]
                        }
                    else:
                        # 如果位置信息不完整，使用随机动作
                        if self.count_steps > keep_steps or self.action is None:
                            self.action = self.action_space.sample()
                            self.count_steps = 0
                        self.tracking_info = None
                else:
                    # 如果没有目标索引，使用随机动作
                    if self.count_steps > keep_steps or self.action is None:
                        self.action = self.action_space.sample()
                        self.count_steps = 0
                    self.tracking_info = None
            else:
                # 非追踪者或被追踪者：使用随机动作（或者可以设置为逃跑策略）
                if self.count_steps > keep_steps or self.action is None:
                    self.action = self.action_space.sample()
                    self.count_steps = 0
                self.tracking_info = None
        
        # 确保总是返回有效的动作
        if self.action is None:
            self.action = self.action_space.sample()
        return self.action

    def reset(self):
        """重置智能体"""
        self.action = self._get_initial_action()
        self.count_steps = 0

class CameraConfig:
    """相机配置类"""
    def __init__(self, name, cam_type, **kwargs):
        """
        初始化相机配置
        
        Args:
            name: 相机名称（用于文件命名）
            cam_type: 相机类型 ('first_person', 'third_person', 'fixed', 'topview')
            **kwargs: 相机参数
                - 对于 'first_person': agent_index (必需), agent_type (可选，默认'player')
                - 对于 'third_person': agent_index, distance, height, angle (必需), agent_type (可选，默认'player')
                - 对于 'fixed': location, rotation (必需)
                - 对于 'topview': location (必需), height (可选，默认1500)
        """
        self.name = name
        self.cam_type = cam_type
        self.params = kwargs
        
        # 验证必需参数
        if cam_type == 'first_person':
            assert 'agent_index' in kwargs, "first_person相机需要agent_index参数"
            # 默认智能体类型为'player'
            if 'agent_type' not in kwargs:
                kwargs['agent_type'] = 'player'
        elif cam_type == 'third_person':
            assert all(k in kwargs for k in ['agent_index', 'distance', 'height', 'angle']), \
                "third_person相机需要agent_index, distance, height, angle参数"
            # 默认智能体类型为'player'
            if 'agent_type' not in kwargs:
                kwargs['agent_type'] = 'player'
        elif cam_type == 'fixed':
            assert all(k in kwargs for k in ['location', 'rotation']), \
                "fixed相机需要location和rotation参数"
        elif cam_type == 'topview':
            assert 'location' in kwargs, "topview相机需要location参数"
        else:
            raise ValueError(f"未知的相机类型: {cam_type}")

def load_camera_configs(config_file):
    """从JSON文件加载相机配置"""
    with open(config_file, 'r', encoding='utf-8') as f:
        configs = json.load(f)
    
    cameras = []
    for cam_config in configs.get('cameras', []):
        cameras.append(CameraConfig(**cam_config))
    
    return cameras

def create_default_camera_configs(num_agents=2, target_agent_index=None, target_agent_type=None):
    """
    创建默认的相机配置（第一人称、第三人称）
    默认只为第一个智能体（索引0）创建相机
    
    Args:
        num_agents: 智能体总数（未使用，保留兼容性）
        target_agent_index: 目标智能体索引（如果指定，为该agent创建相机，否则为第一个agent）
        target_agent_type: 目标智能体类型（如果指定，与target_agent_index一起使用）
    """
    cameras = []
    
    # 确定目标agent索引（默认为第一个智能体，索引0）
    agent_index = target_agent_index if target_agent_index is not None else 0
    agent_type = target_agent_type or 'player'
    
    print(f'只为智能体 {agent_index} ({agent_type}) 创建相机配置')
        
        # 第一人称
    cameras.append(CameraConfig(
        name=f'agent_{agent_index}_first_person',
        cam_type='first_person',
        agent_index=agent_index,
        agent_type=agent_type
    ))
    
    # 第三人称（4个角度：0°, 90°, 180°, 270°）
    for i in range(4):
        cameras.append(CameraConfig(
            name=f'agent_{agent_index}_third_person_{i*90}',
            cam_type='third_person',
            agent_index=agent_index,
            agent_type=agent_type,
            distance=100.0,
            height=100,
            angle=i*90.0
        ))
    
    return cameras

def create_virtual_agent_for_camera(unwrapped_env, agents_by_type, target_agent_type):
    """
    创建虚拟agent以获取额外的相机ID
    
    Args:
        unwrapped_env: 未包装的环境对象
        agents_by_type: 按类型分组的智能体字典
        target_agent_type: 目标智能体类型（用于参考配置）
    
    Returns:
        int: 新创建的相机ID，如果失败返回None
    """
    try:
        # 获取目标agent的配置作为参考
        refer_agent = None
        if agents_by_type and target_agent_type in agents_by_type:
            type_agents = agents_by_type[target_agent_type]
            if len(type_agents) > 0:
                agent_name = type_agents[0]
                if hasattr(unwrapped_env, 'agents') and agent_name in unwrapped_env.agents:
                    refer_agent = unwrapped_env.agents[agent_name]
        
        if refer_agent is None:
            # 如果没有找到参考agent，尝试使用第一个agent
            if hasattr(unwrapped_env, 'agents') and len(unwrapped_env.agents) > 0:
                refer_agent = list(unwrapped_env.agents.values())[0]
        
        if refer_agent is None:
            print(f"  ⚠️  无法创建虚拟agent：找不到参考agent配置")
            return None
        
        # 创建虚拟agent名称（使用时间戳确保唯一性）
        import time
        virtual_name = f"virtual_cam_{int(time.time() * 1000)}"
        
        # 将虚拟agent放置在远离场景的位置（例如，z=10000，使其不可见）
        virtual_loc = [0, 0, 10000]
        
        # 创建虚拟agent
        new_agent = unwrapped_env.add_agent(virtual_name, virtual_loc, refer_agent)
        
        # 获取新创建的相机ID
        new_cam_id = new_agent.get('cam_id', -1)
        
        if new_cam_id >= 0:
            # 将虚拟agent缩放为极小，使其不可见
            try:
                unwrapped_env.unrealcv.set_obj_scale(virtual_name, [0.01, 0.01, 0.01])
            except:
                pass
            print(f"  ✓ 创建虚拟agent '{virtual_name}' 获取相机ID {new_cam_id}")
            return new_cam_id
        else:
            print(f"  ⚠️  虚拟agent创建失败：未获得相机ID")
            return None
    except Exception as e:
        print(f"  ⚠️  创建虚拟agent时出错: {e}")
        return None

def allocate_camera_ids(unwrapped_env, cameras, player_list, agents_by_type=None, target_agent_index=None, target_agent_type=None):
    """
    为相机配置分配相机ID
    只为第一个智能体（索引0）分配相机，使用该智能体绑定的相机ID
    不创建虚拟相机，只使用现有的相机ID
    
    Args:
        unwrapped_env: 未包装的环境对象
        cameras: 相机配置列表
        player_list: 智能体名称列表
        agents_by_type: 按类型分组的智能体字典
        target_agent_index: 目标智能体索引（如果指定，为该agent分配相机，否则为第一个agent）
        target_agent_type: 目标智能体类型（如果指定，与target_agent_index一起使用）
    
    Returns:
        dict: 相机名称到物理相机ID的映射字典
    """
    camera_id_map = {}
    
    # 确定目标agent索引（默认为第一个智能体，索引0）
    agent_index = target_agent_index if target_agent_index is not None else 0
    agent_type = target_agent_type or 'player'
    
    print(f"\n分配相机ID（只为智能体 {agent_index} ({agent_type}) 分配相机）:")
    print(f"  策略：收集所有智能体的相机ID，分配给第一个智能体的多个相机使用")
    print(f"  注意：其他智能体不使用相机，它们的相机ID被第一个智能体使用")
    
    # 收集所有智能体的相机ID
    all_cam_ids = []
    for i in range(len(player_list)):
        if hasattr(unwrapped_env, 'cam_list') and i < len(unwrapped_env.cam_list):
            cam_id = unwrapped_env.cam_list[i]
            if cam_id is not None:
                all_cam_ids.append(cam_id)
                print(f"  智能体 {i} 的相机ID: {cam_id} {'(第一个智能体使用)' if i == agent_index else '(分配给第一个智能体使用)'}")
    
    if len(all_cam_ids) == 0:
        print(f"  ⚠️  警告：无法获取任何智能体的相机ID")
        return camera_id_map
    
    print(f"  共收集到 {len(all_cam_ids)} 个相机ID，将分配给第一个智能体的 {len([c for c in cameras if c.params.get('agent_index', 0) == agent_index])} 个相机")
    
    # 为第一个智能体的相机分配不同的相机ID
    cam_id_index = 0
    for cam in cameras:
        # 只处理目标智能体的相机
        cam_agent_index = cam.params.get('agent_index', 0)
        if cam_agent_index != agent_index:
                    continue  # 跳过非目标agent的相机
            
        if cam.cam_type == 'first_person':
            # 第一人称：使用第一个智能体自己的相机ID
            if len(all_cam_ids) > 0:
                camera_id_map[cam.name] = all_cam_ids[0]
                print(f"  {cam.name}: 使用智能体 {agent_index} 的相机ID {all_cam_ids[0]} (第一人称)")
                cam_id_index = 1  # 从下一个开始
        elif cam.cam_type == 'third_person':
            # 第三人称：使用其他智能体的相机ID（循环使用）
            if cam_id_index < len(all_cam_ids):
                assigned_cam_id = all_cam_ids[cam_id_index]
                camera_id_map[cam.name] = assigned_cam_id
                original_agent = all_cam_ids.index(assigned_cam_id) if assigned_cam_id in all_cam_ids else '?'
                print(f"  {cam.name}: 使用相机ID {assigned_cam_id} (第三人称，来自智能体 {original_agent} 的相机ID)")
                cam_id_index += 1
            else:
                # 如果相机ID不够，循环使用（从第二个开始）
                assigned_cam_id = all_cam_ids[(cam_id_index % len(all_cam_ids)) or 1]  # 跳过第一个，从第二个开始循环
                camera_id_map[cam.name] = assigned_cam_id
                original_agent = all_cam_ids.index(assigned_cam_id) if assigned_cam_id in all_cam_ids else '?'
                print(f"  {cam.name}: 使用相机ID {assigned_cam_id} (第三人称，循环使用智能体 {original_agent} 的相机ID)")
                cam_id_index += 1
        # 跳过fixed和topview相机
    
    print(f"\n相机ID分配总结:")
    print(f"  为智能体 {agent_index} 的 {len(camera_id_map)} 个相机分配了不同的相机ID")
    print(f"  使用的相机ID来自所有智能体，避免视频覆盖")
    
    return camera_id_map

def setup_camera(unwrapped_env, camera_config, player_list, agent_poses=None, step=0, agents_by_type=None, camera_id_map=None):
    """
    设置相机并返回相机ID和图像
    
    Args:
        unwrapped_env: 未包装的环境对象
        camera_config: CameraConfig对象
        player_list: 智能体名称列表（所有智能体）
        agent_poses: 智能体位姿列表（可选）
        step: 当前步数
        agents_by_type: 按类型分组的智能体字典，格式：{'player': [...], 'animal': [...]}
        camera_id_map: 相机名称到物理相机ID的映射字典（可选）
    
    Returns:
        tuple: (cam_id, image) 或 (None, None) 如果失败
    """
    try:
        if camera_config.cam_type == 'first_person':
            # 第一人称：使用智能体绑定的相机
            agent_index = camera_config.params['agent_index']
            agent_type = camera_config.params.get('agent_type', 'player')
            
            # 如果指定了智能体类型，从对应类型的列表中获取
            if agents_by_type and agent_type in agents_by_type:
                type_agents = agents_by_type[agent_type]
                if agent_index < len(type_agents):
                    agent_name = type_agents[agent_index]
                    # 在player_list中找到对应的索引
                    if agent_name in player_list:
                        global_index = player_list.index(agent_name)
                        if hasattr(unwrapped_env, 'cam_list') and global_index < len(unwrapped_env.cam_list):
                            cam_id = unwrapped_env.cam_list[global_index]
                            img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
                            return cam_id, img
            # 否则使用全局索引
            elif agent_index < len(player_list):
                agent_name = player_list[agent_index]
                if hasattr(unwrapped_env, 'cam_list') and agent_index < len(unwrapped_env.cam_list):
                    cam_id = unwrapped_env.cam_list[agent_index]
                    # 获取第一人称图像
                    img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
                    return cam_id, img
            
        elif camera_config.cam_type == 'third_person':
            # 第三人称：相对于智能体的相机
            agent_index = camera_config.params['agent_index']
            agent_type = camera_config.params.get('agent_type', 'player')
            
            # 获取分配的相机ID
            cam_id = None
            if camera_id_map and camera_config.name in camera_id_map:
                cam_id = camera_id_map[camera_config.name]
            else:
                # 向后兼容：如果没有分配，使用智能体的默认相机ID
                if agents_by_type and agent_type in agents_by_type:
                    type_agents = agents_by_type[agent_type]
                    if agent_index < len(type_agents):
                        agent_name = type_agents[agent_index]
                        if agent_name in player_list:
                            global_index = player_list.index(agent_name)
                            if hasattr(unwrapped_env, 'cam_list') and global_index < len(unwrapped_env.cam_list):
                                cam_id = unwrapped_env.cam_list[global_index]
                elif agent_index < len(player_list):
                    if hasattr(unwrapped_env, 'cam_list') and agent_index < len(unwrapped_env.cam_list):
                        cam_id = unwrapped_env.cam_list[agent_index]
            
            if cam_id is None:
                return None, None
            
            # 获取智能体名称和位置
            agent_name = None
            global_agent_index = None  # 全局索引（对应player_list和agent_poses）
            agent_pose = None
            if agents_by_type and agent_type in agents_by_type:
                type_agents = agents_by_type[agent_type]
                if agent_index < len(type_agents):
                    agent_name = type_agents[agent_index]
                    # 转换为全局索引
                    if agent_name in player_list:
                        global_agent_index = player_list.index(agent_name)
            elif agent_index < len(player_list):
                agent_name = player_list[agent_index]
                global_agent_index = agent_index
            
            if agent_name is None or global_agent_index is None:
                return None, None
            
            # 获取智能体的位置和旋转（使用全局索引）
            if agent_poses and global_agent_index < len(agent_poses):
                agent_pose = agent_poses[global_agent_index]
            else:
                # 如果agent_poses不可用，尝试从环境获取
                try:
                    agent_loc = unwrapped_env.unrealcv.get_obj_location(agent_name)
                    agent_rot = unwrapped_env.unrealcv.get_obj_rotation(agent_name)
                    agent_pose = list(agent_loc) + list(agent_rot)
                except:
                    # 如果无法获取，使用set_cam方法（会使用智能体绑定的相机）
                    distance = camera_config.params['distance']
                    height = camera_config.params['height']
                    angle = camera_config.params['angle']
                    angle_rad = np.radians(angle)
                    x_offset = -distance * np.cos(angle_rad)
                    y_offset = distance * np.sin(angle_rad)
                    z_offset = height
                    pitch = -15
                    unwrapped_env.unrealcv.set_cam(agent_name, 
                                                  loc=[x_offset, y_offset, z_offset],
                                                  rot=[0, pitch, 0])
                    img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
                    return cam_id, img
            
            if agent_pose is None or len(agent_pose) < 6:
                return None, None
            
            # 获取智能体的位置和旋转
            agent_x, agent_y, agent_z = agent_pose[0], agent_pose[1], agent_pose[2]
            agent_roll, agent_pitch, agent_yaw = agent_pose[3], agent_pose[4], agent_pose[5]
            
            distance = camera_config.params['distance']
            height = camera_config.params['height']
            angle = camera_config.params['angle']
            
            # 计算相对于agent的相机位置（在智能体的局部坐标系中）
            # 在Unreal Engine中：
            # - angle=0°: 后方（agent的-X方向）
            # - angle=90°: 右侧（agent的+Y方向）
            # - angle=-90°: 左侧（agent的-Y方向）
            # - angle=180°: 前方（agent的+X方向）
            angle_rad = np.radians(angle)
            # 在agent的局部坐标系中计算偏移
            # 注意：在Unreal中，agent朝向X轴正方向时，Y轴正方向是右侧
            local_x = -distance * np.cos(angle_rad)  # X方向：负=后方，正=前方
            local_y = distance * np.sin(angle_rad)    # Y方向：正=右侧，负=左侧
            local_z = height
            
            # 将局部坐标转换为世界坐标（考虑智能体的旋转）
            # agent_yaw是agent的朝向角度（相对于X轴正方向）
            agent_yaw_rad = np.radians(agent_yaw)
            cos_yaw = np.cos(agent_yaw_rad)
            sin_yaw = np.sin(agent_yaw_rad)
            
            # 旋转局部偏移到世界坐标系
            # Unreal坐标系（左手系）：X前，Y右，Z上
            # Yaw角度：0=X, 90=Y。从上往下看，X转到Y是顺时针旋转
            # 而标准数学旋转矩阵 [cos -sin; sin cos] 是逆时针旋转
            # 所以我们需要使用顺时针旋转矩阵（或者将角度取反）
            # 顺时针旋转矩阵：
            # [cos  sin]
            # [-sin cos]
            world_x = agent_x + local_x * cos_yaw + local_y * sin_yaw
            world_y = agent_y - local_x * sin_yaw + local_y * cos_yaw
            world_z = agent_z + local_z
            
            # 计算相机朝向（朝向智能体）
            # 相机的yaw应该指向智能体
            # 方法1：计算从相机到agent的方向向量
            dx = agent_x - world_x
            dy = agent_y - world_y
            # 计算相对于X轴的角度（atan2返回的是相对于X轴的角度）
            # 在Unreal中，yaw=0是X轴正方向，所以atan2(dy, dx)应该直接给出yaw
            cam_yaw = np.degrees(np.arctan2(dy, dx))
            
            # 如果计算出的角度有问题，可以尝试使用agent的yaw + 180度（相机在agent后方时）
            # 但这里我们使用计算出的角度，因为相机可能在任意位置
            
            # 设置独立相机的世界坐标位置和旋转
            cam_loc = [world_x, world_y, world_z]
            # UnrealCV set_cam_rotation 参数顺序通常为 [Pitch, Yaw, Roll]
            # Pitch=-15 (向下看), Yaw=cam_yaw (朝向agent), Roll=0 (水平)
            cam_rot = [-15, cam_yaw, 0]
            
            # 每次step都更新相机位置和旋转（确保相机跟随agent移动）
            # 调试信息：在第一步时打印相机位置更新
            if step == 0:
                print(f"    [调试] 第三人称相机 {camera_config.name}: cam_id={cam_id}")
                print(f"      agent位置=({agent_x:.1f}, {agent_y:.1f}, {agent_z:.1f}), agent_yaw={agent_yaw:.1f}°")
                print(f"      局部坐标: local_x={local_x:.1f}, local_y={local_y:.1f}, local_z={local_z:.1f}")
                print(f"      世界坐标: world_x={world_x:.1f}, world_y={world_y:.1f}, world_z={world_z:.1f}")
                print(f"      角度参数: angle={angle}°, distance={distance}, height={height}")
                print(f"      相机朝向: cam_yaw={cam_yaw:.1f}° (Pitch=-15, Roll=0)")
            
            try:
                unwrapped_env.unrealcv.set_cam_location(cam_id, cam_loc)
                unwrapped_env.unrealcv.set_cam_rotation(cam_id, cam_rot)
            except Exception as e:
                if step == 0:
                    print(f"    ⚠️  [调试] 设置相机 {camera_config.name} (cam_id={cam_id}) 位置时出错: {e}")
                return None, None
            
            # 使用分配的相机ID获取图像
            try:
                img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
                return cam_id, img
            except Exception as e:
                if step == 0:
                    print(f"    ⚠️  [调试] 获取相机 {camera_config.name} (cam_id={cam_id}) 图像时出错: {e}")
                return None, None
        
        elif camera_config.cam_type == 'fixed':
            print("NONONO\n==================\n=================")
            # 固定位置相机：使用分配的相机ID
            location = camera_config.params['location']
            rotation = camera_config.params['rotation']
            
            # 获取分配的相机ID
            cam_id = None
            if camera_id_map and camera_config.name in camera_id_map:
                cam_id = camera_id_map[camera_config.name]
            else:
                # 向后兼容：如果没有分配，使用third_cam的相机ID
                if hasattr(unwrapped_env, 'cam_id') and len(unwrapped_env.cam_id) > 0:
                    cam_id = unwrapped_env.cam_id[0]
                elif hasattr(unwrapped_env, 'cam_list') and len(unwrapped_env.cam_list) > 0:
                    cam_id = unwrapped_env.cam_list[0]
            
            if cam_id is None:
                return None, None
            
            unwrapped_env.unrealcv.set_cam_location(cam_id, location)
            unwrapped_env.unrealcv.set_cam_rotation(cam_id, rotation)
            img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
            return cam_id, img
        elif camera_config.cam_type == 'topview':
            print("NONONO\n==================\n=================")
            # 俯视图：使用分配的相机ID
            location = camera_config.params['location']
            height = camera_config.params.get('height', 1500.0)
            
            # 如果有agent_poses，计算场景中心
            if agent_poses and len(agent_poses) > 0:
                center_x = np.mean([pose[0] for pose in agent_poses])
                center_y = np.mean([pose[1] for pose in agent_poses])
                center_z = np.mean([pose[2] for pose in agent_poses])
                location = [center_x, center_y, center_z]
            
            # 获取分配的相机ID
            cam_id = None
            if camera_id_map and camera_config.name in camera_id_map:
                cam_id = camera_id_map[camera_config.name]
            else:
                # 向后兼容：如果没有分配，使用third_cam的相机ID
                if hasattr(unwrapped_env, 'cam_id') and len(unwrapped_env.cam_id) > 0:
                    cam_id = unwrapped_env.cam_id[0]
                elif hasattr(unwrapped_env, 'cam_list') and len(unwrapped_env.cam_list) > 0:
                    cam_id = unwrapped_env.cam_list[0]
            
            if cam_id is None:
                return None, None
            
            # 设置俯视相机
            cam_loc = [location[0], location[1], location[2] + height]
            cam_rot = [-90, 0, 0]
            unwrapped_env.unrealcv.set_cam_location(cam_id, cam_loc)
            unwrapped_env.unrealcv.set_cam_rotation(cam_id, cam_rot)
            img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
            return cam_id, img
    
    except Exception as e:
        if step % 100 == 0:
            print(f"  警告：无法设置相机 {camera_config.name}: {e}")
        return None, None
    
    return None, None

def get_camera_trajectory(unwrapped_env, camera_config, cam_id, agent_poses=None, step=0):
    """获取相机轨迹信息"""
    try:
        if cam_id is not None:
            location = unwrapped_env.unrealcv.get_cam_location(cam_id)
            rotation = unwrapped_env.unrealcv.get_cam_rotation(cam_id)
            
            trajectory = {
                'step': step,
                'camera_name': camera_config.name,
                'camera_type': camera_config.cam_type,
                'location': location,
                'rotation': rotation
            }
            
            # 如果是第三人称相机，添加智能体信息
            if camera_config.cam_type == 'third_person' and agent_poses:
                agent_index = camera_config.params['agent_index']
                if agent_index < len(agent_poses):
                    agent_pose = agent_poses[agent_index]
                    trajectory['agent_location'] = agent_pose[:3]
                    trajectory['agent_rotation'] = agent_pose[3:6]
            
            return trajectory
    except Exception as e:
        if step % 100 == 0:
            print(f"  警告：无法获取相机 {camera_config.name} 的轨迹: {e}")
    
    return None

def check_agent_types_in_scene(env_id, requested_agents):
    """
    检查场景配置文件中是否包含请求的智能体类型
    
    Args:
        env_id: 环境ID，格式如 'UnrealTrack-MiddleEast-ContinuousColor-v0'
        requested_agents: 请求的智能体类型列表，如 ['player', 'animal']
    
    Returns:
        tuple: (is_valid, available_types, missing_types)
            - is_valid: 是否所有请求的类型都存在
            - available_types: 场景中可用的智能体类型列表
            - missing_types: 缺失的智能体类型列表
    """
    try:
        # 从env_id解析task和map名称
        # 格式: Unreal{task}-{MapName}-{ActionSpace}{ObservationType}-v{version}
        parts = env_id.split('-')
        if len(parts) < 2:
            return False, [], requested_agents
        
        task = parts[0].replace('Unreal', '')  # 去掉'Unreal'前缀
        map_name = parts[1]
        
        # 构建配置文件路径
        setting_file = os.path.join(task, f'{map_name}.json')
        
        # 加载配置文件
        setting = misc.load_env_setting(setting_file)
        
        # 获取场景中可用的智能体类型
        available_types = list(setting.get('agents', {}).keys())
        
        # 检查请求的类型是否都存在
        missing_types = [agent_type for agent_type in requested_agents if agent_type not in available_types]
        is_valid = len(missing_types) == 0
        
        return is_valid, available_types, missing_types
    except Exception as e:
        # 如果无法加载配置文件，返回警告但不阻止运行
        print(f"  警告：无法检查场景配置: {e}")
        return True, [], []  # 假设有效，让环境自己处理

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多相机录制脚本')
    parser.add_argument("-e", "--env_id", nargs='?', 
                        default='UnrealTrack-MiddleEast-ContinuousColor-v0',
                        help='环境ID（场景名称）')
    parser.add_argument("-o", "--output_dir", default='./recordings',
                        help='输出目录')
    parser.add_argument("-c", "--camera_config", type=str, default=None,
                        help='相机配置文件路径（JSON格式）。如果不提供，将使用默认配置')
    parser.add_argument("-s", "--save_images", action='store_true',
                        help='保存每一帧的图片（可选）')
    parser.add_argument("--num_agents", type=int, default=6,
                        help='智能体数量（仅在未提供相机配置时使用）。如果不指定，将根据任务类型自动设置（Track任务默认2，其他任务默认1）')
    parser.add_argument("--resolution", type=int, nargs=2, default=[640, 480],
                        help='视频分辨率 [width height]，默认640x480')
    parser.add_argument("--fps", type=float, default=30.0,
                        help='视频帧率，默认30.0')
    parser.add_argument("--video_codec", type=str, default='avc1',
                        choices=['avc1', 'H264', 'XVID', 'mp4v'],
                        help='视频编码格式：avc1/H264(H.264，推荐，兼容性好), XVID, mp4v。默认avc1')
    parser.add_argument("--agents", type=str, nargs='+', default=['player'],
                        help='智能体类型列表，可选: player, animal, drone, car, motorbike。例如: --agents player animal')
    parser.add_argument("--duration", type=float, default=10,
                        help='录制时长（秒），默认10秒')
    parser.add_argument("--max_steps", type=int, default=None,
                        help='最大步数（如果指定，将覆盖duration）')
    parser.add_argument("--action_mode", type=str, default='track',
                        choices=['random', 'forward', 'circle', 'idle', 'track'],
                        help='动作模式: random(随机), forward(向前), circle(圆形), idle(静止), track(追踪)，默认track')
    parser.add_argument("--save_trajectory", action='store_true',
                        help='保存相机轨迹信息（位置和旋转）到JSON文件')
    parser.add_argument("--sample_rate", type=int, default=1,
                        help='采样频率：每N步采样一次（默认1，即每步都采样）。设置为2表示每2步采样一次，以此类推')
    parser.add_argument("--time_dilation", type=float, default=0.1,
                        help='时间膨胀因子：控制模拟速度（默认5.0，即5倍速）。值越小越慢，1.0为正常速度，>1.0为加速。设置为-1禁用时间膨胀')
    parser.add_argument("--action_smooth", type=int, default=1,
                        help='动作平滑度：每N步更新一次动作（默认1，即每步更新）。增大此值可使动作更平滑连续，但会让视频看起来更慢')
    parser.add_argument("--target_agent_index", type=int, default=None,
                        help='目标智能体索引（如果指定，只为该agent分配多个相机，其他agent和topview相机不分配）。在追踪模式下，该agent将作为被追踪者，第一个agent(索引0)将作为追踪者。例如：--target_agent_index 2')
    parser.add_argument("--target_agent_type", type=str, default="player",
                        help='目标智能体类型（与--target_agent_index一起使用）。例如：--target_agent_type player')
    parser.add_argument("--use_mixed_actions", action='store_true',
                        help='如果指定，将尝试使用Mixed动作空间的环境（支持jump、crouch等动画动作）。如果不指定，将使用环境ID中指定的动作类型')
    parser.add_argument("--debug_track", action='store_true',
                        help='启用追踪调试输出，显示追踪者和被追踪者的位置、距离、角度等信息')
    parser.add_argument("--debug_track_interval", type=int, default=10,
                        help='追踪调试输出的间隔步数（默认10步输出一次）')
    parser.add_argument("--auto_switch_target", action='store_true',
                        help='启用自动切换目标：当追到一个目标后，自动切换到下一个目标')
    parser.add_argument("--catch_distance", type=float, default=100.0,
                        help='追到目标的距离阈值（默认100单位）')
    parser.add_argument("--catch_angle", type=float, default=45.0,
                        help='追到目标的角度阈值（默认45度）')
    use_mixed_actions = True
    args = parser.parse_args()
    
    # 如果指定了使用Mixed动作，修改环境ID
    if args.use_mixed_actions or use_mixed_actions:
        # 将环境ID中的动作类型替换为Mixed
        # 例如：UnrealTrack-MiddleEast-ContinuousColor-v0 -> UnrealTrack-MiddleEast-MixedColor-v0
        if 'Continuous' in args.env_id:
            args.env_id = args.env_id.replace('Continuous', 'Mixed')
            print(f"⚠️  已修改环境ID为Mixed动作类型: {args.env_id}")
        elif 'Discrete' in args.env_id:
            args.env_id = args.env_id.replace('Discrete', 'Mixed')
            print(f"⚠️  已修改环境ID为Mixed动作类型: {args.env_id}")
        else:
            print(f"⚠️  警告：环境ID {args.env_id} 中未找到Continuous或Discrete，无法自动替换为Mixed")
            print(f"   请手动指定Mixed动作类型的环境，例如：--env_id UnrealTrack-MiddleEast-MixedColor-v0")
    
    # 检查场景中是否包含请求的智能体类型
    is_valid, available_types, missing_types = check_agent_types_in_scene(args.env_id, args.agents)
    
    if not is_valid:
        print(f"\n❌ 错误：场景 '{args.env_id}' 中不包含以下智能体类型: {missing_types}")
        print(f"   场景中可用的智能体类型: {available_types}")
        print(f"\n   请使用以下命令之一：")
        if available_types:
            print(f"   # 使用场景中可用的类型")
            for agent_type in available_types:
                print(f"   python example/multi_camera_recorder.py --agents {agent_type} ...")
            if len(available_types) > 1:
                print(f"   # 或使用多种类型")
                print(f"   python example/multi_camera_recorder.py --agents {' '.join(available_types)} ...")
        else:
            print(f"   # 场景中没有配置任何智能体，请检查场景配置文件")
        print(f"\n   提示：场景配置文件位于 gym_unrealcv/envs/setting/ 目录下")
        exit(1)
    
    if available_types:
        print(f"✓ 场景 '{args.env_id}' 包含智能体类型: {available_types}")
    
    # 根据任务类型自动设置智能体数量
    # 从env_id中提取任务类型
    parts = args.env_id.split('-')
    task_type = None
    if len(parts) > 0:
        task_str = parts[0]
        if task_str.startswith('Unreal'):
            task_type = task_str.replace('Unreal', '')
    
    # 如果用户没有指定num_agents，根据任务类型设置默认值
    if args.num_agents is None:
        if task_type == 'Track':
            # Tracking任务需要至少2个智能体（一个追踪者，一个被追踪者）
            args.num_agents = 2
            print(f"检测到Track任务，自动设置智能体数量为: {args.num_agents}")
        else:
            # 其他任务默认1个智能体
            args.num_agents = 1
            print(f"自动设置智能体数量为: {args.num_agents}")
    else:
        # 如果用户指定了num_agents，检查Track任务是否至少2个
        if task_type == 'Track' and args.num_agents < 2:
            print(f"⚠️  警告：Track任务通常需要至少2个智能体（一个追踪者，一个被追踪者）")
            print(f"   当前设置: {args.num_agents}，建议至少设置为2")
            print(f"   自动调整为2个智能体")
            args.num_agents = 2
    
    # 生成包含环境ID、时间戳等信息的文件夹名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 从env_id中提取场景名称（简化处理）
    # 格式: Unreal{task}-{MapName}-{ActionSpace}{ObservationType}-v{version}
    # 注意：parts和task_type已经在上面解析过了
    if len(parts) >= 2:
        # 提取任务和地图名称
        task = task_type if task_type else (parts[0].replace('Unreal', '') if parts[0].startswith('Unreal') else parts[0])
        map_name = parts[1]
        env_name = f"{task}_{map_name}"
    else:
        # 如果格式不符合预期，使用简化处理
        env_name = args.env_id.replace('Unreal', '').replace('-', '_')
    
    # 限制长度，避免文件名过长
    if len(env_name) > 40:
        env_name = env_name[:40]
    
    # 构建文件夹名：env_name_agents_action_mode_timestamp
    agents_str = '_'.join(args.agents)
    folder_name = f"{env_name}_{agents_str}_{args.action_mode}_{timestamp}"
    
    # 创建完整的输出路径
    full_output_dir = os.path.join(args.output_dir, folder_name)
    os.makedirs(full_output_dir, exist_ok=True)
    
    print(f"\n输出文件夹: {full_output_dir}")
    
    # 加载相机配置
    if args.camera_config and os.path.exists(args.camera_config):
        print(f"从文件加载相机配置: {args.camera_config}")
        cameras = load_camera_configs(args.camera_config)
        # 如果指定了目标agent，过滤掉非目标agent的相机和topview相机
        if args.target_agent_index is not None:
            target_agent_type = args.target_agent_type or (args.agents[0] if args.agents else 'player')
            print(f"过滤相机配置：只为智能体 {args.target_agent_index} ({target_agent_type}) 保留相机")
            filtered_cameras = []
            for cam in cameras:
                # 保留目标agent的第一人称和第三人称相机
                if cam.cam_type in ['first_person', 'third_person']:
                    cam_agent_index = cam.params.get('agent_index')
                    cam_agent_type = cam.params.get('agent_type', 'player')
                    if cam_agent_index == args.target_agent_index and cam_agent_type == target_agent_type:
                        filtered_cameras.append(cam)
                # 跳过fixed和topview相机
                elif cam.cam_type in ['fixed', 'topview']:
                    continue
            cameras = filtered_cameras
            print(f"过滤后剩余 {len(cameras)} 个相机配置")
    else:
        print("使用默认相机配置")
        if args.target_agent_index is not None:
            target_agent_type = args.target_agent_type or (args.agents[0] if args.agents else 'player')
            print(f"只为智能体 {args.target_agent_index} ({target_agent_type}) 创建相机配置")
        cameras = create_default_camera_configs(
            args.num_agents, 
            target_agent_index=args.target_agent_index,
            target_agent_type=args.target_agent_type or (args.agents[0] if args.agents else 'player')
        )
    
    # 如果指定了目标agent，确保只显示目标agent的相机
    if args.target_agent_index is not None:
        print(f"\n✓ 只为目标智能体 {args.target_agent_index} 配置了 {len(cameras)} 个相机:")
    else:
        print(f"\n配置了 {len(cameras)} 个相机:")
    for cam in cameras:
        print(f"  - {cam.name} ({cam.cam_type})")
    
    # 创建环境
    print(f"\n创建环境: {args.env_id}")
    env = gym.make(args.env_id)
    
    # 移除TimeLimit包装器（如果存在），避免环境因步数限制提前结束
    # gym在注册环境时如果设置了max_episode_steps，会自动添加TimeLimit包装器
    def remove_time_limit(env):
        """递归移除TimeLimit包装器"""
        if isinstance(env, TimeLimit):
            max_steps = getattr(env, '_max_episode_steps', 'unknown')
            print(f"  检测到TimeLimit包装器（max_episode_steps={max_steps}），正在移除...")
            return env.env  # 返回内部环境，移除TimeLimit包装器
        elif hasattr(env, 'env'):
            # 递归检查内部环境
            env.env = remove_time_limit(env.env)
        return env
    
    # 检查并移除TimeLimit包装器
    original_env = env
    env = remove_time_limit(env)
    if env is not original_env:
        print(f"  ✓ 已移除TimeLimit包装器，环境将不会因步数限制提前结束")
    
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=tuple(args.resolution))
    
    # 设置智能体类型
    env.unwrapped.agents_category = args.agents
    print(f"智能体类型: {args.agents}")
    
    # 应用时间膨胀包装器（控制模拟速度）
    if args.time_dilation > 0:
        print(f"应用时间膨胀: {args.time_dilation}x")
        # 先设置全局时间膨胀（在应用包装器之前）
        unwrapped_env = env.unwrapped
        if hasattr(unwrapped_env, 'unrealcv'):
            unwrapped_env.unrealcv.set_global_time_dilation(args.time_dilation)
        # 使用TimeDilationWrapper来更好地控制时间膨胀
        # reference_fps参数控制目标FPS，这里我们通过时间膨胀来间接控制
        # 如果time_dilation=0.5，意味着模拟速度是正常的一半
        env = time_dilation.TimeDilationWrapper(env, reference_fps=30.0 * args.time_dilation, update_steps=10, update_dilation=True)
    elif args.time_dilation < 0:
        print("时间膨胀已禁用")
    
    # 应用包装器
    env = augmentation.RandomPopulationWrapper(env, args.num_agents, args.num_agents, random_target=False)
    
    # 计算最大步数
    if args.max_steps is not None:
        max_steps = args.max_steps
    else:
        max_steps = int(args.duration * args.fps)
    
    # 根据采样频率调整视频FPS
    # 如果采样频率是2，实际帧数减半，但视频时长不变，所以FPS也要减半
    actual_video_fps = args.fps / args.sample_rate
    
    print(f"录制设置:")
    print(f"  - 时长: {args.duration}秒 ({max_steps}步)")
    print(f"  - 动作模式: {args.action_mode}")
    print(f"  - 采样频率: 每 {args.sample_rate} 步采样一次")
    print(f"  - 时间膨胀: {args.time_dilation}x (模拟速度)" if args.time_dilation > 0 else "  - 时间膨胀: 已禁用")
    print(f"  - 动作平滑度: 每 {args.action_smooth} 步更新一次动作")
    if args.sample_rate > 1:
        actual_frames = max_steps // args.sample_rate
        print(f"  - 实际采样帧数: 约 {actual_frames} 帧")
        print(f"  - 视频FPS: {actual_video_fps:.2f} (已根据采样频率调整)")
    else:
        print(f"  - 视频FPS: {args.fps}")
    
    print(f"\n开始录制，输出目录: {full_output_dir}")
    
    # 重置环境（必须在创建agents_list之前，因为RandomPopulationWrapper在reset时可能改变agent数量）
    obs = env.reset()
    
    # 获取初始位置信息（从环境的obj_poses获取）
    initial_agent_poses = None
    if hasattr(env.unwrapped, 'obj_poses') and env.unwrapped.obj_poses is not None:
        initial_agent_poses = env.unwrapped.obj_poses
    
    # 确定追踪者和被追踪者
    tracker_id = None
    target_id = None
    
    # 优先级1：如果指定了target_agent_index，将其设为被追踪者，第一个agent(索引0)设为追踪者
    if args.target_agent_index is not None and args.action_mode == 'track':
        if args.target_agent_index < len(env.action_space):
            tracker_id = 0  # 第一个agent作为追踪者
            target_id = args.target_agent_index  # 被相机跟踪的agent作为被追踪者
            
            # 检查追踪者和被追踪者是否是同一个agent
            if tracker_id == target_id:
                print(f"  ⚠️  警告：--target_agent_index={args.target_agent_index} 与追踪者(索引0)相同")
                print(f"  ⚠️  追踪者和被追踪者不能是同一个agent，将使用默认追踪关系（追踪者=0，被追踪者=1）")
                # 使用默认逻辑：第一个追踪第二个
                if len(env.action_space) >= 2:
                    tracker_id = 0
                    target_id = 1
                    print(f"  ✓ 已自动调整为：追踪者=智能体0，被追踪者=智能体1")
                else:
                    print(f"  ⚠️  智能体数量不足，将使用随机模式")
                    args.action_mode = 'random'
                    tracker_id = None
                    target_id = None
            else:
                print(f"自动设置追踪关系：")
                print(f"  - 追踪者：智能体 0（第一个agent）")
                print(f"  - 被追踪者：智能体 {target_id}（被相机跟踪的agent，--target_agent_index={args.target_agent_index}）")
    # 优先级2：如果是Track环境，使用环境提供的tracker_id和target_id
    elif hasattr(env.unwrapped, 'tracker_id') and hasattr(env.unwrapped, 'target_id'):
        tracker_id = env.unwrapped.tracker_id
        target_id = env.unwrapped.target_id
        if tracker_id is not None and target_id is not None:
            print(f"检测到Track环境：追踪者ID={tracker_id}, 被追踪者ID={target_id}")
    # 优先级3：默认情况，第一个agent追踪第二个agent
    else:
        if len(env.action_space) >= 2:
            tracker_id = 0
            target_id = 1
            print(f"默认设置：第一个agent(索引0)追踪第二个agent(索引1)")
        else:
            print(f"警告：智能体数量不足，无法使用追踪模式，将使用随机模式")
            if args.action_mode == 'track':
                args.action_mode = 'random'
                print(f"已自动切换为随机模式")
    
    # 创建智能体（必须在reset之后，确保action_space长度与实际agent数量匹配）
    agents_list = []
    for i in range(len(env.action_space)):
        is_tracker = (args.action_mode == 'track' and i == tracker_id)
        agents_list.append(ActionAgent(env.action_space[i], args.action_mode, agent_index=i, is_tracker=is_tracker))
    
    if args.action_mode == 'track' and tracker_id is not None and target_id is not None:
        print(f"创建了 {len(agents_list)} 个动作智能体：")
        print(f"  - 智能体 {tracker_id}: 追踪者（使用PoseTracker）")
        print(f"  - 智能体 {target_id}: 被追踪者（使用随机动作）")
        if len(agents_list) > 2:
            print(f"  - 其他智能体: 使用随机动作")
    else:
        print(f"创建了 {len(agents_list)} 个动作智能体（与env.action_space长度匹配）")
    
    # 获取环境对象和智能体列表
    unwrapped_env = env.unwrapped
    player_list = unwrapped_env.player_list if hasattr(unwrapped_env, 'player_list') else []
    
    # 按类型分组智能体
    agents_by_type = {}
    if hasattr(unwrapped_env, 'agents'):
        for agent_name in player_list:
            if agent_name in unwrapped_env.agents:
                agent_type = unwrapped_env.agents[agent_name].get('agent_type', 'player')
                if agent_type not in agents_by_type:
                    agents_by_type[agent_type] = []
                agents_by_type[agent_type].append(agent_name)
    
    print(f"智能体分组: {[(k, len(v)) for k, v in agents_by_type.items()]}")
    
    # 为相机配置分配相机ID
    # 默认只为第一个智能体（索引0）分配相机
    target_agent_index_for_allocation = args.target_agent_index if args.target_agent_index is not None else 0
    target_agent_type_for_allocation = args.target_agent_type or (args.agents[0] if args.agents else 'player')
    
    print(f"只为智能体 {target_agent_index_for_allocation} ({target_agent_type_for_allocation}) 分配相机")
    
    camera_id_map = allocate_camera_ids(
        unwrapped_env, 
        cameras, 
        player_list, 
        agents_by_type,
        target_agent_index=target_agent_index_for_allocation,
        target_agent_type=target_agent_type_for_allocation
    )
    
    # 初始化视频写入器（只为已分配相机的相机配置创建）
    video_writers = {}
    allocated_cameras = [cam for cam in cameras if cam.name in camera_id_map]
    
    # 检查相机ID是否有重复（这是导致覆盖问题的关键）
    cam_id_to_names = {}
    for cam_name, cam_id in camera_id_map.items():
        if cam_id not in cam_id_to_names:
            cam_id_to_names[cam_id] = []
        cam_id_to_names[cam_id].append(cam_name)
    
    # 检查是否有重复的相机ID
    duplicate_cam_ids = {cam_id: names for cam_id, names in cam_id_to_names.items() if len(names) > 1}
    if duplicate_cam_ids:
        print(f"\n⚠️  警告：发现重复的相机ID分配（这会导致视频覆盖问题）:")
        for cam_id, names in duplicate_cam_ids.items():
            print(f"  相机ID {cam_id} 被多个相机共享: {names}")
        print(f"  这会导致后面的相机设置覆盖前面的相机设置，导致视频不动！")
    else:
        print(f"\n✓ 所有相机都分配了唯一的相机ID")
    
    # 验证相机分配（默认只为第一个智能体）
    print(f"\n验证相机分配（目标agent: {target_agent_index_for_allocation}, 类型: {target_agent_type_for_allocation}）:")
    all_correct = True
    for cam in allocated_cameras:
        if cam.cam_type in ['first_person', 'third_person']:
            cam_agent_index = cam.params.get('agent_index')
            cam_agent_type = cam.params.get('agent_type', 'player')
        if cam_agent_index == target_agent_index_for_allocation and cam_agent_type == target_agent_type_for_allocation:
                cam_id = camera_id_map.get(cam.name, 'N/A')
                if cam.cam_type == 'third_person':
                    distance = cam.params.get('distance', 0)
                    height = cam.params.get('height', 0)
                    angle = cam.params.get('angle', 0)
                    print(f"  ✓ {cam.name}: agent={cam_agent_index}({cam_agent_type}), cam_id={cam_id}, 相对坐标(distance={distance}, height={height}, angle={angle}°)")
                else:
                    print(f"  ✓ {cam.name}: agent={cam_agent_index}({cam_agent_type}), cam_id={cam_id}, 第一人称")
        else:
            print(f"  ⚠️  警告：相机 {cam.name} 不属于目标agent (agent_index={cam_agent_index}, agent_type={cam_agent_type})")
            all_correct = False
    if all_correct:
        print(f"  ✓ 所有相机都正确分配给目标agent")
    else:
        print(f"  ⚠️  部分相机分配不正确")
    
    print(f"\n✓ 为智能体 {target_agent_index_for_allocation} ({target_agent_type_for_allocation}) 创建 {len(allocated_cameras)} 个视频文件:")
    for cam in allocated_cameras:
        video_path = os.path.join(full_output_dir, f'{cam.name}.mp4')
        # 使用用户指定的编码格式，默认使用H.264（avc1）以获得更好的兼容性
        fourcc = cv2.VideoWriter_fourcc(*args.video_codec)
        # 使用调整后的FPS
        video_writer = cv2.VideoWriter(video_path, fourcc, actual_video_fps, tuple(args.resolution))
        
        # 检查VideoWriter是否成功创建
        if not video_writer.isOpened():
            print(f"  ⚠️  警告：无法使用编码格式 '{args.video_codec}' 创建视频写入器")
            print(f"     尝试使用备用编码格式...")
            # 尝试备用编码格式
            backup_codecs = ['H264', 'XVID', 'mp4v'] if args.video_codec != 'H264' else ['XVID', 'mp4v']
            for backup_codec in backup_codecs:
                try:
                    backup_fourcc = cv2.VideoWriter_fourcc(*backup_codec)
                    backup_writer = cv2.VideoWriter(video_path, backup_fourcc, actual_video_fps, tuple(args.resolution))
                    if backup_writer.isOpened():
                        video_writer = backup_writer
                        print(f"  ✓ 使用备用编码格式 '{backup_codec}' 成功创建视频写入器")
                        break
                except:
                    continue
            if not video_writer.isOpened():
                print(f"  ❌ 错误：无法创建视频写入器，跳过相机 {cam.name}")
                continue
        video_writers[cam.name] = video_writer
        print(f"  ✓ {cam.name}: {video_path} (FPS: {actual_video_fps:.2f})")
    
    # 如果需要保存图片，只为已分配相机的相机配置创建子目录
    if args.save_images:
        for cam in allocated_cameras:
            os.makedirs(os.path.join(full_output_dir, cam.name), exist_ok=True)
    
    # 初始化轨迹记录（只为已分配相机的相机配置）
    camera_trajectories = {cam.name: [] for cam in allocated_cameras}
    agent_trajectories = []  # 记录智能体轨迹
    
    # 开始录制
    start_time = time.time()
    step = 0
    current_actions = None  # 存储当前动作，用于动作平滑
    agent_poses = initial_agent_poses  # 使用初始位置信息
    caught_targets = set()  # 记录已追到的目标（用于自动切换目标）
    target_switch_count = 0  # 目标切换次数统计
    
    try:
        while step < max_steps:
            # 检查agents_list长度是否与env.action_space匹配
            if len(agents_list) != len(env.action_space):
                print(f"  ⚠️  警告：agents_list长度({len(agents_list)})与env.action_space长度({len(env.action_space)})不匹配，重新创建agents_list")
                # 重新确定追踪者和被追踪者
                if args.action_mode == 'track':
                    if args.target_agent_index is not None and args.target_agent_index < len(env.action_space):
                        tracker_id = 0
                        target_id = args.target_agent_index
                    elif hasattr(env.unwrapped, 'tracker_id') and hasattr(env.unwrapped, 'target_id'):
                        tracker_id = env.unwrapped.tracker_id
                        target_id = env.unwrapped.target_id
                    elif len(env.action_space) >= 2:
                        tracker_id = 0
                        target_id = 1
                    else:
                        tracker_id = None
                        target_id = None
                
                # 重新创建agents_list
                agents_list = []
                for i in range(len(env.action_space)):
                    is_tracker = (args.action_mode == 'track' and i == tracker_id)
                    agents_list.append(ActionAgent(env.action_space[i], args.action_mode, agent_index=i, is_tracker=is_tracker))
                print(f"  ✓ 已重新创建 {len(agents_list)} 个动作智能体")
            
            # 根据动作平滑度决定是否更新动作
            # 如果action_smooth > 1，动作会保持多个step，使动作更平滑连续
            if step % args.action_smooth == 0 or current_actions is None:
                # 更新动作
                # 如果是追踪模式，传递位置信息
                if args.action_mode == 'track' and agent_poses and tracker_id is not None and target_id is not None:
                    current_actions = []
                    for i in range(len(agents_list)):
                        action = agents_list[i].act(agent_poses=agent_poses, target_index=target_id)
                        if action is None:
                            # 如果返回None，使用随机动作作为后备
                            action = agents_list[i].action_space.sample()
                        current_actions.append(action)
                else:
                    current_actions = []
                    for i in range(len(agents_list)):
                        action = agents_list[i].act()
                        if action is None:
                            # 如果返回None，使用随机动作作为后备
                            action = agents_list[i].action_space.sample()
                        current_actions.append(action)
                
                # 确保current_actions长度与env.action_space匹配
                if len(current_actions) != len(env.action_space):
                    print(f"  ⚠️  警告：生成的current_actions长度({len(current_actions)})与env.action_space长度({len(env.action_space)})不匹配")
                    # 如果长度不匹配，补齐或截断
                    if len(current_actions) < len(env.action_space):
                        # 补齐：使用随机动作
                        for i in range(len(current_actions), len(env.action_space)):
                            current_actions.append(env.action_space[i].sample())
                    else:
                        # 截断：只保留前N个
                        current_actions = current_actions[:len(env.action_space)]
            
            # 执行动作（使用当前动作，可能是新动作也可能是保持的动作）
            # 再次检查长度，确保安全
            if len(current_actions) != len(env.action_space):
                print(f"  ❌ 错误：current_actions长度({len(current_actions)})与env.action_space长度({len(env.action_space)})不匹配，无法执行step")
                print(f"  跳过此步，使用随机动作")
                current_actions = [env.action_space[i].sample() for i in range(len(env.action_space))]
            
            obs, _, done, info = env.step(current_actions)
            
            # 获取智能体位置信息（优先从info获取，如果没有则从环境的obj_poses获取）
            agent_poses = info.get('Pose', [])
            if not agent_poses and hasattr(env.unwrapped, 'obj_poses') and env.unwrapped.obj_poses is not None:
                agent_poses = env.unwrapped.obj_poses
            
            # 追踪模式：检测是否追到目标并自动切换（无论是否启用调试）
            if args.action_mode == 'track' and args.auto_switch_target and tracker_id is not None and target_id is not None and agent_poses:
                if tracker_id < len(agents_list) and hasattr(agents_list[tracker_id], 'tracking_info') and agents_list[tracker_id].tracking_info is not None:
                    tracking_info = agents_list[tracker_id].tracking_info
                    distance = tracking_info['distance']
                    direction_angle = tracking_info['direction_angle']
                    
                    # 检测是否追到目标（距离<阈值且角度<阈值）
                    is_caught = (distance < args.catch_distance) and (abs(direction_angle) < args.catch_angle)
                    
                    # 自动切换目标逻辑
                    if is_caught and target_id not in caught_targets:
                        caught_targets.add(target_id)
                        target_switch_count += 1
                        print(f"\n🎯 [Step {step}] 追到目标 {target_id}！")
                        print(f"  距离: {distance:.1f} < {args.catch_distance}, 角度: {abs(direction_angle):.1f}° < {args.catch_angle}°")
                        
                        # 寻找下一个未追到的目标
                        next_target_id = None
                        for i in range(len(env.action_space)):
                            if i != tracker_id and i not in caught_targets:
                                next_target_id = i
                                break
                        
                        if next_target_id is not None:
                            old_target_id = target_id
                            target_id = next_target_id
                            print(f"  ✓ 自动切换到下一个目标: Agent {old_target_id} -> Agent {target_id}")
                            print(f"  已追到的目标: {sorted(caught_targets)}")
                        else:
                            print(f"  ⚠️  所有目标都已追到！共追到 {len(caught_targets)} 个目标")
                            # 可以选择重置或继续
                            if len(caught_targets) >= len(env.action_space) - 1:  # 除了追踪者自己
                                print(f"  🎉 所有目标都已追到，重置已追到列表，重新开始")
                                caught_targets.clear()
                                # 选择下一个目标（排除追踪者）
                                for i in range(len(env.action_space)):
                                    if i != tracker_id:
                                        target_id = i
                                        break
            
            # 追踪模式调试输出（如果启用）
            if args.debug_track and args.action_mode == 'track' and tracker_id is not None and target_id is not None and agent_poses:
                if tracker_id < len(agents_list) and hasattr(agents_list[tracker_id], 'tracking_info') and agents_list[tracker_id].tracking_info is not None:
                    tracking_info = agents_list[tracker_id].tracking_info
                    distance = tracking_info['distance']
                    direction_angle = tracking_info['direction_angle']
                    distance_diff = tracking_info['distance_diff']
                    action_angle = tracking_info['action_angle']
                    action_velocity = tracking_info['action_velocity']
                    tracker_pos = tracking_info['tracker_pos']
                    target_pos = tracking_info['target_pos']
                    
                    # 检测是否追到目标（用于调试输出）
                    is_caught = (distance < args.catch_distance) and (abs(direction_angle) < args.catch_angle)
                    
                    # 按指定间隔输出，或者追到目标时输出
                    if step % args.debug_track_interval == 0 or is_caught:
                        print(f"[追踪调试] Step {step}:")
                        print(f"  追踪者(Agent {tracker_id})位置: ({tracker_pos[0]:.1f}, {tracker_pos[1]:.1f}, {tracker_pos[2]:.1f})")
                        print(f"  被追踪者(Agent {target_id})位置: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
                        print(f"  距离: {distance:.1f} 单位 (期望: {tracking_info['expected_distance']:.1f}, 差值: {distance_diff:+.1f})")
                        print(f"  方向角: {direction_angle:+.1f}° (期望: 0°, 偏差: {abs(direction_angle):.1f}°)")
                        print(f"  计算动作: 角度={action_angle:+.2f}, 速度={action_velocity:+.2f}")
                        if is_caught:
                            if target_id in caught_targets:
                                print(f"  ✓ 已追到目标 {target_id} (之前已追到过)")
                            else:
                                print(f"  ⚠️  碰撞检测: 距离<{args.catch_distance}且角度<{args.catch_angle}° (已追到!)")
                        else:
                            # 判断距离状态
                            if distance_diff > 50:
                                print(f"  📍 状态: 距离过远，需要加速接近")
                            elif distance_diff < -50:
                                print(f"  📍 状态: 距离过近，需要减速远离")
                            else:
                                print(f"  📍 状态: 距离合适")
                        print()
            
            # 根据采样频率决定是否采样图像
            # 每个step都执行动作以保持动作连续性，但只在满足采样频率时才采样图像
            should_sample = (step % args.sample_rate == 0)
            
            # 记录智能体轨迹（只在采样时记录，或每步都记录取决于需求）
            if args.save_trajectory and agent_poses:
                if should_sample:  # 只在采样时记录轨迹
                    agent_traj_frame = {
                        'step': step,
                        'timestamp': time.time() - start_time,
                        'agents': []
                    }
                    for i, pose in enumerate(agent_poses):
                        if len(pose) >= 6:
                            agent_traj_frame['agents'].append({
                                'agent_index': i,
                                'agent_name': player_list[i] if i < len(player_list) else f'agent_{i}',
                                'location': [float(pose[0]), float(pose[1]), float(pose[2])],
                                'rotation': [float(pose[3]), float(pose[4]), float(pose[5])]
                            })
                    agent_trajectories.append(agent_traj_frame)
            
            # step()之后，相机可能被重置，需要重新设置所有相机
            # 重要：相机位置必须在每个step都更新，以确保相机跟随agent移动
            # 但图像只在满足采样频率时获取（节省计算资源）
            
            # 为所有第三人称相机更新位置（每个step都需要，确保跟随agent）
            for cam_config in allocated_cameras:
                if cam_config.cam_type == 'third_person':
                    # 获取分配的相机ID
                    assigned_cam_id = camera_id_map.get(cam_config.name, None)
                    if assigned_cam_id is None:
                        continue
                    
                    # 更新相机位置（不获取图像，只更新位置）
                    agent_index = cam_config.params['agent_index']
                    agent_type = cam_config.params.get('agent_type', 'player')
                    
                    # 获取agent的全局索引
                    global_agent_index = None
                    if agents_by_type and agent_type in agents_by_type:
                        type_agents = agents_by_type[agent_type]
                        if agent_index < len(type_agents):
                            agent_name = type_agents[agent_index]
                            if agent_name in player_list:
                                global_agent_index = player_list.index(agent_name)
                    elif agent_index < len(player_list):
                        global_agent_index = agent_index
                    
                    if global_agent_index is not None and agent_poses and global_agent_index < len(agent_poses):
                        agent_pose = agent_poses[global_agent_index]
                        if agent_pose and len(agent_pose) >= 6:
                            agent_x, agent_y, agent_z = agent_pose[0], agent_pose[1], agent_pose[2]
                            agent_yaw = agent_pose[5]
                            
                            distance = cam_config.params['distance']
                            height = cam_config.params['height']
                            angle = cam_config.params['angle']
                            
                            # 计算相对于agent的相机位置（在智能体的局部坐标系中）
                            # 在Unreal Engine中：
                            # - angle=0°: 后方（agent的-X方向）
                            # - angle=90°: 右侧（agent的+Y方向）
                            # - angle=-90°: 左侧（agent的-Y方向）
                            # - angle=180°: 前方（agent的+X方向）
                            angle_rad = np.radians(angle)
                            local_x = -distance * np.cos(angle_rad)  # X方向：负=后方，正=前方
                            local_y = distance * np.sin(angle_rad)   # Y方向：正=右侧，负=左侧
                            local_z = height
                            
                            # 将局部坐标转换为世界坐标（考虑智能体的旋转）
                            agent_yaw_rad = np.radians(agent_yaw)
                            cos_yaw = np.cos(agent_yaw_rad)
                            sin_yaw = np.sin(agent_yaw_rad)
                            
                            # 旋转局部偏移到世界坐标系
                            # 使用顺时针旋转矩阵适应Unreal坐标系
                            world_x = agent_x + local_x * cos_yaw + local_y * sin_yaw
                            world_y = agent_y - local_x * sin_yaw + local_y * cos_yaw
                            world_z = agent_z + local_z
                            
                            # 计算相机朝向（朝向智能体）
                            # 相机的yaw应该指向智能体
                            dx = agent_x - world_x
                            dy = agent_y - world_y
                            cam_yaw = np.degrees(np.arctan2(dy, dx))
                            
                            # 更新相机位置和旋转
                            cam_loc = [world_x, world_y, world_z]
                            # UnrealCV set_cam_rotation 参数顺序通常为 [Pitch, Yaw, Roll]
                            cam_rot = [-15, cam_yaw, 0]
                            
                            try:
                                unwrapped_env.unrealcv.set_cam_location(assigned_cam_id, cam_loc)
                                unwrapped_env.unrealcv.set_cam_rotation(assigned_cam_id, cam_rot)
                            except Exception as e:
                                if step == 0:
                                    print(f"  ⚠️  更新相机 {cam_config.name} (cam_id={assigned_cam_id}) 位置时出错: {e}")
            
            # 只在满足采样频率时获取图像
            if should_sample:
                # 只为已分配相机的相机配置获取图像
                for cam_config in allocated_cameras:
                    # 获取分配的相机ID（用于调试）
                    assigned_cam_id = camera_id_map.get(cam_config.name, None)
                    if assigned_cam_id is None:
                        # 如果相机没有分配ID，跳过
                        if step == 0:
                            print(f"  ⚠️  警告：相机 {cam_config.name} 没有分配相机ID，跳过")
                        continue
                    
                    # 调试信息：在第一步时打印相机ID分配情况
                    if step == 0 and cam_config.cam_type == 'third_person':
                        print(f"  [调试] 设置第三人称相机 {cam_config.name}: cam_id={assigned_cam_id}, agent_index={cam_config.params.get('agent_index')}, agent_type={cam_config.params.get('agent_type')}")
                    
                    cam_id, img = setup_camera(unwrapped_env, cam_config, player_list, agent_poses, step, agents_by_type, camera_id_map)
                    
                    # 调试信息：检查返回的相机ID是否匹配
                    if step == 0 and cam_id != assigned_cam_id:
                        print(f"  ⚠️  [调试] 相机 {cam_config.name} 返回的cam_id={cam_id} 与分配的cam_id={assigned_cam_id} 不匹配！")
                    
                    # 调试信息：检查相机ID是否匹配
                    if step == 0:
                        if cam_id != assigned_cam_id:
                            print(f"  ⚠️  警告：相机 {cam_config.name} 分配的ID是 {assigned_cam_id}，但实际使用的是 {cam_id}")
                        else:
                            if cam_config.cam_type == 'third_person':
                                print(f"  ✓ 第三人称相机 {cam_config.name} 使用相机ID {cam_id}")
                            elif cam_config.cam_type == 'first_person':
                                print(f"  ✓ 第一人称相机 {cam_config.name} 使用相机ID {cam_id}")
                    
                    if img is not None:
                        # 转换为BGR格式
                        # img_bgr = cv2.cvtColor(img,cv2.RGB) if len(img.shape) == 3 else img
                        img_bgr = img
                        # 记录相机轨迹
                        if args.save_trajectory:
                            trajectory = get_camera_trajectory(unwrapped_env, cam_config, cam_id, agent_poses, step)
                            if trajectory:
                                trajectory['timestamp'] = time.time() - start_time
                                camera_trajectories[cam_config.name].append(trajectory)
                        
                        # 保存图片
                        if args.save_images:
                            img_path = os.path.join(full_output_dir, cam_config.name, 
                                                  f'{cam_config.name}_step_{step:05d}.png')
                            cv2.imwrite(img_path, img_bgr)
                        
                        # 写入视频
                        if cam_config.name in video_writers:
                            # 调整图像大小以匹配视频分辨率
                            if img_bgr.shape[:2] != tuple(reversed(args.resolution)):
                                img_bgr = cv2.resize(img_bgr, tuple(args.resolution))
                            video_writers[cam_config.name].write(img_bgr)
            
            step += 1
            
            # 打印进度
            if step % 50 == 0:
                elapsed = time.time() - start_time
                remaining_steps = max_steps - step
                # 考虑采样频率，计算实际剩余时间
                actual_remaining_steps = remaining_steps // args.sample_rate if args.sample_rate > 1 else remaining_steps
                estimated_remaining = actual_remaining_steps / actual_video_fps if actual_video_fps > 0 else 0
                sampled_frames = step // args.sample_rate if args.sample_rate > 1 else step
                total_frames = max_steps // args.sample_rate if args.sample_rate > 1 else max_steps
                print(f"  进度: {step}/{max_steps} 步 ({step/max_steps*100:.1f}%), 已采样: {sampled_frames}/{total_frames} 帧, 已用时: {elapsed:.1f}秒, 预计剩余: {estimated_remaining:.1f}秒")
            
            # 如果环境提前结束，可以选择重置或退出
            if done and step < max_steps:
                print(f"  环境提前结束，重置环境继续录制...")
                obs = env.reset()
                
                # 获取重置后的位置信息
                agent_poses = None
                if hasattr(env.unwrapped, 'obj_poses') and env.unwrapped.obj_poses is not None:
                    agent_poses = env.unwrapped.obj_poses
                
                # 重新确定追踪者和被追踪者（优先级与初始化时相同）
                # 优先级1：如果指定了target_agent_index，将其设为被追踪者
                if args.target_agent_index is not None and args.action_mode == 'track':
                    if args.target_agent_index < len(env.action_space):
                        tracker_id = 0  # 第一个agent作为追踪者
                        target_id = args.target_agent_index  # 被相机跟踪的agent作为被追踪者
                # 优先级2：如果是Track环境，使用环境提供的tracker_id和target_id
                elif hasattr(env.unwrapped, 'tracker_id') and hasattr(env.unwrapped, 'target_id'):
                    tracker_id = env.unwrapped.tracker_id
                    target_id = env.unwrapped.target_id
                # 优先级3：默认情况，第一个agent追踪第二个agent
                elif len(env.action_space) >= 2:
                    tracker_id = 0
                    target_id = 1
                else:
                    tracker_id = None
                    target_id = None
                
                # 检查并重新创建agents_list（如果action_space长度发生变化）
                if len(agents_list) != len(env.action_space):
                    print(f"  警告：环境重置后agent数量发生变化 ({len(agents_list)} -> {len(env.action_space)})，重新创建agents_list")
                    agents_list = []
                    for i in range(len(env.action_space)):
                        is_tracker = (args.action_mode == 'track' and i == tracker_id)
                        agents_list.append(ActionAgent(env.action_space[i], args.action_mode, agent_index=i, is_tracker=is_tracker))
                else:
                    # 更新每个agent的is_tracker标志
                    for i, agent in enumerate(agents_list):
                        agent.is_tracker = (args.action_mode == 'track' and i == tracker_id)
                        if agent.is_tracker and agent.pose_tracker is None:
                            try:
                                agent.pose_tracker = PoseTracker(agent.action_space, expected_distance=250, expected_angle=0)
                            except:
                                pass
                        agent.reset()
                
                # 重置current_actions，强制重新生成
                current_actions = None
        
        # 释放视频写入器
        for writer in video_writers.values():
            writer.release()
    
    except KeyboardInterrupt:
        print(f"\n\n⚠️  录制被用户中断 (Ctrl+C)")
        print(f"  - 已录制步数: {step}")
        elapsed_time = time.time() - start_time
        print(f"  - 已录制时长: {elapsed_time:.2f}秒")
    except Exception as e:
        print(f"\n\n❌ 录制过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        elapsed_time = time.time() - start_time
        print(f"  - 已录制步数: {step}")
        print(f"  - 已录制时长: {elapsed_time:.2f}秒")
    finally:
        # 确保资源被正确释放
        print(f"\n正在清理资源...")
        
        # 关闭视频写入器
        for cam_name, writer in video_writers.items():
            try:
                writer.release()
            except:
                pass
        
        # 安全关闭环境
        try:
            if hasattr(env, 'close'):
                # 检查是否在主线程中，避免在接收线程中调用 disconnect
                import threading
                if threading.current_thread() is threading.main_thread():
                    env.close()
                else:
                    # 如果在其他线程中，只关闭二进制文件，不调用 disconnect
                    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'ue_binary'):
                        try:
                            env.unwrapped.ue_binary.close()
                        except:
                            pass
        except Exception as e:
            print(f"  警告: 关闭环境时出错: {e}")
            # 忽略线程相关的错误，这些是预期的
    
    elapsed_time = time.time() - start_time
    print(f"\n录制完成！")
    if args.target_agent_index is not None:
        print(f"  ✓ 只为目标智能体 {args.target_agent_index} ({target_agent_type_for_allocation}) 创建了 {len(allocated_cameras)} 个视频文件")
    print(f"  - 总步数: {step}")
    print(f"  - 实际时长: {elapsed_time:.2f}秒")
    print(f"  - 数据已保存到: {full_output_dir}")
    print(f"  - 视频文件列表:")
    for cam in allocated_cameras:
        print(f"    ✓ {cam.name}.mp4")
    
    # 保存相机轨迹
    if args.save_trajectory:
        trajectory_path = os.path.join(full_output_dir, 'camera_trajectories.json')
        # 将numpy数组转换为列表
        trajectory_data = {}
        for cam_name, trajectory in camera_trajectories.items():
            trajectory_data[cam_name] = []
            for frame in trajectory:
                frame_data = {'step': frame['step']}
                
                # 转换location和rotation
                if isinstance(frame['location'], (list, np.ndarray)):
                    frame_data['location'] = [float(x) for x in frame['location']]
                else:
                    frame_data['location'] = frame['location']
                
                if isinstance(frame['rotation'], (list, np.ndarray)):
                    frame_data['rotation'] = [float(x) for x in frame['rotation']]
                else:
                    frame_data['rotation'] = frame['rotation']
                
                # 添加其他信息
                frame_data['camera_name'] = frame.get('camera_name', cam_name)
                frame_data['camera_type'] = frame.get('camera_type', 'unknown')
                frame_data['timestamp'] = frame.get('timestamp', 0.0)
                
                if 'agent_location' in frame:
                    frame_data['agent_location'] = [float(x) for x in frame['agent_location']]
                if 'agent_rotation' in frame:
                    frame_data['agent_rotation'] = [float(x) for x in frame['agent_rotation']]
                
                trajectory_data[cam_name].append(frame_data)
        
        with open(trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        print(f"  相机轨迹已保存到: {trajectory_path}")
    
    # 保存智能体轨迹
    if args.save_trajectory and agent_trajectories:
        agent_trajectory_path = os.path.join(full_output_dir, 'agent_trajectories.json')
        with open(agent_trajectory_path, 'w', encoding='utf-8') as f:
            json.dump(agent_trajectories, f, indent=2, ensure_ascii=False)
        print(f"  智能体轨迹已保存到: {agent_trajectory_path}")
    
    # 保存录制信息到输出目录（供参考）
    info_path = os.path.join(full_output_dir, 'recording_info.json')
    # 计算实际采样帧数
    actual_sampled_frames = step // args.sample_rate if args.sample_rate > 1 else step
    
    recording_info = {
        'env_id': args.env_id,
        'agents': args.agents,
        'action_mode': args.action_mode,
        'duration_seconds': args.duration,
        'max_steps': max_steps,
        'actual_steps': step,
        'actual_duration': elapsed_time,
        'fps': args.fps,
        'actual_video_fps': actual_video_fps,
        'sample_rate': args.sample_rate,
        'action_smooth': args.action_smooth,
        'time_dilation': args.time_dilation,
        'actual_sampled_frames': actual_sampled_frames,
        'resolution': args.resolution,
        'cameras': [{'name': cam.name, 'type': cam.cam_type, 'camera_id': camera_id_map.get(cam.name)} for cam in allocated_cameras],
        'target_agent_index': args.target_agent_index,
        'target_agent_type': target_agent_type_for_allocation,
        'has_trajectory': args.save_trajectory,
        'has_images': args.save_images,
        'timestamp': timestamp,
        'folder_name': folder_name,
        'auto_switch_target': args.auto_switch_target,
        'target_switch_count': target_switch_count if args.auto_switch_target else 0,
        'catch_distance': args.catch_distance,
        'catch_angle': args.catch_angle
    }
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(recording_info, f, indent=2, ensure_ascii=False)
    print(f"录制信息已保存到: {info_path}")
    
    # 保存相机配置到输出目录（只保存已分配相机的配置）
    config_output_path = os.path.join(full_output_dir, 'camera_config.json')
    config_data = {
        'target_agent_index': args.target_agent_index,
        'target_agent_type': target_agent_type_for_allocation,
        'cameras': [
            {
                'name': cam.name,
                'cam_type': cam.cam_type,
                'camera_id': camera_id_map.get(cam.name),
                **cam.params
            }
            for cam in allocated_cameras
        ]
    }
    with open(config_output_path, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"相机配置已保存到: {config_output_path}")
    
    # 保存完整的命令行参数配置到录制文件夹中
    args_output_path = os.path.join(full_output_dir, 'args_config.json')
    # 将args对象转换为字典，处理各种类型
    args_dict = {}
    for key, value in vars(args).items():
        # 处理各种类型，确保可以JSON序列化
        if isinstance(value, (str, int, float, bool, type(None))):
            args_dict[key] = value
        elif isinstance(value, (list, tuple)):
            # 列表和元组转换为列表
            args_dict[key] = [v if isinstance(v, (str, int, float, bool, type(None))) else str(v) for v in value]
        elif isinstance(value, np.ndarray):
            # numpy数组转换为列表
            args_dict[key] = value.tolist()
        else:
            # 其他类型转换为字符串
            args_dict[key] = str(value)
    
    # 添加额外的元数据
    args_dict['command_line'] = ' '.join(sys.argv)  # 保存完整的命令行
    args_dict['use_mixed_actions_override'] = use_mixed_actions  # 保存是否强制使用Mixed动作
    args_dict['final_env_id'] = args.env_id  # 保存最终使用的环境ID（可能被修改过）
    args_dict['output_folder'] = folder_name  # 保存对应的录制文件夹名
    args_dict['full_output_dir'] = full_output_dir  # 保存完整的输出路径
    
    with open(args_output_path, 'w', encoding='utf-8') as f:
        json.dump(args_dict, f, indent=2, ensure_ascii=False)
    print(f"参数配置已保存到: {args_output_path}")

