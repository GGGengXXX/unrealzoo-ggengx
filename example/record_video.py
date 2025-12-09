# ============================================================================
# 录制视频和保存图片示例脚本
# 功能：保存智能体的第一人称、第三人称视角视频和图片
# ============================================================================

import os
import argparse
import gym_unrealcv
import gym
import cv2
import time
import numpy as np
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, agents, augmentation, configUE

class RandomAgent(object):
    """随机动作智能体"""
    def __init__(self, action_space):
        self.action_space = action_space
        self.count_steps = 0
        self.action = self.action_space.sample()

    def act(self, observation, keep_steps=10):
        self.count_steps += 1
        if self.count_steps > keep_steps:
            self.action = self.action_space.sample()
            self.count_steps = 0
        return self.action

    def reset(self):
        self.action = self.action_space.sample()
        self.count_steps = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='录制视频和保存图片')
    parser.add_argument("-e", "--env_id", nargs='?', 
                        default='UnrealTrack-MiddleEast-ContinuousColor-v0',
                        help='环境ID')
    parser.add_argument("-o", "--output_dir", default='./recordings',
                        help='输出目录')
    parser.add_argument("-s", "--save_images", action='store_true',
                        help='保存每一帧的图片')
    parser.add_argument("-v", "--save_video", action='store_true',
                        help='保存视频文件')
    parser.add_argument("--episodes", type=int, default=1,
                        help='录制的episode数量')
    parser.add_argument("--third_person_distance", type=float, default=300.0,
                        help='第三人称相机距离角色的距离（单位）')
    parser.add_argument("--third_person_height", type=float, default=150.0,
                        help='第三人称相机相对角色的高度偏移（单位）')
    parser.add_argument("--third_person_angle", type=float, default=0.0,
                        help='第三人称相机角度（0=后方，90=右侧，-90=左侧，180=前方）')
    parser.add_argument("--hide_face", action='store_true',
                        help='隐藏第一人称视角中的人脸（通过前移相机位置）')
    parser.add_argument("--face_offset", type=float, default=200.0,
                        help='第一人称相机前移距离，用于隐藏人脸（单位，默认200，可尝试150-300）')
    parser.add_argument("--face_pitch", type=float, default=-15.0,
                        help='第一人称相机pitch角度，向下看以隐藏人脸（默认-15度，可尝试-10到-30）')
    parser.add_argument("--save_trajectory", action='store_true',
                        help='保存相机轨迹（位置和旋转）')
    parser.add_argument("--save_skeleton", action='store_true',
                        help='保存骨骼序列（动作序列）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建环境
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=False, resolution=(640, 480))
    env.unwrapped.agents_category = ['player']
    
    # 应用包装器
    env = augmentation.RandomPopulationWrapper(env, 2, 2, random_target=False)
    
    # 初始化视频写入器
    video_writers = []
    third_person_writers = []  # 第三人称跟随视角
    if args.save_video:
        # 为每个智能体创建视频写入器（第一人称视角）
        for i in range(len(env.action_space)):
            video_path = os.path.join(args.output_dir, f'agent_{i}_first_person.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (640, 480))
            video_writers.append(video_writer)
        
        # 第三人称跟随视角（从角色后方/侧面）
        for i in range(len(env.action_space)):
            third_person_path = os.path.join(args.output_dir, f'agent_{i}_third_person.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            third_person_writer = cv2.VideoWriter(third_person_path, fourcc, 30.0, (640, 480))
            third_person_writers.append(third_person_writer)
        
        # 第三人称视角（俯视图）视频
        topview_path = os.path.join(args.output_dir, 'topview.mp4')
        topview_writer = cv2.VideoWriter(topview_path, fourcc, 30.0, (640, 480))
    
    # 创建智能体
    agents = [RandomAgent(env.action_space[i]) for i in range(len(env.action_space))]
    
    print(f"开始录制，输出目录: {args.output_dir}")
    
    for episode in range(args.episodes):
        print(f"\nEpisode {episode + 1}/{args.episodes}")
        obs = env.reset()
        
        # 创建episode目录（无论是否保存图片都需要创建，用于保存info.json）
        episode_dir = os.path.join(args.output_dir, f'episode_{episode:03d}')
        os.makedirs(episode_dir, exist_ok=True)
        
        # 如果需要保存图片，创建子目录
        if args.save_images:
            os.makedirs(os.path.join(episode_dir, 'first_person'), exist_ok=True)
            os.makedirs(os.path.join(episode_dir, 'third_person'), exist_ok=True)
            os.makedirs(os.path.join(episode_dir, 'topview'), exist_ok=True)
        
        # 获取环境对象和智能体列表
        unwrapped_env = env.unwrapped
        player_list = unwrapped_env.player_list if hasattr(unwrapped_env, 'player_list') else []
        
        # 保存原始相机设置（用于恢复第一人称视角）
        original_cam_settings = {}
        first_person_cam_settings = {}  # 第一人称相机设置（可能已调整以隐藏人脸）
        
        if hasattr(unwrapped_env, 'agents') and len(player_list) > 0:
            for i, agent_name in enumerate(player_list):
                if agent_name in unwrapped_env.agents:
                    orig_loc = unwrapped_env.agents[agent_name].get('relative_location', [0, 0, 0]).copy()
                    orig_rot = unwrapped_env.agents[agent_name].get('relative_rotation', [0, 0, 0]).copy()
                    
                    # 保存原始设置
                    original_cam_settings[agent_name] = {
                        'loc': orig_loc.copy(),
                        'rot': orig_rot.copy()
                    }
                    
                    # 如果启用隐藏人脸，前移相机位置并稍微向下看
                    if args.hide_face:
                        # 前移相机（x方向，正值为前方，远离角色身体）
                        first_person_loc = orig_loc.copy()
                        first_person_loc[0] = first_person_loc[0] + args.face_offset
                        
                        # 向下看（pitch为负值），避免看到角色身体和脸
                        first_person_rot = orig_rot.copy()
                        first_person_rot[1] = first_person_rot[1] + args.face_pitch  # pitch向下（负值）
                        
                        first_person_cam_settings[agent_name] = {
                            'loc': first_person_loc,
                            'rot': first_person_rot
                        }
                    else:
                        first_person_cam_settings[agent_name] = {
                            'loc': orig_loc.copy(),
                            'rot': orig_rot.copy()
                        }
        
        # 初始化轨迹记录
        camera_trajectories = {}  # 存储所有相机的轨迹
        if args.save_trajectory:
            for i in range(len(player_list)):
                camera_trajectories[f'agent_{i}_first_person'] = []
                camera_trajectories[f'agent_{i}_third_person'] = []
            camera_trajectories['topview'] = []
        
        # 初始化骨骼序列记录
        skeleton_sequences = {}  # 存储所有智能体的骨骼序列
        if args.save_skeleton:
            for i in range(len(player_list)):
                skeleton_sequences[f'agent_{i}'] = []
        
        step = 0
        while True:
            # 在step之前，先设置第一人称相机（如果启用hide_face，使用调整后的设置）
            if hasattr(unwrapped_env, 'unrealcv') and len(player_list) > 0:
                for i, agent_name in enumerate(player_list):
                    if agent_name in first_person_cam_settings:
                        # 使用第一人称相机设置（可能已调整以隐藏人脸）
                        cam_loc = first_person_cam_settings[agent_name]['loc']
                        cam_rot = first_person_cam_settings[agent_name]['rot']
                        unwrapped_env.unrealcv.set_cam(agent_name, loc=cam_loc, rot=cam_rot)
            
            # 获取动作
            actions = [agents[i].act(obs[i]) for i in range(len(agents))]
            
            # 执行动作
            obs, rewards, done, info = env.step(actions)
            
            # step()之后，相机可能被重置（step内部会调用set_cam重置相机）
            # 如果启用hide_face，需要重新设置第一人称相机并重新获取观察
            if args.hide_face and hasattr(unwrapped_env, 'unrealcv') and len(player_list) > 0:
                for i, agent_name in enumerate(player_list):
                    if agent_name in first_person_cam_settings and i < len(obs):
                        cam_loc = first_person_cam_settings[agent_name]['loc']
                        cam_rot = first_person_cam_settings[agent_name]['rot']
                        # 重新设置第一人称相机（前移，隐藏人脸）
                        unwrapped_env.unrealcv.set_cam(agent_name, loc=cam_loc, rot=cam_rot)
                        
                        # 重新获取第一人称观察（不包含人脸）
                        if hasattr(unwrapped_env, 'cam_list') and i < len(unwrapped_env.cam_list):
                            cam_id = unwrapped_env.cam_list[i]
                            try:
                                # 获取更新后的第一人称图像（相机已前移，不包含人脸）
                                first_person_img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
                                if first_person_img is not None:
                                    obs[i] = first_person_img
                            except Exception as e:
                                # 如果获取失败，使用原始obs
                                if step % 100 == 0:
                                    print(f"  警告：无法重新获取智能体{i}的第一人称图像: {e}")
            
            # 获取第一人称视角（每个智能体的观察）
            # 注意：如果启用了hide_face，obs已经更新为不包含人脸的图像
            for i, agent_obs in enumerate(obs):
                # 转换为BGR格式（OpenCV使用BGR）
                img_bgr = cv2.cvtColor(agent_obs, cv2.COLOR_RGB2BGR) if len(agent_obs.shape) == 3 else agent_obs
                
                # 记录第一人称相机轨迹
                if args.save_trajectory and hasattr(unwrapped_env, 'unrealcv'):
                    try:
                        if hasattr(unwrapped_env, 'cam_list') and i < len(unwrapped_env.cam_list):
                            cam_id = unwrapped_env.cam_list[i]
                            cam_location = unwrapped_env.unrealcv.get_cam_location(cam_id)
                            cam_rotation = unwrapped_env.unrealcv.get_cam_rotation(cam_id)
                            camera_trajectories[f'agent_{i}_first_person'].append({
                                'step': step,
                                'location': cam_location,
                                'rotation': cam_rotation
                            })
                    except Exception as e:
                        if step % 100 == 0:
                            print(f"  警告：无法获取智能体{i}的第一人称相机轨迹: {e}")
                
                # 保存图片
                if args.save_images:
                    img_path = os.path.join(episode_dir, 'first_person', 
                                          f'agent_{i}_step_{step:05d}.png')
                    cv2.imwrite(img_path, img_bgr)
                
                # 写入视频
                if args.save_video and i < len(video_writers):
                    # 调整图像大小以匹配视频分辨率
                    if img_bgr.shape[:2] != (480, 640):
                        img_bgr = cv2.resize(img_bgr, (640, 480))
                    video_writers[i].write(img_bgr)
            
            # 获取智能体位置信息（用于第三人称相机轨迹和骨骼序列）
            agent_poses = info.get('Pose', [])
            
            # 记录骨骼序列（动作序列）
            if args.save_skeleton and hasattr(unwrapped_env, 'unrealcv') and len(player_list) > 0:
                for i, agent_name in enumerate(player_list):
                    try:
                        skeleton_data = {
                            'step': step,
                            'agent_name': agent_name
                        }
                        
                        # 1. 获取智能体的位置和旋转（根骨骼）
                        if i < len(agent_poses) and len(agent_poses[i]) >= 6:
                            skeleton_data['root_location'] = [float(agent_poses[i][0]), 
                                                              float(agent_poses[i][1]), 
                                                              float(agent_poses[i][2])]
                            skeleton_data['root_rotation'] = [float(agent_poses[i][3]), 
                                                              float(agent_poses[i][4]), 
                                                              float(agent_poses[i][5])]
                        
                        # 2. 尝试通过vbp命令获取骨骼数据（如果UnrealCV支持）
                        # 常见的骨骼获取命令可能是：
                        # - vbp {agent_name} get_bone_positions
                        # - vbp {agent_name} get_skeleton
                        # - vbp {agent_name} get_joints
                        bone_positions = {}
                        bone_rotations = {}
                        
                        # 尝试获取常见骨骼的位置和旋转
                        common_bones = [
                            'pelvis', 'spine_01', 'spine_02', 'spine_03', 'neck_01', 'head',
                            'clavicle_l', 'upperarm_l', 'lowerarm_l', 'hand_l',
                            'clavicle_r', 'upperarm_r', 'lowerarm_r', 'hand_r',
                            'thigh_l', 'calf_l', 'foot_l',
                            'thigh_r', 'calf_r', 'foot_r'
                        ]
                        
                        for bone_name in common_bones:
                            try:
                                # 尝试获取骨骼位置
                                cmd_pos = f'vbp {agent_name} get_bone_location {bone_name}'
                                res_pos = unwrapped_env.unrealcv.client.request(cmd_pos, timeout=0.1)
                                if res_pos and 'error' not in res_pos.lower():
                                    # 解析位置（通常是 "x y z" 格式）
                                    try:
                                        pos_values = [float(x) for x in res_pos.strip().split()]
                                        if len(pos_values) >= 3:
                                            bone_positions[bone_name] = pos_values[:3]
                                    except:
                                        pass
                                
                                # 尝试获取骨骼旋转
                                cmd_rot = f'vbp {agent_name} get_bone_rotation {bone_name}'
                                res_rot = unwrapped_env.unrealcv.client.request(cmd_rot, timeout=0.1)
                                if res_rot and 'error' not in res_rot.lower():
                                    # 解析旋转（通常是 "roll pitch yaw" 或 "x y z w" 格式）
                                    try:
                                        rot_values = [float(x) for x in res_rot.strip().split()]
                                        if len(rot_values) >= 3:
                                            bone_rotations[bone_name] = rot_values[:3]
                                    except:
                                        pass
                            except:
                                # 如果命令不存在或失败，跳过这个骨骼
                                pass
                        
                        # 如果获取到了骨骼数据，添加到skeleton_data
                        if bone_positions:
                            skeleton_data['bone_positions'] = bone_positions
                        if bone_rotations:
                            skeleton_data['bone_rotations'] = bone_rotations
                        
                        # 3. 尝试获取所有骨骼（如果支持批量获取）
                        try:
                            cmd_all = f'vbp {agent_name} get_all_bones'
                            res_all = unwrapped_env.unrealcv.client.request(cmd_all, timeout=0.1)
                            if res_all and 'error' not in res_all.lower():
                                # 解析所有骨骼数据（格式可能因实现而异）
                                skeleton_data['all_bones'] = res_all
                        except:
                            pass
                        
                        # 4. 尝试获取动画状态（当前播放的动画）
                        try:
                            cmd_anim = f'vbp {agent_name} get_current_animation'
                            res_anim = unwrapped_env.unrealcv.client.request(cmd_anim, timeout=0.1)
                            if res_anim and 'error' not in res_anim.lower():
                                skeleton_data['current_animation'] = res_anim.strip()
                        except:
                            pass
                        
                        # 5. 尝试获取动画播放进度
                        try:
                            cmd_progress = f'vbp {agent_name} get_animation_progress'
                            res_progress = unwrapped_env.unrealcv.client.request(cmd_progress, timeout=0.1)
                            if res_progress and 'error' not in res_progress.lower():
                                try:
                                    skeleton_data['animation_progress'] = float(res_progress.strip())
                                except:
                                    pass
                        except:
                            pass
                        
                        # 保存骨骼数据
                        skeleton_sequences[f'agent_{i}'].append(skeleton_data)
                        
                    except Exception as e:
                        if step % 100 == 0:
                            print(f"  警告：无法获取智能体{i}的骨骼数据: {e}")
            
            # 获取第三人称跟随视角（使用set_cam绑定到agent，自动跟随）
            if hasattr(unwrapped_env, 'unrealcv') and len(player_list) > 0:
                for i, agent_name in enumerate(player_list):
                    try:
                        # 设置第三人称相机：相对于agent的位置
                        distance = args.third_person_distance
                        height_offset = args.third_person_height
                        angle_offset = args.third_person_angle  # 0=后方，90=右侧，-90=左侧
                        
                        # 计算相对于agent的相机位置（Unreal坐标系）
                        # set_cam的loc是相对于agent的：[x, y, z]
                        # 根据UnrealCV文档和代码，通常：
                        # x: 前后（正=前方，负=后方），y: 左右（正=右侧），z: 高度
                        angle_rad = np.radians(angle_offset)
                        x_offset = -distance * np.cos(angle_rad)  # 后方为负
                        y_offset = distance * np.sin(angle_rad)   # 右侧为正
                        z_offset = height_offset
                        
                        # 设置相机旋转（朝向agent）
                        pitch = -15  # 稍微向下看
                        yaw = 0  # 朝向agent前方
                        roll = 0
                        
                        # 使用set_cam设置相对于agent的相机（会自动跟随agent移动）
                        unwrapped_env.unrealcv.set_cam(agent_name, 
                                                      loc=[x_offset, y_offset, z_offset],
                                                      rot=[roll, pitch, yaw])
                        
                        # 获取agent的相机ID
                        if hasattr(unwrapped_env, 'cam_list') and i < len(unwrapped_env.cam_list):
                            cam_id = unwrapped_env.cam_list[i]
                            
                            # 获取第三人称视角图像（set_cam已经设置了相对位置，会自动跟随）
                            third_person_img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
                            
                            if third_person_img is not None:
                                third_person_bgr = cv2.cvtColor(third_person_img, cv2.COLOR_RGB2BGR)
                                
                                # 记录第三人称相机轨迹
                                if args.save_trajectory and len(agent_poses) > 0:
                                    try:
                                        # 获取第三人称相机的实际位置（世界坐标）
                                        # 由于set_cam是相对位置，我们需要计算世界坐标
                                        agent_pose = agent_poses[i] if i < len(agent_poses) else None
                                        if agent_pose and len(agent_pose) >= 6:
                                            agent_x, agent_y, agent_z = agent_pose[0], agent_pose[1], agent_pose[2]
                                            agent_yaw = agent_pose[4]
                                            
                                            # 计算第三人称相机的世界坐标
                                            total_yaw = agent_yaw + angle_offset
                                            yaw_rad = np.radians(total_yaw)
                                            third_cam_x = agent_x - distance * np.sin(yaw_rad)
                                            third_cam_y = agent_y - distance * np.cos(yaw_rad)
                                            third_cam_z = agent_z + height_offset
                                            
                                            # 计算第三人称相机旋转
                                            third_cam_pitch = -15
                                            third_cam_yaw = agent_yaw + 180
                                            third_cam_roll = 0
                                            
                                            camera_trajectories[f'agent_{i}_third_person'].append({
                                                'step': step,
                                                'location': [third_cam_x, third_cam_y, third_cam_z],
                                                'rotation': [third_cam_roll, third_cam_pitch, third_cam_yaw],
                                                'agent_location': [agent_x, agent_y, agent_z],
                                                'agent_rotation': agent_pose[3:6]  # roll, pitch, yaw
                                            })
                                    except Exception as e:
                                        if step % 100 == 0:
                                            print(f"  警告：无法记录智能体{i}的第三人称相机轨迹: {e}")
                                
                                # 保存第三人称图片
                                if args.save_images:
                                    third_person_path = os.path.join(episode_dir, 'third_person',
                                                                    f'agent_{i}_step_{step:05d}.png')
                                    cv2.imwrite(third_person_path, third_person_bgr)
                                
                                # 写入第三人称视频
                                if args.save_video and i < len(third_person_writers):
                                    if third_person_bgr.shape[:2] != (480, 640):
                                        third_person_bgr = cv2.resize(third_person_bgr, (640, 480))
                                    third_person_writers[i].write(third_person_bgr)
                    except Exception as e:
                        # 如果获取第三人称视角失败，跳过
                        if step % 100 == 0:  # 每100步打印一次警告
                            print(f"  警告：无法获取智能体{i}的第三人称视角: {e}")
            
            # 获取第三人称视角（俯视图）
            topview_img = env.render(mode='rgb_array')
            if topview_img is not None:
                topview_bgr = cv2.cvtColor(topview_img, cv2.COLOR_RGB2BGR)
                
                # 记录俯视图相机轨迹
                if args.save_trajectory and hasattr(unwrapped_env, 'unrealcv'):
                    try:
                        third_cam_id = unwrapped_env.cam_id[0] if hasattr(unwrapped_env, 'cam_id') else 0
                        topview_location = unwrapped_env.unrealcv.get_cam_location(third_cam_id)
                        topview_rotation = unwrapped_env.unrealcv.get_cam_rotation(third_cam_id)
                        camera_trajectories['topview'].append({
                            'step': step,
                            'location': topview_location,
                            'rotation': topview_rotation
                        })
                    except Exception as e:
                        if step % 100 == 0:
                            print(f"  警告：无法获取俯视图相机轨迹: {e}")
                
                # 保存俯视图
                if args.save_images:
                    topview_path = os.path.join(episode_dir, 'topview', 
                                               f'topview_step_{step:05d}.png')
                    cv2.imwrite(topview_path, topview_bgr)
                
                # 写入俯视图视频
                if args.save_video:
                    if topview_bgr.shape[:2] != (480, 640):
                        topview_bgr = cv2.resize(topview_bgr, (640, 480))
                    topview_writer.write(topview_bgr)
            
            step += 1
            
            # 打印进度
            if step % 50 == 0:
                print(f"  步骤 {step}, 奖励: {rewards}")
            
            if done:
                print(f"  Episode 结束，总步数: {step}, 最终奖励: {rewards}")
                break
        
        # 保存episode信息
        info_path = os.path.join(episode_dir, 'info.json')
        import json
        episode_info = {
            'steps': step,
            'final_rewards': rewards.tolist() if isinstance(rewards, np.ndarray) else rewards,
            'distance': info.get('Distance', None),
            'direction': info.get('Direction', None)
        }
        with open(info_path, 'w') as f:
            json.dump(episode_info, f, indent=2)
        
        # 保存相机轨迹
        if args.save_trajectory:
            trajectory_path = os.path.join(episode_dir, 'camera_trajectories.json')
            # 将numpy数组转换为列表
            trajectory_data = {}
            for key, trajectory in camera_trajectories.items():
                trajectory_data[key] = []
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
                    
                    # 如果有agent信息，也转换
                    if 'agent_location' in frame:
                        frame_data['agent_location'] = [float(x) for x in frame['agent_location']]
                    if 'agent_rotation' in frame:
                        frame_data['agent_rotation'] = [float(x) for x in frame['agent_rotation']]
                    
                    trajectory_data[key].append(frame_data)
            
            with open(trajectory_path, 'w') as f:
                json.dump(trajectory_data, f, indent=2)
            print(f"  相机轨迹已保存到: {trajectory_path}")
        
        # 保存骨骼序列（动作序列）
        if args.save_skeleton:
            skeleton_path = os.path.join(episode_dir, 'skeleton_sequences.json')
            # 将numpy数组转换为列表
            skeleton_data = {}
            for key, sequence in skeleton_sequences.items():
                skeleton_data[key] = []
                for frame in sequence:
                    frame_data = {'step': frame['step'], 'agent_name': frame['agent_name']}
                    
                    # 转换根骨骼位置和旋转
                    if 'root_location' in frame:
                        frame_data['root_location'] = [float(x) for x in frame['root_location']]
                    if 'root_rotation' in frame:
                        frame_data['root_rotation'] = [float(x) for x in frame['root_rotation']]
                    
                    # 转换骨骼位置
                    if 'bone_positions' in frame:
                        frame_data['bone_positions'] = {}
                        for bone_name, pos in frame['bone_positions'].items():
                            frame_data['bone_positions'][bone_name] = [float(x) for x in pos]
                    
                    # 转换骨骼旋转
                    if 'bone_rotations' in frame:
                        frame_data['bone_rotations'] = {}
                        for bone_name, rot in frame['bone_rotations'].items():
                            frame_data['bone_rotations'][bone_name] = [float(x) for x in rot]
                    
                    # 其他信息
                    if 'all_bones' in frame:
                        frame_data['all_bones'] = frame['all_bones']
                    if 'current_animation' in frame:
                        frame_data['current_animation'] = frame['current_animation']
                    if 'animation_progress' in frame:
                        frame_data['animation_progress'] = float(frame['animation_progress'])
                    
                    skeleton_data[key].append(frame_data)
            
            with open(skeleton_path, 'w') as f:
                json.dump(skeleton_data, f, indent=2)
            print(f"  骨骼序列已保存到: {skeleton_path}")
    
    # 释放视频写入器
    if args.save_video:
        for writer in video_writers:
            writer.release()
        for writer in third_person_writers:
            writer.release()
        topview_writer.release()
        print(f"\n视频已保存到: {args.output_dir}")
        print("  - 第一人称视角: agent_X_first_person.mp4")
        print("  - 第三人称跟随视角: agent_X_third_person.mp4")
        print("  - 俯视图: topview.mp4")
    
    env.close()
    print("录制完成！")

