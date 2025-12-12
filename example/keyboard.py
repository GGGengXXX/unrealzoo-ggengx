import argparse
import gym
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, augmentation, configUE,agents
from pynput import keyboard
import time
import cv2
import numpy as np
import os
from datetime import datetime

# 随机模型
class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        # return 2
        return self.action_space.sample()

# 按键控制
# 全局变量
key_state = {
    'i': False,
    'j': False,
    'k': False,
    'l': False,
    'space': False,
    '1': False,
    '2': False,
    'head_up': False,
    'head_down': False,
    'q': False,  # 用于结束录制
    'esc': False  # 用于结束录制
}

# 键盘按下
def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = True
        elif key == keyboard.Key.up:
            key_state['head_up'] = True
        elif key == keyboard.Key.down:
            key_state['head_down'] = True
        elif key == keyboard.Key.esc:
            key_state['esc'] = True

# 键盘抬起来
def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = False
        elif key == keyboard.Key.up:
            key_state['head_up'] = False
        elif key == keyboard.Key.down:
            key_state['head_down'] = False
        elif key == keyboard.Key.esc:
            key_state['esc'] = False
            
# 开启监听进程
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()
def get_key_action():
    action = ([0, 0], 0, 0)
    action = list(action)  # Convert tuple to list for modification
    action[0] = list(action[0])  # Convert inner tuple to list for modification

    if key_state['i']:
        action[0][1] = 100
    if key_state['k']:
        action[0][1] = -100
    if key_state['j']:
        action[0][0] = -30
    if key_state['l']:
        action[0][0] = 30
    if key_state['space']:
        action[2] = 1
    if key_state['1']:
        action[2] = 3
    if key_state['2']:
        action[2] = 4
    if key_state['head_up']:
        action[1] = 1
    if key_state['head_down']:
        action[1] = 2

    action[0] = tuple(action[0])  # Convert inner list back to tuple
    action = tuple(action)  # Convert list back to tuple
    return action

def allocate_camera_ids(unwrapped_env, player_list):
    """收集所有agent的相机ID，分配给第0个agent使用（排除0号相机）"""
    all_cam_ids = []
    for i in range(len(player_list)):
        if hasattr(unwrapped_env, 'cam_list') and i < len(unwrapped_env.cam_list):
            cam_id = unwrapped_env.cam_list[i]
            if cam_id is not None and cam_id != 0:  # 排除0号相机
                all_cam_ids.append(cam_id)
    return all_cam_ids

def setup_third_person_camera(unwrapped_env, cam_id, agent_pose, distance=100.0, height=100, angle=0):
    """设置第三人称相机位置"""
    if agent_pose is None or len(agent_pose) < 6:
        return None
    
    agent_x, agent_y, agent_z = agent_pose[0], agent_pose[1], agent_pose[2]
    agent_yaw = agent_pose[5]
    
    # 计算相对于agent的相机位置
    angle_rad = np.radians(angle)
    local_x = -distance * np.cos(angle_rad)
    local_y = distance * np.sin(angle_rad)
    local_z = height
    
    # 转换为世界坐标
    agent_yaw_rad = np.radians(agent_yaw)
    cos_yaw = np.cos(agent_yaw_rad)
    sin_yaw = np.sin(agent_yaw_rad)
    
    world_x = agent_x + local_x * cos_yaw + local_y * sin_yaw
    world_y = agent_y - local_x * sin_yaw + local_y * cos_yaw
    world_z = agent_z + local_z
    
    # 计算相机朝向
    dx = agent_x - world_x
    dy = agent_y - world_y
    cam_yaw = np.degrees(np.arctan2(dy, dx))
    
    cam_loc = [world_x, world_y, world_z]
    cam_rot = [-15, cam_yaw, 0]
    
    try:
        unwrapped_env.unrealcv.set_cam_location(cam_id, cam_loc)
        unwrapped_env.unrealcv.set_cam_rotation(cam_id, cam_rot)
        img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
        return img
    except:
        return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-track_train-ContinuousMask-v4',
    #                     help='Select the environment to run')
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-MiddleEast-MixedColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=10, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=-1,
                        help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-o", '--output', dest='output', default='./recordings', help='output directory for videos')

    args = parser.parse_args()
    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=True, resolution=(240, 240))
    env.unwrapped.agents_category=['player'] #choose the agent type in the scene

    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    env = augmentation.RandomPopulationWrapper(env, 6, 6, random_target=False)
    env = agents.NavAgents(env, mask_agent=False)
    agent = RandomAgent(env.action_space[0])
    
    # 分配相机ID给第0个agent
    unwrapped_env = env.unwrapped
    player_list = unwrapped_env.player_list if hasattr(unwrapped_env, 'player_list') else []
    all_cam_ids = allocate_camera_ids(unwrapped_env, player_list)
    
    if len(all_cam_ids) < 5:
        print(f"警告：只有 {len(all_cam_ids)} 个相机ID，需要至少5个（1个第一人称+4个第三人称）")
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"keyboard_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化视频写入器
    resolution = (240, 240)
    fps = 30.0
    
    def create_video_writer(path):
        """创建视频写入器，尝试多个编码格式"""
        codecs = ['H264', 'XVID', 'mp4v', 'avc1']
        for codec in codecs:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(path, fourcc, fps, resolution)
                if writer.isOpened():
                    return writer
            except:
                continue
        return None
    
    video_writers = {}
    
    # 第一人称相机
    if len(all_cam_ids) > 0:
        writer = create_video_writer(os.path.join(output_dir, 'agent_0_first_person.mp4'))
        if writer:
            video_writers['first_person'] = writer
    
    # 第三人称相机（4个角度：0°, 90°, 180°, 270°）
    third_person_angles = [0, 90, 180, 270]
    for i, angle in enumerate(third_person_angles):
        if i + 1 < len(all_cam_ids):
            writer = create_video_writer(os.path.join(output_dir, f'agent_0_third_person_{angle}.mp4'))
            if writer:
                video_writers[f'third_person_{angle}'] = writer
    
    print(f'输出目录: {output_dir}')
    print(f'已分配 {len(all_cam_ids)} 个相机ID，创建 {len(video_writers)} 个视频文件')
    
    rewards = 0
    done = False
    Total_rewards = 0
    count_step = 0
    env.seed(int(args.seed))
    obs = env.reset()
    t0 = time.time()
    print('Use the "I", "J", "K", and "L" keys to control the agent  movement.')
    print('Press "Q" or "ESC" to stop recording and save videos.')
    
    try:
        while True:
            # 检查是否按下结束录制按键
            if key_state['q'] or key_state['esc']:
                print('\n用户按下停止键，结束录制...')
                break
            
            action = get_key_action()
            obs, rewards, done, info = env.step([action])
            
            # 显示第一人称视角
            if obs is not None and len(obs) > 0:
                cv2.imshow('obs', obs[0])
                cv2.waitKey(1)
            
            # 获取agent位置
            agent_poses = info.get('Pose', [])
            if not agent_poses and hasattr(env.unwrapped, 'obj_poses'):
                agent_poses = env.unwrapped.obj_poses
            
            agent_pose = agent_poses[0] if agent_poses and len(agent_poses) > 0 else None
            
            # 录制第一人称视频
            if 'first_person' in video_writers and obs is not None and len(obs) > 0:
                img = obs[0]
                if img.shape[:2] != tuple(reversed(resolution)):
                    img = cv2.resize(img, resolution)
                video_writers['first_person'].write(img)
            
            # 录制第三人称视频
            if agent_pose is not None:
                for i, angle in enumerate(third_person_angles):
                    cam_name = f'third_person_{angle}'
                    if cam_name in video_writers and i + 1 < len(all_cam_ids):
                        cam_id = all_cam_ids[i + 1]
                        img = setup_third_person_camera(unwrapped_env, cam_id, agent_pose, 
                                                       distance=100.0, height=100, angle=angle)
                        if img is not None:
                            if img.shape[:2] != tuple(reversed(resolution)):
                                img = cv2.resize(img, resolution)
                            video_writers[cam_name].write(img)
            
            count_step += 1
            if done:
                fps = count_step / (time.time() - t0)
                print('Success')
                print('Fps:' + str(fps))
                break
    finally:
        # 释放视频写入器
        print('\n正在保存视频...')
        for writer in video_writers.values():
            writer.release()
        print(f'✓ 视频已保存到: {output_dir}')
        print(f'  共保存 {len(video_writers)} 个视频文件')
    
    env.close()