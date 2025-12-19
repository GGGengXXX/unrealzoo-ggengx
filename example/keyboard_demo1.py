import argparse
from gc import enable
from cv2.gapi import video
import gym
from gym_unrealcv.envs.wrappers import time_dilation, early_done, monitor, augmentation, configUE,agents
from pynput import keyboard
import time
import cv2 as cv
from datetime import datetime
import os
import numpy as np

class RandomAgent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        # return 2
        return self.action_space.sample()


key_state = {
    'a': False,
    's': False,
    'd': False,
    'w': False,
    'space': False,
    '0':False,
    '1': False,
    '2': False,
    '3': False,
    '4': False,
    '5': False,
    '6': False,
    '7': False,
    '8': False,
    '9': False,
    'head_up': False,
    'head_down': False,
    'esc': False,
    'q': False
}

# 键盘控制
def on_press(key):
    try:
        if key.char in key_state:
            key_state[key.char] = True
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = True
        if key == keyboard.Key.up:
            key_state['head_up'] = True
        if key == keyboard.Key.down:
            key_state['head_down'] = True
        elif key == keyboard.Key.esc:
            key_state['esc'] = True


def on_release(key):
    try:
        if key.char in key_state:
            key_state[key.char] = False
    except AttributeError:
        if key == keyboard.Key.space:
            key_state['space'] = False
        if key == keyboard.Key.up:
            key_state['head_up'] = False
        if key == keyboard.Key.down:
            key_state['head_down'] = False
        elif key == keyboard.Key.esc:
            key_state['esc'] = False
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

def getP(rate):
    if np.random.rand() < rate:
        return True
    else:
        return False

def get_random_action():
    rol, forward = 0, 0
    headup = 0
    ani = 0
    if getP(0.1):
        rol = -30
    elif getP(0.13):
        rol = 30
    else:
        rol = 0

    if getP(0.8):
        forward = 100
    elif getP(0.2):
        forward = -100

    if getP(0.1):
        headup = 1
    elif getP(0.15):
        headup = 2
    else:
        headup = 0

    if getP(0.1):
        ani = np.random.randint(0, 4)

    return ([rol, forward], headup, ani)


def get_key_action():
    action = ([0, 0], 0, 0)
    action = list(action)  # Convert tuple to list for modification
    action[0] = list(action[0])  # Convert inner tuple to list for modification

    if key_state['w']:
        action[0][1] = 100
    if key_state['s']:
        action[0][1] = -100
    if key_state['a']:
        action[0][0] = -30
    if key_state['d']:
        action[0][0] = 30
    if key_state['space']:
        action[2] = 1
    if key_state['0']:
        action[2] = 0
    if key_state['1']:
        action[2] = 1
    if key_state['2']:
        action[2] = 2
    if key_state['3']:
        action[2] = 3
    if key_state['4']:
        action[2] = 4
    if key_state['5']:
        action[2] = 5
    if key_state['6']:
        action[2] = 6
    if key_state['7']:
        action[2] = 7
    if key_state['8']:
        action[2] = 8
    if key_state['9']:
        action[2] = 9
    if key_state['head_up']:
        action[1] = 1
    if key_state['head_down']:
        action[1] = 2

    action[0] = tuple(action[0])  # Convert inner list back to tuple
    action = tuple(action)  # Convert list back to tuple
    return action

def get_agent_view_at_angle(unwrapped_env, cam_id, agent_pose, distance=100.0, height=100, angle=0):
    """
    获取指定 agent 在指定方位的第三人称视角图像

    参数:
        unwrapped_env: 环境的 unwrapped 部分，用于设置相机和获取图像
        cam_id: 相机的ID
        agent_pose: agent 当前的位姿（长度≥6，[x, y, z, roll, pitch, yaw]）
        distance: 相机距离 agent 的距离（默认100.0）
        height: 相机距离地面的高度（默认100）
        angle: 方位角度（以 agent 朝向为基准，默认0）

    返回:
        相机捕获的图像（可能为None）
    """
    if agent_pose is None or len(agent_pose) < 6:
        return None
    
    agent_x, agent_y, agent_z = agent_pose[0], agent_pose[1], agent_pose[2]
    agent_yaw = agent_pose[5]
    
    # 计算相对于 agent 的相机位置（指定的方位角）
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

def getImageFromCamera(unwrapped_env, cam_id, agent_pose, distance=100.0, height=100, angle=0):
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

def getDepthImageFromCamera(unwrapped_env, cam_id, agent_pose, distance=100.0, height=100, angle=0):
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
        # img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
        img = unwrapped_env.unrealcv.get_depth(cam_id,show=False)
        # img = unwrapped_env.unrealcv.get_image(cam_id, 'depth', 'bmp')
        # print(img)
        # print(img.shape)
        return img
    except:
        return None
    

def debug_camera():
    for i in range(len(player_list)):
        if hasattr(unwrapped_env, 'cam_list') and i < len(unwrapped_env.cam_list):
            cam_id = unwrapped_env.cam_list[i]
            print(f'cam_id: {cam_id}')

def get_360_part(x):
    while x < 0:
        x = x + 360
    while x > 360:
        x = x - 360
    return 360 - x

def record_frame(video_writers, fst_img, agent_pose, unwrapped_env, rsolution):
    if "first_person" in video_writers:
        img = fst_img
        if img.shape[:2] != tuple(reversed(resolution)):
            img = cv.resize(img, resolution)
        video_writers['first_person'].write(img)

    for i in range(4):
        cam_name = f"third_person_{i}"
        if cam_name in video_writers:
            img = getImageFromCamera(unwrapped_env, 1, agent_pose,angle=get_360_part(agent_pose[4]+i*90))
            
            if img is not None:
                if img.shape[:2] != tuple(reversed(resolution)):
                    img = cv.resize(img, resolution)
                video_writers[cam_name].write(img)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    # parser.add_argument("-e", "--env_id", nargs='?', default='UnrealTrack-track_train-ContinuousMask-v4',
    #                     help='Select the environment to run')
    parser.add_argument("-e", "--env_id", nargs='?', default='UnrealAgent-Greek_Island-MixedColor-v0',
                        help='Select the environment to run')
    parser.add_argument("-r", '--render', dest='render', action='store_true', help='show env using cv2')
    parser.add_argument("-s", '--seed', dest='seed', default=10, help='random seed')
    parser.add_argument("-t", '--time-dilation', dest='time_dilation', default=50,
                        help='time_dilation to keep fps in simulator')
    parser.add_argument("-d", '--early-done', dest='early_done', default=-1, help='early_done when lost in n steps')
    parser.add_argument("-m", '--monitor', dest='monitor', action='store_true', help='auto_monitor')
    parser.add_argument("-o", '--output', dest='output', default='./recordings', help='output directory for videos')

    # argparse 是 Python 标准库中用于解析命令行参数的模块，可以让我们从命令行传递参数给脚本，提升程序通用性和灵活性。
    # parser.add_argument 用于定义程序可以接受的参数。
    # 下面的例子添加了一个 "--record" 参数，用于控制是否启用录制功能。如果在命令行使用 --record，则 args.enable_record 为 True。
    parser.add_argument("--record", dest='enable_record', action='store_true', help='enable recording functionality')
    parser.add_argument("--step", action="store", type=int, default=100000, help="number of steps to run the environment")


    args = parser.parse_args()

    resolution = [1920, 1280]
    fps = 60.0

    from config_set import config
    myconfig = config()
    # args.time_dilation, resolution, fps = myconfig.get_high_fps_1920x1080_config()
    args.time_dilation, resolution, fps = myconfig.get_normal_config()

    AGENT_TYPE = 'player'

    enable_record = args.enable_record
    step_count = args.step

    env = gym.make(args.env_id)
    env = configUE.ConfigUEWrapper(env, offscreen=True, resolution=(1920, 1080))
    env = configUE.ConfigUEWrapper(env, offscreen=True, resolution=resolution)
    env.unwrapped.agents_category=[AGENT_TYPE] #choose the agent type in the scene

    if int(args.time_dilation) > 0:  # -1 means no time_dilation
        print(f"应用时间膨胀: {args.time_dilation}x")
        env = time_dilation.TimeDilationWrapper(env, int(args.time_dilation))
    if int(args.early_done) > 0:  # -1 means no early_done
        env = early_done.EarlyDoneWrapper(env, int(args.early_done))
    if args.monitor:
        env = monitor.DisplayWrapper(env)
    env = augmentation.RandomPopulationWrapper(env, 1, 1, random_tracker=False)
    # env = agents.NavAgents(env, mask_agent=False)

    agent = RandomAgent(env.action_space[0])

    unwrapped_env = env.unwrapped
    player_list = unwrapped_env.player_list if hasattr(unwrapped_env, 'player_list') else []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output, f"keyboard_{timestamp}")
    if enable_record:
        os.makedirs(output_dir, exist_ok=True)



    def create_video_writer(path):
        """创建视频写入器，尝试多个编码格式"""
        codecs = ['H264', 'avc1', 'XVID', 'mp4v']
        for codec in codecs:
            try:
                fourcc = cv.VideoWriter_fourcc(*codec)
                writer = cv.VideoWriter(path, fourcc, fps, resolution)
                if writer.isOpened():
                    
                    return writer
            except:
                continue
        return None

    video_writers = {}

    writer = None

    if enable_record:
        writer = create_video_writer(os.path.join(output_dir, 'first_person.mp4'))
        if writer:
            video_writers["first_person"] = writer
        for i in range(4):
            writer = create_video_writer(os.path.join(output_dir, f"third_person_{i}.mp4"))
            if writer:
                video_writers[f"third_person_{i}"] = writer

    rewards = 0
    done = False
    Total_rewards = 0
    count_step = 0
    env.seed(int(args.seed))
    obs = env.reset()
    t0 = time.time()
    # 每个 action 是一个 tuple，包含三个元素，分别是：
    # 1. 移动方向 2.头部动作 3.具体动作
    debug_action = (
        [0,0],
        0,
        1
    )

    debug_animal_action = (
        [0, 100],
        None,
        None
    )

    from random_controller import randomPeopleController, randomAnimalController
    random_controller = randomPeopleController()
    random_controller = randomAnimalController()

    # debug_camera()
    # action = get_random_action()
    # cnt = 0
    # nowdelay = np.random.randint(1,5)
    dimgls = []
    iter = 0
    print(env.unwrapped.player_list)
    # 随机人物外观
    env.unwrapped.random_app()
    try:
        while True:
            if key_state['q'] or key_state['esc'] or iter == step_count:
                print('\n用户按下停止键，结束录制...')
                break

            iter = iter + 1
            if iter % 1000 == 0:
                print(f"{iter}/{step_count} steps completed.")
            # action = get_key_action()
            # action = get_random_action()
            # cnt = cnt + 1
            # if cnt == nowdelay:
            #     action = get_random_action()
            #     cnt = 0
            #     nowdelay = np.random.randint(1,5)
            print(random_controller.get_random_action())
            action = random_controller.get_action()
            print(action)
            obs, rewards, done, info = env.step([action])
            # obs, rewards, done, info = env.step([debug_animal_action])

            # 获取agent位置
            agent_poses = info.get('Pose', [])
            if not agent_poses and hasattr(env.unwrapped, 'obj_poses'):
                agent_poses = env.unwrapped.obj_poses
            
            agent_pose = agent_poses[0] if agent_poses and len(agent_poses) > 0 else None

            # print(agent_pose)

            # img = getImageFromCamera(unwrapped_env, 1, agent_pose)
            img = getImageFromCamera(unwrapped_env, 1, agent_pose, angle=get_360_part(agent_pose[4]))
            # dimg = getDepthImageFromCamera(unwrapped_env, 1, agent_pose,angle=get_360_part(agent_pose[4]))
            # dimgls.append(dimg)
            # if img is not None:
            #     cv.imshow('img',img)
            #     cv.waitKey(1)
            # if obs is not None and len(obs) > 0:
            #     cv.imshow('obs', obs[0])
            #     cv.waitKey(1)
            # if dimg is not None:
            #     cv.imshow('dimg',dimg/dimg.max())
            #     cv.waitKey(1)

            record_frame(video_writers, obs[0], agent_pose, unwrapped_env, resolution)
    finally:
        if enable_record:
            for writer in video_writers.values():
                writer.release()
            print(f'视频已保存到: {output_dir}')
            print(f'共保存 {len(video_writers)} 个视频文件')
        # npls = np.array(dimgls[0])[:,:,0]
        # print(npls.shape)
        # np.save(os.path.join(output_dir, 'depth_images.npy'), npls)
        # print(f'深度图像数据已保存到: {output_dir}/depth_images.npy')
        # np.savetxt('my_data.csv', npls, fmt='%d', delimiter=',')
        env.close()