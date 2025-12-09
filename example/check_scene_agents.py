# ============================================================================
# 场景智能体类型检查脚本
# 功能：查看指定场景支持哪些智能体类型及其详细信息
# ============================================================================

import os
import argparse
import json
from gym_unrealcv.envs.utils import misc

def parse_env_id(env_id):
    """
    从env_id解析task和map名称
    
    Args:
        env_id: 环境ID，格式如 'UnrealTrack-MiddleEast-ContinuousColor-v0'
    
    Returns:
        tuple: (task, map_name) 或 (None, None) 如果解析失败
    """
    # 格式: Unreal{task}-{MapName}-{ActionSpace}{ObservationType}-v{version}
    parts = env_id.split('-')
    if len(parts) < 2:
        return None, None
    
    task = parts[0].replace('Unreal', '')  # 去掉'Unreal'前缀
    map_name = parts[1]
    
    return task, map_name

def check_scene_agents(env_id=None, task=None, map_name=None):
    """
    检查场景中可用的智能体类型
    
    Args:
        env_id: 环境ID（可选）
        task: 任务类型（可选，如 'Track', 'Navigation'）
        map_name: 地图名称（可选，如 'MiddleEast', 'Greek_Island'）
    
    Returns:
        dict: 包含场景信息的字典，如果失败返回None
    """
    # 如果提供了env_id，从中解析task和map_name
    if env_id:
        task, map_name = parse_env_id(env_id)
        if not task or not map_name:
            print(f"❌ 无法解析环境ID: {env_id}")
            print(f"   期望格式: Unreal{{task}}-{{MapName}}-{{ActionSpace}}{{ObservationType}}-v{{version}}")
            print(f"   例如: UnrealTrack-MiddleEast-ContinuousColor-v0")
            return None
    
    if not task or not map_name:
        print("❌ 错误：必须提供 env_id 或同时提供 task 和 map_name")
        return None
    
    # 构建配置文件路径
    setting_file = os.path.join(task, f'{map_name}.json')
    
    try:
        # 加载配置文件
        setting = misc.load_env_setting(setting_file)
        
        # 获取场景信息
        scene_info = {
            'env_name': setting.get('env_name', map_name),
            'task': task,
            'map_name': map_name,
            'agents': setting.get('agents', {}),
            'safe_start': setting.get('safe_start', []),
            'reset_area': setting.get('reset_area', []),
        }
        
        return scene_info
    except FileNotFoundError:
        print(f"❌ 错误：找不到场景配置文件: {setting_file}")
        print(f"   请检查场景名称是否正确")
        return None
    except Exception as e:
        print(f"❌ 错误：加载场景配置失败: {e}")
        return None

def print_agent_info(scene_info):
    """
    打印智能体信息
    
    Args:
        scene_info: 场景信息字典
    """
    print(f"\n{'='*60}")
    print(f"场景信息: {scene_info['env_name']}")
    print(f"任务类型: {scene_info['task']}")
    print(f"地图名称: {scene_info['map_name']}")
    print(f"{'='*60}\n")
    
    agents = scene_info['agents']
    
    if not agents:
        print("⚠️  该场景没有配置任何智能体")
        return
    
    print(f"可用的智能体类型 ({len(agents)} 种):\n")
    
    # 按类型显示智能体信息
    for agent_type, agent_config in agents.items():
        names = agent_config.get('name', [])
        num_agents = len(names)
        
        print(f"  📌 {agent_type.upper()}")
        print(f"     - 数量: {num_agents}")
        
        if num_agents > 0:
            print(f"     - 智能体名称:")
            for i, name in enumerate(names[:5]):  # 最多显示5个
                print(f"       [{i}] {name}")
            if num_agents > 5:
                print(f"       ... 还有 {num_agents - 5} 个")
        
        # 显示配置信息
        internal_nav = agent_config.get('internal_nav', False)
        scale = agent_config.get('scale', [1, 1, 1])
        relative_location = agent_config.get('relative_location', [0, 0, 0])
        
        print(f"     - 内部导航: {'是' if internal_nav else '否'}")
        print(f"     - 缩放: {scale}")
        print(f"     - 相机相对位置: {relative_location}")
        
        # 显示动作空间信息
        if 'move_action' in agent_config:
            move_actions = agent_config['move_action']
            print(f"     - 离散动作数: {len(move_actions)}")
        
        if 'move_action_continuous' in agent_config:
            print(f"     - 连续动作: 支持")
        
        print()  # 空行分隔
    
    # 显示安全起始点信息
    safe_start = scene_info.get('safe_start', [])
    if safe_start:
        print(f"安全起始点数量: {len(safe_start)}")
    
    # 显示重置区域
    reset_area = scene_info.get('reset_area', [])
    if reset_area and len(reset_area) >= 6:
        print(f"重置区域: X[{reset_area[0]}, {reset_area[1]}], "
              f"Y[{reset_area[2]}, {reset_area[3]}], "
              f"Z[{reset_area[4]}, {reset_area[5]}]")
    
    print(f"\n{'='*60}")

def list_available_scenes():
    """
    列出所有可用的场景
    """
    import gym_unrealcv
    
    # 从__init__.py中获取maps列表
    try:
        from gym_unrealcv import __init__ as gym_init
        # 尝试读取maps列表
        maps = [
            'track_train', 'Greek_Island', 'MiddleEast', 'Hospital', 'Old_Town',
            'ContainerYard_Night', 'SuburbNeighborhood_Night', 'AbandonedDistrict', 'FlexibleRoom'
        ]
        
        print(f"\n可用的场景列表（部分）:")
        print(f"  提示：使用 --env_id 参数查看具体场景的智能体类型")
        print(f"  例如：python example/check_scene_agents.py --env_id UnrealTrack-Greek_Island-ContinuousColor-v0\n")
        
    except:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='查看场景支持的智能体类型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用环境ID
  python example/check_scene_agents.py --env_id UnrealTrack-MiddleEast-ContinuousColor-v0
  
  # 直接指定任务和地图
  python example/check_scene_agents.py --task Track --map MiddleEast
  
  # 列出所有场景
  python example/check_scene_agents.py --list
        """
    )
    
    parser.add_argument("--env_id", "-e", type=str, default=None,
                        help='环境ID，格式: Unreal{task}-{MapName}-{ActionSpace}{ObservationType}-v{version}')
    parser.add_argument("--task", "-t", type=str, default=None,
                        help='任务类型（如 Track, Navigation）')
    parser.add_argument("--map", "-m", type=str, default=None,
                        help='地图名称（如 MiddleEast, Greek_Island）')
    parser.add_argument("--list", "-l", action='store_true',
                        help='列出所有可用场景')
    parser.add_argument("--json", action='store_true',
                        help='以JSON格式输出（便于脚本处理）')
    
    args = parser.parse_args()
    
    # 如果请求列出场景
    if args.list:
        list_available_scenes()
        exit(0)
    
    # 检查场景智能体
    scene_info = check_scene_agents(
        env_id=args.env_id,
        task=args.task,
        map_name=args.map
    )
    
    if scene_info:
        if args.json:
            # JSON格式输出
            output = {
                'env_name': scene_info['env_name'],
                'task': scene_info['task'],
                'map_name': scene_info['map_name'],
                'agent_types': list(scene_info['agents'].keys()),
                'agents': {}
            }
            
            for agent_type, agent_config in scene_info['agents'].items():
                output['agents'][agent_type] = {
                    'count': len(agent_config.get('name', [])),
                    'names': agent_config.get('name', []),
                    'internal_nav': agent_config.get('internal_nav', False),
                    'scale': agent_config.get('scale', [1, 1, 1]),
                    'relative_location': agent_config.get('relative_location', [0, 0, 0])
                }
            
            print(json.dumps(output, indent=2, ensure_ascii=False))
        else:
            # 人类可读格式输出
            print_agent_info(scene_info)
            
            # 给出使用建议
            agent_types = list(scene_info['agents'].keys())
            if agent_types:
                # 构建默认env_id
                default_env_id = f"Unreal{scene_info['task']}-{scene_info['map_name']}-ContinuousColor-v0"
                env_id_to_use = args.env_id if args.env_id else default_env_id
                
                print(f"\n💡 使用建议:")
                print(f"   可以使用以下命令录制该场景:")
                if len(agent_types) == 1:
                    print(f"   python example/multi_camera_recorder.py --agents {agent_types[0]} --env_id {env_id_to_use}")
                else:
                    print(f"   # 使用单个类型")
                    for agent_type in agent_types:
                        print(f"   python example/multi_camera_recorder.py --agents {agent_type} --env_id {env_id_to_use}")
                    print(f"   # 或使用多种类型")
                    print(f"   python example/multi_camera_recorder.py --agents {' '.join(agent_types)} --env_id {env_id_to_use}")
    else:
        exit(1)

