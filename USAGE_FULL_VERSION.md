# 使用 UnrealZoo_UE5_5_Linux_v1.0.5 完整版本指南

## 1. 环境准备

您的完整版本已下载到：
```
~/tmp/UnrealEnv/UnrealZoo_UE5_5_Linux_v1.0.5/
```

可执行文件路径：
```
UnrealZoo_UE5_5_Linux_v1.0.5/Linux/UnrealZoo_UE5_5/Binaries/Linux/UnrealZoo_UE5_5
```

## 2. 生成场景配置文件

完整版本包含 **100+ 个场景**，您需要为每个想使用的场景生成配置文件。

### 步骤 1：修改 generate_env_config.py

编辑 `generate_env_config.py`，修改默认路径（第11行和第164行）：

```python
# 第11行：设置 UnrealEnv 路径
os.environ['UnrealEnv'] = '/home/ggengx/tmp/UnrealEnv/'

# 第164行：设置可执行文件路径（相对于 UnrealEnv）
parser.add_argument('--env-bin', 
    default='UnrealZoo_UE5_5_Linux_v1.0.5/Linux/UnrealZoo_UE5_5/Binaries/Linux/UnrealZoo_UE5_5', 
    help='The path to the UE binary')
```

### 步骤 2：生成单个场景的配置文件

```bash
cd ~/tmp/unrealzoo-gym

# 为特定场景生成配置（例如：Old_Town）
python generate_env_config.py \
    --env-bin UnrealZoo_UE5_5_Linux_v1.0.5/Linux/UnrealZoo_UE5_5/Binaries/Linux/UnrealZoo_UE5_5 \
    --env-map Old_Town \
    --target_dir gym_unrealcv/envs/setting/Track

# 或者用于 Navigation 任务
python generate_env_config.py \
    --env-bin UnrealZoo_UE5_5_Linux_v1.0.5/Linux/UnrealZoo_UE5_5/Binaries/Linux/UnrealZoo_UE5_5 \
    --env-map Demo_Roof \
    --target_dir gym_unrealcv/envs/setting/Navigation
```

### 步骤 3：批量生成所有场景配置（可选）

```bash
# 生成所有场景的配置（会启动UE并遍历所有场景，耗时较长）
python generate_env_config.py \
    --env-bin UnrealZoo_UE5_5_Linux_v1.0.5/Linux/UnrealZoo_UE5_5/Binaries/Linux/UnrealZoo_UE5_5 \
    --env-map all \
    --target_dir gym_unrealcv/envs/setting/Track
```

## 3. 可用场景列表

完整版本包含 100+ 个场景，部分场景名称包括：

### 常见场景
- `Old_Town` - 老城区
- `MiddleEast` - 中东场景
- `Demo_Roof` - 屋顶城市
- `Map_ChemicalPlant_1` - 化工厂
- `Greek_Island` - 希腊岛屿
- `ContainerYard_Day` / `ContainerYard_Night` - 集装箱场
- `SuburbNeighborhood_Day` / `SuburbNeighborhood_Night` - 郊区社区

### 更多场景
参考 `doc/EnvLists.md` 或运行 `generate_env_config.py --env-map all` 查看完整列表。

## 4. 使用场景运行任务

### 运行 Tracking 任务

```bash
# 使用键盘控制
python example/keyboard_agent.py -e UnrealTrack-Old_Town-ContinuousColor-v0

# 使用随机智能体
python example/random_agent_multi.py -e UnrealTrack-Old_Town-ContinuousColor-v0
```

### 运行 Navigation 任务

```bash
# 使用键盘控制导航
python example/Keyboard_NavigationAgent.py -e UnrealNavigation-Demo_Roof-MixedColor-v0
```

### 环境命名规则

格式：`Unreal{任务}-{场景名}-{动作空间}{观察类型}-v{版本}`

- **任务类型**：`Track`（追踪）、`Navigation`（导航）
- **场景名**：如 `Old_Town`、`MiddleEast` 等
- **动作空间**：`Discrete`（离散）、`Continuous`（连续）、`Mixed`（混合）
- **观察类型**：`Color`、`Depth`、`Rgbd`、`Gray`、`Mask` 等

## 5. 在代码中使用

```python
import gym
import gym_unrealcv

# 创建环境
env = gym.make('UnrealTrack-Old_Town-ContinuousColor-v0')

# 或者使用 Navigation
env = gym.make('UnrealNavigation-Demo_Roof-MixedColor-v0')

# 使用环境
obs = env.reset()
for _ in range(100):
    action = env.action_space.sample()  # 随机动作
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

## 6. 注意事项

1. **首次运行**：首次运行某个场景时，UE 需要加载资源，可能需要较长时间
2. **内存要求**：完整版本场景较大，确保有足够内存（建议 16GB+）
3. **GPU**：建议使用 NVIDIA GPU 以获得更好的性能
4. **配置文件位置**：生成的 JSON 配置文件保存在：
   - Tracking 任务：`gym_unrealcv/envs/setting/Track/`
   - Navigation 任务：`gym_unrealcv/envs/setting/Navigation/`

## 7. 故障排除

### 问题：找不到可执行文件
- 检查路径是否正确：`~/tmp/UnrealEnv/UnrealZoo_UE5_5_Linux_v1.0.5/Linux/UnrealZoo_UE5_5/Binaries/Linux/UnrealZoo_UE5_5`
- 确保文件有执行权限：`chmod +x UnrealZoo_UE5_5`

### 问题：场景加载失败
- 检查场景名称是否正确（区分大小写）
- 查看 UE 日志：`~/tmp/UnrealEnv/UnrealZoo_UE5_5_Linux_v1.0.5/Linux/UnrealZoo_UE5_5/Saved/Logs/`

### 问题：配置文件不存在
- 先运行 `generate_env_config.py` 生成配置文件
- 检查配置文件是否在正确的目录（Track 或 Navigation）

## 8. 快速开始示例

```bash
# 1. 生成 Old_Town 场景的配置
cd ~/tmp/unrealzoo-gym
python generate_env_config.py \
    --env-bin UnrealZoo_UE5_5_Linux_v1.0.5/Linux/UnrealZoo_UE5_5/Binaries/Linux/UnrealZoo_UE5_5 \
    --env-map Old_Town \
    --target_dir gym_unrealcv/envs/setting/Track

# 2. 运行测试
python example/keyboard_agent.py -e UnrealTrack-Old_Town-ContinuousColor-v0
```

祝您使用愉快！


