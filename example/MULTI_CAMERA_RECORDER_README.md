# 多相机录制脚本使用说明

## 功能概述

`multi_camera_recorder.py` 是一个功能强大的录制脚本，支持：

1. **多相机配置**：可以同时配置多个相机，包括：
   - 第一人称视角（绑定到智能体）
   - 第三人称跟随视角（从不同角度跟随智能体）
   - 固定位置相机（场景中的固定观察点）
   - 俯视图相机（从上方观察整个场景）

2. **场景选择**：通过 `env_id` 参数选择不同的 Unreal 场景

3. **数据导出**：
   - 视频文件（每个相机独立视频）
   - 图片序列（每帧保存为PNG）
   - 相机位姿序列（位置和旋转的JSON文件）

## 基本用法

### 1. 使用默认相机配置

```bash
# 使用默认配置（2个智能体的第一人称、第三人称和俯视图）
python example/multi_camera_recorder.py \
    -e UnrealTrack-MiddleEast-ContinuousColor-v0 \
    -o ./recordings \
    --save_video \
    --save_images \
    --save_trajectory \
    --num_agents 2
```

### 2. 使用自定义相机配置文件

```bash
# 使用自定义相机配置
python example/multi_camera_recorder.py \
    -e UnrealTrack-Greek_Island-ContinuousColor-v0 \
    -o ./recordings \
    -c example/camera_config_example.json \
    --save_video \
    --save_images \
    --save_trajectory
```

## 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-e, --env_id` | 环境ID（场景名称） | `UnrealTrack-MiddleEast-ContinuousColor-v0` |
| `-o, --output_dir` | 输出目录 | `./recordings` |
| `-c, --camera_config` | 相机配置文件路径（JSON） | 无（使用默认配置） |
| `-s, --save_images` | 保存每一帧的图片 | False |
| `-v, --save_video` | 保存视频文件 | False |
| `--episodes` | 录制的episode数量 | 1 |
| `--num_agents` | 智能体数量（仅默认配置时使用） | 2 |
| `--save_trajectory` | 保存相机轨迹 | False |
| `--resolution` | 视频分辨率 [width height] | 640 480 |
| `--fps` | 视频帧率 | 30.0 |
| `--agents` | 智能体类型列表 | `['player']` |

### 智能体类型配置

通过 `--agents` 参数可以指定场景中使用的智能体类型，支持的类型包括：
- `player`: 人类角色（默认）
- `animal`: 动物
- `drone`: 无人机
- `car`: 汽车
- `motorbike`: 摩托车

**示例：**

```bash
# 只使用人类角色
python example/multi_camera_recorder.py \
    -e UnrealTrack-MiddleEast-ContinuousColor-v0 \
    --agents player \
    --save_video

# 使用人类和动物
python example/multi_camera_recorder.py \
    -e UnrealTrack-MiddleEast-ContinuousColor-v0 \
    --agents player animal \
    --save_video

# 使用多种类型
python example/multi_camera_recorder.py \
    -e UnrealTrack-MiddleEast-ContinuousColor-v0 \
    --agents player animal drone \
    --save_video
```

## 相机配置文件格式

相机配置文件是一个JSON文件，包含一个 `cameras` 数组。每个相机配置包含以下字段：

### 通用字段
- `name`: 相机名称（用于文件命名）
- `cam_type`: 相机类型（`first_person`, `third_person`, `fixed`, `topview`）

### 第一人称相机 (`first_person`)
```json
{
  "name": "player_0_first_person",
  "cam_type": "first_person",
  "agent_index": 0,
  "agent_type": "player"  // 可选，默认为"player"。可选值：player, animal, drone, car, motorbike
}
```

**注意：** `agent_index` 是在指定类型内的索引。例如，如果配置了 `--agents player animal`，那么：
- `agent_index: 0, agent_type: "player"` 表示第一个人类角色
- `agent_index: 0, agent_type: "animal"` 表示第一个动物
- `agent_index: 1, agent_type: "player"` 表示第二个人类角色

### 第三人称相机 (`third_person`)
```json
{
  "name": "player_0_third_person_back",
  "cam_type": "third_person",
  "agent_index": 0,
  "agent_type": "player",  // 可选，默认为"player"
  "distance": 300.0,       // 距离智能体的距离（单位）
  "height": 150.0,         // 相对智能体的高度偏移（单位）
  "angle": 0.0             // 角度：0=后方，90=右侧，-90=左侧，180=前方
}
```

### 固定位置相机 (`fixed`)
```json
{
  "name": "fixed_camera_1",
  "cam_type": "fixed",
  "location": [0.0, 0.0, 500.0],      // 世界坐标位置 [x, y, z]
  "rotation": [0.0, -45.0, 0.0]       // 旋转角度 [roll, pitch, yaw]
}
```

### 俯视图相机 (`topview`)
```json
{
  "name": "topview",
  "cam_type": "topview",
  "location": [0.0, 0.0, 0.0],       // 场景中心位置（会在运行时自动更新）
  "height": 1500.0                    // 俯视高度（单位）
}
```

## 输出文件结构

```
recordings/
├── episode_000/
│   ├── info.json                          # Episode信息
│   ├── camera_trajectories.json          # 相机轨迹（如果启用）
│   ├── agent_0_first_person/             # 第一人称图片序列
│   │   ├── agent_0_first_person_step_00000.png
│   │   └── ...
│   ├── agent_0_third_person_back/        # 第三人称图片序列
│   │   └── ...
│   └── ...
├── agent_0_first_person.mp4              # 第一人称视频
├── agent_0_third_person_back.mp4         # 第三人称视频
├── topview.mp4                           # 俯视图视频
└── camera_config.json                    # 使用的相机配置（供参考）
```

## 示例场景

### 示例1：多角度观察单个智能体（人类）

```json
{
  "cameras": [
    {
      "name": "player_first_person",
      "cam_type": "first_person",
      "agent_index": 0,
      "agent_type": "player"
    },
    {
      "name": "player_third_person_back",
      "cam_type": "third_person",
      "agent_index": 0,
      "agent_type": "player",
      "distance": 300.0,
      "height": 150.0,
      "angle": 0.0
    },
    {
      "name": "player_third_person_right",
      "cam_type": "third_person",
      "agent_index": 0,
      "agent_type": "player",
      "distance": 300.0,
      "height": 150.0,
      "angle": 90.0
    },
    {
      "name": "topview",
      "cam_type": "topview",
      "location": [0.0, 0.0, 0.0],
      "height": 1500.0
    }
  ]
}
```

### 示例2：同时观察人类和动物

```json
{
  "cameras": [
    {
      "name": "player_0_first_person",
      "cam_type": "first_person",
      "agent_index": 0,
      "agent_type": "player"
    },
    {
      "name": "player_0_third_person",
      "cam_type": "third_person",
      "agent_index": 0,
      "agent_type": "player",
      "distance": 300.0,
      "height": 150.0,
      "angle": 0.0
    },
    {
      "name": "animal_0_first_person",
      "cam_type": "first_person",
      "agent_index": 0,
      "agent_type": "animal"
    },
    {
      "name": "animal_0_third_person",
      "cam_type": "third_person",
      "agent_index": 0,
      "agent_type": "animal",
      "distance": 400.0,
      "height": 100.0,
      "angle": 0.0
    },
    {
      "name": "topview",
      "cam_type": "topview",
      "location": [0.0, 0.0, 0.0],
      "height": 1500.0
    }
  ]
}
```

使用时需要指定智能体类型：
```bash
python example/multi_camera_recorder.py \
    -e UnrealTrack-MiddleEast-ContinuousColor-v0 \
    -c camera_config.json \
    --agents player animal \
    --save_video
```

### 示例3：固定观察点 + 智能体视角

```json
{
  "cameras": [
    {
      "name": "player_0_first_person",
      "cam_type": "first_person",
      "agent_index": 0,
      "agent_type": "player"
    },
    {
      "name": "fixed_overview",
      "cam_type": "fixed",
      "location": [0.0, 0.0, 800.0],
      "rotation": [0.0, -60.0, 0.0]
    },
    {
      "name": "fixed_side_view",
      "cam_type": "fixed",
      "location": [1000.0, 0.0, 500.0],
      "rotation": [0.0, -30.0, 90.0]
    }
  ]
}
```

## 注意事项

1. **相机数量限制**：实际可用的相机数量取决于环境配置和硬件性能。建议同时使用的相机数量不超过10个。

2. **固定位置相机**：固定位置相机需要环境支持额外的相机ID。如果环境不支持，可能需要修改环境配置。

3. **第三人称相机角度**：
   - `angle = 0`: 后方视角
   - `angle = 90`: 右侧视角
   - `angle = -90`: 左侧视角
   - `angle = 180`: 前方视角

4. **性能考虑**：
   - 同时录制多个相机会影响性能
   - 建议根据需求选择必要的相机
   - 如果只需要视频，可以关闭 `--save_images` 选项

5. **场景选择**：确保选择的 `env_id` 对应的场景已正确配置并可用。

## 与 record_video.py 的区别

| 特性 | record_video.py | multi_camera_recorder.py |
|------|----------------|-------------------------|
| 相机配置 | 硬编码（第一人称、第三人称、俯视图） | 灵活的JSON配置文件 |
| 场景选择 | 支持 | 支持 |
| 多相机支持 | 有限（每个智能体固定视角） | 完全支持（任意配置） |
| 固定位置相机 | 不支持 | 支持 |
| 配置灵活性 | 低 | 高 |

## 故障排除

1. **相机无法获取图像**：
   - 检查相机配置是否正确
   - 确认智能体索引是否有效
   - 查看控制台警告信息

2. **视频文件损坏**：
   - 确保有足够的磁盘空间
   - 检查视频编码器是否支持（默认使用mp4v）

3. **性能问题**：
   - 减少相机数量
   - 降低分辨率
   - 关闭图片保存（仅保存视频）

