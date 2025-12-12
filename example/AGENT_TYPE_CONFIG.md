# 智能体类型配置指南

## 概述

`multi_camera_recorder.py` 支持配置不同类型的智能体作为录制主体，包括人类（player）、动物（animal）、无人机（drone）、汽车（car）和摩托车（motorbike）。

## 快速开始

### 1. 只使用人类角色（默认）

```bash
python example/multi_camera_recorder.py \
    -e UnrealTrack-MiddleEast-ContinuousColor-v0 \
    --agents player \
    --save_video
```

### 2. 使用人类和动物

```bash
python example/multi_camera_recorder.py \
    -e UnrealTrack-MiddleEast-ContinuousColor-v0 \
    --agents player animal \
    --save_video
```

### 3. 只使用动物

```bash
python example/multi_camera_recorder.py \
    -e UnrealTrack-MiddleEast-ContinuousColor-v0 \
    --agents animal \
    --save_video
```

## 支持的智能体类型

| 类型 | 说明 | 示例 |
|------|------|------|
| `player` | 人类角色 | 默认类型，支持18种不同外观 |
| `animal` | 动物 | 支持27种不同动物模型 |
| `drone` | 无人机 | 支持飞行控制 |
| `car` | 汽车 | 支持车辆控制 |
| `motorbike` | 摩托车 | 支持摩托车控制 |

## 相机配置中的智能体类型

在相机配置JSON文件中，可以为每个相机指定 `agent_type` 和 `agent_index`：

### 示例：观察人类角色

```json
{
  "name": "player_0_first_person",
  "cam_type": "first_person",
  "agent_index": 0,
  "agent_type": "player"
}
```

### 示例：观察动物

```json
{
  "name": "animal_0_first_person",
  "cam_type": "first_person",
  "agent_index": 0,
  "agent_type": "animal"
}
```

### 示例：同时观察人类和动物

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
      "name": "animal_0_first_person",
      "cam_type": "first_person",
      "agent_index": 0,
      "agent_type": "animal"
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
      "name": "animal_0_third_person",
      "cam_type": "third_person",
      "agent_index": 0,
      "agent_type": "animal",
      "distance": 400.0,
      "height": 100.0,
      "angle": 0.0
    }
  ]
}
```

## 重要说明

1. **索引规则**：`agent_index` 是在指定类型内的索引，不是全局索引。
   - 如果配置了 `--agents player animal`
   - `agent_index: 0, agent_type: "player"` 表示第一个人类
   - `agent_index: 0, agent_type: "animal"` 表示第一个动物
   - `agent_index: 1, agent_type: "player"` 表示第二个人类

2. **类型匹配**：相机配置中的 `agent_type` 必须在 `--agents` 参数中指定，否则无法找到对应的智能体。

3. **默认类型**：如果相机配置中未指定 `agent_type`，默认为 `"player"`。

## 完整示例

### 场景：追踪人类和动物的交互

**命令行：**
```bash
python example/multi_camera_recorder.py \
    -e UnrealTrack-Greek_Island-ContinuousColor-v0 \
    -c camera_config.json \
    --agents player animal \
    --num_agents 4 \
    --save_video \
    --save_images \
    --save_trajectory
```

**相机配置文件 (camera_config.json)：**
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

这将创建：
- 2个人类角色和2个动物
- 5个相机视角（2个第一人称、2个第三人称、1个俯视图）
- 所有相机的视频、图片和轨迹数据

## 常见问题

**Q: 如何知道场景中有哪些类型的智能体？**  
A: 查看场景的JSON配置文件（如 `gym_unrealcv/envs/setting/Track/Greek_Island.json`），其中的 `agents` 字段列出了可用的智能体类型。

**Q: 可以混合使用不同类型的智能体吗？**  
A: 可以！通过 `--agents player animal drone` 可以同时使用多种类型。

**Q: 为什么相机找不到指定的智能体？**  
A: 确保：
1. 相机配置中的 `agent_type` 在 `--agents` 参数中已指定
2. `agent_index` 不超过该类型智能体的数量
3. 场景配置文件中包含该类型的智能体


























