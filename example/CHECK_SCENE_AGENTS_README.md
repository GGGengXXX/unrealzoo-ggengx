# 场景智能体类型检查工具

## 功能概述

`check_scene_agents.py` 是一个实用工具脚本，用于查看指定场景支持哪些智能体类型及其详细信息。

## 基本用法

### 1. 使用环境ID查看

```bash
python example/check_scene_agents.py --env_id UnrealTrack-MiddleEast-ContinuousColor-v0
```

### 2. 直接指定任务和地图

```bash
python example/check_scene_agents.py --task Track --map MiddleEast
```

### 3. JSON格式输出（便于脚本处理）

```bash
python example/check_scene_agents.py --env_id UnrealTrack-Greek_Island-ContinuousColor-v0 --json
```

### 4. 列出可用场景

```bash
python example/check_scene_agents.py --list
```

## 命令行参数

| 参数 | 简写 | 说明 | 示例 |
|------|------|------|------|
| `--env_id` | `-e` | 环境ID | `UnrealTrack-MiddleEast-ContinuousColor-v0` |
| `--task` | `-t` | 任务类型 | `Track`, `Navigation` |
| `--map` | `-m` | 地图名称 | `MiddleEast`, `Greek_Island` |
| `--list` | `-l` | 列出所有可用场景 | - |
| `--json` | - | JSON格式输出 | - |

## 输出示例

### 人类可读格式

```
============================================================
场景信息: MiddleEast
任务类型: Track
地图名称: MiddleEast
============================================================

可用的智能体类型 (1 种):

  📌 PLAYER
     - 数量: 1
     - 智能体名称:
       [0] BP_Character_C_1
     - 内部导航: 是
     - 缩放: [1, 1, 1]
     - 相机相对位置: [20, 0, 0]
     - 离散动作数: 7
     - 连续动作: 支持

安全起始点数量: 1
重置区域: X[0, 0], Y[0, 0], Z[0, 0]

============================================================

💡 使用建议:
   可以使用以下命令录制该场景:
   python example/multi_camera_recorder.py --agents player --env_id UnrealTrack-MiddleEast-ContinuousColor-v0
```

### JSON格式

```json
{
  "env_name": "MiddleEast",
  "task": "Track",
  "map_name": "MiddleEast",
  "agent_types": [
    "player"
  ],
  "agents": {
    "player": {
      "count": 1,
      "names": [
        "BP_Character_C_1"
      ],
      "internal_nav": true,
      "scale": [1, 1, 1],
      "relative_location": [20, 0, 0]
    }
  }
}
```

## 使用场景

### 场景1：快速查看场景支持的智能体类型

```bash
python example/check_scene_agents.py --env_id UnrealTrack-Greek_Island-ContinuousColor-v0
```

### 场景2：在脚本中获取智能体类型列表

```bash
# 获取JSON输出并解析
python example/check_scene_agents.py --env_id UnrealTrack-Greek_Island-ContinuousColor-v0 --json | jq -r '.agent_types[]'
```

### 场景3：检查场景是否支持特定智能体类型

```bash
# 检查是否支持animal类型
python example/check_scene_agents.py --env_id UnrealTrack-Greek_Island-ContinuousColor-v0 --json | jq -e '.agent_types | contains(["animal"])' > /dev/null && echo "支持animal" || echo "不支持animal"
```

## 常见问题

**Q: 如何知道场景的env_id格式？**  
A: env_id格式为：`Unreal{task}-{MapName}-{ActionSpace}{ObservationType}-v{version}`
   - `task`: Track, Navigation, Rendezvous 等
   - `MapName`: MiddleEast, Greek_Island 等
   - `ActionSpace`: Discrete, Continuous, Mixed
   - `ObservationType`: Color, Depth, Rgbd 等
   - `version`: 0-5

**Q: 如果场景配置文件不存在怎么办？**  
A: 脚本会显示错误信息，提示检查场景名称是否正确。场景配置文件位于 `gym_unrealcv/envs/setting/{task}/{MapName}.json`

**Q: 如何批量检查多个场景？**  
A: 可以编写简单的shell脚本：
```bash
#!/bin/bash
for scene in "MiddleEast" "Greek_Island" "Hospital"; do
    echo "=== $scene ==="
    python example/check_scene_agents.py --task Track --map $scene
    echo
done
```

## 与其他脚本的配合使用

这个工具与 `multi_camera_recorder.py` 配合使用，可以：

1. **先检查场景支持的智能体类型**
   ```bash
   python example/check_scene_agents.py --env_id UnrealTrack-Greek_Island-ContinuousColor-v0
   ```

2. **根据输出结果使用正确的智能体类型进行录制**
   ```bash
   python example/multi_camera_recorder.py --agents player animal --env_id UnrealTrack-Greek_Island-ContinuousColor-v0 --save_video
   ```

这样可以避免因为使用了场景不支持的智能体类型而导致的错误。






















