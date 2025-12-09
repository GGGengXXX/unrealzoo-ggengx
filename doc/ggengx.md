## `env-id` 格式
Unreal{task}-{MapName}-{ActionSpace}{ObservationType}-v{version}

## 添加新的 Unreal 环境

本节将逐步介绍如何在 gym-unrealzoo 中添加新环境以完成交互任务。

### 1. 下载/打包 UE 二进制文件
下载或打包一个集成了 UnrealCV Server 的 UE 二进制文件，并将其移动到 `UnrealEnv` 文件夹（这是二进制文件的默认位置）。文件夹结构如下：

```
gym-unrealcv/  
|-- docs/                  
|-- example/                
|-- gym_unrealcv/              
|   |-- envs/    
|   |   |-- agent/     
|   |   |-- UnrealEnv/                    # 二进制文件默认位置
|   |   |   |-- Collection_WinNoEditor/   # 二进制文件夹
|   |-- setting/
|   |   |-- env_config/                   # 环境配置 json 文件位置  
...
generate_env_config.py                    # 生成环境配置 json 文件
...
```

### 2. 生成配置文件
运行 **generate_env_config.py** 自动生成并存储所需地图的配置 JSON 文件：

```bash
python generate_env_config.py --env-bin {binary relative path} --env-map {map name}  
# binary relative path : 相对于 UnrealEnv 文件夹的可执行文件路径
# map name: 用户想要运行的地图名称

# 示例:
python generate_env_config.py --env-bin Collection_WinNoEditor\\WindowsNoEditor\\Collection\\Binaries\\Win64\\Collection.exe --env-map track_train
```

**如何获取可用的地图名称？**

有以下几种方法可以获取可用的地图名称：

1. **查看代码中预定义的地图列表**：
   - 查看 `gym_unrealcv/__init__.py` 文件中的 `maps` 列表（第99-116行），其中包含了所有预定义的地图名称

2. **查看文档**：
   - 查看 `doc/EnvLists.md` 文件，其中列出了所有可用的地图名称
   - 查看 `README.md` 中的 "Available Map Name in Exemplar Binary" 部分，了解示例二进制文件中的地图

3. **查看二进制文件文档**：
   - 如果您使用的是官方提供的二进制文件，请查看其文档说明，了解该二进制文件包含哪些地图
   - 常见的地图名称包括：`track_train`、`Greek_Island`、`MiddleEast`、`Old_Town`、`ContainerYard_Day`、`ContainerYard_Night` 等

4. **尝试运行并查看错误**：
   - 如果地图名称不正确，运行 `generate_env_config.py` 时会报错，您可以根据错误信息调整地图名称

5. **查看已生成的配置文件**：
   - 查看 `gym_unrealcv/envs/setting/` 目录下已有的 JSON 配置文件，文件名即为地图名称（不含 `.json` 扩展名）

**注意**：地图名称必须与 Unreal Engine 二进制文件中实际包含的地图名称完全匹配（区分大小写）。

### 3. 创建环境类文件
在 ```/gym-unrealcv/gym_unrealcv/envs``` 目录下创建一个新的 Python 文件，并在其中编写您的环境类。您可以继承 [base_env.py](./gym_unrealcv/envs/base_env.py) 中的基类，它提供了 gym 环境的基本功能。您也可以参考同一文件夹中的现有环境以获取更多详细信息。**注意：您需要在新环境中编写自己的奖励函数。**

### 4. 导入环境类
在集合的 ```__init__.py``` 文件中导入您的环境。该文件位于 ```/gym-unrealcv/gym_unrealcv/envs/__init__.py```。添加 ```from gym_unrealcv.envs.{your_env_file} import {Your_Env_Class}``` 到该文件中。

### 5. 注册环境
在 ```gym-unrealcv/gym_unrealcv/__init__.py``` 中注册您的环境。

我们预定义了命名规则来定义不同的环境及其对应的任务接口，建议您遵循此规则来命名您的环境。命名规则如下：

```Unreal{task}-{MapName}-{ActionSpace}{ObservationType}-v{version}```

- ```{task}```: 任务名称，我们目前支持：```Track```（追踪）、```Navigation```（导航）、```Rendezvous```（会合）。
- ```{MapName}```: 您想要运行的地图名称，如 ```track_train```、```Greek_Island``` 等。
- ```{ActionSpace}```: 智能体的动作空间，```Discrete```（离散）、```Continuous```（连续）、```Mixed```（混合）。（只有 Mixed 类型支持交互动作）
- ```{ObservationType}```: 智能体的观察类型，```Color```（彩色）、```Depth```（深度）、```Rgbd```（RGBD）、```Gray```（灰度）、```CG```、```Mask```（掩码）、```Pose```（位姿）、```MaskDepth```、```ColorMask```。
- ```{version}```: 版本号，在 ```track_train``` 地图上，```0-5``` 表示不同的增强因子（光照、障碍物、布局、纹理）。

### 6. 测试环境
通过运行随机智能体来测试您的环境：

```bash
python example/random_agent.py -e YOUR_ENV_NAME
```

如果您的环境是为多智能体设计的，可以运行以下命令：

```bash
python example/random_agent_multi.py -e YOUR_ENV_NAME
```
