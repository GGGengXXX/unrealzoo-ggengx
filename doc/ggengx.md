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

# 开发设置

## 深度图的获取

```python
img = unwrapped_env.unrealcv.get_depth(cam_id,show=False)
```

这里返回的是一个 `(h,w,1)` 的矩阵，深度的值可能在 几万到几百不等

在unrealcv的 `api.py` 中可以找到具体的实现

```python
def get_depth(self, cam_id, inverse=False, return_cmd=False, show=False):  # get depth from unrealcv in npy format
        """
        Get the depth image from a camera.

        Args:
            cam_id (int): The camera ID.
            inverse (bool): Whether to inverse the depth. Default is False.
            return_cmd (bool): Whether to return the command instead of executing it. Default is False.
            show (bool): Whether to display the image. Default is False.

        Returns:
            np.ndarray: The depth image.
        """
        cmd = f'vget /camera/{cam_id}/depth npy'
        if return_cmd:
            return cmd
        res = self.client.request(cmd)
        depth = self.decoder.decode_depth(res, inverse)
        if show:
            cv2.imshow('image', depth/depth.max())  # normalize the depth image
            cv2.waitKey(10)
        return depth
```

可以通过除以 `numpy.max()` 进行归一化，得到深度图进行imshow

## 多视角获取

首先怎么获取图片，可以调用`api.py` 中的 `get_image` 函数来实现

源码：

```python
def get_image(self, cam_id, viewmode, mode='bmp', return_cmd=False, show=False, inverse=False):
        """
        Get an image from a camera.

        Args:
            cam_id (int): The camera ID.
            viewmode (str): The view mode (e.g., 'lit', 'normal', 'object_mask', 'depth').
            mode (str): The image format (e.g., 'bmp', 'png', 'npy'). Default is 'bmp'.
            return_cmd (bool): Whether to return the command instead of executing it. Default is False.
            show (bool): Whether to display the image. Default is False.
            inverse (bool): Whether to inverse the depth. Default is False.
        Returns:
            np.ndarray: The image.
        """
        if viewmode == 'depth':
            return self.get_depth(cam_id, return_cmd=return_cmd, show=show)
        cmd = f'vget /camera/{cam_id}/{viewmode} {mode}'
        if return_cmd:
            return cmd
        image = self.decoder.decode_img(self.client.request(cmd), mode, inverse)
        if show:
            cv2.imshow('image_'+viewmode, image)
            cv2.waitKey(1)
        return image
```

只需要这样就可以获取图片

```python
img = unwrapped_env.unrealcv.get_image(cam_id, 'lit', 'png')
```

关键在于角度的计算

这里有一个 ai 写好的函数，只要设置水瓶方位角即可，具体的，我们把每一次人物的朝向传入函数

即可得到追随agent的固定角度的图片

```python

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
```

## agent外观的改变

看到 `base_env.py` 中的一个实现

```python
# 随机分配外观
    def random_app(self):
        """
        Randomly assign an appearance to each agent in the player list based on their category.

        The appearance is selected from a predefined range of IDs for each category.

        Categories:
            - player: IDs from 1 to 18
            - animal: IDs from 0 to 26
        """
        app_map = {
            'player': range(1, 19),
            'animal': range(0, 27),
            'drone':range(0,1)
        }
        for obj in self.player_list:
            category = self.agents[obj]['agent_type']
            if category not in app_map.keys():
                continue
            app_id = np.random.choice(app_map[category])
            self.unrealcv.set_appearance(obj, app_id)
```

是的，你只要轻松的一句话就可以改变外观

```python
self.unrealcv.set_appearance(obj, app_id)
```

或者通过一句话，为代码中的所有agent随机外观

```python
env.unwrapped.random_app()
```

## 调分辨率

- 包括两个部分，一个是采样的时候，一个是写入视频的时候，当然，我们可以采取同样的分辨率即可

- 具体来说，调用ConfigUEWrapper进行调节

  - ```python
    env = configUE.ConfigUEWrapper(env, offscreen=True, resolution=resolution
    ```

- 注意分辨率调大了会影响运行的速度

## 调节运行的速度

有两个值需要同步的调整，那就是 `fps` 和 `time_dilation` 前者是调节视频写入的帧率，后者调节游戏运行的速度。

- 当游戏运行正常，但是写入视频的帧率大的时候，视频就会不连贯。而且动作很快。原因在于，由于机器的限制，采样的频率是不变的，而游戏运行正常，导致采样其实比较稀疏。
- 当游戏运行慢的时候，视频写入帧率同时提高。这是采样频率不变，可以对慢动作进行充分的采样。使我们想要达到的效果

在 `example/config_set.py` 中设置了两套配置

```python
    def get_high_fps_1920x1080_config(self):
        return 50, [1920, 1280], 60.0

    def get_normal_config(self):
        return 1, [240,240], 5
```

# 需求分析

## 人物外观的设置

需要方便的设置

使用以下的某一种agent

- player
- animal
- car

同时可以随机外观或者指定外观

## 数据导出的规范化

需要同时导出深度图和图像，而且要有时间尺度

可以考虑numpy的导出格式 npy

同时要方便的加载npy

## 方便的设置导出的帧率和分辨率

包括运行的流畅度，视频的速度等等参数

能够轻易调节

## 多个视角的调整

对于不同的主体，高矮胖瘦不一，可能需要微微调整相机的位置

从而保证达到好的拍摄效果
