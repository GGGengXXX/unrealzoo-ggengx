# 视频信息查看工具使用说明

## 功能
`video_info.py` 用于查看mp4等视频文件的基本信息，包括：
- 文件路径和大小
- 分辨率（宽x高）
- 帧率（FPS）
- 总帧数和时长
- 编码格式（FourCC）

## 使用方法

### 基本用法
```bash
# 查看视频信息
python3 scripts/video_info.py <视频文件路径>

# 示例：查看recordings目录下的视频
python3 scripts/video_info.py recordings/keyboard_20251215_212718/first_person.mp4
```

### 详细模式
```bash
# 显示更详细的信息
python3 scripts/video_info.py <视频文件路径> --verbose
# 或
python3 scripts/video_info.py <视频文件路径> -v
```

### 使用绝对路径
```bash
python3 scripts/video_info.py /home/ggengx/tmp/unrealzoo-gym/recordings/keyboard_20251215_212718/third_person_0.mp4
```

### 查看帮助
```bash
python3 scripts/video_info.py --help
```

## 输出示例

```
============================================================
视频文件信息
============================================================
文件路径: /path/to/video.mp4
文件名: video.mp4
文件格式: .mp4
文件大小: 15.23 MB (15.23 MB)

视频属性:
  分辨率: 480 x 480
  帧率: 60.00 fps
  总帧数: 3600
  时长: 01:00:00.000 (60.00 秒)
  编码格式 (FourCC): H264 (0x34363248)
============================================================
```

## 注意事项
- 需要安装 opencv-python: `pip install opencv-python`
- 支持常见的视频格式：mp4, avi, mov, mkv 等
- 如果视频文件无法打开，会显示错误信息
