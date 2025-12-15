import numpy as np
import cv2 as cv
import os
import sys

# =================================================================
# 1. 视频配置参数
# =================================================================
# 随机视频文件的路径
OUTPUT_VIDEO_PATH = 'random_demo_video.mp4'
# 帧率 (Frames Per Second)，您提供的例子中是 60 fps，这里也用 30 fps
FPS = 30.0 
# 视频分辨率 (宽度 x 高度)
WIDTH = 640
HEIGHT = 480
# 生成的帧数 (例如，生成 3 秒钟的视频)
NUM_FRAMES = int(FPS * 3) 
# 分辨率元组
RESOLUTION = (WIDTH, HEIGHT)


def create_video_writer(path, fps, resolution):
    """
    创建视频写入器，尝试多个编码格式。
    
    Args:
        path (str): 输出文件路径
        fps (float): 帧率
        resolution (tuple): (宽度, 高度)
        
    Returns:
        cv.VideoWriter or None: 成功创建的写入器对象，否则为 None
    """
    # 优先尝试 avc1 和 H264 (H.264)，以保证在 VS Code/网页上的预览兼容性
    # 其次尝试 XVID，以确保在大多数环境下都能成功写入（兼容性最高）
    codecs = ['avc1', 'H264','mp4v', 'XVID'] 
    
    # 注意：确保 path 路径的后缀与编码器匹配，avc1/H264 通常搭配 .mp4
    if not path.lower().endswith('.mp4'):
         print(f"⚠️ 警告: 推荐将输出路径后缀改为 '.mp4' 以配合 H.264 编码器。")

    print(f"尝试使用 {fps:.1f} fps 和 {resolution[0]}x{resolution[1]} 分辨率创建写入器...")
    
    for codec in codecs:
        try:
            fourcc = cv.VideoWriter_fourcc(*codec)
            writer = cv.VideoWriter(path, fourcc, fps, resolution)
            
            if writer.isOpened():
                print(f"✅ 成功使用 FourCC '{codec}' (0x{fourcc:08X}) 创建视频写入器。")
                return writer
            else:
                # 即使 isOpened() 失败，也可能因为环境或依赖缺失，继续尝试下一个
                print(f"❌ FourCC '{codec}' 无法打开写入器 (可能是环境缺少依赖)。尝试下一个...")
                
        except Exception as e:
            # 捕获异常，例如 FourCC 字符串解析错误（通常不会发生）
            print(f"异常: 尝试 FourCC '{codec}' 时发生错误: {e}")
            continue
            
    return None

def generate_random_video(output_path, fps, resolution, num_frames):
    """
    生成随机帧视频的主函数。
    """
    writer = create_video_writer(output_path, fps, resolution)
    
    if writer is None:
        print("\n🚫 错误: 无法创建任何可用的视频写入器。请检查 OpenCV 依赖和环境配置。")
        return

    print(f"\n开始生成 {num_frames} 帧随机视频...")
    
    try:
        width, height = resolution
        for i in range(num_frames):
            # 创建一个随机的彩色帧 (3通道: BGR)
            # np.uint8 是 OpenCv 图像的标准数据类型 (0-255)
            frame = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
            
            # (可选) 在帧上添加文字，方便查看帧数
            cv.putText(frame, 
                       f"Frame: {i+1}/{num_frames}", 
                       (10, 30), 
                       cv.FONT_HERSHEY_SIMPLEX, 
                       1, 
                       (255, 255, 255), # 白色字体
                       2, 
                       cv.LINE_AA)
            
            # 写入帧
            writer.write(frame)
            
            # 打印进度 (每 1/10 进度打印一次)
            if (i + 1) % (num_frames // 10) == 0 or i == num_frames - 1:
                print(f"   -> 已写入 {i+1} 帧...")

        print("\n🎉 视频生成完成！")
        print(f"文件位置: {os.path.abspath(output_path)}")
        
    except Exception as e:
        print(f"写入视频时发生错误: {e}")
        
    finally:
        # 释放写入器资源
        writer.release()
        print("写入器已释放。")


if __name__ == '__main__':
    # 检查是否安装了 OpenCV
    try:
        import cv2
    except ImportError:
        print("=" * 60)
        print("❌ 错误: 缺少 'opencv-python' 库。")
        print("请在命令行中运行: pip install opencv-python")
        print("=" * 60)
        sys.exit(1)
        
    # 运行生成函数
    generate_random_video(OUTPUT_VIDEO_PATH, FPS, RESOLUTION, NUM_FRAMES)