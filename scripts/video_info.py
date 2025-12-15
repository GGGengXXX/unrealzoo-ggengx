#!/usr/bin/env python3
"""
视频文件信息查看工具
用于查看mp4等视频文件的基本信息，包括编码格式、分辨率、帧率等
"""

import argparse
import cv2
import os
import sys
from pathlib import Path


def get_video_info(video_path):
    """
    获取视频文件的详细信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        dict: 包含视频信息的字典
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    try:
        # 获取视频属性
        info = {
            'file_path': os.path.abspath(video_path),
            'file_name': os.path.basename(video_path),
            'file_size': os.path.getsize(video_path),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': 0,  # 将在后面计算
            'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
            'codec': '',  # 将在后面解析
            'format': '',  # 文件格式
        }
        
        # 计算视频时长（秒）
        if info['fps'] > 0:
            info['duration'] = info['frame_count'] / info['fps']
        
        # 解析编码格式（FourCC）
        fourcc_int = info['fourcc']
        fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
        info['codec'] = fourcc_str
        
        # 获取文件格式
        info['format'] = Path(video_path).suffix.lower()
        
        # 格式化文件大小
        size_mb = info['file_size'] / (1024 * 1024)
        info['file_size_mb'] = size_mb
        
        return info
        
    finally:
        cap.release()


def format_duration(seconds):
    """格式化时长显示"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
    else:
        return f"{minutes:02d}:{secs:02d}.{ms:03d}"


def format_size(size_bytes):
    """格式化文件大小显示"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def print_video_info(info, verbose=False):
    """打印视频信息"""
    print("=" * 60)
    print("视频文件信息")
    print("=" * 60)
    print(f"文件路径: {info['file_path']}")
    print(f"文件名: {info['file_name']}")
    print(f"文件格式: {info['format']}")
    print(f"文件大小: {format_size(info['file_size'])} ({info['file_size_mb']:.2f} MB)")
    print()
    print("视频属性:")
    print(f"  分辨率: {info['width']} x {info['height']}")
    print(f"  帧率: {info['fps']:.2f} fps")
    print(f"  总帧数: {info['frame_count']}")
    print(f"  时长: {format_duration(info['duration'])} ({info['duration']:.2f} 秒)")
    print(f"  编码格式 (FourCC): {info['codec']} (0x{info['fourcc']:08X})")
    
    if verbose:
        print()
        print("详细信息:")
        print(f"  FourCC 数值: {info['fourcc']}")
        print(f"  宽: {info['width']} 像素")
        print(f"  高: {info['height']} 像素")
        print(f"  文件大小: {info['file_size']} 字节")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='查看视频文件的基本信息（编码格式、分辨率、帧率等）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s video.mp4
  %(prog)s /path/to/video.mp4
  %(prog)s video.mp4 --verbose
        """
    )
    
    parser.add_argument(
        'video_path',
        type=str,
        help='视频文件路径'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='显示详细信息'
    )
    
    args = parser.parse_args()
    
    try:
        info = get_video_info(args.video_path)
        print_video_info(info, verbose=args.verbose)
        
    except FileNotFoundError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"未知错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
