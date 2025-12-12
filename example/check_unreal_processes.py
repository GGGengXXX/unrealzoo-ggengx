#!/usr/bin/env python3
"""
检查和管理 Unreal 进程的辅助脚本
用于诊断 UnrealCV 连接问题
"""
import subprocess
import sys
import argparse

def check_unreal_processes():
    """检查当前运行的 Unreal 进程"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        unreal_processes = []
        for line in lines:
            if 'UnrealZoo' in line or 'Unreal' in line:
                unreal_processes.append(line)
        
        if unreal_processes:
            print(f"发现 {len(unreal_processes)} 个 Unreal 相关进程:")
            print("-" * 80)
            for i, proc in enumerate(unreal_processes, 1):
                print(f"{i}. {proc}")
            print("-" * 80)
            return unreal_processes
        else:
            print("未发现运行中的 Unreal 进程")
            return []
    except Exception as e:
        print(f"检查进程时出错: {e}")
        return []

def kill_unreal_processes(force=False):
    """终止所有 Unreal 进程"""
    processes = check_unreal_processes()
    if not processes:
        print("没有需要终止的进程")
        return
    
    if not force:
        response = input(f"\n确定要终止这 {len(processes)} 个进程吗? (y/N): ")
        if response.lower() != 'y':
            print("已取消")
            return
    
    try:
        # 提取进程ID
        pids = []
        for proc in processes:
            parts = proc.split()
            if len(parts) > 1:
                try:
                    pid = int(parts[1])
                    pids.append(pid)
                except ValueError:
                    pass
        
        if pids:
            print(f"\n正在终止进程: {pids}")
            for pid in pids:
                try:
                    subprocess.run(['kill', str(pid)], check=True)
                    print(f"  已终止进程 {pid}")
                except subprocess.CalledProcessError as e:
                    print(f"  终止进程 {pid} 失败: {e}")
            
            # 等待一下，然后检查是否还有进程
            import time
            time.sleep(2)
            remaining = check_unreal_processes()
            if remaining:
                print(f"\n警告: 仍有 {len(remaining)} 个进程未终止，可能需要使用 kill -9")
        else:
            print("无法提取进程ID")
    except Exception as e:
        print(f"终止进程时出错: {e}")

def check_port(port=9090):
    """检查端口是否被占用"""
    try:
        # 尝试使用 netstat 或 ss
        for cmd in [['netstat', '-tuln'], ['ss', '-tuln']]:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
                if str(port) in result.stdout:
                    print(f"端口 {port} 已被占用:")
                    for line in result.stdout.split('\n'):
                        if str(port) in line:
                            print(f"  {line}")
                    return True
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
                continue
        print(f"端口 {port} 未被占用（或无法检查）")
        return False
    except Exception as e:
        print(f"检查端口时出错: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='检查和管理 Unreal 进程')
    parser.add_argument('--kill', action='store_true', help='终止所有 Unreal 进程')
    parser.add_argument('--force', action='store_true', help='强制终止（不询问确认）')
    parser.add_argument('--port', type=int, default=9090, help='检查的端口号（默认9090）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Unreal 进程诊断工具")
    print("=" * 80)
    
    if args.kill:
        kill_unreal_processes(force=args.force)
    else:
        print("\n1. 检查 Unreal 进程:")
        processes = check_unreal_processes()
        
        print(f"\n2. 检查端口 {args.port}:")
        port_in_use = check_port(args.port)
        
        if processes:
            print(f"\n建议:")
            print(f"  - 如果有多个 Unreal 进程在运行，可能导致端口冲突")
            print(f"  - 使用 'python {sys.argv[0]} --kill' 来终止所有进程")
            print(f"  - 或者手动终止: pkill -f UnrealZoo")
        
        if port_in_use and not processes:
            print(f"\n警告: 端口 {args.port} 被占用，但没有发现 Unreal 进程")
            print(f"  可能是其他程序占用了该端口")

if __name__ == '__main__':
    main()





















