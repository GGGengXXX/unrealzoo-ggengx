# 查看并询问是否删除
echo "当前Unreal进程："
ps aux | grep -i unreal | grep -v grep
read -p "是否删除所有Unreal进程？(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    pkill -f "UnrealZoo_UE5_5"
    echo "已删除所有Unreal进程"
fi