import torch.hub

# 查看 PyTorch Hub 默认缓存目录
cache_dir = torch.hub.get_dir()
print("🔧 PyTorch Hub 缓存目录:", cache_dir)

# 权重文件的完整路径
import os
weight_path = os.path.join(cache_dir, 'checkpoints', 'pt_inception-2015-12-05-6726825d.pth')
print("📥 权重文件完整路径:", weight_path)

# 检查文件是否存在
if os.path.exists(weight_path):
    size = os.path.getsize(weight_path)
    print(f"✅ 文件存在，大小: {size} 字节 ({size / (1024**2):.2f} MB)")
else:
    print("❌ 文件不存在，需要下载")