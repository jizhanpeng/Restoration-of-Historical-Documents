import os
import torch
import numpy as np
from PIL import Image
import lpips
from pytorch_fid import fid_score
import torchvision.transforms as transforms

def calculate_lpips(path1, path2, device='cuda'):
    """计算两组图像的平均LPIPS值"""
    # 获取图像列表
    true_list = sorted(os.listdir(path1))
    out_list = sorted(os.listdir(path2))
    
    if len(true_list) != len(out_list):
        raise ValueError(f"两个文件夹中的图像数量不一致: {len(true_list)} vs {len(out_list)}")
    
    # 初始化LPIPS模型
    lpips_model = lpips.LPIPS(net='alex').to(device)
    lpips_model.eval()
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # LPIPS默认输入尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    total_lpips = 0.0
    best_lpips = float('inf')
    worst_lpips = 0.0
    best_name = None
    worst_name = None
    
    # 处理每对图像
    for name1, name2 in zip(true_list, out_list):
        try:
            # 读取图像并转换为RGB
            img1 = Image.open(os.path.join(path1, name1)).convert('RGB')
            img2 = Image.open(os.path.join(path2, name2)).convert('RGB')
            
            # 预处理为LPIPS输入格式
            img1_tensor = transform(img1).unsqueeze(0).to(device)
            img2_tensor = transform(img2).unsqueeze(0).to(device)
            
            # 计算LPIPS
            with torch.no_grad():
                dist = lpips_model(img1_tensor, img2_tensor).item()
            
            total_lpips += dist
            
            # 更新最佳和最差结果
            if dist < best_lpips:
                best_lpips = dist
                best_name = name1
            if dist > worst_lpips:
                worst_lpips = dist
                worst_name = name1
                
        except Exception as e:
            print(f"处理图像 {name1} 和 {name2} 时出错: {e}")
            continue
    
    # 计算平均值
    avg_lpips = total_lpips / len(true_list)
    
    print(f"LPIPS 平均值: {avg_lpips:.5f}")
    print(f"最佳 LPIPS 图像: {best_name} ({best_lpips:.5f})")
    print(f"最差 LPIPS 图像: {worst_name} ({worst_lpips:.5f})")
    
    return avg_lpips, best_lpips, worst_lpips, best_name, worst_name

if __name__ == '__main__':
    real_images_folder = '/home/dqxy/xyj/xyjjzp/DocDiff-main-1/dataset/HDR28K/test'
    generated_images_folder = '/home/dqxy/xyj/xyjjzp/DocDiff-main-4/results/init_predict'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 计算FID
    print("计算 FID 指标...")
    fid_value = fid_score.calculate_fid_given_paths(
        [real_images_folder, generated_images_folder],
        batch_size=32,  # 根据GPU内存调整
        device=device,
        dims=2048,
        num_workers=8
    )
    print(f'FID 值: {fid_value:.5f}')
    
    # 计算LPIPS
    print("\n计算 LPIPS 指标...")
    avg_lpips, best_lpips, worst_lpips, best_name, worst_name = calculate_lpips(
        real_images_folder, generated_images_folder, device
    )