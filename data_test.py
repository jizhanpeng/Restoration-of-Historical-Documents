

import os


def rename_files(directory):
    try:
        # 遍历指定目录下的所有文件和文件夹
        for root, dirs, files in os.walk(directory):
            for file in files:
                # 获取文件的完整路径
                old_file_path = os.path.join(root, file)
                # 进行文件名替换
                new_file = file.replace('para', 'char')
                # 生成新的文件完整路径
                new_file_path = os.path.join(root, new_file)
                # 重命名文件
                os.rename(old_file_path, new_file_path)
                print(f"已将 {old_file_path} 重命名为 {new_file_path}")
    except FileNotFoundError:
        print(f"错误：指定的目录 {directory} 未找到。")
    except Exception as e:
        print(f"发生未知错误：{e}")


if __name__ == "__main__":
    # 这里可以替换为你想要处理的目录路径
    target_directory = 'D:\\0 ScientificResearch\\5 数据集\\HDR28K\\paper_damage\\train\\content_images'
    rename_files(target_directory)



