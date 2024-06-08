import os

def rename_colon_to_underscore(directory):
    for root, dirs, files in os.walk(directory, topdown=False):
        # 先处理文件
        for name in files:
            if ':' in name:
                new_name = name.replace(':', '_')
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed file: {old_path} -> {new_path}")

        # 再处理文件夹
        for name in dirs:
            if ':' in name:
                new_name = name.replace(':', '_')
                old_path = os.path.join(root, name)
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f"Renamed directory: {old_path} -> {new_path}")

if __name__ == "__main__":
    directory = '.'  # 可以替换为任何你想要处理的目录
    rename_colon_to_underscore(directory)