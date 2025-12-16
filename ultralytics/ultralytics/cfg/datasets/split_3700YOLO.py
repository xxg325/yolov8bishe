import os
import random
from sklearn.model_selection import train_test_split

# --- 配置参数 ---
# 数据集根目录
DATASET_ROOT = "/Users/zhixingong/Desktop/bishe/yolov8bishe/yolov8bishe/ultralytics/ultralytics/cfg/datasets/3700YOLO"
# 结果 txt 文件保存目录
OUTPUT_DIR = os.path.join(DATASET_ROOT, "datasets")
# 原始图片的相对路径 (相对于 DATASET_ROOT)
IMAGE_SUBDIR_TRAIN = "train/images"
IMAGE_SUBDIR_VAL = "val/images"

# 划分比例
TRAIN_RATIO = 0.6
VAL_RATIO = 0.3
TEST_RATIO = 0.1
# 注意：TRAIN_RATIO + VAL_RATIO + TEST_RATIO 必须等于 1.0

def collect_image_paths(root_dir, image_subdir_train, image_subdir_val):
    """
    收集所有图片的路径。YOLOv8 模型要求路径为相对于数据集根目录的路径。
    """
    all_image_paths = []

    # 1. 收集 train 目录下的图片
    train_images_path = os.path.join(root_dir, image_subdir_train)
    if os.path.exists(train_images_path):
        for filename in os.listdir(train_images_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                # 保存相对于 DATASET_ROOT 的路径
                all_image_paths.append(os.path.join(image_subdir_train, filename))
    
    # 2. 收集 val 目录下的图片
    val_images_path = os.path.join(root_dir, image_subdir_val)
    if os.path.exists(val_images_path):
        for filename in os.listdir(val_images_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                # 保存相对于 DATASET_ROOT 的路径
                all_image_paths.append(os.path.join(image_subdir_val, filename))
                
    if not all_image_paths:
        print(f"⚠️ 警告：在 {train_images_path} 和 {val_images_path} 中没有找到任何图片。请检查路径和文件。")
    
    return all_image_paths

def split_data(all_paths, train_ratio, val_ratio, test_ratio):
    """
    使用 sklearn 的 train_test_split 进行三路划分：训练集、验证集、测试集。
    """
    if len(all_paths) == 0:
        return [], [], []

    # 1. 首先，将数据分为 (训练集+验证集) 和 测试集
    # test_size = test_ratio
    # 剩下的一部分 (train_val_paths) 占总体的 (1 - test_ratio)
    train_val_paths, test_paths = train_test_split(
        all_paths, 
        test_size=test_ratio, 
        random_state=42 # 确保每次划分结果一致
    )

    # 2. 然后，将 (训练集+验证集) 分为 训练集 和 验证集
    # 计算 val_ratio 相对于 train_val_paths 的比例
    # 新的 val_size = val_ratio / (train_ratio + val_ratio)
    if train_ratio + val_ratio > 0:
        val_size_relative = val_ratio / (train_ratio + val_ratio)
    else:
        # 如果 train_ratio 和 val_ratio 都为 0，则全部为 test (这应该不会发生)
        val_size_relative = 0

    train_paths, val_paths = train_test_split(
        train_val_paths, 
        test_size=val_size_relative, 
        random_state=42 
    )
    
    return train_paths, val_paths, test_paths

def write_txt_files(output_dir, train_paths, val_paths, test_paths):
    """
    将划分结果写入指定的 txt 文件中。
    """
    os.makedirs(output_dir, exist_ok=True) # 确保输出目录存在

    data_splits = {
        "train.txt": train_paths,
        "val.txt": val_paths,
        "test.txt": test_paths
    }

    print("\n--- 写入文件结果 ---")
    for filename, paths in data_splits.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for path in paths:
                # 写入的是相对于 DATASET_ROOT 的路径
                f.write(path + '\n')
        
        print(f"✅ 文件写入成功: {filepath}")
        print(f"   包含图片数量: {len(paths)}")

def main():
    print(f"🚀 开始处理数据集: {DATASET_ROOT}")

    # 1. 收集所有图片路径
    all_image_paths = collect_image_paths(DATASET_ROOT, IMAGE_SUBDIR_TRAIN, IMAGE_SUBDIR_VAL)
    total_images = len(all_image_paths)
    
    if total_images == 0:
        print("❌ 无法继续，因为没有找到任何图片文件。")
        return

    print(f"✅ 成功收集到 {total_images} 张图片路径。")
    random.shuffle(all_image_paths) # 打乱顺序

    # 2. 划分数据集
    train_paths, val_paths, test_paths = split_data(
        all_image_paths, 
        TRAIN_RATIO, 
        VAL_RATIO, 
        TEST_RATIO
    )

    print("\n--- 划分结果摘要 ---")
    print(f"总图片数: {total_images}")
    print(f"训练集 (60%): {len(train_paths)} 张")
    print(f"验证集 (30%): {len(val_paths)} 张")
    print(f"测试集 (10%): {len(test_paths)} 张")
    
    # 简单校验
    if len(train_paths) + len(val_paths) + len(test_paths) != total_images:
        print("⚠️ 警告：划分后的总数与原始图片总数不匹配！")

    # 3. 写入 txt 文件
    write_txt_files(OUTPUT_DIR, train_paths, val_paths, test_paths)
    
    print("\n🎉 数据集划分及 TXT 文件生成完毕！")


if __name__ == "__main__":
    # 需要安装 scikit-learn
    # pip install scikit-learn
    main()