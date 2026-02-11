import os
import glob
import re


def modify_labels_by_filename(label_root_dir: str):
    """
    根据标签文件的文件名开头 (shipXX) 批量修改文件内容中的类别索引。
    - ship01 开头的文件，类别改为 0
    - ship02 开头的文件，类别改为 1
    - 以此类推

    Args:
        label_root_dir: 包含所有 .txt 标签文件的根目录路径。
    """
    print(f"--- 🚀 开始批量修改 YOLO 标签类别 ---")
    print(f"标签目录: {label_root_dir}")
    print("-" * 35)

    # 使用 glob 递归查找所有 .txt 文件
    label_files = glob.glob(os.path.join(label_root_dir, '**', '*.txt'), recursive=True)

    if not label_files:
        print("✅ 检查完毕：未找到任何 .txt 标签文件。")
        return

    modified_count = 0

    # 编译正则表达式，用于匹配文件名开头的 shipXX
    # 例如：匹配 ship01, ship02, ship10, ship99 等
    pattern = re.compile(r'ship(\d{2})_.*\.txt$', re.IGNORECASE)

    for label_path in label_files:
        # 获取文件名 (e.g., ship01_image.txt)
        filename = os.path.basename(label_path)

        match = pattern.match(filename)

        if match:
            # 提取数字部分 (e.g., '01', '02')
            ship_number_str = match.group(1)
            ship_number = int(ship_number_str)  # 1, 2, 3, ...

            # 你的规则：ship01 -> 0, ship02 -> 1, ...
            # 目标类别索引 = 船舶编号 - 1
            target_class_index = ship_number - 1

            # 读取文件内容
            lines = []
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
            except Exception as e:
                print(f"🚨 **读取文件失败:** {label_path}, 错误: {e}")
                continue

            new_lines = []
            file_modified = False

            for line in lines:
                line = line.strip()
                if not line:  # 跳过空行
                    new_lines.append('\n')
                    continue

                parts = line.split()

                # 检查是否符合 YOLO 格式
                if len(parts) == 5:
                    try:
                        # 确保类别是数字
                        old_class_index = int(parts[0])

                        # 只有当旧类别索引与目标索引不一致时才修改
                        if old_class_index != target_class_index:
                            parts[0] = str(target_class_index)  # 修改类别索引
                            new_line = " ".join(parts) + '\n'
                            new_lines.append(new_line)
                            file_modified = True
                        else:
                            # 类别索引已经正确，保持不变
                            new_lines.append(line + '\n')

                    except ValueError:
                        # 如果第一列不是整数，则跳过修改，保持原样
                        print(f"⚠️ **跳过:** 文件 {filename} 第 {len(new_lines) + 1} 行类别非整数，未修改。")
                        new_lines.append(line + '\n')
                else:
                    # 如果格式不符合 YOLO (不是5个元素)，保持原样
                    new_lines.append(line + '\n')

            # 如果文件内容发生变化，则写回文件
            if file_modified:
                try:
                    with open(label_path, 'w') as f:
                        f.writelines(new_lines)
                    modified_count += 1
                    # 打印修改信息
                    print(f"✅ 已修改: {filename} -> 类别索引全部设为 {target_class_index}。")
                except Exception as e:
                    print(f"🚨 **写入文件失败:** {label_path}, 错误: {e}")
            # else:
            #     print(f"跳过: {filename} (类别索引已正确)。")

    print("-" * 35)
    print(f"🎉 批量修改完成! 总共修改了 {modified_count} 个标签文件。")


# --- 用户配置区 ---
# 替换为你的标签文件所在的根目录，**确保这个目录只包含你想修改的标签文件**
# 推荐使用你上次提供的路径（但请确保它是包含所有 .txt 文件的文件夹）
LABEL_ROOT_DIRECTORY = r"D:\gzx\yolov8bishe\ultralytics\ultralytics\cfg\datasets\CCCShip\labels"

# --- 运行修改 ---
if __name__ == "__main__":
    # 再次强调备份
    print("---!!! 运行前请再次确认已备份标签文件 !!!---")

    if not os.path.isdir(LABEL_ROOT_DIRECTORY):
        print(f"🚨 错误：指定的标签根目录不存在: {LABEL_ROOT_DIRECTORY}")
    else:
        modify_labels_by_filename(LABEL_ROOT_DIRECTORY)