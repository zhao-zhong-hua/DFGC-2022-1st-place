import os
# 将以.jpg结尾的文件夹，输出其绝对（相对）路径和标签到txt文件中
def process_folder(folder_path, output_file, label1, label2, label3):
    for root, dirs, files in os.walk(folder_path):
        for file in files: #对于文件，修改下面的folder为file
        # for folder in dirs: #对于文件夹，修改下面的file为folder
            if file.lower().endswith('.jpg'):
                folder_path = os.path.abspath(os.path.join(root, file)) #输出完整路径
                # file_path = os.path.join(root, folder)  #输出相对路径
                labels_str = f"{label1} {label2} {label3}"
                output_line = f"{folder_path} {labels_str}\n"
                output_file.write(output_line)

if __name__ == "__main__":
    input_folder = "test_video_output"  # 替换成你的文件夹路径
    output_file_path = "test_video_output/output.txt"  # 替换成你想要的输出文件路径
    label1 = 0  # 替换成第一个标签的值 (0或1)
    label2 = 1  # 替换成第二个标签的值 (0或1)
    label3 = 0  # 替换成第三个标签的值 (0或1)

    with open(output_file_path, 'w') as output_file:
        process_folder(input_folder, output_file, label1, label2, label3)
