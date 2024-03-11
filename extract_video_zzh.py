import os
import cv2

def extract_frames(input_folder, output_folder, x, y):
    # 遍历输入文件夹中的所有视频文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):  # 可根据需要添加其他视频格式
                video_path = os.path.join(root, file)

                # 创建输出文件夹，与源视频同名
                output_subfolder = os.path.join(output_folder, os.path.splitext(file)[0])
                os.makedirs(output_subfolder, exist_ok=True)

                # 使用OpenCV读取视频
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # 计算间隔帧数
                interval = max(total_frames // y, 1)

                # 抽取帧并保存到输出文件夹
                count = 0
                for i in range(0, total_frames, interval):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()

                    if ret:
                        frame_name = f"frame_{count:04d}.jpg"
                        frame_path = os.path.join(output_subfolder, frame_name)
                        cv2.imwrite(frame_path, frame)
                        count += 1

                        if count == y:
                            break

                cap.release()

if __name__ == "__main__":
    input_folder = "输入文件夹的路径"
    output_folder = "输出文件夹的路径"
    x = 10  # 间隔x帧
    y = 5   # 总共抽取y帧

    extract_frames(input_folder, output_folder, x, y)
