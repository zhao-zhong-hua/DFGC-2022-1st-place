import os
import cv2

#对于输入文件夹，及其子文件夹中的视频文件，均匀抽取所需数量的帧之后，输出到与视频文件同名的文件夹中。
def extract_frames(input_folder, output_folder, total_frames_per_video):
    # 遍历输入文件夹中的所有视频文件
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov')):  # 可根据需要添加其他视频格式
                video_path = os.path.join(root, file)

                # 创建输出文件夹，与源视频同名
                # output_subfolder = os.path.join(output_folder, os.path.relpath(root, input_folder),os.path.splitext(file)[0]) #不含文件扩展名
                output_subfolder = os.path.join(output_folder,os.path.relpath(root, input_folder), file)    #含有文件扩展名

                os.makedirs(output_subfolder, exist_ok=True)

                # 使用OpenCV读取视频
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # 判断总帧数是否小于要抽取的帧数
                if total_frames <= total_frames_per_video:
                    interval = 1
                else:
                    # 计算均匀间隔帧数
                    interval = max(total_frames // total_frames_per_video, 1)

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

                        if count == total_frames_per_video or i + interval >= total_frames:
                            break

                cap.release()

if __name__ == "__main__":
    input_folder = "/big-data/dataset-academic/FaceForensics++/manipulated_sequences/DeepFakeDetection/c23/"
    output_folder = "/big-data/temp/face_crop/FF++/manipulated_sequences/DeepFakeDtection/c23/"
    total_frames_per_video = 5  # 每个视频中均匀抽取的帧数

    extract_frames(input_folder, output_folder, total_frames_per_video)
