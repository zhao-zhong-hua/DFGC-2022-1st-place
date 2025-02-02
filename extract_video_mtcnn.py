"""
Author: HanChen
Date: 21.06.2022
"""
import torch
from tqdm import tqdm
from facenet_pytorch import MTCNN
from PIL import Image
import numpy as np
from collections import OrderedDict
import cv2
import argparse

import os



def extract_video(input_dir, model, scale=1.3):
    reader = cv2.VideoCapture(input_dir)
    frames_num = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    batch_size = 32
    face_boxes = []
    face_images = []
    face_index = []
    original_frames = OrderedDict()
    halve_frames = OrderedDict()
    index_frames = OrderedDict()
    for i in range(frames_num):
        reader.grab()
        success, frame = reader.retrieve()
            
        if not success:
            continue
        original_frames[i] = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = frame.resize(size=[s // 2 for s in frame.size])
        halve_frames[i] = frame
        index_frames[i] = i


    original_frames = list(original_frames.values())
    halve_frames = list(halve_frames.values())
    index_frames = list(index_frames.values())
    for i in range(0, len(halve_frames), batch_size):
        batch_boxes, batch_probs, batch_points = model.detect(halve_frames[i:i + batch_size], landmarks=True)
        None_array = np.array([], dtype=np.int16)
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                pass
            else:
                batch_boxes[index] = None_array

        batch_boxes, batch_probs, batch_points = model.select_boxes(batch_boxes, batch_probs, batch_points,
                                                                    halve_frames[i:i + batch_size],
                                                                    method="probability")
                                                                    # method="largest")
        for index, bbox in enumerate(batch_boxes):
            if bbox is not None:
                xmin, ymin, xmax, ymax = [int(b * 2) for b in bbox[0, :]]
                w = xmax - xmin
                h = ymax - ymin
                # p_h = h // 3
                # p_w = w // 3
                size_bb = int(max(w, h) * scale)
                center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2

                # Check for out of bounds, x-y top left corner
                xmin = max(int(center_x - size_bb // 2), 0)
                ymin = max(int(center_y - size_bb // 2), 0)
                # Check for too big bb size for given x, y
                size_bb = min(original_frames[i:i + batch_size][index].shape[1] - xmin, size_bb)
                size_bb = min(original_frames[i:i + batch_size][index].shape[0] - ymin, size_bb)

                # crop = original_frames[index][max(ymin - p_h, 0):ymax + p_h, max(xmin - p_w, 0):xmax + p_w]
                # crop = original_frames[i:i + batch_size][index][ymin:ymin+size_bb, xmin:xmin+size_bb]
                face_index.append(index_frames[i:i + batch_size][index])
                face_boxes.append([ymin, ymin + size_bb, xmin, xmin + size_bb])
                crop = original_frames[i:i + batch_size][index][ymin:ymin + size_bb, xmin:xmin + size_bb]
                face_images.append(crop)
            else:
                pass
				

    return face_images, face_boxes, face_index


def parse_args():
    parser = argparse.ArgumentParser(description='Extract face from videos')
           
    parser.add_argument('--input_root_path', type=str, default='/data/linyz/DFGC2022/DFGC2022_Test/Det_data_3')           
    parser.add_argument('--output_root_path', type=str, default='/data/linyz/DFGC2022/DFGC2022_Test/images_mtcnn')             
    parser.add_argument('--gpu_id', type=int, default=5)

    parser.add_argument('--scale', type=float, default=1.3)
    parser.add_argument('--id_num', type=int, default=61)

    args = parser.parse_args()
    return args



def main():
    args = parse_args()    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    model = MTCNN(margin=0, thresholds=[0.6, 0.7, 0.7], device=device)
    scale = args.scale

    if not os.path.isdir(args.output_root_path):
        os.mkdir(args.output_root_path)

    for video_name in tqdm(os.listdir(args.input_root_path)):
        input_video_path = os.path.join(args.input_root_path, video_name)
        output_video_path = os.path.join(args.output_root_path, video_name)
        if not os.path.isdir(output_video_path):
            os.mkdir(output_video_path)

        face_images, face_boxes, face_index = extract_video(input_video_path, model, scale=scale)   
        
        for j, index in enumerate(face_index):
            cv2.imwrite(os.path.join(output_video_path, "%04d.png" % index), face_images[j])        
        
        
if __name__ == "__main__":
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    main()






