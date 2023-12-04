import cv2
import os


video_path = './val.mp4'

output_dir = './directoryFinal/Alex'
# 1 frame per second is what I'm going for
extract_frequency = 1


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
interval = int(fps * extract_frequency)

frame_number = 0
extracted_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # extracting the frames
    if frame_number % interval == 0:
        frame_file = os.path.join(output_dir, f'Alex_frame_{extracted_count}.jpg')
        cv2.imwrite(frame_file, frame)
        extracted_count += 1

    frame_number += 1

cap.release()
print(f'Extracted {extracted_count} frames and saved in {output_dir}')
