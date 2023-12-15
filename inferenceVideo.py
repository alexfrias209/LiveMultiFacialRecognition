from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
from PIL import Image, ImageDraw
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

# loading model
def load_model(checkpoint_path, num_classes, device):
    model = InceptionResnetV1(classify=True, num_classes=num_classes).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# face tensor for InceptionResnet input
def prepare_face_tensor(face):
    trans = transforms.Compose([
        transforms.Resize((500, 500)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return trans(face).unsqueeze(0)

# device agnostic code

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(keep_all=True, device=device)
model = load_model('./model.pth', num_classes=2, device = device)
model = model.to(device)
classes = ["Alex", "Alexandra"]

probability_threshold = 0.8

# loading video
video_path = 'Test.mp4'
cap = cv2.VideoCapture(video_path)

frames_tracked = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #converting BGR to RGB for MTCNN
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # detecting faces
    boxes, _ = mtcnn.detect(frame_pil)

    if boxes is not None:
        for box in boxes:

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            # predicting class for each face
            face = frame_pil.crop(box)
            face_tensor = prepare_face_tensor(face).to(device)
            prediction = model(face_tensor)
            probabilities = F.softmax(prediction, dim=1)
            top_probability, predicted_class = torch.max(probabilities, dim=1)


            if top_probability.item() > probability_threshold:
                label = f"{classes[predicted_class.item()]}: {top_probability.item():.2f}"
            else:
                label = "Uncertain"

            # drawing the labels
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    frames_tracked.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()

# saving videos with all our drawn stuff
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_trackedF.mp4', fourcc, 25.0, (frames_tracked[0].shape[1], frames_tracked[0].shape[0]))
for frame in frames_tracked:
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
