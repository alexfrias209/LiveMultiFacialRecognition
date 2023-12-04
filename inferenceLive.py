from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import cv2
from PIL import Image, ImageDraw
import numpy as np
import torch.nn.functional as F
from torchvision import transforms

def load_model(checkpoint_path, device,num_classes,):
    model = InceptionResnetV1(classify=True, num_classes=num_classes).eval()
    # I trained on cuda so if using cpu will convert pt with map_location parameter
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# transforming image parameters
def prepare_face_tensor(face):
    trans = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return trans(face).unsqueeze(0)

# checking if gpu is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# creating MTCNN and InceptionResnetV1 models
mtcnn = MTCNN(keep_all=True, device=device)
model = load_model('./checkpoints/model_checkpoint_epoch_5.pth',device, num_classes=2)
model = model.to(device)
classes = ["alex", "alexandra"]

# Allowed probability threshold for putting out a name
probability_threshold = 0.8

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert BGR to RGB for MTCNN
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)

    # detecting faces
    boxes, _ = mtcnn.detect(frame_pil)

    if boxes is not None:
        for box in boxes:
            # drawing rectangles
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            # predicting who the peope are
            face = frame_pil.crop(box)
            face_tensor = prepare_face_tensor(face).to(device)
            prediction = model(face_tensor)
            probabilities = F.softmax(prediction, dim=1)
            top_probability, predicted_class = torch.max(probabilities, dim=1)

            # checking prediction probability with threshold
            if top_probability.item() > probability_threshold:
                label = f"{classes[predicted_class.item()]}: {top_probability.item():.2f}"
            else:
                label = "None"

            # drawing the labels
            cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # displaying the labels and boxes
    cv2.imshow('Live Video', frame)

    # exit condition of live
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
