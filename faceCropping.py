import torch
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

data_dir = './directory'  
batch_size = 32
epochs = 8
workers = 0 if os.name == 'nt' else 4

# using cuda for my desktop but cpu for laptop
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# initializing MTCNN 
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

#using MTCNN facial detection and saving the cropped faces 
dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
]

loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)

for i, (x, y) in enumerate(loader):
    for j in range(len(x)):
        try:
            mtcnn(x[j:j+1], save_path=y[j:j+1])
        except Exception as e:
            print(f"Error processing image: {y[j]}: {e}")
            continue  # I was having an error so created this try and except. But hasnt came up so will test out and remove later
    print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')

