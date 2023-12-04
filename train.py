# import torch
# from torch import optim
# from torch.utils.data import DataLoader, SubsetRandomSampler
# from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms
# import numpy as np
# import os
# from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training

# # Define run parameters
# data_dir = './directory_cropped1'  # Update to your cropped dataset path
# batch_size = 4
# epochs = 8
# workers = 0 if os.name == 'nt' else 4

# # Determine if an NVIDIA GPU is available
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('Running on device: {}'.format(device))

# # Define Inception Resnet V1 module for face recognition
# resnet = InceptionResnetV1(
#     classify=True,
#     pretrained='vggface2',
#     num_classes=2
# ).to(device)

# # Define optimizer, scheduler, dataset, and dataloader
# optimizer = optim.Adam(resnet.parameters(), lr=0.0001)
# scheduler = MultiStepLR(optimizer, [5, 10])

# trans = transforms.Compose([
#     np.float32,
#     transforms.ToTensor(),
#     fixed_image_standardization
# ])
# dataset = datasets.ImageFolder(data_dir, transform=trans)
# img_inds = np.arange(len(dataset))
# np.random.shuffle(img_inds)
# train_inds = img_inds[:int(0.8 * len(img_inds))]
# val_inds = img_inds[int(0.8 * len(img_inds)):]

# train_loader = DataLoader(
#     dataset,
#     num_workers=workers,
#     batch_size=batch_size,
#     sampler=SubsetRandomSampler(train_inds)
# )
# val_loader = DataLoader(
#     dataset,
#     num_workers=workers,
#     batch_size=batch_size,
#     sampler=SubsetRandomSampler(val_inds)
# )

# # Define loss and evaluation functions
# loss_fn = torch.nn.CrossEntropyLoss()
# metrics = {
#     'fps': training.BatchTimer(),
#     'acc': training.accuracy
# }

# # Function to save a checkpoint
# def save_checkpoint(model, optimizer, epoch, loss, filename='model_checkpoint.pth'):
#     checkpoint = {
#         'epoch': epoch + 1,  # Next epoch
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#     }
#     torch.save(checkpoint, filename)

# # Train model
# writer = SummaryWriter()
# writer.iteration, writer.interval = 0, 10

# print('\n\nInitial')
# print('-' * 10)
# resnet.eval()
# training.pass_epoch(
#     resnet, loss_fn, val_loader,
#     batch_metrics=metrics, show_running=True, device=device,
#     writer=writer
# )

# for epoch in range(epochs):
#     print('\nEpoch {}/{}'.format(epoch + 1, epochs))
#     print('-' * 10)

#     resnet.train()
#     training.pass_epoch(
#         resnet, loss_fn, train_loader, optimizer, scheduler,
#         batch_metrics=metrics, show_running=True, device=device,
#         writer=writer
#     )

#     resnet.eval()
#     validation_loss = training.pass_epoch(
#         resnet, loss_fn, val_loader,
#         batch_metrics=metrics, show_running=True, device=device,
#         writer=writer
#     )

#     save_checkpoint(resnet, optimizer, epoch, validation_loss)

# writer.close()


# import torch
# from torch import optim
# from torch.utils.data import DataLoader, SubsetRandomSampler
# from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
# from torchvision import datasets, transforms
# import numpy as np
# import os
# from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
# import random

# # # Set seeds for reproducibility
# # torch.manual_seed(42)
# # np.random.seed(42)

# # Set seeds for reproducibility
# seed = 0
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # if use multi-GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False

# # Define run parameters
# data_dir = './directory_cropped1'  # Update to your cropped dataset path
# batch_size = 32
# epochs = 8
# workers = 0 if os.name == 'nt' else 4

# # Determine if an NVIDIA GPU is available
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# print('Running on device: {}'.format(device))

# # Define Inception Resnet V1 module for face recognition
# resnet = InceptionResnetV1(
#     classify=True,
#     pretrained='vggface2',
#     num_classes=2
# ).to(device)

# # Define optimizer, scheduler, dataset, and dataloader
# optimizer = optim.Adam(resnet.parameters(), lr=0.00001)
# scheduler = MultiStepLR(optimizer, [5, 10])

# trans = transforms.Compose([
#     np.float32,
#     transforms.ToTensor(),
#     fixed_image_standardization
# ])
# dataset = datasets.ImageFolder(data_dir, transform=trans)
# img_inds = np.arange(len(dataset))
# np.random.shuffle(img_inds)
# train_inds = img_inds[:int(0.8 * len(img_inds))]
# val_inds = img_inds[int(0.8 * len(img_inds)):]

# train_loader = DataLoader(
#     dataset,
#     num_workers=workers,
#     batch_size=batch_size,
#     sampler=SubsetRandomSampler(train_inds)
# )
# val_loader = DataLoader(
#     dataset,
#     num_workers=workers,
#     batch_size=batch_size,
#     sampler=SubsetRandomSampler(val_inds)
# )

# # Define loss and evaluation functions
# loss_fn = torch.nn.CrossEntropyLoss()
# metrics = {
#     'fps': training.BatchTimer(),
#     'acc': training.accuracy
# }

# # Function to save a checkpoint
# def save_checkpoint(model, optimizer, epoch, loss, filename='model_checkpoint.pth'):
#     checkpoint = {
#         'epoch': epoch + 1,  # Next epoch
#         'model_state_dict': model.state_dict(),
#         'optimizer_state_dict': optimizer.state_dict(),
#         'loss': loss,
#     }
#     torch.save(checkpoint, filename)

# # Train model
# writer = SummaryWriter()
# writer.iteration, writer.interval = 0, 10

# print('\n\nInitial')
# print('-' * 10)
# resnet.eval()
# training.pass_epoch(
#     resnet, loss_fn, val_loader,
#     batch_metrics=metrics, show_running=True, device=device,
#     writer=writer
# )

# for epoch in range(epochs):
#     print('\nEpoch {}/{}'.format(epoch + 1, epochs))
#     print('-' * 10)

#     resnet.train()
#     training.pass_epoch(
#         resnet, loss_fn, train_loader, optimizer, scheduler,
#         batch_metrics=metrics, show_running=True, device=device,
#         writer=writer
#     )

#     resnet.eval()
#     validation_loss = training.pass_epoch(
#         resnet, loss_fn, val_loader,
#         batch_metrics=metrics, show_running=True, device=device,
#         writer=writer
#     )

#     if (epoch + 1) % 5 == 0:
#         save_checkpoint(resnet, optimizer, epoch, validation_loss)

# writer.close()



import torch
from torch import optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization, training
import random

# seeds so I can reproduce same results
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # was thinking of using multi-gpu so will keep this just in case
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


data_dir = './directory_cropped1'  # my cropped face images directory
batch_size = 32
epochs = 8
workers = 0 if os.name == 'nt' else 4

# using gpu if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# Initializing Inception Resnet V1 model
resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=2
).to(device)

#  optimizer, scheduler, dataset, and dataloader
optimizer = optim.Adam(resnet.parameters(), lr=0.00001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir, transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]  #80/20 split
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

#loss and evaluation functions
loss_fn = torch.nn.CrossEntropyLoss()
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

# saving checkpoint
def save_checkpoint(model, optimizer, epoch, loss, base_filename='model_checkpoint'):
    filename = f'{base_filename}_epoch_{epoch+1}.pth'  
    checkpoint = {
        'epoch': epoch + 1, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")  # Print the name of the saved checkpoint

# training loop code
writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\nInitial')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    validation_loss = training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    if (epoch + 1) % 5 == 0: #Saving every 5th epoch, saves on memory
        save_checkpoint(resnet, optimizer, epoch, validation_loss)

writer.close()
