import os
import warnings
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import sys
sys.path.append('utils')
from config import IMG_SIZE, DATA_PATH_BREEDS, BATCH_SIZE, NUM_WORKERS

warnings.filterwarnings("ignore")

print("Processing Data Breeds...")

transform = {
    'train': transforms.Compose([transforms.Scale(256),
                                 transforms.RandomSizedCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                     mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                                 ]),
    'test': transforms.Compose([transforms.Scale(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
}

dsets = {x: datasets.ImageFolder(os.path.join(DATA_PATH_BREEDS, x), transform[x])
         for x in ['train', 'test']}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
                for x in ['train', 'test']}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'test']}

dset_classes = dsets['train'].classes
