import warnings
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import os
import sys
sys.path.append('../../utils')
from config import IMG_SIZE, BATCH_SIZE, NUM_WORKERS, DATA_PATH

warnings.filterwarnings("ignore")

print("Processing Species...")

transform = {
    'train': transforms.Compose([transforms.RandomSizedCrop(224),
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

dsets = {x: datasets.ImageFolder(os.path.join(DATA_PATH, x), transform[x])
         for x in ['train', 'test']}

dset_loaders = {x: torch.utils.data.DataLoader(dsets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
                for x in ['train', 'test']}

dset_sizes = {x: len(dsets[x]) for x in ['train', 'test']}

dset_classes = dsets['train'].classes
