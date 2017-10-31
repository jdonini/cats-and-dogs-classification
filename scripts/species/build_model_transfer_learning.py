import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torchvision
from torch.autograd import Variable
import sys
sys.path.append('../../scripts')
from species.data_loader import dset_classes, dset_loaders, dset_sizes
sys.path.append('../../utils')
from config import LR, LR_DECAY_EPOCH, NUM_EPOCHS, NUM_IMAGES, MOMENTUM, GAMMA
sys.path.append('../../utils')


print('\nProcessing Model Layers Species\n')


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated


inputs, classes = next(iter(dset_loaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[dset_classes[x] for x in classes])

model_transfer_learning = models.resnet18(pretrained=True)

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            *list(model_transfer_learning.children())[:-5],
            # print(*list(model_transfer_learning.children())[:-7]),
            nn.Conv2d(64, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Linear(512, 2),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def exp_lr_scheduler(optimizer, epoch, init_lr=LR, lr_decay_epoch=LR_DECAY_EPOCH):
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('Learning Rate: {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs=NUM_EPOCHS):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 50)
        results = ('Epoch {}/{}\n'.format(epoch + 1, num_epochs)) + ('--' * 50) + '\n'
        with open('../../results/species/build_model__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
            f.write(results)
        f.close

        for phase in ['train', 'test']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for data in dset_loaders[phase]:
                inputs, labels = data

                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(F.softmax(outputs)[:, 1], labels.float())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.8f} Acc: {:.8f}'.format(phase, epoch_loss, epoch_acc))

            results = ('{} Loss: {:.8f} Acc: {:.8f}\n'.format(phase, epoch_loss, epoch_acc)) + '\n'
            with open('../../results/species/build_model__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
                f.write(results)
            f.close

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:8f}\n'.format(best_acc))

    results = ('\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)) + \
        ('Best val Acc: {:8f}\n'.format(best_acc))
    with open('../../results/species/build_model__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
        f.write(results)
    f.close

    return best_model


def visualize_model(model, num_images=NUM_IMAGES):
    images_so_far = 0
    fig = plt.figure()

    for i, data in enumerate(dset_loaders['test']):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)

        for j in range(inputs.size()[0]):
            images_so_far += 1
            ax = plt.subplot(num_images//2, 2, images_so_far)
            ax.axis('off')
            ax.set_title('predicted: {}'.format(dset_classes[preds[j]]))
            imshow(inputs.cpu().data[j])

            if images_so_far == num_images:
                return


model = CNNModel()

if torch.cuda.is_available():
    model.cuda()

criterion = nn.BCELoss().cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

# visualize_model(model)

plt.ioff()
plt.show()

# torch.save(model.state_dict(), '../../results/species/build_model_species.pkl')
