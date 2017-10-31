import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import sys
sys.path.append('../../scripts')
from cat_breeds.data_loader import dset_classes, dset_loaders, dset_sizes
sys.path.append('../../utils')
from config import LR, LR_DECAY_EPOCH, NUM_EPOCHS, NUM_IMAGES, MOMENTUM
sys.path.append('../../utils')

print('\nProcessing Model 4 Layers Dogs Breeds...\n')


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


# Get a batch of training data
inputs, classes = next(iter(dset_loaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[dset_classes[x] for x in classes])


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        # conv11
        self.conv11 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(64)

        # conv12
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm2d(64)

        # conv13
        self.conv13 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(64)

        # conv21
        self.conv21 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)

        # conv22
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bn22 = nn.BatchNorm2d(128)

        # conv23
        self.conv23 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn23 = nn.BatchNorm2d(128)

        # conv31
        self.conv31 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)

        # conv32
        self.conv32 = nn.Conv2d(256, 256, 3, padding=1, stride=2)
        self.bn32 = nn.BatchNorm2d(256)

        # conv33
        self.conv33 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)

        # conv41
        self.conv41 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)

        # conv42
        self.conv42 = nn.Conv2d(512, 512, 3, padding=1, stride=2)
        self.bn42 = nn.BatchNorm2d(512)

        # conv43
        self.conv43 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)

        # out
        self.drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(512, 256)
        self.fc = nn.Linear(256, 12)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        x = F.relu(self.conv12(x))
        x = self.bn12(x)

        x = F.relu(self.conv13(x))
        x = self.bn13(x)

        x = F.relu(self.conv21(x))
        x = self.bn21(x)

        x = F.relu(self.conv22(x))
        x = self.bn22(x)

        x = F.relu(self.conv23(x))
        x = self.bn23(x)

        x = F.relu(self.conv31(x))
        x = self.bn31(x)

        x = F.relu(self.conv32(x))
        x = self.bn32(x)

        x = F.relu(self.conv33(x))
        x = self.bn33(x)

        x = F.relu(self.conv41(x))
        x = self.bn41(x)

        x = F.relu(self.conv42(x))
        x = self.bn42(x)

        x = F.relu(self.conv43(x))
        x = self.bn43(x)

        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.fc1(x.view(x.size(0), -1))
        x = self.drop(x)
        x = self.fc(x)
        x = self.drop(x)
        x = F.softmax(x)
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
        with open('../../results/dogs/4layers__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
            f.write(results)
        f.close

        # Each epoch has a training and test phase
        for phase in ['train', 'test']:
            if phase == 'train':
                optimizer = lr_scheduler(optimizer, epoch)
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data[0]
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            results = ('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc)) + '\n'
            with open('../../results/dogs/4layers__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
                f.write(results)
            f.close

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}\n'.format(best_acc))

    # save results
    results = ('\nTraining complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60)) + \
        ('Best val Acc: {:4f}\n'.format(best_acc))
    with open('../../results/dogs/4layers__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
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

criterion = nn.CrossEntropyLoss().cuda()

# optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
optimizer = torch.optim.RMSprop(model.parameters())

model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

# visualize_model(model)

plt.ioff()
plt.show()

# Save the Trained Model
# torch.save(model.state_dict(), '../../results/dogs/model_4layers_dogs_breeds.pkl')
