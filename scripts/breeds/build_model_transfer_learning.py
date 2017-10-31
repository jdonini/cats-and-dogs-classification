import copy
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import math
from torchvision import models
from torch.autograd import Variable
import sys
sys.path.append('../../scripts')
from breeds.data_loader import dset_classes, dset_loaders, dset_sizes
sys.path.append('../../utils')
from config import LR, LR_DECAY_EPOCH, NUM_EPOCHS, NUM_IMAGES, MOMENTUM

print('\nProcessing Model 4 Layers Breeds...\n')


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)


# Get a batch of training data_species
inputs, classes = next(iter(dset_loaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[dset_classes[x] for x in classes])

model_transfer_learning = models.resnet50(pretrained=True)


class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        ch = [3, 8, 16, 32, 48, 64, 92, 128, 156, 192, 256, 394, 512, 792, 1024]
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'bias': True}
        self.conv1 = nn.Sequential(  # (3, 64, 64)
            *list(model_transfer_learning.children())[:-5],
            # print(*list(model_transfer_learning.children())[:-5]),
            nn.Conv2d(256, 256, **kwargs)
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.1, inplace=True),
        #
        #     nn.Conv2d(128, 128, **kwargs),
        #     nn.BatchNorm2d(128),
        #     nn.LeakyReLU(0.1, inplace=True)
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(128, 192, **kwargs),
        #     nn.BatchNorm2d(192),
        #     nn.LeakyReLU(0.1, inplace=True),
        #
        #     nn.Conv2d(192, 256, **kwargs),
        #     nn.BatchNorm2d(256),
        #     nn.LeakyReLU(0.1, inplace=True),
        #
        #     # nn.MaxPool2d(kernel_size=2),  # 32
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 256, **kwargs),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(kernel_size=2),  # 16

            nn.Conv2d(256, 394, **kwargs),
            nn.BatchNorm2d(394),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(394, 394, **kwargs),
            nn.BatchNorm2d(394),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(394, 512, **kwargs),

            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            # nn.MaxPool2d(kernel_size=2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, **kwargs),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(512, 512, **kwargs),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),

            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 2048),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.2),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(0.6),
            nn.Linear(512, 37),
            nn.LogSoftmax(),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


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
        with open('../../results/breeds/4layers__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
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

            # Iterate over data_species.
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
            with open('../../results/breeds/4layers__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
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
    with open('../../results/breeds/4layers__Epoch__ ' + str(num_epochs) + '__LR__' + str(LR) + '.txt', 'a') as f:
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

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=NUM_EPOCHS)

# visualize_model(model)

plt.ioff()
plt.show()

# Save the Trained Model
# torch.save(model.state_dict(), '../../results/breeds/model_4layers_breeds.pkl')
