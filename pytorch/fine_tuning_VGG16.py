import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

vgg = models.vgg16(pretrained=True)

print(vgg.summary())
# shows the model is divide into 2
# 1. features - the five blocks with (Conv2d + Relu + MaxPool2d), this the one we'll freeze
# 2. classifiers - makes the final decision, contains (Linear Layer + Relu + Dropout)

# freeze all the layers in the features model
for param in vgg.features.parameters(): param.requires_grad = False

# model is trained to classify 1000 images, change that to only 2 
vgg.classifier[6].out_features = 2

# we only need the classifier params to be trained, not the features, hence we feed only those to the optimizer
optimizer = optim.SGD(vgg.classifier.parameters(), lr=0.0001, momentum=0.5)
criterion = nn.NLLLoss()


def fit(model, data_loader, phase='training'):
        if phase == 'training':
                model.train()
        if phase == 'validation':
                model.eval()

        running_loss = 0.0
        running_correct = 0
        for batch_idx , (data,target) in enumerate(data_loader):
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                data,target = data.to(device), target.to(device)

                if phase == 'training': optimizer.zero_grad()
                output = model(data)
                loss = criterion(output,target)   
                running_loss += criterion(output,target,size_average=False).data[0]
                preds = output.data.max(dim=1,keepdim=True)[1]
                running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
                if phase == 'training':
                    loss.backward()
                    optimizer.step()
                
        loss = running_loss/len(data_loader.dataset)
        accuracy = 100. * running_correct/len(data_loader.dataset)
        
        print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset())}')
        return loss,accuracy


train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_data_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)


# train the model
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,20):
    epoch_loss, epoch_accuracy = fit(epoch, vgg, train_data_loader, phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch, vgg, valid_data_loader, phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)




# if you wanna do stuff say changing value of classifier dropout from .5 to .2
for layer in vgg.classifier.children():
       if (type(layer) == nn.Dropout):
              layer.p = 0.2

#Training
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,3):
    epoch_loss, epoch_accuracy = fit(epoch,vgg,train_data_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,vgg,valid_data_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)



# or maybe we wanna do data augmentation
train_transform =transforms.Compose([transforms.Resize((224,224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(0.2),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

simple_transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])

train = ImageFolder('dogsandcats/train/',train_transform)
valid = ImageFolder('dogsandcats/valid/',simple_transform)

#Training
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,3):
    epoch_loss, epoch_accuracy = fit(epoch,vgg,train_data_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,vgg,valid_data_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)