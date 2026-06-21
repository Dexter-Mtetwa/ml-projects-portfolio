import torch
import numpy as np
from torchvision import models
from torch.utils.data import DataLoader, Dataset

vgg = models.vgg16(pretrained=True)
device = 'cuda' if torch.cuda.is_available else 'cpu'
vgg = vgg.to(device)
features = vgg.features

train_data_loader = DataLoader(train,batch_size=32,num_workers=3,shuffle=False)
valid_data_loader = DataLoader(valid,batch_size=32,num_workers=3,shuffle=False)

def preconvfeat(dataset,model):
    conv_features = []
    labels_list = []

    for data in dataset:
        inputs,labels = data
        inputs , labels = inputs.to(device),labels.to(device)
        output = model(inputs)
        conv_features.extend(output.data.cpu().numpy())
        labels_list.extend(labels.data.cpu().numpy())

    conv_features = np.concatenate([[feat] for feat in conv_features])

    return (conv_features,labels_list)

conv_feat_train, labels_train = preconvfeat(train_data_loader,features)
conv_feat_val, labels_val = preconvfeat(valid_data_loader,features)


# let's create a PyTorch 'dataset' and 'DataLoader' classes, which will ease up our training process
class My_dataset(Dataset):
    def __init__(self,feat,labels):
        self.conv_feat = feat
        self.labels = labels

    def __len__(self):
        return len(self.conv_feat)

    def __getitem__(self,idx):
        return self.conv_feat[idx],self.labels[idx]


train_feat_dataset = My_dataset(conv_feat_train,labels_train)
val_feat_dataset = My_dataset(conv_feat_val,labels_val)

train_feat_loader = DataLoader(train_feat_dataset,batch_size=64,shuffle=True)
val_feat_loader = DataLoader(val_feat_dataset,batch_size=64,shuffle=True)


train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,20):
    epoch_loss, epoch_accuracy = fit_numpy(epoch,vgg.classifier,train_feat_loader,phase='training')
    val_epoch_loss , val_epoch_accuracy = fit_numpy(epoch,vgg.classifier,val_feat_loader,phase='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)