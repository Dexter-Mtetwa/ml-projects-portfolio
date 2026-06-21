def fit(epoch,model,data_loader,phase='training',volatile=False):
        if phase == 'training':
                model.train()
        if phase == 'validation':
                model.eval()
                volatile=True
        running_loss = 0.0
        running_correct = 0
        for batch_idx , (data,target) in enumerate(data_loader):
                if is_cuda:
                        data,target = data.cuda(),target.cuda()
                data , target = Variable(data,volatile),Variable(target)
                if phase == 'training':
                        optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output,target)
                
                running_loss += F.nll_loss(output,target,size_average=False).data[0]
                preds = output.data.max(dim=1,keepdim=True)[1]
                running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
                if phase == 'training':
                        loss.backward()
                        optimizer.step()
                
        loss = running_loss/len(data_loader.dataset)
        accuracy = 100. * running_correct/len(data_loader.dataset)
        
        print(f'{phase} loss is {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}')
        return loss,accuracy
        
        
# calling the training loop
model = Net()
if is_cuda:
        model.cuda()
        
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,20):
        epoch_loss, epoch_accuracy = fit(epoch,model,train_loader,phase='training')
        val_epoch_loss , val_epoch_accuracy = fit(epoch,model,test_loader,phase='validation')
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        
# plot
plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'training loss')
plt.plot(range(1,len(val_losses)+1),val_losses,'r',label = 'validation loss')
plt.legend()
