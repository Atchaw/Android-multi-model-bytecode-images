from accuracy import calculate_accuracy

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

# The evaluation loop

def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
        
        for (data, target) in iterator:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
              data, target = data.cuda(), target.cuda()
        
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # calculate accuracy
            acc = calculate_accuracy(output, target)

            # update evaluation loss and evaluation accuracy
            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)