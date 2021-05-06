from training import train
from evaluation import evaluate
from optimiz import optimizer, criterion
from calculateTime import epoch_time

from model import Net
from dataset import getData

# number of epochs to train the model
EPOCHS = 500

print("*"*50)
print('                   START THE ANDEOID MANIFEST PART')
print("*"*50)
# create a complete CNN
model = Net()
print(model)

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

# move tensors and criterion to GPU if CUDA is available
if train_on_gpu:
    model = model.cuda()
    criterion = criterion.cuda()

train_loader, valid_loader = getData()


# track change in validation loss
best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    
    start_time = time.monotonic()
    
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
    
    end_time = time.monotonic()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)


    print(f'Epoch: {epoch+1:03}/{EPOCHS} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')