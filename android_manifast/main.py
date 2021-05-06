import training
import evaluation
import model
import dataset
import optimizer
import calculateTime
# number of epochs to train the model
EPOCHS = 500

model = model.Net()
train_loader, valid_loader = dataset.getData()
optimizer = optimizer.getOptimizer()
criterion = optimizer.getCriterion()

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
# move tensors and criterion to GPU if CUDA is available
if train_on_gpu:
    model = model.cuda()
    criterion = criterion.cuda()


# track change in validation loss
best_valid_loss = float('inf')

for epoch in range(EPOCHS):
    
    start_time = time.monotonic()
    
    train_loss, train_acc = training.train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluation.evaluate(model, valid_loader, criterion)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'model.pt')
    
    end_time = time.monotonic()

    epoch_mins, epoch_secs = calculateTime.epoch_time(start_time, end_time)


    print(f'Epoch: {epoch+1:03}/{EPOCHS} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')