import torch
import torch.nn as nn

import sys
import time

from training import train
from evaluation import evaluate
from optimiz import getOptimizer
from calculateTime import epoch_time, total_time

from model import Net
from dataset import getData

from csvFile import createCsvTrain, updateCsv

def main(argv):
    # create a csv file for the training loop to save information
    createCsvTrain()

    # number of epochs to train the model
    EPOCHS = int(argv[1])

    print("*"*90)
    print('                            START THE ANDEOID MANIFEST PART')
    print("*"*90)
    # create a complete CNN
    model = Net()
    print(model)

    # check if CUDA is available
    train_on_gpu = torch.cuda.is_available()

    # move tensors and criterion to GPU if CUDA is available
    if train_on_gpu:
        model = model.cuda()

    criterion =  nn.NLLLoss()
    if train_on_gpu:
        criterion = criterion.cuda()

    train_loader, valid_loader = getData(argv[2],int(argv[3]))
    optimizer = getOptimizer(float(argv[4]),float(argv[5]))

    # track change in validation loss
    best_valid_loss = float('inf')

    print("~"*90)

    total_start_time = time.monotonic()
    for epoch in range(EPOCHS):
        
        start_time = time.monotonic()
        
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        valid_loss, valid_acc = evaluate(model, valid_loader, criterion)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './output/model.pt')
        
        end_time = time.monotonic()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)


        print(f'Epoch: {epoch+1}/{EPOCHS} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

        # save result to csv file
        row = [epoch+1, epoch_mins, epoch_secs, train_loss, valid_loss, train_acc*100, valid_acc*100]
        updateCsv(row)

    total_end_time = time.monotonic()

    print("~"*90)
    total_hours, total_mins, total_secs = total_time(total_start_time, total_end_time)
    print(f'Total Time: {total_hours}h {total_mins}m {total_secs}s')

if __name__== "__main__":
    main(sys.argv)
