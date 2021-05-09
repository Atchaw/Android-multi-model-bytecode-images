
import csv

# name of train csv file 
filename_train = "./output/training.csv"

def createCsvTrain():
    # field names of the csv file 
    fields = ['Epoch', 'Epoch Mins', 'Epoch secs', 'Train Loss', 'Val Loss', 'Train Acc', 'Val Acc']
    # writing to csv file 
    with open(filename_train, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)    
        # writing the fields 
        csvwriter.writerow(fields)
    csvfile.close()

def updateCsv(row):
    # writing to csv file 
    with open(filename_train, 'a') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)
        # writing the data row 
        csvwriter.writerow(row)
    csvfile.close()

def csvTest(row):
    # field names 
    fields = ['Test set size', 'Test Mins', 'Test secs', 'Test Loss', 'Test Acc']
    # name of csv file 
    filename = "./output/test.csv"
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the fields 
        csvwriter.writerow(fields)
        # writing the data rows 
        csvwriter.writerow(row)
    csvfile.close()
