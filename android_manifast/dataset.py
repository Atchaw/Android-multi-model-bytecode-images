import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data

# convert data to a normalized torch.FloatTensor
transform = transforms.Compose([
    transforms.Resize((40,40)),
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

def getData(data_dir, batch_size):
    #load data from file
    train_data = datasets.ImageFolder(data_dir + '/train', transform=transform)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=transform)

    # prepare data loaders
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = data.DataLoader(val_data, batch_size=batch_size)

    return train_loader, valid_loader

def getDataTest(data_dir, batch_size):
    #load data from file
    val_data = datasets.ImageFolder(data_dir + '/val', transform=transform)

    # prepare data loaders
    test_loader = data.DataLoader(test_data, batch_size=batch_size)

    return test_loader