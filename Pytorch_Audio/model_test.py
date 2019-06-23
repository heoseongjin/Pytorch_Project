import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from test import CustomConvNet, CustomImageDataset, test


######hyper_param_epoch = 20
hyper_param_epoch = 20
hyper_param_batch = 8
hyper_param_learning_rate = 0.001

transforms_train = transforms.Compose([transforms.Resize((128, 128)), transforms.RandomRotation(10.), transforms.ToTensor()])
transforms_test = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

train_data_set = CustomImageDataset(data_set_path="./data/train", transforms=transforms_train)
train_loader = DataLoader(train_data_set, batch_size=hyper_param_batch, shuffle=True)
test_data_set = CustomImageDataset(data_set_path="./data/test", transforms=transforms_test)
test_loader = DataLoader(test_data_set, batch_size=hyper_param_batch, shuffle=True)
custom_test_data_set = CustomImageDataset(data_set_path="./data/custom", transforms=transforms_test)
custom_test_loader = DataLoader(custom_test_data_set, batch_size=hyper_param_batch, shuffle=True)

if not (train_data_set.num_classes == test_data_set.num_classes):
    print("error: Numbers of class in training set and test set are not equal")
    exit()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
num_classes = train_data_set.num_classes
custom_model = CustomConvNet(num_classes=num_classes).to(device)

# custom_model.load_state_dict(torch.load("./model/test_model.pth"))
custom_model.load_state_dict(torch.load('./model/test_model.pth', map_location=lambda storage, loc: storage))

test()
