from main import *
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse
import torch.backends.cudnn as cudnn
# from dongkeun import resnet
from dongkeun2 import efficientnet_b0
import matplotlib.pyplot as plt
from datasetfolder2 import CustomImageDataset2,CustomImageDataset2_test
from datasetfolder2 import CustomImageDataset3
from dongkeun8 import dongkeun
# from convolution3d import EfficientNetB0
# from dongkeun4 import NLBlockND
# from dongkeun5 import Nonlocal
import torch.cuda
import copy
# from attention import NonLocalSparseAttention
from convolution3d import Attentionmap
from fcanet import FcaBasicBlock
# from dongkeun8 import dongkeun
def log(text, LOGGER_FILE):
    with open(LOGGER_FILE, 'a') as f:
        f.write(text)
        f.close()
parser = argparse.ArgumentParser(description='material classification')
parser.add_argument('--hyper_param_batch',type=int,default=8,help='batch size')
parser.add_argument('--hyper_param_epoch',type=int,default=2000,help='epoch number')
parser.add_argument('--hyper_param_learning_rate',type=float,default=0.0001,help='learning rate')
parser.add_argument('--start_epoch',type=int,default=1,help='start epoch number')
parser.add_argument('--cuda', default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
opt = parser.parse_args()

cudnn.benchmark = True    ###내장된 cudnn 자동 튜너를 활성화하여, 하드웨어에 맞게 사용할 최상의 알고리즘을 찾는다.
torch.manual_seed(opt.seed)  ## 초기화 할때 마다 동일한 random num 을 사용하기 위함.
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)
torch.cuda.empty_cache()
classes = ('can','fabric','paper','plastic','pottery','robber','wood')  ##class 7개
transforms_train = transforms.Compose([transforms.RandomRotation(10.),
                                       transforms.RandomVerticalFlip(p=0.5),
                                       transforms.ToTensor()])
transforms_test = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# train_data_set = CustomImageDataset(data_set_path="D:/data_mat1/train/nir_3ch", path="D:/data_mat1/train/nir_1ch",
#                                     path2="D:/data_mat1/train/rgb", path3 = "D:/data_mat1/train/nir_8ch", transforms=None)
train_data_set2 = CustomImageDataset2(data_set_path="D:/ICEIC/classification/train/nir_56ch", path="D:/ICEIC/classification/train/nir_1ch",
                                    transforms = transforms_train)
# train_data_set2 = CustomImageDataset3(data_set_path="D:/data_mat1/train/non_local", path="D:/data_mat1/train/rgb", transforms = transforms_train)

# print(len(train_data_set))
print(len(train_data_set2))
train_loader = DataLoader(train_data_set2, batch_size=opt.hyper_param_batch, shuffle=True,pin_memory = True, drop_last=True)

test_data_set = CustomImageDataset2(data_set_path="D:/ICEIC/classification/test/nir_56ch", path="D:/ICEIC/classification/test/nir_1ch",
                                    transforms = transforms_test)
# test_data_set = CustomImageDataset3(data_set_path="D:/data_mat1/test/non_local", path="D:/data_mat1/test/rgb",transforms=transforms_test)

test_loader = DataLoader(test_data_set, batch_size=opt.hyper_param_batch, shuffle=True, pin_memory = True)
print(len(test_data_set))

# custom_model = CustomConvNet(num_classes=num_classes).to(device)
# custom_model = AlexNet().to(device)
# custom_model = resnet().to(device)
# custom_model = efficientnet_b0().to(device)
# custom_model = dongkeun().to(device)
custom_model = efficientnet_b0().to(device)
# custom_model2 = NonLocalSparseAttention().to(device)
custom_model = torch.nn.DataParallel(custom_model)  ## 당신의 데이터를 자동으로 분할하고 여러 gpu에 있는 다수의 모델에 작업을 지시합니다.
# custom_model2 = torch.nn.DataParallel(custom_model2)
custom_model3 = Attentionmap(ch=1).to(device)
custom_model3 = torch.nn.DataParallel(custom_model3)
# custom_model3 = FcaBasicBlock().to(device)
# custom_model3 = torch.nn.DataParallel(custom_model3)
# custom_model = dongkeun().to(device)
# custom_model = torch.nn.DataParallel(custom_model)

# custom_model2 = Nonlocal().to(device)
# custom_model2 = torch.nn.DataParallel(custom_model2)

# Loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(custom_model.parameters(), lr=opt.hyper_param_learning_rate,betas=(opt.beta1, 0.999))

PATH = './weight/3dconv_1ch'
if not os.path.exists(PATH):
    os.makedirs(PATH)
net_path1 = PATH + "/net_epoch_{}.pth".format(opt.start_epoch-1)

if os.path.isfile(net_path1):
    checkpoint_load = torch.load(net_path1)
    custom_model.load_state_dict(checkpoint_load['model_state_dict'])
    custom_model3.load_state_dict(checkpoint_load['model_state_dict2'])
    print('successfully loaded checkpoints!')
loss_list0 = []
loss_list = []
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}
best_epoch = 0
best_accuarcy = 0
best_weights = copy.deepcopy(custom_model.state_dict())
logfile = PATH + '/eval.txt'
def eval(e):
    custom_model.eval()
    custom_model3.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for item1 in test_loader:
            images1 = item1[0].to(device)
            labels1 = item1[1].to(device)
            outputs2 = custom_model3(images1)
            outputs = custom_model(outputs2)
            _,predicted = torch.max(outputs.data,1)
            total += len(labels1)
            correct += (predicted ==labels1).sum().item()
            for label, prediction in zip(labels1,predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]]+=1
        print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct/total))
        log('Epoch[%d] : Test Avg loss = %.4f \n' % (e,100 * correct/total), logfile)
    return correct, total




for e in range(opt.start_epoch,opt.hyper_param_epoch+1):
    for i_batch, item in enumerate(train_loader):

        images = item[0].to(device)
        labels = item[1].to(device)
        # Forward pass
        outputs2 = custom_model3(images)
        outputs = custom_model(outputs2)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [{}]({}/{}), Loss: {:.12f}'.format(e,i_batch,len(train_loader), loss.item()))
        loss_list0.append(loss.item())

    avg_loss = sum(loss_list0)/len(train_loader)
    loss_list0.clear()
    loss_list.append(avg_loss)
    if e %50 ==0:
        correct, total = eval(e)
        if correct > best_accuarcy:
            best_epoch = e
            best_accuarcy = correct
            best_weights = copy.deepcopy(custom_model.state_dict())
            best_weights2 = copy.deepcopy(custom_model3.state_dict())
        net_path = PATH + "/net_epoch_{}.pth".format(e)
        torch.save({'epoch': e,
                    'model_state_dict':custom_model.state_dict(),
                    'optimizer_state_dict':optimizer.state_dict(),
                    'model_state_dict2':custom_model3.state_dict(),
                    },net_path)
        print("Checkpoint saved to {}".format(net_path))

    if e == 300:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))
    if e == 700:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

print('Best epoch: {}, accuarcy :{} %'.format(best_epoch, (best_accuarcy/total)*100))
net_path2 = PATH + "/best_{}.pth".format(best_epoch)
torch.save({'model_state_dict':best_weights,
            # 'model_state_dict2':best_weights2,
            'optimizer_state_dict':optimizer.state_dict()
            } , PATH + "/best.pth")
print('successfully save best weights pth file')



# num_epochs = opt.hyper_param_epoch
# plt.title('Train loss')
# plt.plot(range(1,num_epochs+1),loss_list,label = 'train')
# plt.ylabel('loss')
# plt.xlabel('training epochs')
# plt.legend()
# plt.show()

