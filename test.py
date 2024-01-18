from main import *
import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import easydict
# from dongkeun import resnet
from dongkeun2 import efficientnet_b0
from datasetfolder2 import CustomImageDataset2,CustomImageDataset2_test
from datasetfolder2 import CustomImageDataset3
# from attention import NonLocalSparseAttention

from convolution3d import Attentionmap



def save_img(image_tensor,filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = image_numpy
    image_numpy = image_numpy.clip(0,255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))
def save_gray_img(image_tensor,filename):
    image_numpy = image_tensor.float().numpy()

    image_numpy = np.asarray(image_numpy)*255.
    image_numpy = image_numpy.clip(0,255)
    image_numpy = image_numpy.astype(np.uint8)

    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


test_epoch = 1750
opt = easydict.EasyDict({
    "hyper_param_batch" : 1,
    "cuda" : True,
})
print(opt)


# classes = ('can', 'fabric', 'fruit', 'paper','pottery', 'silicon', 'vegetable', 'wood')  ##class 8개
classes = ('can','fabric','paper','plastic','pottery','robber','wood')  ##class 7개
#
transforms_test = transforms.Compose([transforms.ToTensor()])

# transforms_test = transforms.Compose([transforms.RandomRotation(10.),
#                                        transforms.RandomVerticalFlip(p=0.5),
#                                        transforms.ToTensor()])
#

test_data_set = CustomImageDataset2_test(data_set_path="D:/ICEIC/classification/test/nir_56ch", path="D:/ICEIC/classification/test/nir_1ch",
                                    transforms = transforms_test)

test_loader = DataLoader(test_data_set, batch_size=opt.hyper_param_batch, shuffle=True, pin_memory = True)
print(len(test_data_set))

device = torch.device('cuda:0' if opt.cuda else 'cpu')


custom_model = efficientnet_b0().to(device)
custom_model = torch.nn.DataParallel(custom_model)
custom_model3 = Attentionmap(ch=1).to(device)
custom_model3 = torch.nn.DataParallel(custom_model3)



model_path = "./weight/3dconv_1ch/net_epoch_{}.pth".format(test_epoch)
# model_path = "./weight/3dconv_8ch/best.pth"
if os.path.isfile(model_path):
    checkpoint = torch.load(model_path)
    custom_model.load_state_dict(checkpoint['model_state_dict'])
    # custom_model2.load_state_dict(checkpoint['model_state_dict2'])
    custom_model3.load_state_dict(checkpoint['model_state_dict2'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print('successfully loaded checkpoints!')

# 각 분류(class)에 대한 예측값 계산을 위해 준비.convert('rgb')
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}


custom_model.eval()
with torch.no_grad():
    correct = 0
    total = 0

    for i_batch, item in enumerate(test_loader):

        images = item[0].to(device)
        labels = item[1].to(device)

        original = images
        outputs2 = custom_model3(images)
        outputs = custom_model(outputs2)
        _, predicted = torch.max(outputs.data, 1)
        total += len(labels)
        correct += (predicted == labels).sum().item()
        # 각 분류별로 올바른 예측 수를 모읍니다
        for label, prediction in zip(labels, predicted):
            if label == prediction:
                correct_pred[classes[label]] += 1
            else:
                print(classes[label.item()], classes[prediction.item()])
            total_pred[classes[label]] += 1
    print('Test Accuracy of the model on the {} test images: {} %'.format(total, 100 * correct / total))

# 각 분류별 정확도(accuracy)를 출력합니다
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


