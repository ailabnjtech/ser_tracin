"""引入依赖项"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim      #  进行了优化操作
from torch.utils.data import Dataset,DataLoader  # 批量提取数据，并且输出和标签相对应输出
import numpy as np
import librosa         # 与音频处理相关的库
from tqdm import tqdm  # 使用进度条，方便显示
          #  glob 文件名模式匹配，不用遍历整个目录判断每个文件是不是符合。
import os
import pickle          # pickle是二进制序列化格式;
import random          # 随机的概念: 在某个范围内取到的每一个值的概率是相同的
import logging         # 日志文件
from model import Cnn_Transformer
from data_getpy import get_data
from feture_extractor import FeatureExtractor
from torch.autograd import grad

from pathlib import Path
from torch.autograd import Variable
import time
from pif.influence_functions_new import pick_gradient,param_vec_dot_product
from pif.utils import save_json
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
# from tensorflow.keras.models import load_model

# 用3个checkpoint生成3个字典


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
"""定义注意力卷积模型"""
setup_seed(2222220)  # seed( ) 用于指定随机数生成时所用算法开始的整数值
attention_head = 4
attention_hidden = 32
learning_rate = 0.001
BATCH_SIZE = 1
FEATURES_TO_USE_NEXT = 'mfcc'  # {'mfcc' , 'logfbank','fbank','spectrogram','melspectrogram'}
FEATURES_TO_USE = 'logfbank'
impro_or_script = 'impro'
featuresFileName = 'features_{}_{}.pkl'.format(FEATURES_TO_USE, impro_or_script)
featuresExist = False
toSaveFeatures = True
WAV_PATH = "IEMOCAP/"
RATE = 16000
MODEL_NAME = 'Cnn_Transformer'    # 使用上面定义的
MODEL_PATH = '{}_{}_{}_222222.pth'.format(MODEL_NAME, FEATURES_TO_USE, FEATURES_TO_USE_NEXT) # 定义的模型的路径
topk=100

dict = {
    'neutral': torch.Tensor([0]),
    'happy': torch.Tensor([1]),
    'sad': torch.Tensor([2]),
    'angry': torch.Tensor([3]),
    'boring': torch.Tensor([4]),
    'fear': torch.Tensor([5]),
}
getdata=get_data(featuresExist,featuresFileName,WAV_PATH,RATE,
                FEATURES_TO_USE,toSaveFeatures,BATCH_SIZE,impro_or_script,topk)

if (featuresExist == True):
    with open(featuresFileName, 'rb')as f:
        features = pickle.load(f)
    train_X_features = features['train_X']
    train_X_features_NEXT = features['train_X_1']
    train_y = features['train_y']
    train_z = features['train_z']
    test_X_features = features['test_X']
    test_X_features_NEXT = features['test_X_1']
    test_y = features['test_y']
    test_z = features['test_z']
else:
    logging.info("creating meta dict...")
    train_X, train_y, train_z,  test_X, test_y,test_z = getdata.getdata_Tracin(WAV_PATH)
    print(train_X.shape)


    print("getting features")
    logging.info('getting features')
    feature_extractor = FeatureExtractor(rate=RATE)
    train_X_features = feature_extractor.get_features(FEATURES_TO_USE, train_X)
    train_X_features_NEXT = feature_extractor.get_features(FEATURES_TO_USE_NEXT, train_X)
    test_X_features = feature_extractor.get_features(FEATURES_TO_USE, test_X)
    test_X_features_NEXT = feature_extractor.get_features(FEATURES_TO_USE_NEXT, test_X)
    valid_features_dict = {}
    if (toSaveFeatures == True):
        features = {'train_X': train_X_features,'train_X_1': train_X_features_NEXT, 'train_y': train_y,'train_z':train_z,
                    'test_X': test_X_features,'test_X_1': test_X_features_NEXT, 'test_y': test_y,'test_z': test_z,}
        with open(featuresFileName, 'wb') as f:
            pickle.dump(features, f)

class DataSet(object):
    def __init__(self, X,X_1,Y,Z):
        self.X = X
        self.X_1 = X_1
        self.Y = Y
        self.Z = Z

    def __getitem__(self, index):
        x = self.X[index]
        x = torch.from_numpy(x)
        x = x.float()
        x_1 = self.X_1[index]
        x_1 = torch.from_numpy(x_1)
        x_1 = x_1.float()
        y = self.Y[index]
        y = dict[y]
        y = y.long()
        z = self.Z[index]


        return x,x_1,y,z

    def __len__(self):
        return len(self.X)

train_data = DataSet(train_X_features,train_X_features_NEXT, train_y,train_z)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_data = DataSet(test_X_features,test_X_features_NEXT, test_y,test_z) # 初始化了X和Y的值
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
model = Cnn_Transformer(num_emotions=4)  # 调用模型

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_weight_path = "Tracin_second_epcho_1"
model.load_state_dict(torch.load("pth/Tracin_second_epcho_1.pth",map_location="cpu"))
if torch.cuda.is_available():
  model = model.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6) # 更新参数优化
outdir = Path("result_Tracin")
print(device)
model.train()
running_loss = 0.0
time_start = time.perf_counter()
influence_results={}

for i, batch_i in enumerate(tqdm(train_loader)):
    inputs,inputs_1,labels,name = batch_i[0], batch_i[1], batch_i[2], batch_i[3]
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        inputs_1 = inputs_1.cuda()
        labels = labels.cuda()
    _, outputs = model(inputs.unsqueeze(1),inputs_1.unsqueeze(1))

    loss = loss_function(outputs, labels.squeeze(1))
    grad_z_test = grad(loss, model.parameters())
    grad_z_test = pick_gradient(grad_z_test, model)
    train_influences={}
    for j, batch_j in enumerate((test_loader)):
        inputs_train,inputs_train_1, labels_train ,name_train = batch_j[0], batch_j[1] , batch_j[2], batch_j[3] # 获取训练集的语音和标签
        # inputs_train = Variable(torch.unsqueeze(inputs_train, dim=0).float(), requires_grad=False)
        if torch.cuda.is_available():
            inputs_train = inputs_train.cuda()
            inputs_train_1 = inputs_train_1.cuda()
            labels_train = labels_train.cuda()
        # labels2=labels.clone()
        # labels2[0] = 9
        # forward + backward + optimize

        _, outputs_train = model(inputs_train.unsqueeze(1),inputs_train_1.unsqueeze(1))  # 正向传播
        # labels_train = labels_train.view(1)
        max_value, max_idx = torch.max(outputs_train, dim=1)
        # print(max_idx)
        m = labels_train.squeeze(1)
        loss_train = loss_function(outputs_train, m)  # 计算损失
        grad_z_train = grad(loss_train, model.parameters())
        grad_z_train = pick_gradient(grad_z_train, model)
        score = param_vec_dot_product(grad_z_test, grad_z_train)
        # if max_idx == m:
        #     if score >= meann:
        #         score = score * 1
        #     elif score < meann and score >=0:
        #         score = score * 0.5
        #     elif score < 0:
        #         score = 0
        # elif max_idx != m:
        #     if score >= meann:
        #         score = 0
        #     elif score < meann and score >=0:
        #         score = score * 0
        #     elif score < 0:
        #         score = score * (-0.25)

        if j not in train_influences:    #加入json文件保存
            train_influences[j] = {'test_dat': (str(test_loader.dataset.Z[j])),
                                      'if': float(score)}
    if i not in influence_results:
        influence_results[i] = {'train_dat': (str(train_loader.dataset.Z[i])),
                                  'ifs': train_influences}
    if i == len(train_loader)-1:

        save_json(influence_results, outdir.joinpath(f'Tracin-{i}-parallel_{model_weight_path}.json'))
time_end=time.perf_counter()
print(time_end-time_start)


