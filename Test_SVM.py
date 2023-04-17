import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from torch_geometric.datasets import planetoid

dataset_Cora = planetoid.Planetoid(root='./Test_PyTorch/data_Cora/Cora', name='Cora')
# print(dataset_Cora)
print(dataset_Cora.data)
# 数据准备
X = dataset_Cora.data.x.data.numpy()
Y = dataset_Cora.data.y.data.numpy()
train_mask = dataset_Cora.data.train_mask.data.numpy()
test_mask = dataset_Cora.data.test_mask.data.numpy()
# 准备训练数据和测试数据
train_x = X[0:140, :]
train_y = Y[train_mask]
test_x = X[1708:2708, :]
test_y = Y[test_mask]
# 使用训练集训练SVM模型，并使用测试数据进行测试
svmModel = SVC()
svmModel.fit(train_x, train_y)  # 训练
preLab = svmModel.predict(test_x)  # 预测
print("*****SVM的预测精度：", accuracy_score(test_y, preLab))




