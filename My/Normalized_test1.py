import numpy as np
import torch
from sklearn.preprocessing import StandardScaler,MinMaxScaler

# 生成一些随机数据作为示例
true_data = np.array([3, 8, 12, 15, 21, 24, 30])

# 创建标准化器

scaler_train = StandardScaler()
scaler_test = StandardScaler()

# scaler_train = MinMaxScaler(feature_range=(0, 1))
# scaler_test = MinMaxScaler(feature_range=(0, 1))

# 训练集和测试集划分
train_size = 0.8
test_size = 0.2
train_data = true_data[:int(train_size * len(true_data))]
test_data = true_data[-int(test_size * len(true_data)):]
print("训练集尺寸:", len(train_data))
print("测试集尺寸:", len(test_data))

# 进行标准化处理
train_data_normalized = scaler_train.fit_transform(train_data.reshape(-1, 1))
test_data_normalized = scaler_test.fit_transform(test_data.reshape(-1, 1))

# 转化为深度学习模型需要的类型Tensor
train_data_normalized = torch.FloatTensor(train_data_normalized)
test_data_normalized = torch.FloatTensor(test_data_normalized)

print("标准化后的训练集数据:")
print(train_data_normalized)

print("标准化后的测试集数据:")
print(test_data_normalized)

mean_value = scaler_train.mean_
std_deviation = np.sqrt(scaler_train.var_)
print("训练集均值:", mean_value)
print("训练集标准差:", std_deviation)
