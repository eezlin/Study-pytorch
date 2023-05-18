import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 读取 CSV 文件并按时间升序排列
df = pd.read_csv('12.csv', parse_dates=[0], index_col=[0]).sort_index()

# 将功率列转换为 NumPy 数组
power = df.iloc[:, 1].to_numpy()

# 将时间列转换为时间戳并计算时间间隔
timestamps = pd.to_datetime(df.index).values.astype(np.int64) // 10 ** 9
time_diff = np.diff(timestamps)

# 将数据组织成三元组 (时间间隔，动态序列，目标值)
seq_len = 50  # 设置每个动态序列的长度为 50
data = []
for i in range(seq_len, len(power)):
    data.append((time_diff[i - seq_len:i], power[i - seq_len:i], power[i]))


# 定义自定义 Dataset 类
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        time_diff, sequence, target = self.data[index]
        time_diff = torch.from_numpy(time_diff).float()
        sequence = torch.from_numpy(sequence).unsqueeze(1).float()  # 将动态序列转换为二维张量
        target = torch.Tensor([target]).float()
        return time_diff, sequence, target

    def __len__(self):
        return len(self.data)


# 实例化自定义 Dataset 类
dataset = MyDataset(data)

# 使用 DataLoader 加载数据集
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)


# 定义 LSTM 模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, h0, c0):
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out, hn, cn


# 定义输入、隐藏和输出层的维度
input_dim = 1  # 动态序列和目标值都是一维
hidden_dim = 32
output_dim = 1
num_layers = 2

# 实例化模型
model = LSTMPredictor(input_dim, hidden_dim, output_dim, num_layers)

# 定义损失函数和优化器
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 将模型移动到 GPU 上（如果可用的话）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# 定义训练函数
def train(model, dataloader, loss_function, optimizer, device):
    model.train()

    train_loss = 0.0
    for i, (time_diff, sequence, target) in enumerate(dataloader):
        time_diff = time_diff.to(device)
        sequence = sequence.to(device)
        target = target.to(device)

        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(num_layers, sequence.size(0), hidden_dim).to(device)
        c0 = torch.zeros(num_layers, sequence.size(0), hidden_dim).to(device)

        # 前向传播和计算损失
        optimizer.zero_grad()
        output, hn, cn = model(sequence, h0, c0)
        loss = loss_function(output, target)

        # 反向传播和更新权重
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(dataloader)

    return train_loss


# 定义测试函数
def test(model, dataloader, loss_function, device):
    model.eval()

    test_loss = 0.0
    with torch.no_grad():
        for i, (time_diff, sequence, target) in enumerate(dataloader):
            time_diff = time_diff.to(device)
            sequence = sequence.to(device)
            target = target.to(device)

            # 初始化隐藏状态和细胞状态
            h0 = torch.zeros(num_layers, sequence.size(0), hidden_dim).to(device)
            c0 = torch.zeros(num_layers, sequence.size(0), hidden_dim).to(device)

            # 前向传播和计算损失
            output, hn, cn = model(sequence, h0, c0)
            loss = loss_function(output, target)

            test_loss += loss.item()

    test_loss /= len(dataloader)

    return test_loss


# 开始训练模型
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train(model, dataloader, loss_function, optimizer, device)
    test_loss = test(model, dataloader, loss_function, device)
    print('Epoch [{}/{}], Train Loss: {:.4f}, Test Loss: {:.4f}'.format(epoch+1, num_epochs, train_loss, test_loss))

# 使用模型进行预测
test_data = [...] # 测试数据
with torch.no_grad():
    input_sequence = [...] # 输入动态序列
    time_diff = [...] # 时间间隔
    input_sequence = torch.from_numpy(input_sequence).unsqueeze(1).float().to(device)
    time_diff = torch.from_numpy(time_diff).unsqueeze(0).float().to(device)
    h0 = torch.zeros(num_layers, 1, hidden_dim).to(device) # 初始化隐藏状态
    c0 = torch.zeros(num_layers, 1, hidden_dim).to(device) # 初始化细胞状态
    output, hn, cn = model(input_sequence, h0, c0)
    prediction = output.item() # 预测值
    print('Prediction: {}'.format(prediction))