import torch
import torch.nn as nn

# 定义 LSTM 模型，包括输入、隐藏和输出层
class LSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device=x.device)
        
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        out = self.fc(out[:, -1, :])
        
        return out

# 定义输入、隐藏和输出层的维度
input_dim = 1
hidden_dim = 32
output_dim = 1
num_layers = 2

# 实例化模型
model = LSTMPredictor(input_dim, hidden_dim, output_dim, num_layers)

# 定义损失函数和优化器
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 定义训练数据和标签
train_data = [...] # 训练数据
train_labels = [...] # 训练标签

# 训练模型
for epoch in range(num_epochs):
    inputs = torch.from_numpy(train_data).float().unsqueeze(0) # 将数据转换为 PyTorch 张量
    labels = torch.from_numpy(train_labels).float().unsqueeze(0) # 将标签转换为 PyTorch 张量
    
    optimizer.zero_grad() # 梯度归零
    
    outputs = model(inputs) # 前向传播
    loss = loss_function(outputs, labels) # 计算损失
    
    loss.backward() # 反向传播
    optimizer.step() # 更新权重
    
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# 使用模型进行预测
test_data = [...] # 测试数据

with torch.no_grad():
    inputs = torch.from_numpy(test_data).float().unsqueeze(0) # 将测试数据转换为 PyTorch 张量
    outputs = model(inputs) # 前向传播
    
print(outputs.item()) # 输出预测值