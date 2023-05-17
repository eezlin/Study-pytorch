import pandas as pd
import numpy as np
import statsmodels.api as sm

# 读取 CSV 文件并按时间升序排列
df = pd.read_csv('12.csv', parse_dates=[0], index_col=[0]).sort_index()

# 将功率列转换为 NumPy 数组
power = df.iloc[:, 1].values

# 将时间列转换为时间戳并计算时间间隔
timestamps = pd.to_datetime(df.index).values.astype(np.int64) // 10 ** 9
time_diff = np.diff(timestamps)

# 定义训练集和测试集的划分比例
train_ratio = 0.8
test_ratio = 1 - train_ratio

# 计算训练集和测试集的划分点
split_point = int(len(power) * train_ratio)

# 将功率数据进行分割
train_power = power[:split_point]
test_power = power[split_point:]

# 使用 diff() 函数计算差分，并去除第一个 NaN 值
train_diff = np.diff(train_power)
train_diff = np.concatenate(([train_power[0]], train_diff))
test_diff = np.diff(test_power)
test_diff = np.concatenate(([train_power[-1]], test_diff))

# 构建 ARIMA 模型
model = sm.tsa.ARIMA(endog=train_diff, order=(2, 1, 2))

# 用训练数据进行拟合，并得到预测结果
result = model.fit()
forecast = result.predict(start=split_point, end=len(power)-1)

# 将预测结果转换为原始功率数据的格式
forecast_power = np.zeros(len(power))
forecast_power[:split_point] = train_power[:split_point]
forecast_power[split_point:] = test_diff.cumsum() + forecast.cumsum()[0]

# 输出预测结果
print(forecast_power)

# 计算预测误差（MSE）
mse = np.mean((test_power - forecast_power[split_point:]) ** 2)
print("MSE: %.2f" % mse)