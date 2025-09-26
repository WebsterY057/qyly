import tensorflow as tf
from tensorflow.keras import layers

# 一个简单的GAN生成器模型
def build_generator(latent_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=latent_dim))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(28*28*1, activation='tanh'))  # 假设生成28x28的灰度图像
    model.add(layers.Reshape((28, 28, 1)))
    return model

# 示例：构建生成器
latent_dim = 100
generator = build_generator(latent_dim)
generator.summary()


#######简单神经网络实现
import numpy as np
import matplotlib.pyplot as plt


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # 输入到隐藏层的权重
        self.b1 = np.zeros((1, hidden_size))  # 隐藏层偏置
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # 隐藏到输出层的权重
        self.b2 = np.zeros((1, output_size))  # 输出层偏置

    def sigmoid(self, x):
        # Sigmoid激活函数
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))  # 防止数值溢出

    def sigmoid_derivative(self, x):
        # Sigmoid函数的导数
        return x * (1 - x)

    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1  # 隐藏层输入
        self.a1 = self.sigmoid(self.z1)  # 隐藏层激活输出
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # 输出层输入
        self.a2 = self.sigmoid(self.z2)  # 最终输出
        return self.a2

    def backward(self, X, y, output, learning_rate=0.1):
        # 反向传播
        m = X.shape[0]  # 样本数量

        # 计算输出层误差
        dZ2 = output - y  # 输出层误差
        dW2 = (1 / m) * np.dot(self.a1.T, dZ2)  # 输出层权重梯度
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)  # 输出层偏置梯度

        # 计算隐藏层误差
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)  # 隐藏层误差
        dW1 = (1 / m) * np.dot(X.T, dZ1)  # 隐藏层权重梯度
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)  # 隐藏层偏置梯度

        # 更新参数
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        # 训练网络
        losses = []
        for i in range(epochs):
            output = self.forward(X)
            loss = np.mean((output - y) ** 2)  # 均方误差
            losses.append(loss)

            self.backward(X, y, output, learning_rate)

            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

        return losses


# 示例：解决XOR问题
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 输入特征
y = np.array([[0], [1], [1], [0]])  # 目标输出（XOR结果）

# 创建并训练网络
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
losses = nn.train(X, y, epochs=1000, learning_rate=0.1)

# 测试训练结果
print("\n测试结果:")
for i in range(len(X)):
    prediction = nn.forward(X[i:i + 1])
    print(f"输入: {X[i]}, 预测: {prediction[0][0]:.4f}, 实际: {y[i][0]}")

# 绘制损失曲线
plt.plot(losses)
plt.title('训练损失曲线')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.show()