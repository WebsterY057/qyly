import tensorflow as tf
from tensorflow.keras import layers
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义一个简单的GAN生成器模型
def build_generator(latent_dim):
    # 创建序列模型（层按顺序堆叠）
    model = tf.keras.Sequential()

    # 第一层全连接层：从潜在空间维度到256个神经元
    model.add(layers.Dense(256, input_dim=latent_dim))
    # 添加LeakyReLU激活函数，负值区域有小的斜率（0.2）
    model.add(layers.LeakyReLU(alpha=0.2))
    # 批量归一化层，加速训练并提高稳定性，momentum控制移动平均的衰减率
    model.add(layers.BatchNormalization(momentum=0.8))

    # 第二层全连接层：扩展到512个神经元
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    # 第三层全连接层：进一步扩展到1024个神经元
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))

    # 输出层：生成28x28=784个像素值，使用tanh激活函数将输出限制在[-1,1]范围
    model.add(layers.Dense(28 * 28 * 1, activation='tanh'))  # 假设生成28x28的灰度图像
    # 将一维向量重塑为二维图像格式（高度，宽度，通道数）
    model.add(layers.Reshape((28, 28, 1)))
    return model


# 示例：构建生成器
latent_dim = 100  # 潜在空间的维度（噪声向量的长度）
generator = build_generator(latent_dim)
# 打印模型结构摘要
generator.summary()

####简单神经网络
import numpy as np
import matplotlib.pyplot as plt


class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        # 输入层到隐藏层的权重矩阵，使用小随机数初始化
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        # 隐藏层的偏置向量，初始化为零
        self.b1 = np.zeros((1, hidden_size))
        # 隐藏层到输出层的权重矩阵
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        # 输出层的偏置向量
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        # Sigmoid激活函数，将输入映射到(0,1)范围
        # 使用np.clip防止指数运算时数值溢出
        return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

    def sigmoid_derivative(self, x):
        # Sigmoid函数的导数，用于反向传播
        # 对于sigmoid(x)，其导数为sigmoid(x)*(1-sigmoid(x))
        # 这里x已经是sigmoid的输出，所以直接计算x*(1-x)
        return x * (1 - x)

    def forward(self, X):
        # 前向传播过程
        # 隐藏层输入：输入数据与权重矩阵相乘加上偏置
        self.z1 = np.dot(X, self.W1) + self.b1
        # 隐藏层激活输出：应用sigmoid函数
        self.a1 = self.sigmoid(self.z1)
        # 输出层输入：隐藏层输出与权重矩阵相乘加上偏置
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # 最终输出：应用sigmoid函数
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate=0.1):
        # 反向传播过程，计算梯度并更新参数
        m = X.shape[0]  # 样本数量（批量大小）

        # 计算输出层误差（损失函数对输出层输入的导数）
        # 使用均方误差的导数：dL/dz2 = (a2 - y) * sigmoid_derivative(a2)
        # 但由于sigmoid_derivative在后续计算中会用到，这里先计算基础误差
        dZ2 = output - y  # 输出层误差

        # 计算输出层权重的梯度：dL/dW2 = (1/m) * a1^T · dZ2
        dW2 = (1 / m) * np.dot(self.a1.T, dZ2)
        # 计算输出层偏置的梯度：dL/db2 = (1/m) * sum(dZ2)
        db2 = (1 / m) * np.sum(dZ2, axis=0, keepdims=True)

        # 计算隐藏层误差：dL/dz1 = dL/da1 * da1/dz1
        # 先计算dL/da1 = dZ2 · W2^T
        dA1 = np.dot(dZ2, self.W2.T)
        # 再乘以sigmoid的导数得到隐藏层误差
        dZ1 = dA1 * self.sigmoid_derivative(self.a1)

        # 计算隐藏层权重的梯度：dL/dW1 = (1/m) * X^T · dZ1
        dW1 = (1 / m) * np.dot(X.T, dZ1)
        # 计算隐藏层偏置的梯度：dL/db1 = (1/m) * sum(dZ1)
        db1 = (1 / m) * np.sum(dZ1, axis=0, keepdims=True)

        # 使用梯度下降更新参数
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        # 训练网络
        losses = []  # 存储每个epoch的损失值
        for i in range(epochs):
            # 前向传播得到预测输出
            output = self.forward(X)
            # 计算均方误差损失
            loss = np.mean((output - y) ** 2)
            losses.append(loss)

            # 反向传播更新参数
            self.backward(X, y, output, learning_rate)

            # 每100个epoch打印一次损失
            if i % 100 == 0:
                print(f"Epoch {i}, Loss: {loss:.4f}")

        return losses


# 示例：解决XOR问题
# XOR问题的输入特征：四种可能的输入组合
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# XOR问题的目标输出：相同为0，不同为1
y = np.array([[0], [1], [1], [0]])

# 创建神经网络实例：2个输入，4个隐藏神经元，1个输出
nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1)
# 训练网络1000个epoch
losses = nn.train(X, y, epochs=1000, learning_rate=0.1)

# 测试训练结果
print("\n测试结果:")
for i in range(len(X)):
    # 对每个测试样本进行预测（保持二维形状）
    prediction = nn.forward(X[i:i + 1])
    print(f"输入: {X[i]}, 预测: {prediction[0][0]:.4f}, 实际: {y[i][0]}")

# 绘制训练损失曲线
plt.plot(losses)
plt.title('训练损失曲线')
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.show()