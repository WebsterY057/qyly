from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator  # 更新后的导入方式
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
# 创建量子电路：3个量子比特，3个经典比特
qc = QuantumCircuit(3, 3)

# 添加H门到第一个量子比特，创建叠加态
qc.h(0)

# 添加CNOT门，创建纠缠态
qc.cx(0, 1)
qc.cx(1, 2)

# 测量所有量子比特到经典比特
qc.measure([0,1,2], [0,1,2])

# 绘制电路图
print("量子电路:")
print(qc.draw())

# 使用AerSimulator运行
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circuit)
print("\n计数结果:", counts)

# 可视化
plot_histogram(counts)
plt.title('GHZ态量子电路测量结果')
plt.show()