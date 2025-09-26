from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram

# 创建量子电路：3个量子比特，3个经典比特
qc = QuantumCircuit(3, 3)

# 添加H门到第一个量子比特，创建叠加态
qc.h(0)

# 添加CNOT门，创建纠缠态
qc.cx(0, 1)
qc.cx(1, 2)

# 测量所有量子比特到经典比特
qc.measure([0,1,2], [0,1,2])

# 模拟器运行
simulator = QasmSimulator()
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts(compiled_circuit)
print("计数结果:", counts)

# 可视化
plot_histogram(counts)

#####量子电路模拟（Python + Qiskit）
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# 创建量子电路
qc = QuantumCircuit(3, 3)  # 3个量子比特，3个经典比特

# 应用量子门
qc.h(0)  # 在第一个量子比特上应用Hadamard门（创建叠加态）
qc.cx(0, 1)  # 应用CNOT门，创建纠缠（0控制，1目标）
qc.cx(1, 2)  # 应用CNOT门，扩展纠缠（1控制，2目标）
qc.measure([0, 1, 2], [0, 1, 2])  # 测量所有量子比特到经典比特

# 绘制电路图
print("量子电路:")
print(qc.draw())

# 使用模拟器运行
simulator = AerSimulator()
compiled_circuit = transpile(qc, simulator)
job = simulator.run(compiled_circuit, shots=1000)
result = job.result()
counts = result.get_counts()

print("\n测量结果:")
print(counts)

# 可视化结果
plot_histogram(counts)
plt.title('量子电路测量结果')
plt.show()