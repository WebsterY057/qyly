import numpy as np

def lennard_jones_potential(r, epsilon=1.0, sigma=1.0):
    """
    计算Lennard-Jones势能
    r: 原子间距离
    epsilon: 势阱深度
    sigma: 势能为零时的距离
    """
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)

# 示例：计算不同距离下的势能
r_values = np.linspace(0.9, 3.0, 100)
potentials = [lennard_jones_potential(r) for r in r_values]

import matplotlib.pyplot as plt
plt.plot(r_values, potentials)
plt.title('Lennard-Jones势能')
plt.xlabel('原子间距离')
plt.ylabel('势能')
plt.grid(True)
plt.show()

####分子动力学模拟
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MolecularDynamics:
    def __init__(self, num_particles=50, box_size=10.0, temperature=300.0):
        self.num_particles = num_particles
        self.box_size = box_size
        self.temperature = temperature
        self.kB = 8.617333262145e-5  # 玻尔兹曼常数 (eV/K)

        # 初始化粒子位置和速度
        self.positions = np.random.rand(num_particles, 3) * box_size
        self.velocities = np.random.randn(num_particles, 3) * np.sqrt(self.kB * temperature)

        # 力场参数 (Lennard-Jones势能)
        self.epsilon = 1.0  # 势能深度
        self.sigma = 1.0  # 粒子直径

        self.trajectory = []  # 存储轨迹

    def lennard_jones_force(self, r):
        """计算Lennard-Jones力"""
        if r == 0:
            return 0
        return 24 * self.epsilon * (2 * (self.sigma / r) ** 12 - (self.sigma / r) ** 6) / r

    def calculate_forces(self):
        """计算所有粒子间的力"""
        forces = np.zeros((self.num_particles, 3))

        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                # 计算粒子间距离（考虑周期性边界条件）
                r_vec = self.positions[j] - self.positions[i]
                r_vec = r_vec - self.box_size * np.round(r_vec / self.box_size)
                r = np.linalg.norm(r_vec)

                if r > 0 and r < self.box_size / 2:  # 截断半径
                    # 计算力的大小
                    force_magnitude = self.lennard_jones_force(r)
                    # 计算力向量
                    force_vec = force_magnitude * (r_vec / r)

                    forces[i] += force_vec
                    forces[j] -= force_vec

        return forces

    def kinetic_energy(self):
        """计算动能"""
        return 0.5 * np.sum(self.velocities ** 2)

    def potential_energy(self):
        """计算势能"""
        potential = 0.0

        for i in range(self.num_particles):
            for j in range(i + 1, self.num_particles):
                r_vec = self.positions[j] - self.positions[i]
                r_vec = r_vec - self.box_size * np.round(r_vec / self.box_size)
                r = np.linalg.norm(r_vec)

                if r > 0 and r < self.box_size / 2:
                    # Lennard-Jones势能
                    potential += 4 * self.epsilon * ((self.sigma / r) ** 12 - (self.sigma / r) ** 6)

        return potential

    def total_energy(self):
        """计算总能量"""
        return self.kinetic_energy() + self.potential_energy()

    def verlet_integration(self, dt=0.001):
        """Verlet积分算法更新位置和速度"""
        forces = self.calculate_forces()

        # 更新位置
        self.positions += self.velocities * dt + 0.5 * forces * dt ** 2

        # 应用周期性边界条件
        self.positions = self.positions % self.box_size

        # 计算新力
        new_forces = self.calculate_forces()

        # 更新速度
        self.velocities += 0.5 * (forces + new_forces) * dt

        # 温度控制（简单的速度缩放）
        current_temperature = (2 / 3) * self.kinetic_energy() / (self.num_particles * self.kB)
        if current_temperature > 0:
            scaling_factor = np.sqrt(self.temperature / current_temperature)
            self.velocities *= scaling_factor

    def simulate(self, steps=1000, save_interval=10):
        """运行模拟"""
        energies = []

        for step in range(steps):
            self.verlet_integration()

            if step % save_interval == 0:
                self.trajectory.append(self.positions.copy())

                ke = self.kinetic_energy()
                pe = self.potential_energy()
                te = self.total_energy()
                energies.append((ke, pe, te))

                if step % 100 == 0:
                    print(f"步骤 {step}: 动能={ke:.4f}, 势能={pe:.4f}, 总能量={te:.4f}")

        return np.array(energies)

    def visualize(self, step_interval=10):
        """可视化模拟结果"""
        fig = plt.figure(figsize=(15, 5))

        # 3D轨迹图
        ax1 = fig.add_subplot(131, projection='3d')
        for i in range(min(10, len(self.trajectory))):  # 显示前10帧
            frame = self.trajectory[i * step_interval]
            ax1.scatter(frame[:, 0], frame[:, 1], frame[:, 2], alpha=0.5)
        ax1.set_title('粒子轨迹')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # 能量图
        ax2 = fig.add_subplot(132)
        energies = self.simulate(steps=100, save_interval=1)
        time_steps = np.arange(len(energies)) * step_interval
        ax2.plot(time_steps, energies[:, 0], label='动能')
        ax2.plot(time_steps, energies[:, 1], label='势能')
        ax2.plot(time_steps, energies[:, 2], label='总能量')
        ax2.set_title('能量随时间变化')
        ax2.set_xlabel('时间步')
        ax2.set_ylabel('能量')
        ax2.legend()

        # 径向分布函数
        ax3 = fig.add_subplot(133)
        rdf = self.calculate_radial_distribution()
        ax3.plot(rdf[0], rdf[1])
        ax3.set_title('径向分布函数')
        ax3.set_xlabel('距离')
        ax3.set_ylabel('g(r)')

        plt.tight_layout()
        plt.show()

    def calculate_radial_distribution(self, max_r=None, dr=0.1):
        """计算径向分布函数"""
        if max_r is None:
            max_r = self.box_size / 2

        r_bins = np.arange(0, max_r, dr)
        g_r = np.zeros(len(r_bins))

        # 统计粒子对距离
        for frame in self.trajectory:
            for i in range(self.num_particles):
                for j in range(i + 1, self.num_particles):
                    r_vec = frame[j] - frame[i]
                    r_vec = r_vec - self.box_size * np.round(r_vec / self.box_size)
                    r = np.linalg.norm(r_vec)

                    if r < max_r:
                        bin_index = int(r / dr)
                        if bin_index < len(g_r):
                            g_r[bin_index] += 2  # 每对计数两次

        # 归一化
        volume_shell = 4 * np.pi * r_bins ** 2 * dr
        ideal_gas_density = self.num_particles / (self.box_size ** 3)
        normalization = self.num_particles * ideal_gas_density * volume_shell * len(self.trajectory)

        g_r = g_r / normalization

        return r_bins, g_r


# 运行模拟
md = MolecularDynamics(num_particles=20, box_size=8.0, temperature=300)
md.visualize()