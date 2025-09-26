from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.constants import GM_earth
import numpy as np

# 定义轨道参数：半长轴（单位：米），偏心率
a = 7000 * 1000  # 7000 km
ecc = 0.1

# 创建轨道
orb = Orbit.from_classical(Earth, a, ecc, 0, 0, 0, 0)

# 打印轨道周期
period = orb.period.to('h')
print("轨道周期: ", period)


####卫星轨道模拟
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class SatelliteOrbit:
    def __init__(self):
        self.G = 6.67430e-11  # 万有引力常数 (m³/kg/s²)
        self.M_earth = 5.972e24  # 地球质量 (kg)
        self.R_earth = 6371e3  # 地球半径 (m)
        self.mu = self.G * self.M_earth  # 地球引力参数

    def orbital_velocity(self, altitude):
        """计算圆形轨道速度"""
        r = self.R_earth + altitude
        return np.sqrt(self.mu / r)

    def orbital_period(self, altitude):
        """计算轨道周期"""
        r = self.R_earth + altitude
        return 2 * np.pi * np.sqrt(r ** 3 / self.mu)

    def two_body_equations(self, t, y):
        """二体问题运动方程"""
        x, y, z, vx, vy, vz = y
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # 加速度 (重力)
        ax = -self.mu * x / r ** 3
        ay = -self.mu * y / r ** 3
        az = -self.mu * z / r ** 3

        return [vx, vy, vz, ax, ay, az]

    def simulate_orbit(self, altitude, inclination=0, eccentricity=0, duration=None):
        """模拟卫星轨道"""
        if duration is None:
            duration = self.orbital_period(altitude) * 2  # 模拟两个轨道周期

        # 初始条件 (在赤道平面)
        r = self.R_earth + altitude
        v_circular = self.orbital_velocity(altitude)

        # 考虑轨道倾角
        inclination_rad = np.radians(inclination)

        # 初始位置和速度
        r0 = r * (1 - eccentricity)  # 近地点距离（椭圆轨道）
        v0 = np.sqrt(self.mu * (2 / r0 - 1 / r))  # 速度大小

        x0 = r0
        y0 = 0
        z0 = 0

        vx0 = 0
        vy0 = v0 * np.cos(inclination_rad)
        vz0 = v0 * np.sin(inclination_rad)

        # 初始状态向量
        y0 = [x0, y0, z0, vx0, vy0, vz0]

        # 时间范围
        t_span = (0, duration)
        t_eval = np.linspace(0, duration, 1000)

        # 求解微分方程
        solution = solve_ivp(self.two_body_equations, t_span, y0, t_eval=t_eval, method='RK45')

        return solution.t, solution.y

    def plot_orbit(self, altitude=400e3, inclination=45, eccentricity=0.1):
        """绘制轨道"""
        t, y = self.simulate_orbit(altitude, inclination, eccentricity)
        x, y, z = y[0], y[1], y[2]

        fig = plt.figure(figsize=(15, 5))

        # 3D轨道图
        ax1 = fig.add_subplot(131, projection='3d')

        # 绘制地球
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_earth = self.R_earth * np.outer(np.cos(u), np.sin(v))
        y_earth = self.R_earth * np.outer(np.sin(u), np.sin(v))
        z_earth = self.R_earth * np.outer(np.ones(np.size(u)), np.cos(v))
        ax1.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3)

        # 绘制轨道
        ax1.plot(x, y, z, 'r-', linewidth=1)
        ax1.scatter(x[0], y[0], z[0], color='green', s=50, label='起始点')
        ax1.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='结束点')

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'卫星轨道\n高度: {altitude / 1000:.0f} km, 倾角: {inclination}°')
        ax1.legend()

        # 2D投影图
        ax2 = fig.add_subplot(132)
        ax2.plot(x, y, 'b-')
        ax2.add_patch(plt.Circle((0, 0), self.R_earth, color='blue', alpha=0.3))
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY平面投影')
        ax2.axis('equal')

        # 高度随时间变化
        ax3 = fig.add_subplot(133)
        altitude_km = (np.sqrt(x ** 2 + y ** 2 + z ** 2) - self.R_earth) / 1000
        time_hours = t / 3600
        ax3.plot(time_hours, altitude_km)
        ax3.set_xlabel('时间 (小时)')
        ax3.set_ylabel('高度 (km)')
        ax3.set_title('高度随时间变化')
        ax3.grid(True)

        plt.tight_layout()
        plt.show()

        # 打印轨道参数
        period = self.orbital_period(altitude) / 60  # 转换为分钟
        velocity = self.orbital_velocity(altitude) / 1000  # 转换为km/s

        print(f"轨道参数:")
        print(f"- 高度: {altitude / 1000:.0f} km")
        print(f"- 轨道周期: {period:.1f} 分钟")
        print(f"- 轨道速度: {velocity:.2f} km/s")
        print(f"- 偏心率: {eccentricity}")
        print(f"- 倾角: {inclination}°")

    def hohmann_transfer(self, r1, r2):
        """计算霍曼转移轨道参数"""
        # 转移椭圆轨道的半长轴
        a_transfer = (r1 + r2) / 2

        # 速度增量
        v1_circular = np.sqrt(self.mu / r1)
        v1_transfer = np.sqrt(self.mu * (2 / r1 - 1 / a_transfer))
        delta_v1 = v1_transfer - v1_circular

        v2_circular = np.sqrt(self.mu / r2)
        v2_transfer = np.sqrt(self.mu * (2 / r2 - 1 / a_transfer))
        delta_v2 = v2_circular - v2_transfer

        total_delta_v = abs(delta_v1) + abs(delta_v2)
        transfer_time = np.pi * np.sqrt(a_transfer ** 3 / self.mu)

        return {
            'delta_v1': delta_v1,
            'delta_v2': delta_v2,
            'total_delta_v': total_delta_v,
            'transfer_time': transfer_time
        }


# 运行轨道模拟
orbit_sim = SatelliteOrbit()

# 绘制低地球轨道示例
orbit_sim.plot_orbit(altitude=400e3, inclination=45, eccentricity=0.01)

# 计算霍曼转移
r_low = orbit_sim.R_earth + 400e3  # 400km低轨道
r_high = orbit_sim.R_earth + 35786e3  # 地球静止轨道高度
transfer = orbit_sim.hohmann_transfer(r_low, r_high)

print(f"\n霍曼转移参数:")
print(f"第一次加速 Δv: {transfer['delta_v1'] / 1000:.2f} km/s")
print(f"第二次加速 Δv: {transfer['delta_v2'] / 1000:.2f} km/s")
print(f"总速度增量: {transfer['total_delta_v'] / 1000:.2f} km/s")
print(f"转移时间: {transfer['transfer_time'] / 3600:.1f} 小时")