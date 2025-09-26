from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.constants import GM_earth
import numpy as np
from astropy import units as u  # 导入astropy单位模块

# 定义轨道参数：半长轴（需要带单位），偏心率
a = 7000 * u.km  # 半长轴，单位公里
ecc = 0.1 * u.one  # 偏心率，无量纲

# 创建轨道 - 修复：所有角度参数需要带单位
# Orbit.from_classical参数：地球，半长轴，偏心率，倾角，升交点赤经，近地点幅角，真近点角
orb = Orbit.from_classical(Earth, a, ecc, 0*u.deg, 0*u.deg, 0*u.deg, 0*u.deg)  # 修复：角度参数添加单位

# 打印轨道周期
period = orb.period.to('h')  # 将周期转换为小时单位
print("轨道周期: ", period)


####卫星轨道模拟
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SatelliteOrbit:
    def __init__(self):
        self.G = 6.67430e-11  # 万有引力常数 (m³/kg/s²)
        self.M_earth = 5.972e24  # 地球质量 (kg)
        self.R_earth = 6371e3  # 地球半径 (m)
        self.mu = self.G * self.M_earth  # 地球引力参数

    def orbital_velocity(self, altitude):
        """计算圆形轨道速度
        参数: altitude - 海拔高度 (m)
        返回: 轨道速度 (m/s)
        """
        r = self.R_earth + altitude  # 计算地心距离
        return np.sqrt(self.mu / r)  # 圆形轨道速度公式: v = sqrt(GM/r)

    def orbital_period(self, altitude):
        """计算轨道周期
        参数: altitude - 海拔高度 (m)
        返回: 轨道周期 (秒)
        """
        r = self.R_earth + altitude  # 计算地心距离
        return 2 * np.pi * np.sqrt(r ** 3 / self.mu)  # 开普勒第三定律: T = 2π√(r³/μ)

    def two_body_equations(self, t, y):
        """二体问题运动方程 - 描述卫星在地球引力场中的运动
        参数:
            t - 时间
            y - 状态向量 [x, y, z, vx, vy, vz]
        返回: 状态向量的导数 [vx, vy, vz, ax, ay, az]
        """
        x, y, z, vx, vy, vz = y  # 解包状态向量
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)  # 计算当前位置的地心距离

        # 计算加速度 (牛顿万有引力定律: a = -GM*r/|r|³)
        ax = -self.mu * x / r ** 3
        ay = -self.mu * y / r ** 3
        az = -self.mu * z / r ** 3

        return [vx, vy, vz, ax, ay, az]  # 返回状态向量的导数

    def simulate_orbit(self, altitude, inclination=0, eccentricity=0, duration=None):
        """模拟卫星轨道
        参数:
            altitude - 轨道高度 (m)
            inclination - 轨道倾角 (度)
            eccentricity - 偏心率
            duration - 模拟时长 (秒)，默认2个轨道周期
        返回: 时间数组和状态向量
        """
        if duration is None:
            duration = self.orbital_period(altitude) * 2  # 默认模拟两个轨道周期

        # 初始条件设置
        r = self.R_earth + altitude  # 圆形轨道半径
        v_circular = self.orbital_velocity(altitude)  # 圆形轨道速度

        # 考虑椭圆轨道：近地点距离
        r0 = r * (1 - eccentricity)  # 近地点距离（椭圆轨道）
        # 椭圆轨道速度公式: v = sqrt(μ(2/r - 1/a))，其中a是半长轴
        v0 = np.sqrt(self.mu * (2 / r0 - 1 / r))  # 速度大小

        # 考虑轨道倾角
        inclination_rad = np.radians(inclination)  # 将角度转换为弧度

        # 初始位置和速度设置
        x0 = r0  # 初始在x轴正方向（近地点）
        y0 = 0
        z0 = 0

        vx0 = 0  # 初始x方向速度为零
        vy0 = v0 * np.cos(inclination_rad)  # y方向速度（考虑倾角）
        vz0 = v0 * np.sin(inclination_rad)  # z方向速度（考虑倾角）

        # 初始状态向量 [x, y, z, vx, vy, vz]
        y0 = [x0, y0, z0, vx0, vy0, vz0]

        # 时间范围设置
        t_span = (0, duration)  # 时间区间
        t_eval = np.linspace(0, duration, 1000)  # 评估点，用于输出结果

        # 求解微分方程（使用Runge-Kutta方法）
        solution = solve_ivp(self.two_body_equations, t_span, y0,
                           t_eval=t_eval, method='RK45', rtol=1e-8)

        return solution.t, solution.y

    def plot_orbit(self, altitude=400e3, inclination=45, eccentricity=0.1):
        """绘制轨道
        参数:
            altitude - 轨道高度 (m)
            inclination - 轨道倾角 (度)
            eccentricity - 偏心率
        """
        t, y = self.simulate_orbit(altitude, inclination, eccentricity)
        x, y, z = y[0], y[1], y[2]  # 提取位置坐标

        fig = plt.figure(figsize=(15, 5))

        # 1. 3D轨道图
        ax1 = fig.add_subplot(131, projection='3d')

        # 绘制地球球体
        u_angle = np.linspace(0, 2 * np.pi, 100)  # 经度角度
        v_angle = np.linspace(0, np.pi, 100)  # 纬度角度
        x_earth = self.R_earth * np.outer(np.cos(u_angle), np.sin(v_angle))
        y_earth = self.R_earth * np.outer(np.sin(u_angle), np.sin(v_angle))
        z_earth = self.R_earth * np.outer(np.ones(np.size(u_angle)), np.cos(v_angle))
        ax1.plot_surface(x_earth, y_earth, z_earth, color='blue', alpha=0.3)

        # 绘制卫星轨道
        ax1.plot(x, y, z, 'r-', linewidth=1)
        ax1.scatter(x[0], y[0], z[0], color='green', s=50, label='起始点')
        ax1.scatter(x[-1], y[-1], z[-1], color='red', s=50, label='结束点')

        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'卫星轨道\n高度: {altitude / 1000:.0f} km, 倾角: {inclination}°')
        ax1.legend()

        # 2. 2D投影图（XY平面）
        ax2 = fig.add_subplot(132)
        ax2.plot(x, y, 'b-')  # 绘制轨道在XY平面的投影
        # 绘制地球截面
        ax2.add_patch(plt.Circle((0, 0), self.R_earth, color='blue', alpha=0.3))
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('XY平面投影')
        ax2.axis('equal')  # 保持纵横比相等

        # 3. 高度随时间变化图
        ax3 = fig.add_subplot(133)
        altitude_km = (np.sqrt(x ** 2 + y ** 2 + z ** 2) - self.R_earth) / 1000  # 计算高度(km)
        time_hours = t / 3600  # 将时间转换为小时
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
        """计算霍曼转移轨道参数（最节能的轨道转移方式）
        参数:
            r1 - 初始轨道半径 (m)
            r2 - 目标轨道半径 (m)
        返回: 转移轨道参数字典
        """
        # 转移椭圆轨道的半长轴
        a_transfer = (r1 + r2) / 2

        # 第一次速度增量（从圆轨道到转移轨道）
        v1_circular = np.sqrt(self.mu / r1)  # 初始圆轨道速度
        v1_transfer = np.sqrt(self.mu * (2 / r1 - 1 / a_transfer))  # 转移轨道在近地点速度
        delta_v1 = v1_transfer - v1_circular  # 第一次速度增量

        # 第二次速度增量（从转移轨道到目标圆轨道）
        v2_circular = np.sqrt(self.mu / r2)  # 目标圆轨道速度
        v2_transfer = np.sqrt(self.mu * (2 / r2 - 1 / a_transfer))  # 转移轨道在远地点速度
        delta_v2 = v2_circular - v2_transfer  # 第二次速度增量

        total_delta_v = abs(delta_v1) + abs(delta_v2)  # 总速度增量
        transfer_time = np.pi * np.sqrt(a_transfer ** 3 / self.mu)  # 转移时间（半个椭圆周期）

        return {
            'delta_v1': delta_v1,
            'delta_v2': delta_v2,
            'total_delta_v': total_delta_v,
            'transfer_time': transfer_time
        }


# 运行轨道模拟
print("=== 使用poliastro库计算轨道 ===")
orbit_sim = SatelliteOrbit()

# 绘制低地球轨道示例
print("\n=== 卫星轨道模拟 ===")
orbit_sim.plot_orbit(altitude=400e3, inclination=45, eccentricity=0.01)

# 计算霍曼转移（从低轨道到地球静止轨道）
print("\n=== 霍曼转移计算 ===")
r_low = orbit_sim.R_earth + 400e3  # 400km低轨道半径
r_high = orbit_sim.R_earth + 35786e3  # 地球静止轨道高度半径
transfer = orbit_sim.hohmann_transfer(r_low, r_high)

print(f"霍曼转移参数:")
print(f"第一次加速 Δv: {transfer['delta_v1'] / 1000:.2f} km/s")
print(f"第二次加速 Δv: {transfer['delta_v2'] / 1000:.2f} km/s")
print(f"总速度增量: {transfer['total_delta_v'] / 1000:.2f} km/s")
print(f"转移时间: {transfer['transfer_time'] / 3600:.1f} 小时")