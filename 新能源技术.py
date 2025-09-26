import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 模拟电池放电曲线（电压随时间下降）
time = np.linspace(0, 10, 100)
voltage = 3.7 - 0.1 * time  # 线性下降模型

plt.plot(time, voltage)
plt.title('电池放电曲线')
plt.xlabel('时间 (小时)')
plt.ylabel('电压 (V)')
plt.grid(True)
plt.show()

####太阳能电池性能模拟
import numpy as np
import matplotlib.pyplot as plt


class SolarCell:
    def __init__(self, efficiency=0.15, area=1.0, temperature_coeff=-0.004):
        self.efficiency = efficiency  # 转换效率
        self.area = area  # 面积 (m²)
        self.temperature_coeff = temperature_coeff  # 温度系数

    def calculate_power(self, solar_irradiance, temperature=25):
        # 计算输出功率
        # 标准测试条件: 1000W/m², 25°C
        standard_irradiance = 1000  # W/m²
        standard_temp = 25  # °C

        # 温度修正
        temp_correction = 1 + self.temperature_coeff * (temperature - standard_temp)

        # 计算功率
        power = (solar_irradiance / standard_irradiance) * self.efficiency * self.area * temp_correction * 1000  # 转换为瓦特

        return max(0, power)  # 功率不能为负

    def daily_energy(self, hourly_irradiance, temperatures):
        # 计算日发电量
        total_energy = 0
        for irradiance, temp in zip(hourly_irradiance, temperatures):
            power = self.calculate_power(irradiance, temp)
            total_energy += power  # 每小时功率累加（假设每小时数据）

        return total_energy / 1000  # 转换为千瓦时


# 模拟一天的光照数据（每小时）
hours = np.arange(24)
# 模拟光照强度（W/m²），呈正态分布
solar_irradiance = 1000 * np.exp(-0.5 * ((hours - 12) / 3) ** 2)
# 模拟温度变化
temperatures = 20 + 10 * np.sin((hours - 6) * np.pi / 12)

# 创建太阳能电池实例
solar_cell = SolarCell(efficiency=0.18, area=2.0)

# 计算每小时功率
hourly_power = [solar_cell.calculate_power(irrad, temp) for irrad, temp in zip(solar_irradiance, temperatures)]

# 计算日发电量
daily_energy = solar_cell.daily_energy(solar_irradiance, temperatures)

print(f"日发电量: {daily_energy:.2f} kWh")

# 绘制结果
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 光照和温度
ax1.plot(hours, solar_irradiance, 'r-', label='光照强度')
ax1.set_ylabel('光照强度 (W/m²)')
ax1.legend(loc='upper left')

ax1_twin = ax1.twinx()
ax1_twin.plot(hours, temperatures, 'b-', label='温度')
ax1_twin.set_ylabel('温度 (°C)')
ax1_twin.legend(loc='upper right')
ax1.set_title('日照和温度变化')

# 发电功率
ax2.plot(hours, hourly_power, 'g-', label='发电功率')
ax2.set_xlabel('小时')
ax2.set_ylabel('功率 (W)')
ax2.set_title('太阳能电池发电功率')
ax2.legend()

plt.tight_layout()
plt.show()

