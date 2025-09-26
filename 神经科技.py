import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 生成模拟脑电信号（包含噪声）
t = np.linspace(0, 10, 1000)
eeg_signal = np.sin(2 * np.pi * 10 * t)  # 10Hz的脑电信号
noise = 0.5 * np.random.normal(size=1000)  # 随机噪声
eeg_with_noise = eeg_signal + noise

# 设计一个带通滤波器，过滤非10Hz的信号
b, a = signal.butter(4, [9, 11], 'bandpass', fs=100)  # 采样频率100Hz
filtered_eeg = signal.filtfilt(b, a, eeg_with_noise)

# 绘制结果
plt.figure(figsize=(12, 6))
plt.subplot(3,1,1)
plt.plot(t, eeg_signal)
plt.title('原始脑电信号')
plt.subplot(3,1,2)
plt.plot(t, eeg_with_noise)
plt.title('带噪声的脑电信号')
plt.subplot(3,1,3)
plt.plot(t, filtered_eeg)
plt.title('滤波后的脑电信号')
plt.tight_layout()
plt.show()


######脑电信号处理
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


class EEGProcessor:
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate  # 采样率 (Hz)

    def generate_sample_eeg(self, duration=10, noise_level=0.5):
        # 生成模拟脑电信号
        t = np.linspace(0, duration, int(duration * self.sampling_rate))

        # 不同频段的脑电波
        delta = 0.5 * np.sin(2 * np.pi * 2 * t)  # δ波 (1-4 Hz)
        theta = 0.3 * np.sin(2 * np.pi * 6 * t)  # θ波 (4-8 Hz)
        alpha = 0.8 * np.sin(2 * np.pi * 10 * t)  # α波 (8-13 Hz)
        beta = 0.2 * np.sin(2 * np.pi * 20 * t)  # β波 (13-30 Hz)

        # 组合信号 + 噪声
        eeg_signal = delta + theta + alpha + beta
        noise = noise_level * np.random.normal(size=len(t))

        return t, eeg_signal + noise

    def bandpass_filter(self, data, lowcut, highcut, order=4):
        # 带通滤波器
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    def compute_band_power(self, data, band):
        # 计算频带功率
        freqs, psd = signal.welch(data, self.sampling_rate, nperseg=1024)

        if band == 'delta':
            band_mask = (freqs >= 1) & (freqs <= 4)
        elif band == 'theta':
            band_mask = (freqs >= 4) & (freqs <= 8)
        elif band == 'alpha':
            band_mask = (freqs >= 8) & (freqs <= 13)
        elif band == 'beta':
            band_mask = (freqs >= 13) & (freqs <= 30)
        else:
            band_mask = (freqs >= 1) & (freqs <= 30)  # 宽频

        return np.trapz(psd[band_mask], freqs[band_mask])


# 示例使用
eeg_processor = EEGProcessor(sampling_rate=250)

# 生成模拟脑电信号
t, raw_eeg = eeg_processor.generate_sample_eeg(duration=10, noise_level=0.3)

# 滤波
alpha_eeg = eeg_processor.bandpass_filter(raw_eeg, 8, 13)

# 计算频带功率
bands = ['delta', 'theta', 'alpha', 'beta']
powers = {band: eeg_processor.compute_band_power(raw_eeg, band) for band in bands}

print("各频带功率:")
for band, power in powers.items():
    print(f"{band}: {power:.4f}")

# 可视化
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, raw_eeg)
plt.title('原始脑电信号')
plt.ylabel('振幅')

plt.subplot(3, 1, 2)
plt.plot(t, alpha_eeg)
plt.title('Alpha波段 (8-13 Hz)')
plt.ylabel('振幅')

plt.subplot(3, 1, 3)
plt.bar(powers.keys(), powers.values())
plt.title('各频带功率')
plt.ylabel('功率')

plt.tight_layout()
plt.show()