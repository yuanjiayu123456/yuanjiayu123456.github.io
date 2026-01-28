# -*- coding: utf-8 -*-
"""
STEP 6 (Final): EQ_SYNC 同步模块 - Chip 级精确匹配版
功能：
1. [关键] 针对 Chip 序列 (111111111110) 进行匹配滤波。
2. 采样率 31.25kHz 刚好对应 1 Chip = 32us (无需过采样)。
3. 跳过 HPF 瞬态，精准定位微弱信号的起始位置。
"""
import numpy as np
import matplotlib.pyplot as plt
import os

CONFIG = {
    'fs': 31250.0,
    
    # 违例码 (Violation Code): 11个1，1个0
    # 对应 Chip 序列: High * 11 + Low * 1
    'eq_pattern': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    
    # 跳过 HPF 上电瞬态 (2ms Idle = 64 points)
    # 建议设为 80，留出足够余量
    'start_index': 80,  
    
    'search_len': 2000,
    
    'in_file': 'data/data_hpf_out.dat',
    'out_file': 'data/data_sync_out.dat',
    'plot_file': 'output/step6_eq_sync_chip_match.png'
}

def float_to_fixed(val, total_bits, frac_bits, signed=True):
    scaling = 1 << frac_bits
    if signed:
        max_val = (1 << (total_bits - 1)) - 1
        min_val = -(1 << (total_bits - 1))
    else:
        max_val = (1 << total_bits) - 1
        min_val = 0
    int_val = int(round(val * scaling))
    return max(min(int_val, max_val), min_val)

def generate_mf_coeffs(pattern):
    """
    生成匹配滤波器系数 (Chip Level)
    输入 Pattern: [1, 1, 0...] (逻辑电平)
    输出 Coeffs:  [+1, +1, -1...] (用于卷积的电压电平)
    """
    waveform = []
    # 映射规则：逻辑 1 -> 正电压(+1), 逻辑 0 -> 负电压(-1)
    # 因为 HPF 已经去除了直流，高电平是正，低电平是负
    for chip in pattern:
        val = 1.0 if chip == 1 else -1.0
        waveform.append(val)
        
    # 匹配滤波器核心：时间反转 (Time Reversal)
    coeffs = np.array(waveform)[::-1]
    
    # 归一化 (防止计算溢出，保持增益为 1)
    return coeffs / len(coeffs)

class MF_Hardware_Model:
    """硬件 FIR 滤波器模型"""
    def __init__(self, coeffs):
        # 模拟 16-bit 系数
        self.coeffs_int = [float_to_fixed(c, 16, 15, signed=True) for c in coeffs]
        self.buffer = [0] * len(coeffs)

    def process(self, x_float):
        # 输入量化 Q1.19
        x_int = float_to_fixed(x_float, 20, 19, signed=True)
        
        self.buffer.pop()
        self.buffer.insert(0, x_int)
        
        acc = 0
        for i in range(len(self.buffer)):
            acc += self.buffer[i] * self.coeffs_int[i]
            
        # 输出处理: 累加器 -> Q1.16
        # 假设内部累加后右移 18 位 (根据之前 IQ 模块的经验)
        out_int = acc >> 18
        
        sat_max = (1 << 16) - 1
        sat_min = -(1 << 16)
        if out_int > sat_max: out_int = sat_max
        elif out_int < sat_min: out_int = sat_min
            
        return out_int / (1 << 16)

def main():
    if not os.path.exists('data'): os.makedirs('data')
    if not os.path.exists(CONFIG['in_file']): 
        print("❌ 找不到 HPF 输出文件，请先运行 Step 5")
        return

    hpf_data = np.loadtxt(CONFIG['in_file'])
    
    # 1. 截取搜索窗口 (模拟硬件 inputs[start:end])
    start = CONFIG['start_index']
    end = min(len(hpf_data), start + CONFIG['search_len'])
    mf_input_slice = hpf_data[start:end]
    
    print(f"✂️ 搜索窗口: Index {start} -> {end}")

    # 2. 生成系数并初始化 MF
    coeffs = generate_mf_coeffs(CONFIG['eq_pattern'])
    mf = MF_Hardware_Model(coeffs)
    
    # 3. 运行匹配滤波
    mf_out = []
    for val in mf_input_slice:
        mf_out.append(mf.process(val))
    mf_out = np.array(mf_out)
    
    # 4. 寻找峰值 (Peak Finding)
    # 因为我们的系数是 +1/-1 匹配，完全重合时会出现正峰值
    peak_rel_idx = np.argmax(mf_out) 
    peak_val = mf_out[peak_rel_idx]
    
    # 计算绝对位置
    peak_abs_idx = start + peak_rel_idx
    
    # 理论位置推算:
    # Idle(2ms = 63点) + EQ长(12点) + HPF延迟(13点) + MF延迟(12点) ≈ 100 左右
    print(f"✅ 同步锁定: AbsIndex={peak_abs_idx} (Rel={peak_rel_idx}), Value={peak_val:.6f}")
    
    # 5. 对齐数据输出
    # 通常取峰值点作为同步点
    aligned_data = hpf_data[peak_abs_idx:]
    np.savetxt(CONFIG['out_file'], aligned_data, fmt='%.8f')

    # 6. 绘图验证
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), dpi=150)
    
    # Input Data
    ax[0].plot(hpf_data[:end+100], 'gray', alpha=0.5, label='HPF Output')
    ax[0].axvspan(0, start, color='red', alpha=0.1, label='Skipped Transient')
    ax[0].axvspan(start, end, color='green', alpha=0.1, label='Search Window')
    ax[0].set_title('Step 1: Search Window Setup', fontweight='bold')
    ax[0].legend()
    
    # MF Correlation
    ax[1].plot(np.arange(start, end), mf_out, '#d62728', label='MF Correlation')
    ax[1].plot(peak_abs_idx, peak_val, 'x', color='black', markersize=10)
    ax[1].set_title(f'Step 2: Matched Filter Output (Peak @ {peak_abs_idx})', fontweight='bold')
    ax[1].legend()

    # Synced Signal
    # 画出后续的几个 bit 看看形状 (Data 1: 11110000, Data 0: 00001111)
    view_len = min(200, len(aligned_data))
    ax[2].plot(aligned_data[:view_len], '#1f77b4', lw=1.5, label='Synced Payload')
    ax[2].set_title('Step 3: Aligned Signal (Expect 4High-4Low patterns)', fontweight='bold')
    ax[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CONFIG['plot_file'])
    print(f"💾 结果图已保存: {CONFIG['plot_file']}")

if __name__ == "__main__":
    main()
