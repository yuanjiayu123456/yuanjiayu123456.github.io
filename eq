# -*- coding: utf-8 -*-
"""
STEP 6 (Golden Verification): EQ_SYNC 软硬对比仿真
功能：
1. [Golden Model]: 浮点双精度计算，验证理论相关性。
2. [Hardware Model]: Q1.19输入 * Q1.16系数，验证定点化损失。
3. 严格遵循硬件架构图：Window -> FIR -> ArgMax。
"""
import numpy as np
import matplotlib.pyplot as plt
import os

CONFIG = {
    'fs': 31250.0,
    
    # 物理层定义：
    # 违例码: 11个1，1个0
    # 物理电平: 1 -> Low(-), 0 -> High(+)
    'eq_pattern': [0, 0, 1, 1, 0, 0, 1, 1,0, 0, 1, 1, 0, 0, 1, 1,0, 0, 1, 1, 0, 0, 1, 1,0, 0, 1, 1, 0, 0, 1, 1,0, 0, 1, 1, 0, 0, 1, 1,0, 0, 1, 1, 0, 0, 1, 1,0, 0, 1, 1, 0, 0, 1, 1,0, 0, 1, 1, 0, 0, 1, 1],
    
    'start_index':1,  
    'search_len': 1000,
    'in_file': 'data/data_hpf_out.dat',
    'out_file': 'data/data_sync_out.dat',
    'plot_file': 'output/step6_eq_sync_golden_compare.png'
}

# ================= 工具函数 =================
def float_to_fixed(val, total_bits, frac_bits):
    """定点化工具 (带饱和与截断)"""
    scaling = 1 << frac_bits
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))
    
    # 量化
    int_val = int(round(val * scaling))
    # 饱和
    if int_val > max_val: int_val = max_val
    elif int_val < min_val: int_val = min_val
    return int_val

def fixed_to_float(int_val, frac_bits):
    """定点转浮点 (用于绘图观察)"""
    return int_val / (1 << frac_bits)

def generate_mf_coeffs(pattern):
    """
    生成系数 (针对 Load Modulation)
    逻辑 1 (Mod On)  -> 物理 Low  (-) -> 匹配系数 -1.0
    逻辑 0 (Mod Off) -> 物理 High (+) -> 匹配系数 +1.0
    """
    waveform = []
    for chip in pattern:
        val = -1.0 if chip == 1 else 1.0
        waveform.append(val)
    
    # FIR 实现相关运算，系数必须是波形的【时间反转】
    coeffs = np.array(waveform)[::-1]
    
    # 归一化 (保证 Golden 结果幅度合理)
    return coeffs / len(coeffs)

# ================= 模型定义 =================

class MF_Golden_Model:
    """浮点黄金模型 (Ideal Math)"""
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def process_block(self, data_slice):
        # 使用 numpy 的相关函数，mode='valid' 模拟 FIR 滑动
        # 注意: correlate 实际上是不翻转的卷积，
        # 但我们上面 generate_coeffs 已经翻转了，所以这里相当于做相关
        return np.correlate(data_slice, self.coeffs, mode='full')[:len(data_slice)]

class MF_Hardware_Model:
    """定点硬件模型 (Bit Accurate)"""
    def __init__(self, coeffs):
        # 系数 Q1.16 (17 bit)
        self.coeffs_int = [float_to_fixed(c, 17, 16) for c in coeffs]
        self.buffer = [0] * len(coeffs)

    def process(self, x_float):
        # 输入 Q1.19 (20 bit)
        x_int = float_to_fixed(x_float, 20, 19)
        
        # 移位寄存器
        self.buffer.pop()
        self.buffer.insert(0, x_int)
        
        # MAC (乘累加)
        acc = 0
        for i in range(len(self.buffer)):
            acc += self.buffer[i] * self.coeffs_int[i]
            
        # 输出截位处理
        # 输入 Q1.19 * 系数 Q1.16 = Q2.35
        # 目标输出 Q1.16
        # 需要右移 19 位 (35 - 16 = 19)
        out_int = acc >> 19
        
        # 输出饱和 Q1.16 (17 bit)
        max_val = (1 << 16) - 1
        min_val = -(1 << 16)
        if out_int > max_val: out_int = max_val
        elif out_int < min_val: out_int = min_val
            
        return fixed_to_float(out_int, 16)

# ================= 主程序 =================
def main():
    if not os.path.exists(CONFIG['in_file']): return
    hpf_data = np.loadtxt(CONFIG['in_file'])
    
    # 1. 窗口截取
    start = CONFIG['start_index']
    end = min(len(hpf_data), start + CONFIG['search_len'])
    mf_input_slice = hpf_data[start:end]
    print(f"[表情] 数据窗口: {start} -> {end}")

    # 2. 生成系数
    coeffs_float = generate_mf_coeffs(CONFIG['eq_pattern'])
    
    # 3. 运行模型
    # A. Golden Run
    golden_model = MF_Golden_Model(coeffs_float)
    # 注意: correlate 输出会有延迟，我们需要对齐
    # FIR 延迟 = (taps-1)/2, 这里为了简单对比，我们直接画
    golden_out = golden_model.process_block(mf_input_slice)
    
    # B. Hardware Run
    hw_model = MF_Hardware_Model(coeffs_float)
    hw_out = []
    for val in mf_input_slice:
        hw_out.append(hw_model.process(val))
    hw_out = np.array(hw_out)
    
    # 4. 寻找峰值 (以 Hardware 结果为准)
    peak_rel_idx = np.argmax(hw_out) 
    peak_val = hw_out[peak_rel_idx]
    peak_abs_idx = start + peak_rel_idx
    
    # Golden 峰值 (用于对比)
    peak_idx_gold = np.argmax(golden_out)
    peak_val_gold = golden_out[peak_idx_gold]

    print(f"[表情] Golden Peak: Index={start+peak_idx_gold}, Val={peak_val_gold:.6f}")
    print(f"[表情] Hardwr Peak: Index={peak_abs_idx}, Val={peak_val:.6f}")
    
    if peak_val < 0:
        print("[表情] 警告: 硬件匹配峰值为负，极性可能仍错误！")

    # 5. 保存
    aligned_data = hpf_data[peak_abs_idx:]
    np.savetxt(CONFIG['out_file'], aligned_data, fmt='%.8f')

    # 6. 绘图 (详细对比)
    plt.style.use('seaborn-v0_8-paper')
    fig, ax = plt.subplots(3, 1, figsize=(10, 12), dpi=150)
    
    # Input
    ax[0].plot(mf_input_slice, 'gray', alpha=0.5, label='MF Input (from HPF)')
    ax[0].set_title('MF Input Slice')
    ax[0].legend()
    
    # Comparison
    ax[1].plot(golden_out, 'orange', lw=2, alpha=0.6, label='Golden (Float)')
    ax[1].plot(hw_out, '#1f77b4', lw=1.2, label='Hardware (Fixed Q1.16)')
    # 标记峰值
    ax[1].plot(peak_rel_idx, peak_val, 'x', color='blue', markersize=10)
    ax[1].plot(peak_idx_gold, peak_val_gold, '+', color='red', markersize=10)
    
    ax[1].set_title('MF Output Comparison: Golden vs Fixed')
    ax[1].legend(loc=1)
    ax[1].grid(True, alpha=0.3)
    
    # Synced Output
    view_len = min(200, len(aligned_data))
    ax[2].plot(aligned_data[:view_len], 'green')
    ax[2].set_title(f'Synced Output (Starting at {peak_abs_idx})')
    
    plt.tight_layout()
    plt.savefig(CONFIG['plot_file'])
    print(f"[表情] 对比图已保存: {CONFIG['plot_file']}")

if __name__ == "__main__":
    main()
