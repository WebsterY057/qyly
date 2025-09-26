def reverse_complement(dna_sequence):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join(complement[base] for base in reversed(dna_sequence))

# 示例
dna_seq = "ATCGGA"
rev_comp = reverse_complement(dna_seq)
print("原始序列:", dna_seq)
print("反向互补序列:", rev_comp)

#####DNA序列分析
from collections import Counter
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class DNAAnalyzer:
    def __init__(self, sequence):
        self.sequence = sequence.upper()  # 转换为大写
        self.valid_bases = {'A', 'T', 'C', 'G'}  # 有效碱基

    def validate_sequence(self):
        # 验证DNA序列是否有效
        return all(base in self.valid_bases for base in self.sequence)

    def base_composition(self):
        # 计算碱基组成
        counter = Counter(self.sequence)
        total = len(self.sequence)
        composition = {base: (count / total) * 100 for base, count in counter.items()}
        return composition

    def reverse_complement(self):
        # 生成反向互补序列
        complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        return ''.join(complement[base] for base in reversed(self.sequence))

    def gc_content(self):
        # 计算GC含量
        gc_count = self.sequence.count('G') + self.sequence.count('C')
        return (gc_count / len(self.sequence)) * 100

    def find_pattern(self, pattern):
        # 查找模式出现位置
        pattern = pattern.upper()
        positions = []
        start = 0
        while True:
            pos = self.sequence.find(pattern, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        return positions


# 示例使用
dna_seq = "ATCGATCGATCGATCGATCG"  # 示例DNA序列
analyzer = DNAAnalyzer(dna_seq)

print(f"序列: {analyzer.sequence}")
print(f"序列有效: {analyzer.validate_sequence()}")
print(f"碱基组成: {analyzer.base_composition()}")
print(f"GC含量: {analyzer.gc_content():.2f}%")
print(f"反向互补: {analyzer.reverse_complement()}")
print(f"模式'ATC'位置: {analyzer.find_pattern('ATC')}")

# 可视化碱基组成
composition = analyzer.base_composition()
plt.bar(composition.keys(), composition.values())
plt.title('DNA碱基组成')
plt.ylabel('百分比 (%)')
plt.show()

