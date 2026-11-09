# Hybrid CIM 中 ViT-B Transformer Encoder Block 计算与数据移动模拟实验报告（初版）

> 版本：v0.1  
> 整理日期：2026-11-10  
> 实验对象：ViT-B/16 的单个 Pre-LN Transformer Encoder Block  
> 报告性质：算子级逻辑模拟报告，不是芯片周期、延迟、能耗或模型精度报告

## 摘要

本实验建立了一个由 RRAM CIM、SRAM CIM、数字处理单元和数字缓冲区组成的 Hybrid CIM 逻辑系统，并将 ViT-B/16 的单个 Transformer Encoder Block 映射到该系统。冻结的主干权重矩阵乘法主要由 RRAM CIM 执行；LoRA 低秩分支以及动态 Attention 矩阵乘法由 SRAM CIM 执行；LayerNorm、Softmax、GELU、偏置、分支合并、张量变形与残差连接由数字单元执行。

在当前配置下，一个 Encoder Block 共执行 18 次矩阵乘法、1,463,637,504 MAC、27 个数字算子事件和 29 次跨逻辑硬件边界的数据传输，传输总量为 26,873,952 B。RRAM 承担 95.27% 的 MAC，说明冻结主干决定了主要计算量；SRAM 只承担 4.73% 的 MAC，但与 SRAM 有关的输入和输出传输占总通信量的 58.90%。这表明当前映射呈现出一个重要特征：LoRA 和动态 Attention 的计算量不大，但它们需要完整激活或完整输出跨越数字单元与 SRAM 的边界，因此其通信占比明显高于计算占比。

这一结果支持对 Hybrid CIM 数据移动问题的进一步研究，但尚不足以判断 Hybrid CIM 的实际延迟、能耗或性能是否优于纯数字、纯 RRAM 或其他混合映射方案。

## 报告逻辑

本报告按照以下顺序组织实验结果：

\[
\text{研究问题}
\rightarrow
\text{系统映射与实验配置}
\rightarrow
\text{统计口径}
\rightarrow
\text{总体结果}
\rightarrow
\text{逐硬件、逐阶段、逐部件分解}
\rightarrow
\text{数据解释}
\rightarrow
\text{局限性与下一步实验}
\]

这种组织方式先说明“统计的是什么”，再展示“统计结果是什么”，最后才讨论“这些结果能够支持什么结论”，从而避免把逻辑传输量直接等同于真实芯片的时间或能耗。

## 1. 研究目的与问题

本实验不是 HaLoRA 精度复现实验，而是面向 HaLoRA/Hybrid CIM 执行架构的计算与数据通路模拟。它试图回答四个问题：

1. 一个 ViT-B Encoder Block 的计算量在 RRAM CIM、SRAM CIM 和数字单元之间如何分配？
2. Attention 与 FFN 中，哪些部件产生最多 MAC，哪些部件产生最多跨硬件数据移动？
3. LoRA 的低秩结构在减少计算量的同时，是否也同比减少了接口通信量？
4. 当前统计能否为后续的数据复用、局部缓存、QKV 融合和流式 Attention 优化提供基线？

## 2. 系统结构与计算映射

### 2.1 Encoder Block 结构

本实验使用 Pre-LN Transformer Encoder Block：

\[
Y=X+\operatorname{MSA}(\operatorname{LN}_1(X))
\]

\[
Z=Y+\operatorname{FFN}(\operatorname{LN}_2(Y))
\]

完整的硬件映射图见：

- [Encoder Block 硬件映射图（PDF）](../../output/pdf/encoder_block_hardware_mapping.pdf)
- [硬件映射图 TikZ 源文件](../figures/encoder_block_hardware_mapping.tex)

### 2.2 Hybrid 线性层

对于带 LoRA 的线性层，冻结主干和低秩增量分开执行：

\[
Y=XW_0+X\Delta W+b,
\qquad
\Delta W=BA
\]

其中：

- \(XW_0\) 由 RRAM CIM 执行；
- \(XBA\) 的两次低秩矩阵乘法由 SRAM CIM 执行；
- 两支结果合并以及偏置加法由数字单元执行。

### 2.3 当前实验的部件映射

| 模型部件 | 当前映射 | 原因或角色 |
|---|---|---|
| Q、K、V 投影的冻结主干 | RRAM CIM | 静态、大规模权重矩阵乘法 |
| Q、K、V 投影的 LoRA 分支 | SRAM CIM | 小规模、可更新低秩权重 |
| \(QK^T\)、\(PV\) | SRAM CIM | 两个输入都随样本动态变化 |
| Attention 输出投影 | RRAM CIM | 当前配置为冻结主干、无 SRAM LoRA 分支 |
| FFN 升维/降维冻结主干 | RRAM CIM | 静态、大规模权重矩阵乘法 |
| FFN 升维/降维 LoRA 分支 | SRAM CIM | 小规模、可更新低秩权重 |
| LayerNorm、Scale、Softmax、GELU | Digital | 非矩阵乘法和非线性操作 |
| Bias、主干/LoRA 合并、残差 | Digital | 逐元素加法与控制 |
| reshape、transpose、head merge | Digital | 张量布局与数据重排 |

上述映射是本次实验的配置选择，不代表所有 Hybrid CIM 或 HaLoRA 系统都必须使用相同映射。

## 3. 实验配置

### 3.1 模型与数值配置

| 参数 | 数值 |
|---|---:|
| 模型结构 | ViT-B/16 单个 Pre-LN Encoder Block |
| Batch size \(B\) | 1 |
| Token 数 \(N\) | 197 |
| 隐藏维度 \(D\) | 768 |
| Attention head 数 \(H\) | 12 |
| 单 head 维度 \(d_h\) | 64 |
| FFN 中间维度 \(D_{ff}\) | 3072 |
| LoRA rank \(r\) | 4 |
| LayerNorm \(\epsilon\) | \(10^{-5}\) |
| 激活位宽 | 8 bit |
| 权重位宽 | 8 bit |
| 累加器/返回结果位宽 | 32 bit |
| 随机种子 | 2026 |
| 输出张量形状 | \([1,197,768]\) |

实验配置文件见 [vit_base_block_halora.json](../../experiments/configs/vit_base_block_halora.json)。本次运行使用 Python 3.12.6 和 NumPy 2.1.1。随机输入与随机权重只用于验证计算路径、形状和事件统计，不用于评价模型准确率。

### 3.2 权重准备阶段

权重写入/登记与单次前向执行分开统计。准备阶段不计入后文的 29 次执行期数据传输。

| 存储/计算单元 | 权重事件 | 权重字节 | 准备阶段权重占比 |
|---|---:|---:|---:|
| RRAM CIM | 6 | 7,077,888 B | 98.7570% |
| SRAM CIM | 10 | 49,152 B | 0.6858% |
| Digital | 10 | 39,936 B | 0.5572% |
| **合计** | **26** | **7,166,976 B** | **100%** |

原始准备阶段报告见 [preparation.md](../../output/experiments/vit-base-block-halora/raw_reports/preparation.md)。准备阶段没有记录传输事件，只记录 storage/program/load 事件，因此不能把这 7,166,976 B 直接加到执行期通信量中。

## 4. 统计指标与边界

### 4.1 MAC

对于形状为 \((M,K)\) 和 \((K,N)\) 的两个矩阵：

\[
\operatorname{MAC}=MKN
\]

一次 MAC 表示一次乘法-累加。本文只把 RRAM/SRAM 中的矩阵乘法计为 MAC；Digital 的加法、除法、指数、平方根等使用“数字原语次数”单独统计，二者不能直接相加。

### 4.2 逻辑数据传输

每次张量跨越一个逻辑硬件边界记为一次传输事件，字节数为：

\[
\operatorname{Bytes}
=
\operatorname{Elements}\times\frac{\operatorname{BitWidth}}{8}
\]

当前记录五类路径：Buffer→Digital、Digital→RRAM、RRAM→Digital、Digital→SRAM 和 SRAM→Digital。LoRA 在 SRAM 内部的两次低秩乘法之间如果中间张量不离开 SRAM，则不额外记跨硬件传输。

本次原始记录中，初始输入按 8-bit 统计，之后 28 个内部跨单元事件均按 32-bit 中间结果统计；与此同时，矩阵乘法事件仍声明 8-bit 输入。当前模型没有显式记录 32→8 bit 的 requantization 算子。因此，本文通信数据必须理解为“当前 32-bit 中间结果传输假设下的逻辑统计”，而不是完整的 W8A8 硬件数据通路。

### 4.3 算术强度

本文使用的逻辑算术强度为：

\[
I=\frac{\text{MAC}}{\text{跨逻辑硬件边界传输字节}}
\quad(\text{MAC/B})
\]

它用于比较当前抽象模型中的计算/通信关系，不等同于考虑缓存层次和真实 DRAM 流量的硬件 Roofline 指标。

### 4.4 归属规则与统计守恒

每个计算事件和传输事件只归属一个模型部件，因此应满足：

\[
\sum_i \operatorname{MAC}_i=\operatorname{MAC}_{\text{total}}
\]

\[
\sum_i \operatorname{Bytes}_i=\operatorname{Bytes}_{\text{total}}
\]

本次核对结果为 1,463,637,504 MAC 和 26,873,952 B，均与原始事件报告一致。

## 5. 总体实验结果

### 5.1 执行总览

| 指标 | 结果 |
|---|---:|
| 矩阵乘法事件 | 18 |
| 总 MAC | 1,463,637,504 |
| 数字算子事件 | 27 |
| 跨硬件传输事件 | 29 |
| 跨硬件传输总量 | 26,873,952 B |
| 全局算术强度 | 54.4631 MAC/B |
| 每 MAC 对应通信量 | 0.018361 B/MAC |
| 数字原语总量 | 14,591,396 |
| 输出均值（通路校验） | -0.0002646570 |
| 输出标准差（通路校验） | 1.0666647344 |

### 5.2 按硬件单元划分的计算量

| 硬件单元 | 矩阵乘法次数 | MAC | 全局计算占比 |
|---|---:|---:|---:|
| RRAM CIM | 6 | 1,394,343,936 | 95.2657% |
| SRAM CIM | 12 | 69,293,568 | 4.7343% |
| **合计** | **18** | **1,463,637,504** | **100%** |

RRAM 的矩阵乘法调用次数更少，但每次处理的是大规模冻结权重，因此承担绝大多数 MAC。由此可见，调用次数不能替代计算量。

### 5.3 按模型阶段划分的计算量

| 阶段 | MAC | 全局计算占比 |
|---|---:|---:|
| Attention | 528,023,040 | 36.0761% |
| FFN | 935,614,464 | 63.9239% |
| **合计** | **1,463,637,504** | **100%** |

FFN 升维和降维两个大矩阵乘法使 FFN 成为主要计算热点。

### 5.4 按计算角色划分的计算量

| 计算角色 | MAC | 全局计算占比 |
|---|---:|---:|
| 冻结主干 | 1,394,343,936 | 95.2657% |
| LoRA 低秩分支 | 9,682,944 | 0.6616% |
| 动态 Attention（\(QK^T+PV\)） | 59,610,624 | 4.0728% |
| **合计** | **1,463,637,504** | **100%** |

LoRA 的 rank 为 4，因此低秩分支的 MAC 只占 0.66%；这验证了低秩结构对乘法量的压缩效果。

## 6. 数据传输结果

### 6.1 按传输方向划分

| 逻辑路径 | 传输次数 | 传输字节 | 全局通信占比 |
|---|---:|---:|---:|
| Buffer→Digital | 1 | 151,296 B | 0.5630% |
| Digital→RRAM | 6 | 5,446,656 B | 20.2674% |
| RRAM→Digital | 6 | 5,446,656 B | 20.2674% |
| Digital→SRAM | 9 | 8,519,856 B | 31.7030% |
| SRAM→Digital | 7 | 7,309,488 B | 27.1992% |
| **合计** | **29** | **26,873,952 B** | **100%** |

与 RRAM 有关的双向通信量为 10,893,312 B，占 40.53%；与 SRAM 有关的双向通信量为 15,829,344 B，占 58.90%。这里的“SRAM 相关通信”同时包含 LoRA 分支和动态 Attention，并不等于 SRAM 权重本身的大小。

### 6.2 按模型阶段划分

| 阶段 | 传输次数 | 传输字节 | 全局通信占比 |
|---|---:|---:|---:|
| Attention | 20 | 14,618,976 B | 54.3983% |
| FFN | 8 | 12,103,680 B | 45.0387% |
| Block 输入开销 | 1 | 151,296 B | 0.5630% |
| **合计** | **29** | **26,873,952 B** | **100%** |

Attention 只占 36.08% 的 MAC，却占 54.40% 的逻辑通信；FFN 占 63.92% 的 MAC和 45.04% 的通信。因此在当前映射下，Attention 相对更偏通信密集，FFN 相对更偏计算密集。

## 7. Encoder Block 逐部件统计

### 7.1 每个部件的总计算与总通信

| Encoder Block 部件 | MAC | 全局计算占比 | 传输次数 | 传输字节 | 全局通信占比 | 算术强度 |
|---|---:|---:|---:|---:|---:|---:|
| 输入进入 Digital | 0 | 0% | 1 | 151,296 B | 0.5630% | — |
| LayerNorm 1 | 0 | 0% | 0 | 0 B | 0% | — |
| Q 投影 | 117,405,696 | 8.0215% | 4 | 2,420,736 B | 9.0077% | 48.5000 |
| K 投影 | 117,405,696 | 8.0215% | 4 | 2,420,736 B | 9.0077% | 48.5000 |
| V 投影 | 117,405,696 | 8.0215% | 4 | 2,420,736 B | 9.0077% | 48.5000 |
| 分 head 与转置 | 0 | 0% | 0 | 0 B | 0% | — |
| \(QK^T\) | 29,805,312 | 2.0364% | 3 | 3,073,200 B | 11.4356% | 9.6985 |
| Scale 与 Softmax | 0 | 0% | 0 | 0 B | 0% | — |
| \(PV\) | 29,805,312 | 2.0364% | 3 | 3,073,200 B | 11.4356% | 9.6985 |
| 多 head 合并 | 0 | 0% | 0 | 0 B | 0% | — |
| Attention 输出投影 | 116,195,328 | 7.9388% | 2 | 1,210,368 B | 4.5039% | 96.0000 |
| Attention 残差 | 0 | 0% | 0 | 0 B | 0% | — |
| LayerNorm 2 | 0 | 0% | 0 | 0 B | 0% | — |
| FFN 升维 | 467,807,232 | 31.9620% | 4 | 6,051,840 B | 22.5194% | 77.3000 |
| GELU | 0 | 0% | 0 | 0 B | 0% | — |
| FFN 降维 | 467,807,232 | 31.9620% | 4 | 6,051,840 B | 22.5194% | 77.3000 |
| FFN 残差 | 0 | 0% | 0 | 0 B | 0% | — |
| 未分类 | 0 | 0% | 0 | 0 B | 0% | — |
| **合计** | **1,463,637,504** | **100%** | **29** | **26,873,952 B** | **100%** | **54.4631** |

表中 LayerNorm、Softmax、GELU、残差和数据重排显示为 0 MAC，是因为它们不执行本文定义的矩阵乘法；这不表示它们没有数字计算或内部访存开销。

### 7.2 每个矩阵乘法部件的 RRAM/SRAM 计算比例

| 部件 | 总 MAC | RRAM MAC | RRAM 占部件计算 | SRAM MAC | SRAM 占部件计算 |
|---|---:|---:|---:|---:|---:|
| Q 投影 | 117,405,696 | 116,195,328 | 98.9691% | 1,210,368 | 1.0309% |
| K 投影 | 117,405,696 | 116,195,328 | 98.9691% | 1,210,368 | 1.0309% |
| V 投影 | 117,405,696 | 116,195,328 | 98.9691% | 1,210,368 | 1.0309% |
| \(QK^T\) | 29,805,312 | 0 | 0% | 29,805,312 | 100% |
| \(PV\) | 29,805,312 | 0 | 0% | 29,805,312 | 100% |
| Attention 输出投影 | 116,195,328 | 116,195,328 | 100% | 0 | 0% |
| FFN 升维 | 467,807,232 | 464,781,312 | 99.3532% | 3,025,920 | 0.6468% |
| FFN 降维 | 467,807,232 | 464,781,312 | 99.3532% | 3,025,920 | 0.6468% |
| **合计** | **1,463,637,504** | **1,394,343,936** | **95.2657%（全局）** | **69,293,568** | **4.7343%（全局）** |

Q/K/V 和 FFN 的“部件内占比”反映冻结主干与 LoRA 分支在同一个线性层中的计算比例；合计行则是全局硬件占比，两种分母不同。

### 7.3 每个部件的传输方向、次数和数据量

单元格采用“次数 / 字节”表示。未列出的数字部件没有跨本模型定义的硬件边界，但仍可能发生数字单元内部访存。

| 部件 | Buffer→D | D→RRAM | RRAM→D | D→SRAM | SRAM→D | 部件合计 |
|---|---:|---:|---:|---:|---:|---:|
| 输入进入 Digital | 1 / 151,296 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 151,296 |
| Q 投影 | 0 / 0 | 1 / 605,184 | 1 / 605,184 | 1 / 605,184 | 1 / 605,184 | 4 / 2,420,736 |
| K 投影 | 0 / 0 | 1 / 605,184 | 1 / 605,184 | 1 / 605,184 | 1 / 605,184 | 4 / 2,420,736 |
| V 投影 | 0 / 0 | 1 / 605,184 | 1 / 605,184 | 1 / 605,184 | 1 / 605,184 | 4 / 2,420,736 |
| \(QK^T\) | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 1,210,368 | 1 / 1,862,832 | 3 / 3,073,200 |
| \(PV\) | 0 / 0 | 0 / 0 | 0 / 0 | 2 / 2,468,016 | 1 / 605,184 | 3 / 3,073,200 |
| Attention 输出投影 | 0 / 0 | 1 / 605,184 | 1 / 605,184 | 0 / 0 | 0 / 0 | 2 / 1,210,368 |
| FFN 升维 | 0 / 0 | 1 / 605,184 | 1 / 2,420,736 | 1 / 605,184 | 1 / 2,420,736 | 4 / 6,051,840 |
| FFN 降维 | 0 / 0 | 1 / 2,420,736 | 1 / 605,184 | 1 / 2,420,736 | 1 / 605,184 | 4 / 6,051,840 |
| **合计** | **1 / 151,296** | **6 / 5,446,656** | **6 / 5,446,656** | **9 / 8,519,856** | **7 / 7,309,488** | **29 / 26,873,952** |

需要特别说明：

- Q、K、V 当前分别广播一次输入；如果以后实现 QKV 融合或共享广播，传输次数会下降。
- RRAM/SRAM 的输出按 32-bit 累加结果返回，因此返回流量可能显著大于 8-bit 输入流量。
- 最大单次传输量为 2,420,736 B，共有四个事件并列：FFN 升维的 RRAM 输出→Digital、FFN 升维的 SRAM/LoRA 输出→Digital、FFN 降维的 Digital→RRAM 输入以及 FFN 降维的 Digital→SRAM 输入。自动生成的 `analysis.md` 只展示了其中第一项。

## 8. 数字处理单元统计

### 8.1 数字原语总量

| 数字原语 | 元素级操作数 |
|---|---:|
| add_or_subtract | 2,139,026 |
| multiply | 4,701,996 |
| divide | 769,088 |
| sqrt | 394 |
| add | 4,084,992 |
| compare | 463,344 |
| exp | 465,708 |
| tanh | 605,184 |
| metadata_or_layout | 605,184 |
| data_rearrangement | 756,480 |

这些数值是元素级原语估算，不等于 CPU 指令条数，也没有换算为周期或能量。

### 8.2 数字算子按模型部件划分

| 数字部件 | 算子事件数 | 主要元素级原语及数量 | 原语合计 |
|---|---:|---|---:|
| LayerNorm 1 | 1 | add/sub 604,987；multiply 302,592；divide 151,690；sqrt 197 | 1,059,466 |
| Q 投影合并与偏置 | 2 | add 302,592 | 302,592 |
| K 投影合并与偏置 | 2 | add 302,592 | 302,592 |
| V 投影合并与偏置 | 2 | add 302,592 | 302,592 |
| 分 head、Q/K/V 转置与 \(K^T\) | 7 | metadata/layout 453,888；data rearrangement 605,184 | 1,059,072 |
| Scale 与 Softmax | 2 | multiply 465,708；compare 463,344；add/sub 929,052；exp 465,708；divide 465,708 | 2,789,520 |
| head 合并 | 2 | data rearrangement 151,296；metadata/layout 151,296 | 302,592 |
| Attention 输出投影偏置 | 1 | add 151,296 | 151,296 |
| Attention 残差 | 1 | add 151,296 | 151,296 |
| LayerNorm 2 | 1 | add/sub 604,987；multiply 302,592；divide 151,690；sqrt 197 | 1,059,466 |
| FFN 升维合并与偏置 | 2 | add 1,210,368 | 1,210,368 |
| GELU | 1 | multiply 3,631,104；add 1,210,368；tanh 605,184 | 5,446,656 |
| FFN 降维合并与偏置 | 2 | add 302,592 | 302,592 |
| FFN 残差 | 1 | add 151,296 | 151,296 |
| **合计** | **27** | 见上一表的各原语总量 | **14,591,396** |

### 8.3 数字统计按阶段聚合

| 阶段 | Digital kernel 数 | Primitive 合计 | Digital 输出元素数 |
|---|---:|---:|---:|
| Block overhead | 4 | 2,421,524 | 605,184 |
| Attention | 18 | 5,210,256 | 3,352,152 |
| FFN | 5 | 6,959,616 | 2,118,144 |
| **合计** | **27** | **14,591,396** | **6,075,480** |

数字部件显示为“零跨硬件传输”只表示当前统计没有把 Digital 内部的读写视为跨单元事件，不能解释成这些算子不存在数据移动成本。

## 9. 结果分析与核心发现

### 9.1 冻结主干决定计算量

冻结主干贡献 1,394,343,936 MAC，占总 MAC 的 95.27%，并全部映射到 RRAM。由此可以确认，当前系统确实把大规模静态权重计算留给了 RRAM；FFN 又占全部计算的 63.92%，因此未来如果只优化计算阵列，FFN 是首先需要关注的部分。

### 9.2 LoRA 显著压缩计算，但没有同比压缩接口通信

LoRA 只贡献 9,682,944 MAC，占总计算量的 0.66%。但是，Q/K/V 和 FFN 两个线性层中与 LoRA SRAM 分支有关的接口路径共传输：

\[
3\times1{,}210{,}368
+
2\times3{,}025{,}920
=9{,}682{,}944\ \text{B}
\]

占全部逻辑通信的 36.03%。这里的 9,682,944 B 与 LoRA MAC 数值相同只是当前维度和位宽组合产生的数值巧合，两者单位和含义完全不同。

原因是低秩化压缩了 \(D\times D\) 权重乘法，但当前接口仍需要把完整输入送入 SRAM，并把完整输出送回 Digital。低秩中间张量可以留在 SRAM 内部，却不能自动消除完整输入/输出的跨边界传输。

### 9.3 动态 Attention 是低计算量、高通信占比部件

\(QK^T\) 和 \(PV\) 合计只占 4.07% 的 MAC，却占 22.87% 的通信。两者的逻辑算术强度都约为 9.70 MAC/B，明显低于 Q/K/V 投影的 48.5 MAC/B、Attention 输出投影的 96 MAC/B 和 FFN 线性层的 77.3 MAC/B。

这说明在当前数据通路中，动态 Attention 比线性投影更容易受到接口带宽和数据驻留策略影响。后续值得测试的方向包括：在 SRAM 附近保留 Q/K/V、流式计算 score/context、减少 score 矩阵回传，以及对 Softmax 前后数据进行局部消费。

### 9.4 RRAM-only 输出投影提供了内部参照

Attention 输出投影占 7.94% 的 MAC、4.50% 的通信，只产生 2 次传输；单个 Q/K/V 混合投影约占 8.02% 的 MAC、9.01% 的通信，产生 4 次传输。二者计算规模接近，因此该对比说明：在当前逻辑模型中，增加 SRAM LoRA 分支会增加接口数和数据量。

但这只是同一模拟器内部的结构参照。由于输出投影没有配置相同的 LoRA 分支，且实验没有延迟、能耗和精度收益数据，因此不能据此直接断言“带 LoRA 的 Hybrid 投影一定更差”。

### 9.5 次数、字节与实际代价必须区分

SRAM 有 12 次矩阵乘法，RRAM 只有 6 次，但 SRAM 的 MAC 仅占 4.73%；同样，29 次传输中不同事件的数据量差异很大。真实系统的代价还取决于阵列尺寸、并行度、位串行周期、带宽、流水线、启动开销和复用方式。因此后续分析不能只比较“调用次数”或“传输次数”，必须同时保留 MAC、字节数以及硬件参数。

## 10. 结论

在当前 ViT-B 单 Encoder Block 的逻辑映射中，可以得到以下受实验数据支持的结论：

1. RRAM 承担 95.27% 的 MAC，冻结主干是主要计算来源。
2. FFN 占 63.92% 的 MAC，是计算量最大的模型阶段。
3. SRAM 只承担 4.73% 的 MAC，但 SRAM 相关通信占 58.90%，说明 SRAM 在当前系统中更像一个计算量较小、接口活动较频繁的动态/可更新计算单元。
4. LoRA 的低秩结构把乘法量压缩到总 MAC 的 0.66%，但完整输入和输出仍跨越 Digital–SRAM 边界，因此通信量没有按相同比例下降。
5. \(QK^T\) 和 \(PV\) 是通信密集型部件；FFN 是计算密集型部件。二者需要不同的优化策略。

最简洁的总判断是：

> 当前 Hybrid CIM 映射成功地把绝大多数静态 MAC 放入 RRAM，但 LoRA 分支与动态 Attention 使 SRAM 的通信参与度显著高于其计算占比。后续优化的核心不只是减少 MAC，还要提高激活和中间结果在各计算单元附近的驻留与复用。

## 11. 实验局限性

本报告的结论必须限定在以下边界内：

1. 当前是算子级逻辑模拟，没有纯数字、纯 RRAM 或其他 Hybrid 方案作为性能基线。
2. 传输次数是逻辑直连事件；真实系统如果经过共享 Buffer、NoC 或 DMA，多跳次数和物理流量可能更大。
3. 尚未建模阵列尺寸、矩阵分块、ADC/DAC、位串行周期、频率、带宽、流水线、延迟和能耗。
4. Digital 内部的缓存/内存访问没有计入跨硬件通信。
5. 配置标称 W8A8，但初始输入之后的内部跨单元张量按 32-bit 结果统计，且没有显式 requantization 事件；若实际系统在阵列附近量化或融合，通信量会改变。
6. 当前只模拟一个 Encoder Block，不包括 Patch Embedding、12 层级联、分类头和层间缓存复用。
7. 当前使用随机输入和随机权重，只验证执行通路与统计，不评价准确率、RRAM 噪声鲁棒性或 HaLoRA 正则项效果。

因此，本实验能够描述“当前映射发生了多少逻辑计算和数据移动”，但不能单独证明 Hybrid CIM 在真实芯片上更快、更省能或更准确。

## 12. 下一步实验建议

建议按照以下顺序扩展实验：

1. **建立可比较基线**：加入纯 Digital、RRAM-only 线性层、当前 Hybrid 和通信优化 Hybrid 四种映射。
2. **优化数据驻留**：实现 QKV 融合/共享广播、SRAM 近端 QKV 缓存和 Attention 流式执行，比较传输次数与字节变化。
3. **增加参数扫描**：扫描 token 数 \(N\)、LoRA rank \(r\)、批大小 \(B\)、位宽和 FFN 扩展倍率，观察计算/通信拐点。
4. **加入物理代价模型**：引入阵列规模、分块次数、读写带宽、ADC/DAC、单位 MAC 能量和单位字节传输能量。
5. **扩展到完整 ViT**：模拟 12 个 Block 的层间激活驻留和总端到端通信。
6. **与精度实验连接**：把 HaLoRA 的无噪声 LoRA、噪声 LoRA、噪声+正则化 LoRA，以及无噪声+正则化 LoRA 四组精度实验与本报告的硬件统计放在同一研究框架下。

## 附录 A：结果文件索引

### A.1 汇总结果

- [自动生成的总体分析](../../output/experiments/vit-base-block-halora/analysis/analysis.md)
- [总体分析 JSON](../../output/experiments/vit-base-block-halora/analysis/analysis.json)
- [逐部件硬件统计](../../output/experiments/vit-base-block-halora/analysis/component_hardware_breakdown.md)
- [逐部件硬件统计 CSV](../../output/experiments/vit-base-block-halora/analysis/component_hardware_breakdown.csv)
- [逐部件汇总 CSV](../../output/experiments/vit-base-block-halora/analysis/component_breakdown.csv)
- [按硬件计算 CSV](../../output/experiments/vit-base-block-halora/analysis/compute_by_unit.csv)
- [按阶段计算 CSV](../../output/experiments/vit-base-block-halora/analysis/compute_by_stage.csv)
- [按角色计算 CSV](../../output/experiments/vit-base-block-halora/analysis/compute_by_role.csv)
- [按路径通信 CSV](../../output/experiments/vit-base-block-halora/analysis/traffic_by_path.csv)
- [按阶段通信 CSV](../../output/experiments/vit-base-block-halora/analysis/traffic_by_stage.csv)

### A.2 原始事件记录

- [执行阶段报告](../../output/experiments/vit-base-block-halora/raw_reports/execution.md)
- [执行阶段完整 JSON](../../output/experiments/vit-base-block-halora/raw_reports/execution.json)
- [18 个矩阵乘法事件](../../output/experiments/vit-base-block-halora/raw_reports/execution_compute_events.csv)
- [27 个数字算子事件](../../output/experiments/vit-base-block-halora/raw_reports/execution_digital_events.csv)
- [29 个传输事件](../../output/experiments/vit-base-block-halora/raw_reports/execution_transfer_events.csv)
- [执行期存储事件](../../output/experiments/vit-base-block-halora/raw_reports/execution_storage_events.csv)
- [准备阶段报告](../../output/experiments/vit-base-block-halora/raw_reports/preparation.md)
- [准备阶段完整 JSON](../../output/experiments/vit-base-block-halora/raw_reports/preparation.json)
- [准备阶段权重事件](../../output/experiments/vit-base-block-halora/raw_reports/preparation_storage_events.csv)

## 附录 B：复现实验信息

| 项目 | 内容 |
|---|---|
| 正式实验目录 | `output/experiments/vit-base-block-halora/` |
| 生成时间（UTC） | 2026-11-09T17:31:33.304733+00:00 |
| Python | 3.12.6 |
| NumPy | 2.1.1 |
| 随机种子 | 2026 |
| 配置快照 | `experiment_config.json` |
| 元数据 | `experiment_metadata.json` |

本报告中的百分比经过四舍五入，个别表格各行显示值之和可能因舍入出现末位偏差；所有精确总量均以 JSON/CSV 原始结果为准。
