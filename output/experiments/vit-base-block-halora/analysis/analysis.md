# Experiment Analysis

## Overview

| Metric | Value |
|---|---:|
| matmul_operations | 18 |
| total_macs | 1463637504 |
| digital_operations | 27 |
| data_transfers | 29 |
| transferred_bytes | 26873952 |
| arithmetic_intensity_mac_per_byte | 54.46305419 |
| communication_bytes_per_mac | 0.01836107 |

## Compute by Unit

| unit | macs | share | matmul_operations |
|---:|---:|---:|---:|
| rram_cim | 1394343936 | 0.95265661 | 6 |
| sram_cim | 69293568 | 0.04734339 | 12 |

## Compute by Stage

| stage | macs | share |
|---:|---:|---:|
| attention | 528023040 | 0.3607608 |
| feed_forward | 935614464 | 0.6392392 |

## Compute by Role

| role | macs | share |
|---:|---:|---:|
| dynamic_attention | 59610624 | 0.04072772 |
| frozen_backbone | 1394343936 | 0.95265661 |
| lora | 9682944 | 0.00661567 |

## Traffic by Path

| path | transfers | bytes | share |
|---:|---:|---:|---:|
| digital_buffer->digital_processor | 1 | 151296 | 0.00562984 |
| digital_processor->rram_cim | 6 | 5446656 | 0.20267417 |
| digital_processor->sram_cim | 9 | 8519856 | 0.31703026 |
| rram_cim->digital_processor | 6 | 5446656 | 0.20267417 |
| sram_cim->digital_processor | 7 | 7309488 | 0.27199156 |

## Traffic by Stage

| stage | transfers | bytes | share |
|---:|---:|---:|---:|
| attention | 20 | 14618976 | 0.54398311 |
| block_overhead | 1 | 151296 | 0.00562984 |
| feed_forward | 8 | 12103680 | 0.45038705 |

## Encoder Block Component Breakdown

| component_zh | macs | rram_macs | rram_percent | sram_macs | sram_percent | mac_share | transfer_events | transferred_bytes | traffic_share |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 输入进入Digital | 0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 1 | 151296 | 0.00562984 |
| LayerNorm 1 | 0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 |
| Q投影 | 117405696 | 116195328 | 98.9691 | 1210368 | 1.0309 | 0.08021501 | 4 | 2420736 | 0.09007741 |
| K投影 | 117405696 | 116195328 | 98.9691 | 1210368 | 1.0309 | 0.08021501 | 4 | 2420736 | 0.09007741 |
| V投影 | 117405696 | 116195328 | 98.9691 | 1210368 | 1.0309 | 0.08021501 | 4 | 2420736 | 0.09007741 |
| 分头与转置 | 0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 |
| QK^T | 29805312 | 0 | 0.0 | 29805312 | 100.0 | 0.02036386 | 3 | 3073200 | 0.11435609 |
| 缩放与Softmax | 0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 |
| PV | 29805312 | 0 | 0.0 | 29805312 | 100.0 | 0.02036386 | 3 | 3073200 | 0.11435609 |
| 多头合并 | 0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 |
| Attention输出投影 | 116195328 | 116195328 | 100.0 | 0 | 0.0 | 0.07938805 | 2 | 1210368 | 0.04503871 |
| Attention残差 | 0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 |
| LayerNorm 2 | 0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 |
| FFN升维 | 467807232 | 464781312 | 99.3532 | 3025920 | 0.6468 | 0.3196196 | 4 | 6051840 | 0.22519353 |
| GELU | 0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 |
| FFN降维 | 467807232 | 464781312 | 99.3532 | 3025920 | 0.6468 | 0.3196196 | 4 | 6051840 | 0.22519353 |
| FFN残差 | 0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 |
| 未分类 | 0 | 0 | 0.0 | 0 | 0.0 | 0.0 | 0 | 0 | 0.0 |

## Largest Transfer

- Tensor: `transformer_block.feed_forward.up_proj.backbone_output`
- Path: `rram_cim -> digital_processor`
- Shape: `[1, 197, 3072]`
- Bytes: 2420736
