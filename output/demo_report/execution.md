# Simulation Report: execution

## Totals

| Metric | Value |
|---|---:|
| matmul_operations | 20 |
| macs | 4080 |
| weight_events | 0 |
| weight_bytes | 0 |
| digital_operations | 28 |
| digital_primitives | {"add_or_subtract": 400, "multiply": 690, "divide": 150, "sqrt": 10, "add": 800, "metadata_or_layout": 160, "data_rearrangement": 200, "compare": 40, "exp": 50, "tanh": 80} |
| data_transfers | 31 |
| transferred_bytes | 5560 |

## Compute by Unit

| Unit | MatMuls | MACs | Digital Ops | Weight Bytes |
|---|---:|---:|---:|---:|
| rram_cim | 6 | 2560 | 0 | 0 |
| sram_cim | 14 | 1520 | 0 | 0 |
| digital_processor | 0 | 0 | 28 | 0 |

## Transfers by Path

| Path | Transfers | Bytes |
|---|---:|---:|
| digital_buffer->digital_processor | 1 | 40 |
| digital_processor->rram_cim | 6 | 1120 |
| digital_processor->sram_cim | 10 | 1800 |
| rram_cim->digital_processor | 6 | 1120 |
| sram_cim->digital_processor | 8 | 1480 |
