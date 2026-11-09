# Simulation Report: execution

## Totals

| Metric | Value |
|---|---:|
| matmul_operations | 18 |
| macs | 1463637504 |
| weight_events | 0 |
| weight_bytes | 0 |
| digital_operations | 27 |
| digital_primitives | {"add_or_subtract": 2139026, "multiply": 4701996, "divide": 769088, "sqrt": 394, "add": 4084992, "metadata_or_layout": 605184, "data_rearrangement": 756480, "compare": 463344, "exp": 465708, "tanh": 605184} |
| data_transfers | 29 |
| transferred_bytes | 26873952 |

## Compute by Unit

| Unit | MatMuls | MACs | Digital Ops | Weight Bytes |
|---|---:|---:|---:|---:|
| rram_cim | 6 | 1394343936 | 0 | 0 |
| sram_cim | 12 | 69293568 | 0 | 0 |
| digital_processor | 0 | 0 | 27 | 0 |

## Transfers by Path

| Path | Transfers | Bytes |
|---|---:|---:|
| digital_buffer->digital_processor | 1 | 151296 |
| digital_processor->rram_cim | 6 | 5446656 |
| digital_processor->sram_cim | 9 | 8519856 |
| rram_cim->digital_processor | 6 | 5446656 |
| sram_cim->digital_processor | 7 | 7309488 |
