# ViT Hybrid CIM Block Simulator

This project models the execution of one ViT Transformer Encoder Block on a
hybrid RRAM-CIM/SRAM-CIM system. The current milestone implements the two CIM
matrix engines, a lightweight digital processor, and explicit logical data
movement.

## Current abstraction boundary

- RRAM-CIM stores a static weight matrix first, then executes matrix
  multiplication through a resident-weight handle.
- SRAM-CIM supports both resident weights and dynamic matrix multiplication.
- Both units execute real NumPy operations and record operator-level MAC counts.
- The digital processor implements add, residual add, bias, scale, LayerNorm,
  Softmax, GELU, concat, transpose, and reshape with primitive-operation counts.
- DataMover records each logical transfer direction, tensor shape, bit width,
  and byte count. Host arrays are not physically copied by this abstraction.
- HybridLinear composes the low-level units into `XW0 + (XA)B + bias` and owns
  the complete RRAM/SRAM/digital data path for one model linear layer.
- Tiling, latency, energy, detailed buffering, ADC/DAC behavior, and analog
  non-idealities are deliberately deferred.

## Quick start

```powershell
py -3.12 -m pip install -e .
py -3.12 -m unittest discover -s tests -v
py -3.12 examples/basic_usage.py
py -3.12 examples/dataflow_demo.py
py -3.12 examples/hybrid_linear_demo.py
py -3.12 examples/full_block_demo.py
```

## Main interfaces

```python
system = HybridCIMSystem()

rram_weight = system.program_rram_weight("w_q_backbone", weight)
q_backbone = system.rram_matmul(x, rram_weight, op_name="q_backbone")

sram_weight = system.load_sram_weight("lora_a", lora_a)
lora_hidden = system.sram_resident_matmul(
    x, sram_weight, op_name="q_lora_a"
)

scores = system.sram_dynamic_matmul(
    q, k.swapaxes(-1, -2), op_name="attention_qk_t"
)

report = system.summary()
```

The system-level data path uses `TensorPacket` objects:

```python
x = system.create_tensor("x", x_array, bit_width=8, location="digital_buffer")
x_rram = system.transfer(x, system.rram.name)
y_rram = system.rram_matmul_tensor(
    x_rram, rram_weight, output_name="y_backbone"
)
y_digital = system.transfer(y_rram, system.digital.name)
```

Calling a compute function does not implicitly count a transfer. A transfer is
counted only through `DataMover.transfer()` or `DataMover.broadcast()`, so data
reuse and future buffer policies can be represented explicitly.

## Composite HybridLinear

```python
layer = HybridLinear("q_projection", system)
layer.prepare(
    HybridLinearWeights(
        backbone=w0,
        lora_a=a,
        lora_b=b,
        bias=bias,
    )
)

system.reset_stats()  # keep resident weights; isolate runtime statistics
q = layer.forward(x_packet, output_name="q")
```

`prepare()` produces weight-programming/loading statistics. `forward()` produces
runtime compute and activation-transfer statistics. Keeping those phases
separate prevents one-time setup cost from being mixed with per-inference cost.

## Five-layer architecture

```text
Layer 1  TensorPacket and event records
Layer 2  RRAMCIM, SRAMCIM, DigitalProcessor, DataMover
Layer 3  HybridLinear, MultiHeadAttention, FeedForward
Layer 4  TransformerEncoderBlock
Layer 5  HybridCIMBlockSimulator and report exporters
```

The public block-level interface is deliberately small:

```python
simulator = HybridCIMBlockSimulator(config)
preparation_report = simulator.prepare(block_weights)
result = simulator.run(input_array)
simulator.export_report("output/report")
```

`result.output` is the numerical block result. `result.report` contains per-unit
MAC counts, digital primitive counts, detailed transfers, traffic grouped by
path, and setup/runtime storage events. Reports can be exported as JSON,
Markdown, and four detailed CSV files.

## Reproducible experiments

The formal ViT-B/16 single-block profile is stored in:

```text
experiments/configs/vit_base_block_halora.json
```

It fixes `B=1`, `N=197`, `D=768`, `H=12`, `D_ff=3072`, LoRA rank 4,
W8A8/32-bit accumulation, and a HaLoRA-style mapping profile. Run it with:

```powershell
$env:PYTHONPATH = "src"
py -3.12 experiments/run_block_experiment.py
```

Use a custom result directory when comparing configurations:

```powershell
py -3.12 experiments/run_block_experiment.py `
  --config experiments/configs/vit_base_block_halora.json `
  --output output/experiments/vit-base-block-halora
```

Each run writes the exact configuration, environment metadata, raw preparation
and execution events, and an analysis directory containing:

- compute share by hardware unit, Transformer stage, and computation role;
- per-Encoder-Block-component compute, digital operations, transfer events,
  transferred bytes, tensor shapes, and participating hardware;
- per-component RRAM/SRAM MAC split and four directional traffic breakdowns
  (`Digital->RRAM`, `RRAM->Digital`, `Digital->SRAM`, `SRAM->Digital`);
- traffic events/bytes and byte share by path and stage;
- arithmetic intensity in MAC/byte and communication in byte/MAC;
- the largest individual tensor transfer;
- JSON, Markdown, and comparison-ready CSV files.

Matrix multiplication accepts arrays with two or more dimensions and follows
NumPy broadcasting semantics for leading batch dimensions. For
`(..., M, K) @ (..., K, N)`, the recorded MAC count is the number of broadcast
batches multiplied by `M * K * N`.

