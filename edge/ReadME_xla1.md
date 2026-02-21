### XLA Compilation Pipeline

```
┌─────────────────┐
│  JAX Python     │
│  Code           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  JAX IR         │
│  (jaxpr)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  HLO            │
│  (High-Level    │
│   Optimizer)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  XLA            │
│  Optimizations  │
│  - Fusion       │
│  - Layout       │
│  - Algebraic    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLVM IR        │ ← **Target achieved**
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Machine Code   │
│  (PTX/x86)      │
└─────────────────┘
```


### Optimization Stack

```
┌─────────────────────────────────────┐
│  Input: Raw Images                  │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  Memory Transfer Optimizer          │
│  • Batch transfers (97% reduction)  │
│  • Pinned memory                    │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  XLA Graph Optimizer                │
│  • Operator fusion                  │
│  • Layout optimization              │
│  • LLVM IR lowering                 │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  Optimized Inference Engine         │
│  • JIT compilation                  │
│  • Batched execution                │
│  • 30%+ throughput increase         │
└──────────────┬──────────────────────┘
               ↓
┌─────────────────────────────────────┐
│  Model Export                       │
│  • TFLite (mobile/edge)             │
│  • ONNX (cross-platform)            │
└─────────────────────────────────────┘
```
