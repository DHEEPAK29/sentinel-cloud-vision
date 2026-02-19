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
