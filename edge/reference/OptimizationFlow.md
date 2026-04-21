## Optimization Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    BEFORE OPTIMIZATION                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  for each image in images:                                               │
│      ┌─────────────────────────────────────────────┐                    │
│      │ 1. Transfer image to GPU      [3.2ms]       │ ← N transfers      │
│      │ 2. Run Conv2D                 [2.1ms]       │                    │
│      │ 3. Run BiasAdd                [0.5ms]       │                    │
│      │ 4. Run ReLU                   [0.4ms]       │                    │
│      │ 5. Run Conv2D                 [2.8ms]       │                    │
│      │ 6. Run BiasAdd                [0.5ms]       │                    │
│      │ 7. Run ReLU                   [0.4ms]       │                    │
│      │ 8. Run Dense                  [1.2ms]       │                    │
│      │ 9. Transfer result to CPU     [0.8ms]       │ ← N transfers      │
│      └─────────────────────────────────────────────┘                    │
│      Total per image: ~15ms                                              │
│      Total for 100 images: ~1500ms                                       │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘

                                    ▼
                            XLA OPTIMIZATION
                                    ▼

┌──────────────────────────────────────────────────────────────────────────┐
│                     AFTER OPTIMIZATION                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  batch = stack(images)  # Batch assembly on CPU                          │
│      ┌─────────────────────────────────────────────┐                    │
│      │ 1. Transfer batch to GPU      [0.1ms]       │ ← 1 transfer!      │
│      │ 2. Run FusedConvBiasReLU      [1.8ms]       │ ← Fused!           │
│      │ 3. Run FusedConvBiasReLU      [2.2ms]       │ ← Fused!           │
│      │ 4. Run Dense                  [0.9ms]       │                    │
│      │ 5. Transfer results to CPU    [0.8ms]       │ ← 1 transfer!      │
│      └─────────────────────────────────────────────┘                    │
│      Total for batch of 32: ~5.8ms                                       │
│      Per image: ~0.18ms                                                  │
│      Total for 100 images: ~18ms (3 batches + 1 partial)                │
│                                                                           │
│  Improvement: 1500ms → 18ms = 83x faster per image!                      │
│  Throughput increase: ~30-40% (accounting for batching overhead)         │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```
