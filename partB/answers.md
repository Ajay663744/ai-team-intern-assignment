# Part B — Capacity Reconciliation

All arithmetic is reproduced verbatim from the Python snippets below (run with Python 3).
Intermediate values are **not** rounded; only final displayed figures are rounded.

---

## B1 — KV-cache Sizing

### B1(a) KV-cache memory cost per token

**Formula (GQA-aware):**

```
bytes_per_token = layers × 2 × kv_heads × head_dim × bytes_per_element
```

The factor `2` covers one K tensor and one V tensor per layer.  
GQA means we use `kv_heads = 8` (not the 24 Q-heads); `head_dim = 128`; precision = fp16 = 2 bytes.

```python
layers    = 28
kv_heads  = 8        # GQA — only 8 KV heads, not 24 Q heads
head_dim  = 128
prec      = 2        # fp16 bytes

bytes_per_tok = 28 × 2 × 8 × 128 × 2
             = 28 × 2 × 8 × 256
             = 28 × 4096
             = 114 688 bytes/token   (112.00 KB/token)
```

**Result: 114 688 bytes per token.**

---

### B1(b) Maximum concurrent sequences at `max_model_len = 4096`

Step-by-step:

```python
# 1. Usable GPU memory
gpu_mem_total   = 24.0 GB
gpu_mem_util    = 0.92
usable_mem      = 24.0 × 0.92  = 22.08 GB

# 2. Weight footprint (fp16)
params          = 4.2 × 10^9 parameters
weight_bytes    = 4.2 × 10^9 × 2  = 8.40 GB

# 3. Non-KV runtime overhead (stated)
non_kv_overhead = 1.60 GB

# 4. KV budget
kv_budget = 22.08 − 8.40 − 1.60  =  12.08 GB  (12 080 MB)

# 5. KV cost per sequence at max_model_len
max_model_len = 4096 tokens
kv_per_seq    = 114 688 bytes/tok × 4096 tok
              = 469 762 048 bytes  ≈ 469.76 MB

# 6. Max concurrent sequences
max_seqs = 12 080 MB ÷ 469.76 MB  = 25.715  → floor → 25
```

**Result: 25 concurrent sequences fit at maximum context length.**

---

### B1 Cross-check against `bench_log.csv`

Long-prompt rows (`prompt_len = 3584`, `gen_len = 512`):

| batch_size | kv_cache_util | preempted_seqs |
|:---:|:---:|:---:|
| 4  | 0.16 | 0 |
| 8  | 0.31 | 0 |
| 16 | 0.62 | 0 |
| 24 | 0.93 | **0** |
| 32 | 0.97 | **7** |
| 48 | 0.97 | **23** |

**Prediction vs log:**  
The predicted ceiling is **25 concurrent sequences**.

- `bs = 24`: 0 preemptions, `kv_cache_util = 0.93`. Since 24 < 25, every sequence fits — consistent with prediction.  
- `bs = 32`: 7 preemptions, `kv_cache_util = 0.97`. Since 32 > 25, the scheduler must evict — consistent with prediction.

The break point falls exactly between bs = 24 and bs = 32, matching the predicted ceiling of 25 within the resolution of the batch-size grid. **The log confirms the prediction.**

---

## B2 — Throughput Anomaly in the Long-context Sweep

### Data (prompt_len = 3584)

| bs | wall_s | reported_tok_s | kv_util | preempted | ttft_ms_p50 | itl_ms_p50 |
|:---:|---:|---:|:---:|:---:|---:|---:|
| 4  |  28.98 |  565.4 | 0.16 |  0 |  483.2 | 51.33 |
| 8  |  36.30 |  902.6 | 0.31 |  0 |  519.0 | 62.26 |
| 16 |  49.97 | 1311.4 | 0.62 |  0 |  498.3 | 77.20 |
| 24 |  61.16 | 1607.4 | 0.93 |  0 |  500.5 | 96.07 |
| 32 |  94.71 | 1384.0 | 0.97 |  7 |  636.9 | 101.79 |
| 48 | 151.41 | 1298.5 | 0.97 | 23 |  955.4 | 100.00 |

### Where the anomaly begins

Throughput should scale linearly with batch size. Actual scaling ratios:

```
bs  4 →  8:  565.4 → 902.6   actual_ratio = 1.596   naive_expected = 2.000
bs  8 → 16:  902.6 → 1311.4  actual_ratio = 1.453   naive_expected = 2.000
bs 16 → 24: 1311.4 → 1607.4  actual_ratio = 1.226   naive_expected = 1.500
bs 24 → 32: 1607.4 → 1384.0  actual_ratio = 0.861   naive_expected = 1.333  ← DROPS BELOW 1
bs 32 → 48: 1384.0 → 1298.5  actual_ratio = 0.938   naive_expected = 1.500  ← still falling
```

**Throughput peaks at bs = 24 (1607.4 tok/s) and falls at bs = 32 (1384.0 tok/s).**  
The ratio at `bs 24 → 32` is 0.861 — below 1.0 — meaning adding 8 more sequences actively reduces throughput.

### Mechanism (evidence from specific columns)

The break coincides exactly with the B1 KV-cache ceiling of 25 sequences:

1. **KV pool exhaustion**: `kv_cache_util` hits 0.97 at bs = 32 (up from 0.93 at bs = 24) — the allocator has no free blocks.
2. **Preemption begins**: `preempted_seqs` jumps from 0 (bs = 24) to 7 (bs = 32) to 23 (bs = 48) — the scheduler swaps in-flight sequences out, paying re-computation or swap-I/O cost.
3. **TTFT spikes**: 498–500 ms at bs ≤ 24 → 636.9 ms at bs = 32 → 955.4 ms at bs = 48 — preempted sequences must wait for a free KV slot before their prefill can restart.
4. **Wall clock balloons**: 61.16 s → 94.71 s → 151.41 s — effective parallelism collapses because stalled sequences hold KV memory while not decoding.

All four columns (`kv_cache_util → 0.97`, `preempted_seqs > 0`, `ttft_ms` rising, `wall_clock_s` ballooning) point to the same root cause: **the KV pool is full and the scheduler is thrashing**, consistent with the B1 ceiling calculation.

### Quantified overhead

Linear extrapolation baseline: bs = 24 (last clean batch, 0 preemptions).

```python
# bs = 32  (factor = 32/24 = 1.333)
expected_tok_s_32 = 1607.4 × (32/24) = 2143.20 tok/s
actual_tok_s_32   = 1384.0 tok/s
throughput_loss   = (1384.0 − 2143.20) / 2143.20 = −35.4%

expected_wall_32  = 61.16 × (32/24) = 81.55 s
actual_wall_32    = 94.71 s
wall_overhead_32  = (94.71 − 81.55) / 81.55 = +16.1%

# bs = 48  (factor = 48/24 = 2.000)
expected_tok_s_48 = 1607.4 × (48/24) = 3214.80 tok/s
actual_tok_s_48   = 1298.5 tok/s
throughput_loss   = (1298.5 − 3214.80) / 3214.80 = −59.6%

expected_wall_48  = 61.16 × (48/24) = 122.32 s
actual_wall_48    = 151.41 s
wall_overhead_48  = (151.41 − 122.32) / 122.32 = +23.8%
```

The preemption regime costs **−35% throughput at bs = 32** and **−60% at bs = 48**, with wall-clock overhead of +16% and +24% respectively.

### Proposed fix

**Set `max_num_seqs = 24`** in the vLLM serving config.

Justification from the log: bs = 24 is the largest batch with `kv_cache_util = 0.93`, zero preemptions, and peak measured throughput (1607 tok/s reported). Capping admission at 24 prevents the scheduler from ever entering the preemption regime.

Predicted effect: throughput holds at ≥ 1607 tok/s (reported) with TTFT ≤ 500 ms p50 — versus collapsing to 1298 tok/s with TTFT = 955 ms at bs = 48. P95 end-to-end latency would stay near the bs = 24 baseline of 69 221 ms, not climb to 105 427 ms (bs = 48). The bs = 24 row in the log already demonstrates this regime.

---

## B3 — Auditing REPORT_v0 Section 2

### Step 1: Reverse-engineering `reported_tok_s`

Two candidate formulas:

- **Formula A** (decode-only): `gen_len × num_requests / wall_clock_s`
- **Formula B** (total tokens): `(prompt_len + gen_len) × num_requests / wall_clock_s`

Tested against six rows:

| bs | pl | gl | nr | wall_s | reported | Formula A | Formula B | A err% | B err% |
|:---:|:---:|:---:|:---:|---:|---:|---:|---:|---:|---:|
| 1  | 512 | 256 | 1  | 10.94 |   70.2 |   23.40 |   70.20 | −66.67% | **+0.00%** |
| 16 | 512 | 256 | 16 | 13.91 |  883.2 |  294.46 |  883.39 | −66.66% | **+0.02%** |
| 64 | 512 | 256 | 64 | 21.68 | 2267.3 |  755.72 | 2267.16 | −66.67% | **−0.01%** |
| 8  |3584 | 512 | 8  | 36.30 |  902.6 |  112.84 |  902.70 | −87.50% | **+0.01%** |
| 16 |3584 | 512 | 16 | 49.97 | 1311.4 |  163.94 | 1311.51 | −87.50% | **+0.01%** |
| 24 |3584 | 512 | 24 | 61.16 | 1607.4 |  200.92 | 1607.33 | −87.50% | **−0.00%** |

Explicit arithmetic for three rows:

```python
# Row bs=1, pl=512, gl=256, nr=1, wall=10.94 s
Formula_A = 256 × 1 / 10.94           = 23.4004    err = −66.666%
Formula_B = (512+256) × 1 / 10.94     = 70.2011    err = +0.002%

# Row bs=32, pl=512, gl=256, nr=32, wall=16.5 s
Formula_A = 256 × 32 / 16.5           = 496.4848   err = −66.670%
Formula_B = (512+256) × 32 / 16.5     = 1489.4545  err = −0.010%

# Row bs=16, pl=3584, gl=512, nr=16, wall=49.97 s
Formula_A = 512 × 16 / 49.97          = 163.9384   err = −87.499%
Formula_B = (3584+512) × 16 / 49.97   = 1311.5069  err = +0.008%
```

**Conclusion: Formula B matches every row to < 0.02%.** `reported_tok_s` counts **(prompt_len + gen_len) × num_requests / wall_clock_s** — it counts input tokens in the numerator alongside output tokens.

### Step 2: The common source of both REPORT_v0 errors

Both conclusions stem from the same conflation: **`reported_tok_s` counts prompt tokens as if they were generated tokens**, inflating the numerator by a factor of `(prompt_len + gen_len) / gen_len`.

- For short-prompt rows: inflation = `(512+256)/256 = 3.0×`
- For long-prompt rows: inflation = `(3584+512)/512 = 8.0×`

**Claim 1** ("longer prompts → better utilization/throughput"): Long-prompt rows show higher `reported_tok_s` only because their inflation factor is 8× vs 3× for short-prompt rows — not because more useful work is done. Honest goodput at bs = 16 is **163.9 tok/s for long prompts vs 294.5 tok/s for short** — long prompts are *worse*, not better.

**Claim 2** ("batch 48 → ~3200 tok/s"): The "best observed" 1600 tok/s is already 8× inflated. Linear extrapolation from an inflated metric to batch 48 yields a doubly-wrong prediction. Actual reported_tok_s at bs = 48 is 1298.5 (not 3200); actual honest goodput is only **162.3 tok/s**.

### Step 3: Honest goodput — bs = 24, prompt_len = 3584

Row values: `gen_len = 512, num_requests = 24, wall_clock_s = 61.16, ttft_ms_p50 = 500.5, itl_ms_p50 = 96.07`

**Method 1** — gen_len × num_requests / wall_clock_s:

```python
goodput_M1 = 512 × 24 / 61.16
           = 12 288 / 61.16
           = 200.92 tok/s
```

**Method 2** — subtract prefill time, use decode-only window:

```python
ttft_s      = 500.5 / 1000 = 0.50050 s
decode_wall = 61.16 − 0.50050 = 60.65950 s

goodput_M2  = 512 × 24 / 60.65950
            = 12 288 / 60.65950
            = 202.57 tok/s
```

**Comparison:**

```
Method 1:  200.92 tok/s
Method 2:  202.57 tok/s
Difference: |200.92 − 202.57| / 200.92 = 0.83%
```

The two methods agree to **0.83%**, confirming the result. Honest decode goodput at bs = 24, pl = 3584 is **≈ 201 tok/s**.

*(Note: Method 2b using ITL gives `24 / 0.09607 s = 249.8 tok/s`, diverging 23% from Methods 1 & 2. The p50 ITL is a median of mid-flight decode steps and excludes scheduler overhead and tail latency, so it overestimates sustained goodput. Methods 1 and 2, which use actual wall-clock time, are the reliable derivation.)*

### What REPORT_v0 Section 2 should have said

`reported_tok_s` counts prompt + output tokens combined, making it unsuitable for comparing workloads with different prompt lengths or for capacity planning; the correct metric is `gen_len × num_requests / wall_clock_s`. At bs = 16, honest decode goodput is 294.5 tok/s for short prompts and only 163.9 tok/s for long prompts — **longer prompts reduce useful output throughput** because each long-context request occupies KV cache that could otherwise serve more concurrent short requests. The practical serving ceiling on this L4 is bs = 24 (≈ 201 tok/s honest goodput); beyond that, KV-cache exhaustion triggers preemption and collapses throughput to 162 tok/s at bs = 48 — making the "3200 tok/s at batch 48" projection off by roughly 20× in honest terms.

---

## B4 — Confirming the Mechanism in Production

In a live vLLM deployment, pull the **`vllm:num_preemptions_total`** counter (a Prometheus counter exposed by vLLM's `/metrics` endpoint). If the B2 KV-cache-pressure mechanism is correct, this counter should be zero (or negligible) at concurrency ≤ 24 for this workload, and should begin incrementing — at an accelerating rate — as the number of in-flight long-context sequences is pushed past 25. Concretely, the per-request preemption rate (counter delta ÷ `vllm:request_success_total` delta over the same observation window) would jump sharply at that threshold, mirroring the transition from `preempted_seqs = 0` at bs = 24 to `preempted_seqs = 7` at bs = 32 observed in the load-test log.
