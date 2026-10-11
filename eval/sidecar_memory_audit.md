# Sidecar Memory Audit — MLX Inference Loop

**Date:** 2026-10-11  
**Test:** 3 of Bouncer test suite  
**Stack:** MLX 0.32.2 + laya_mlx (Laya v3d checkpoint) on macOS Metal  
**Agent class:** `laya_mlx.agent.Agent` — stateless per call, no retained history

---

## Findings

### 1. No Python-side memory leak
- `pyobj` count stays flat at ~48,869 throughout all phases (drift: 0 objects)
- `active_mb` (live Metal buffers for weights + transients) stays constant at ~803 MB
- **Conclusion:** The inference loop does not accumulate Python objects or hold onto transient tensors.

### 2. MLX cache grows without bound under varied-length requests — this is the fragmentation problem
Under phase A (100 varied-length tasks, no explicit sweep):

| Sample | cache_mb (MLX allocator pool) | Latency (ms/batch) |
|--------|-------------------------------|---------------------|
| 10     | ~8,000                        | 133                 |
| 50     | ~27,600                       | 203                 |
| 100    | ~43,100                       | 871                 |

**The MLX Metal cache (allocator pool of freed-but-not-released buffers) grows from ~1.5 GB to ~43 GB over 100 requests.** This is not a leak — the buffers are pooled for reuse. But because each request has a different input length, the allocator stores differently-sized fragments that never get reused by later requests. Over time, this creates a massive pool of unusable cached memory.

### 3. Fixed-length requests don't fragment
Under phase A2 (100 identical-length tasks):

- Cache fills once (~45 GB) then stays flat at ~45 GB (drift: +0 MB after initial fill)
- Latency drops from ~539 ms (first batch) to ~75 ms (last batch) — the allocator reuses the same shape

This confirms that **the growth is shape-driven fragmentation, not a leak**. With identical shapes, buffers are reused. With varied shapes, each new length adds a new cache entry.

### 4. `mx.clear_cache()` releases everything instantly
After phase A2 accumulated ~45 GB in the cache:
```
post-clear_cache → cache_mb = 0.0 (all released)
```
One call to `mx.clear_cache()` freed **45,260 MB** of pooled buffers. This is not garbage collection — it's a hard reset of MLX's Metal allocator pool.

### 5. Explicit sweep eliminates both fragmentation and latency degradation
Under phase B (varied-length requests, `mx.clear_cache()` + `gc.collect()` every 10):

- Cache stays at ~0 MB throughout
- RSS stays flat at ~155 MB (vs 143 MB in phase A)
- Latency stays stable at ~144 ms/batch (no degradation)

**The sweep prevents cache accumulation entirely.** The cost is that each request pays the full allocation cost — but since MLX allocates small Metal buffers (~1.6 GB peak active), this is fast and bounded.

### 6. Latency penalty from unbounded cache
Phase A shows clear latency degradation as cache grows:

- Requests 10–50: ~130–260 ms (cache growing, allocator searching through larger pool)
- Requests 70–100: ~470–871 ms (allocator spending more time managing a huge fragmented pool)

Phase B stays flat at ~144 ms because the cache never grows.

---

## Root Cause

The `Agent.system_one()` method in `laya_mlx/agent.py` is **stateless per call** — no retained history, no prompt cache (when `cache_prompts=False`). However, every request allocates new Metal tensors via `collate_items()`, and MLX's Metal allocator caches freed buffers by shape.

When the sidecar processes varied-length tasks:
1. Request N allocates tensors of shape S_N → runs forward pass → tensors freed → cached by shape S_N
2. Request N+1 has a different length → new shape S_{N+1} → allocates fresh Metal memory
3. The old shape S_N's buffers sit in the cache, never reused
4. After 100 requests with random lengths (8–320 words), the cache accumulates ~45 GB of shape-specific fragments

This is **not MPS fragmentation** (the question's premise) — it's MLX's Metal allocator caching freed buffers by shape without an eviction policy. The MLX `cache_memory` metric tracks this pool, not actual live usage.

---

## Verdict

| Question | Answer |
|----------|--------|
| Is there a Python-side memory leak? | **NO** — pyobj count flat, active memory constant |
| Does the inference loop accumulate Metal buffers? | **YES** — cache grows to ~43 GB over 100 varied requests |
| Is this fragmentation or a leak? | **Fragmentation** — shape-specific pooling with no reuse |
| Does `mx.clear_cache()` fix it? | **YES** — releases all pooled buffers instantly |
| Is an explicit GC sweep required? | **YES, if processing varied-length requests** — without it, cache grows unbounded and latency degrades 5× |

---

## Recommendation for `serving/sidecar.py`

Add a periodic `mx.clear_cache()` call to the inference loop. The optimal strategy:

```python
import gc
import mlx.core as mx

# Inside the request loop in sidecar.py, after each predict():
result = agent.predict(task, QUESTION_SET)

# Optional: clear cache every N requests to prevent accumulation
if request_count % 10 == 0:
    mx.clear_cache()
    gc.collect()
```

**Why not every request?** Because clearing the cache on every request means you lose all allocator benefits — even fixed-length requests would pay full allocation cost each time. Clearing every 10 requests lets same-shape requests within a batch reuse buffers while preventing unbounded growth.

**Expected impact:**
- Cache stays at ~0 MB (vs 43 GB without sweep)
- RSS stays flat at ~155 MB (vs potential gradual increase)
- Latency stays stable at ~144 ms/batch (vs degrading to 871 ms)
- Memory limit: ~1.6 GB peak active Metal buffers per request

---

## Benchmark Script

Full reproduction script at `eval/bench_sidecar_memory.py` — run with:
```bash
LAYA_MODEL_PATH=/Users/apollo/laya-router/checkpoints/laya-v3d-mlx \
/Users/apollo/laya-mlx-venv/bin/python eval/bench_sidecar_memory.py
```

Or with `N_REQUESTS=200` for longer runs.
