# Qwen3-Coder-30B-A3B Betelgeuse: gfx906 validation record

Date: 2026-11-21

## Scope

This record documents validation of the native HIP pager data path on the
target gfx906 workstation, plus a first end-to-end MoE inference smoke test.
It is the gfx906 analogue of the MSI/RTX-3050 Qwen3.5 record in
`qwen35-msi-validation-2026-11-18.md`, using the closest available model and
parameters.

## Parameter mapping (closest match to the MSI setup)

| Parameter            | MSI reference                | This setup                          |
| -------------------- | ---------------------------- | ----------------------------------- |
| GPU                  | RTX 3050 4 GiB (sm_86, CUDA) | AMD MI50 32 GiB (gfx906, HIP/ROCm)  |
| Model                | Qwen3.5-35B-A3B IQ2_XXS      | Qwen3-Coder-30B-A3B-Instruct Q4_K_M |
| GGUF arch            | `qwen35moe`                  | `qwen3moe`                          |
| Expert layers        | 41 (40 trunk + MTP blk.40)   | 48                                  |
| Experts per layer    | 256                          | 128                                 |
| Routed experts/token | 8                            | 8                                   |
| Backend              | CUDA (`ggml-cuda`)           | HIP (`ggml-hip`), branch gfx906-port|
| Pager test args      | `40 256 8`                   | `48 128 8`                          |

## Target workstation

- pvf (home lab host)
- AMD Ryzen AI 9 HX 370 (Strix Point; Radeon 890M iGPU kept off the test path
  via `HIP_VISIBLE_DEVICES=1`)
- AMD Instinct MI50 32 GiB HBM2, gfx906, internal PCIe Gen4 x16 (16 GT/s)
- 28 GiB system RAM; kernel 7.3.0-rc5-vanilla+
- ROCm 6.2.4 (clang from `/opt/rocm/lib/llvm/bin`)

## Source model

- File: `Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf`
- Size: 18,556,689,568 bytes
- SHA-256: `fadc3e5f8d42bf7e894a785b05082e47daee4df26680389817e2093056f088ad`
- GGUF v3; architecture `qwen3moe` (Unsloth conversion)
- 48 expert-bearing blocks, 128 experts per block, 8 routed experts per token

## Bundle generation and integrity

`tools/moe_pack.py` dry run reported:

- 48 layers
- 128 experts per layer
- 6,144 experts total
- 17,553,162,240 expert payload bytes

`--verify` compared 3 experts per layer (144 experts) byte-for-byte: all
matched. Generated bundle facts:

- Expert pack: 17,553,162,240 bytes (`experts.bin`, 4096-aligned)
- Expert-pack SHA-256: `339bc168986b11cc1701ae3b9afbe8c6a7fd375049cbd0ce398cf609380dd13c`
- Index: `native_experts.txt` (native-experts-v3 format)

## HIP build

CMake configured with:

```bash
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release
```

The pager test target built successfully:

```bash
cmake --build build --target test_moe_pager -j8
```

Build result: `[100%] Built target test_moe_pager`. The test links
`ggml-hip`, `uring` and the ROCm runtime; CUDA calls in the pager and test
are mapped through `src/moe_pager_gpu.h`.

## Native pager result

Command shape:

```bash
HIP_VISIBLE_DEVICES=1 build/bin/test_moe_pager \
  /models/Qwen3-Coder-30B-A3B-q4km/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf \
  48 128 8
```

Observed result:

```text
  fetch(L 0,E  0): compared 3059712 bytes  EXACT MATCH
  fetch(L 0,E  1): compared 3059712 bytes  EXACT MATCH
  fetch(L 7,E123): compared 2654208 bytes  EXACT MATCH
  fetch(L23,E127): compared 3059712 bytes  EXACT MATCH
  fetch(L47,E127): compared 3059712 bytes  EXACT MATCH
  async(L 1,E 19): 3059712 bytes via prefetch+commit  EXACT MATCH (pf_hit=1)
  async(L 8,E 44): 3059712 bytes via prefetch+commit  EXACT MATCH (pf_hit=2)
  async(L11,E127): 3059712 bytes via prefetch+commit  EXACT MATCH (pf_hit=3)
  async(L37,E  7): 2654208 bytes via prefetch+commit  EXACT MATCH (pf_hit=4)
  capacity-rejection: PASS
  size-rejection: PASS
  write-allocate(L24,E 42): 2654208 bytes in host and device EXACT MATCH
  reuse-stability(L24,E 42<->E 43): PASS
  prefetch-batch(L24,E 47): PASS (pf_hit=4->5)
  telemetry: disk_wait_us=10217 dma_ms=5.837 dma_wait_us=4134 dma_bytes=37343232
  batch(L 0,N36,pass0): EXACT MATCH
  batch(L 0,N36,pass1): EXACT MATCH
  batch(L24,N36,pass0): EXACT MATCH
  batch(L24,N36,pass1): EXACT MATCH
  batch(L47,N36,pass0): EXACT MATCH
  batch(L47,N36,pass1): EXACT MATCH

result: PASS   io_ops=687 io_bytes=671858688 pf_hit=8 disk_wait_us=12441 dma_ms=245.245
```

## Defect found and fixed during validation

`fetch(L47,E127)` initially failed: the layer-47 `ffn_up_exps` tensor ends at
the GGUF file end, which is not 4096-aligned. `wait_ok_` required each
O_DIRECT read to return the full padded length, so the physically absent
tail padding produced a false failure on the last expert. The check now
requires only the real payload (`pre + role bytes`), matching the existing
`is_request_complete` semantics. Commit `fe93e5796`.

## End-to-end smoke test (beyond the MSI scope)

`bel-cli` completed a real MoE decode on the MI50 with expert streaming:

```bash
HIP_VISIBLE_DEVICES=1 build/bin/bel-cli \
  -m /models/DeepSeek-V2-Lite-Chat/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf \
  --moe-direct-io -ngl 99 -c 2048 -p "The capital of France is" -n 16
```

Output: `The capital of France is Paris.` (~1.2 tok/s cold, small context,
single slot; not a benchmark).

## What this establishes

- The HIP-compiled pager reads selected experts through io_uring + O_DIRECT
  on gfx906 and matches the GGUF slices byte-for-byte.
- Synchronous, asynchronous prefetch/commit, write-allocate, cache-reuse,
  prefetch-batch and batch-read paths pass on the MI50.
- The gfx906 `bel-cli` binary loads a MoE GGUF and generates tokens with
  `--moe-direct-io` enabled.

## What this does not establish

- No sustained throughput, latency, or warm-pass benchmark for inference is
  claimed. The single decode ran at ~1.2 tok/s cold and was not tuned
  (no slot-pool sizing, no profile, no warm passes).
- No pack-mode (`--moe-pack-dir`) runtime path was exercised by the test;
  pack contents were verified by `moe_pack.py --verify` only.
- Doorbell/predictor device kernels are compiled for gfx906 but their
  PCIe-coherency semantics under sustained load are untested.

## Next engineering gate

Before benchmarking, inspect and document:

1. whether `--moe-pack-dir` activates pack mode on the HIP path;
2. warm-pass tok/s on the 30B model with a bounded slot pool
   (`--moe-slots`, `--moe-cache-mib`) versus the reference envelope;
3. doorbell spin/`glc` coherency under continuous expert traffic;
4. whether the iGPU can serve the dense path in parallel (gfx1150 lacks
   device code in this build).
