# LFM2.5 on the AMD FirePro W9100, in Rust

An experimental, model-specific inference engine for **LFM2.5-8B-A1B Q6_K** on the **16 GB AMD FirePro W9100**. Built and tested on a Dell Precision 5820 running Linux and Mesa RADV. This is not a general replacement for llama.cpp.

The inference graph, GGUF reader, tokenizer, chat formatter and incremental decoder are Rust. Vulkan is driven through `ash`. GPU kernels are authored as typed Rust computation IR and lowered directly to SPIR-V with `rspirv`: no generated GLSL source, shader-text compiler, Python inference or C++ inference backend. A separate Python standard-library HTTP adapter is included as a **machine-specific deployment reference**, not as an all-Rust HTTP server or a turnkey installer.

## What was verified

The accepted implementation was independently reviewed and exercised on the W9100 before publication:

- 808/808 greedy output tokens matched the retained native LFM reference fixtures.
- 24 isolated sessions, and the 4096-token context boundary (3840 prompt + 256 output).
- A ten-minute persistent-engine run: 804 exact-output requests, one model load, no new GPU faults.
- The separate persistent HTTP/SSE integration passed real chat/streaming tests, ten minutes of exact-output requests, and administrative stop/start checks.
- In measured warm workloads, request time was **19–31% lower** than the qualified native Vulkan LFM configuration. Model loading is excluded. This is not a comparison against the separately running Bonsai model.

See [qualification data](validation/qualification.json) for the per-workload times, paired measurements and tested scope. These are retained historical measurements, **not** a new benchmark of the publication package. All computational Rust sources are byte-identical to the accepted source; see [publication notes](PUBLICATION.md).

## Build

Use Rust 1.98.1, Linux, a Vulkan loader and Mesa RADV. Building and the self-contained CPU checks do not require a GPU. Actual GPU inference deliberately requires W9100 device ID `0x67a0` and is not qualified on other cards.

```sh
git clone https://github.com/Amidwestnoob/lfm2-w9100-rust.git
cd lfm2-w9100-rust
cargo build --locked --release --manifest-path engine/Cargo.toml --bins
```

Obtain a compatible LFM2.5-8B-A1B Q6_K GGUF separately and comply with its publisher's license. Model weights are **not** included; this repository's MIT license does not license those weights. The exact artifact used for qualification is identified by SHA-256 in the qualification data.

## CLI inference

```sh
LFM_ATTENTION=coalesced LFM_Q6_ALIGNED=1   engine/target/release/lfm2-greedy   --model /path/to/LFM2.5-8B-A1B-Q6_K.gguf   --text 'Explain why the sky is blue.' --n 64 --stream
```

Choose exactly one of `--text`, `--tokens`, or `--chat-json`. Streaming emits incremental JSON lines. The `--cpu --tokens` path is an explicit diagnostic reference, not an automatic fallback. The engine is greedy-only, model-specific and bounded to 4096 total context tokens; it is not a general sampling, tools or multi-model server.

## Load-once Rust protocol

`api_stream` uses the same accepted GPU engine and tokenizer, loads once, resets session state between requests and emits incremental JSONL results over stdin/stdout:

```sh
LFM_ATTENTION=coalesced LFM_Q6_ALIGNED=1   engine/target/release/api_stream --model /path/to/MODEL.gguf
```

Example input line:

```json
{"id":"example-1","text":"Explain why the sky is blue.","n":64}
```

The device must be the W9100 at `/sys/class/drm/card0/device`, with sufficient free VRAM. `--validate-only MODEL.gguf` validates requests without constructing the GPU engine. Normal GPU execution is not a smoke test to run on an arbitrary card.

## HTTP deployment reference

`service/` contains the tested Python HTTP/SSE adapter and supervisor, plus the Rust bridge source. It serves loopback-only `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`. It supports serialized requests, bounded queuing, request validation and incremental streaming; it does not expose shell/file endpoints.

**Do not run `service/server.py` unmodified on another machine.** Its guard deliberately binds the original deployment's model hash, binary certificate, protected services, cooler configuration and retained fault evidence. The private machine certificates, systemd unit, logs, credentials and model are not published. The model path is redacted. This reference is included so the implementation is available, not to pretend a hardware-specific deployment is portable. The standalone Rust CLI and JSONL binary build independently of these private certificates.

## Self-contained CPU checks

```sh
cargo test --locked --manifest-path engine/Cargo.toml --lib --test dequant_mirror --bin lfm2-greedy
cargo test --locked --manifest-path parser/Cargo.toml --lib
cargo test --locked --manifest-path tokenizer/Cargo.toml --lib
mkdir -p tmp evidence
python3 tests/test_service_cpu.py
```

These are CPU regression checks, not GPU qualification. The full tokenizer integration tests additionally require a separately obtained metadata/oracle packet via `W9100_TOKENIZER_PACKET`. `engine/tests/text_stream.rs` requires `model/LFM2.5-8B-A1B-Q6_K.gguf` and its `model/golden.json` native fixtures. Those model-derived packets are not redistributed. The default full `cargo test` is therefore not a fixture-free command; do not mistake the library-only CI for the full hardware acceptance suite.

## Limits and safety

- Tested target: W9100/Hawaii wave64 and this one Q6_K model, not other GPUs, quantizations or arbitrary GGUFs.
- Measured/predictive GPU timing checks and sticky-stop behavior remain in the accepted Rust implementation. The HTTP supervisor adds temperature, protected-service, exclusive-owner and monitor-freshness checks.
- A timing screen is not a physical watchdog or a guarantee against driver/device faults. Review the hardware-specific controls before live use; do not remove checks merely to make another GPU run.
- Native long-context intermediate-score divergence remains unresolved. Matching tested greedy outputs does not prove universal logit/cache equivalence.
- No cloud inference dependency. Fetching source/dependencies/model is separate from local inference.

## Credits and license

Drew Schuyler / @amidwestnoob, with AI-assisted implementation and independent review. Q6_K arithmetic/layout and tokenizer Unicode data retain llama.cpp/ggml MIT attribution. See [third-party notices](THIRD_PARTY_NOTICES.md). This is an independent project, not an official Liquid AI or AMD product.
