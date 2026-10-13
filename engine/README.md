# Rust W9100 inference engine

This is the accepted LFM2.5 inference source, published as an experimental W9100-specific engine. See the repository README and PUBLICATION.md for runnable build commands and qualification scope.

## Build and run

```sh
cargo build --locked --release --manifest-path engine/Cargo.toml
engine/target/release/lfm2-greedy --model MODEL.gguf --tokens 124894,123 --n 8
engine/target/release/lfm2-greedy --model MODEL.gguf --text 'Hello, world!' --n 8 --stream
engine/target/release/lfm2-greedy --model MODEL.gguf --chat-json settings.json --n 8 --stream
```

Select exactly one input mode. `--text` uses the embedded tokenizer with
`add_special=true, parse_special=false`. `--chat-json` takes the handoff settings
object, renders the GGUF template in Rust, then encodes with
`add_special=false, parse_special=true`. No production subprocess invokes a
reference runner or tokenizer.

Each generated non-EOS token is passed immediately to `StreamDecoder`.
`--stream` writes and flushes JSONL `{"text_delta":"..."}` chunks; the last line
contains `prompt_tokens`, `tokens`, `decoded_text`, `greedy`, `device`, and
`gpu_dispatch_count`. EOS is excluded from generated tokens. Incomplete UTF-8
is buffered; malformed UTF-8 is replaced consistently with lossy full decoding.
The accepted tokenizer's three whole-sequence space-cleanup passes are applied
to completed StreamDecoder text. A short trailing suffix is withheld until it
cannot change, so punctuation and contractions split across tokens remain correct.
Whole-sequence and incremental decoded text are checked for equality. This
stack targets the supplied LFM2 tokenizer, not arbitrary tokenizer cleanup rules.

The retained `--cpu --tokens` mode is a deliberately selected diagnostic
reference. It is not a fallback and cannot be combined with text, chat, or
streaming modes. Normal execution requires the W9100 (device ID `67a0`).

## Compute authoring

`src/kernels.rs` and `src/kernels/resident.rs` contain the seven required operations
and resident-graph helpers in typed Rust IR. Rust scalar
wrappers distinguish unsigned, float, and boolean values; operator overloads
emit typed binary SPIR-V instructions with `rspirv`. Rust closures construct
structured GPU loops. This does not run the tensor arithmetic on the CPU.

The binary SPIR-V standard extended instruction set provides floating-point
exp, sin, cos, power, inverse square root and half unpacking. Its standardized
name is not a shader source language frontend. There is no shader-text compiler,
shader string generation, opaque precompiled module, or shader asset loader.

`GpuOps` builds these kernels for real production dispatches, and `kernel_check`
uses the same `GpuOps` and kernel builders. `kernel-provenance.json` records the
Rust source and emitted SPIR-V SHA-256 values. `hash_dump` prints the hashes;
an optional directory argument exports generated binaries for independent
validation only. Those diagnostics are never used by inference.

Normal execution uses a persistent device-local graph: embedding lookup, routing,
short convolution, residual updates and cache management also execute on the GPU.
The host retains the unchanged greedy argmax and text pipeline. Cooperative Q6_K
uses four rows per Hawaii wave; two-column prefill tiles reuse decoded weights.
RMS keeps the native descending reduction tree, including a subgroup tail.
Command batches retain producer/consumer barriers and an explicit host-read
visibility barrier before completion. Descriptor bindings persist with the graph.
Model layout, signed Q6_K scales, GQA mapping, f16-rounded caches and MoE weighting
are retained. The retained host-staged forward is a diagnostic oracle, not a
fallback. This candidate targets the supplied LFM2.5 model and Hawaii wave64,
4..=64 experts, at most 4096 context positions and subsequent forward batches
no larger than the initial prompt. It is not a general-model engine.

## Verification

```sh
cargo test --locked --manifest-path engine/Cargo.toml --lib --test dequant_mirror --bin lfm2-greedy
# GPU diagnostics are hardware-specific and are NOT CPU CI checks:
engine/target/release/kernel_check --model MODEL.gguf --all
engine/target/release/resident_check MODEL.gguf
```

The historical independent hardware gates checked protected services, real W9100 operations, native greedy-token fixtures and streaming parity. See `../validation/qualification.json` and `../PUBLICATION.md` for the retained acceptance scope and public-package changes. Private deployment gates and model fixtures are not included; the commands above are not new claims of GPU acceptance.
