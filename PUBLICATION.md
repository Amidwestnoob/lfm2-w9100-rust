# Publication identity and scope

The computational source comes from the independently accepted LFM2.5 W9100 deployment, not the later unqualified optimization experiment and not the Bonsai Rust project.

All engine/parser/tokenizer production Rust sources and Cargo dependencies are unchanged from that accepted snapshot. The accepted `api_stream.rs` protocol bridge is added under `engine/src/bin/` for Cargo discovery and also retained in `service/`.

Publication-only edits: replace one private absolute fixture lookup with a repository-relative candidate; redact the historical model path in the machine-specific reference guard; remove the private host prefix from the SPIR-V validation-tool note; replace stale README references and omit a machine-specific timing shell script. No inference arithmetic, quantization, tokenizer semantics, work bounds, cache layout or GPU kernel is changed.

A fresh publication build and self-contained CPU checks were run locally. The W9100 measurements in `validation/qualification.json` are reused, identified historical evidence. No new GPU benchmark or service restart was performed for publication. The exact publicly committed snapshot is the publication identity; historical W9100 binary hashes are not hashes of this fresh local build.

The original machine-specific HTTP supervisor requires private deployment evidence and protected-service configuration; those are intentionally not published. Source availability is not a claim of turnkey deployment on another machine. Model weights and model-derived metadata/oracle packets are excluded and governed by their own licenses.
