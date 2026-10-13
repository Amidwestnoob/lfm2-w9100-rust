//! # w9100_tokenizer
//!
//! Pure-Rust CPU tokenizer, byte decoder, streaming text decoder and chat
//! formatter for the tokenizer embedded in LFM2.5-8B-A1B (GGUF metadata:
//! `tokenizer.ggml.model = gpt2`, `tokenizer.ggml.pre = lfm2`).
//!
//! The behavior follows llama.cpp `llama-vocab.cpp` and `unicode.cpp` at the
//! pinned commit c061df19838ff60970faf54fd7e414953590125d (MIT, see
//! `LICENSE-llama.cpp`). No FFI, no subprocess, no GPU.
//!
//! ```no_run
//! use w9100_tokenizer::{Tokenizer, EncodeOptions, DecodeOptions, ChatTemplate, ChatRequest};
//! let tok = Tokenizer::from_gguf_file("LFM2.5-8B-A1B-Q6_K.gguf")?;
//! let ids = tok.encode(b"Hello", EncodeOptions { add_special: true, parse_special: false })?;
//! let text = tok.decode(&ids, DecodeOptions { remove_special: true, render_special: false })?;
//! let tmpl = ChatTemplate::from_tokenizer(&tok)?;
//! let mut req = ChatRequest::for_tokenizer(&tok, serde_json::json!([{"role": "user", "content": "Hi"}]))?;
//! req.add_generation_prompt = true;
//! let prompt_ids = tok.encode_chat(&tmpl, &req)?;
//! # Ok::<(), w9100_tokenizer::TokenizerError>(())
//! ```

pub mod chat;
pub mod error;
pub mod gguf;
pub mod stream;
pub mod unicode;
mod unicode_data;
pub mod vocab;

pub use chat::{ChatRequest, ChatTemplate};
pub use error::{Result, TokenizerError};
pub use stream::{InvalidUtf8Policy, StreamDecoder, StreamOptions, Utf8Stream};
pub use vocab::{attr, DecodeOptions, EncodeOptions, SpecialTokens, TokenId, Tokenizer};
