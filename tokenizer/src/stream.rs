//! Stateful streaming text decoder.
//!
//! Token boundaries are not UTF-8 boundaries: one token piece can hold part
//! of a character, and a character can span several tokens. The decoder
//! keeps an incomplete trailing UTF-8 sequence until more bytes arrive and
//! never emits U+FFFD for a sequence that can still become valid.
//!
//! Output contract: the concatenation of all text returned by `push_*` and
//! `finish` is byte-identical to the concatenation of the token pieces
//! (`Tokenizer::token_to_piece` with the chosen `render_special`), whenever
//! those bytes are valid UTF-8. Streaming does not apply the whole-sequence
//! `clean_spaces` pass of `Tokenizer::decode` (llama.cpp streaming output
//! also concatenates pieces).
//!
//! Invalid bytes (a sequence that can never become valid UTF-8):
//! - `InvalidUtf8Policy::Error` (default): `push_*` returns
//!   `TokenizerError::InvalidUtf8`; the text before the bad sequence is
//!   included in the error bytes and nothing is emitted. The buffer is kept,
//!   so call `reset()` before reuse.
//! - `InvalidUtf8Policy::Replace`: each maximal invalid subsequence becomes
//!   one U+FFFD (same rule as `String::from_utf8_lossy`).
//!
//! End of stream: `finish()` returns the remaining text. If an incomplete
//! sequence is still buffered, `Error` policy returns
//! `TokenizerError::IncompleteUtf8 { pending }` and `Replace` policy emits one
//! U+FFFD. `finish()` and `reset()` both leave the decoder empty and ready
//! for a new stream.

use crate::error::{Result, TokenizerError};
use crate::vocab::{TokenId, Tokenizer};

/// Handling of byte sequences that can never become valid UTF-8.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InvalidUtf8Policy {
    #[default]
    Error,
    Replace,
}

/// Streaming options.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StreamOptions {
    pub render_special: bool,
    pub invalid: InvalidUtf8Policy,
}

/// Incremental UTF-8 assembler, independent of any tokenizer.
#[derive(Clone, Debug, Default)]
pub struct Utf8Stream {
    pending: Vec<u8>,
    policy: InvalidUtf8Policy,
}

impl Utf8Stream {
    pub fn new(policy: InvalidUtf8Policy) -> Self {
        Utf8Stream { pending: Vec::new(), policy }
    }

    /// Bytes held back because they end inside a UTF-8 sequence.
    pub fn pending(&self) -> &[u8] {
        &self.pending
    }

    /// Add bytes; return the newly completed text.
    pub fn push(&mut self, bytes: &[u8]) -> Result<String> {
        self.pending.extend_from_slice(bytes);
        let mut out = String::new();
        let mut consumed = 0usize;
        loop {
            let rest = &self.pending[consumed..];
            match std::str::from_utf8(rest) {
                Ok(s) => {
                    out.push_str(s);
                    consumed = self.pending.len();
                    break;
                }
                Err(e) => {
                    let good = e.valid_up_to();
                    // SAFETY-free: `good` bytes are valid by definition.
                    out.push_str(std::str::from_utf8(&rest[..good]).unwrap_or_default());
                    match e.error_len() {
                        None => {
                            // incomplete sequence at the end: keep it
                            consumed += good;
                            break;
                        }
                        Some(bad) => match self.policy {
                            InvalidUtf8Policy::Error => {
                                return Err(TokenizerError::InvalidUtf8 {
                                    valid_up_to: consumed + good,
                                    bytes: self.pending.clone(),
                                });
                            }
                            InvalidUtf8Policy::Replace => {
                                out.push('\u{FFFD}');
                                consumed += good + bad;
                            }
                        },
                    }
                }
            }
        }
        self.pending.drain(..consumed);
        Ok(out)
    }

    /// End of stream. See the module documentation.
    pub fn finish(&mut self) -> Result<String> {
        if self.pending.is_empty() {
            return Ok(String::new());
        }
        let pending = std::mem::take(&mut self.pending);
        match self.policy {
            InvalidUtf8Policy::Error => match std::str::from_utf8(&pending) {
                Ok(s) => Ok(s.to_string()),
                Err(e) if e.error_len().is_some() => Err(TokenizerError::InvalidUtf8 {
                    valid_up_to: e.valid_up_to(),
                    bytes: pending,
                }),
                Err(_) => Err(TokenizerError::IncompleteUtf8 { pending }),
            },
            InvalidUtf8Policy::Replace => Ok("\u{FFFD}".to_string()),
        }
    }

    /// Drop buffered bytes.
    pub fn reset(&mut self) {
        self.pending.clear();
    }
}

/// Token-to-text streaming decoder bound to a tokenizer.
#[derive(Clone, Debug)]
pub struct StreamDecoder<'a> {
    tok: &'a Tokenizer,
    opts: StreamOptions,
    utf8: Utf8Stream,
    tokens_seen: usize,
}

impl<'a> StreamDecoder<'a> {
    pub fn new(tok: &'a Tokenizer, opts: StreamOptions) -> Self {
        StreamDecoder { tok, opts, utf8: Utf8Stream::new(opts.invalid), tokens_seen: 0 }
    }

    /// Add one generated token; return the newly completed text (may be "").
    /// An out-of-range ID is an error and leaves the state unchanged.
    pub fn push_token(&mut self, id: TokenId) -> Result<String> {
        let piece = self.tok.token_to_piece(id, self.opts.render_special)?;
        let s = self.utf8.push(piece)?;
        self.tokens_seen += 1;
        Ok(s)
    }

    /// Add several tokens; return the newly completed text.
    /// On a UTF-8 error, retain the entire unreturned batch in the error and
    /// pending buffer; no tokens in this batch are counted as accepted.
    pub fn push_tokens(&mut self, ids: &[TokenId]) -> Result<String> {
        let mut bytes = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(self.tok.token_to_piece(id, self.opts.render_special)?);
        }
        let out = self.utf8.push(&bytes)?;
        self.tokens_seen += ids.len();
        Ok(out)
    }

    /// Bytes held back because they end inside a UTF-8 sequence.
    pub fn pending(&self) -> &[u8] {
        self.utf8.pending()
    }

    /// Number of tokens accepted since creation or the last reset/finish.
    pub fn tokens_seen(&self) -> usize {
        self.tokens_seen
    }

    /// End of stream; the decoder is empty afterwards.
    pub fn finish(&mut self) -> Result<String> {
        self.tokens_seen = 0;
        self.utf8.finish()
    }

    /// Drop all state.
    pub fn reset(&mut self) {
        self.tokens_seen = 0;
        self.utf8.reset();
    }
}
