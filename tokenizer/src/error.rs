//! Error type for the tokenizer crate.

use std::fmt;

/// All failures are reported explicitly. The crate does not panic on bad
/// metadata, bad token IDs or bad input text.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenizerError {
    /// GGUF container could not be read.
    Gguf(String),
    /// A required metadata key is absent.
    MissingKey(&'static str),
    /// A metadata key has the wrong GGUF type (the reference loader throws).
    WrongType { key: String, expected: &'static str, found: String },
    /// Metadata is present but not usable (duplicate tokens, short arrays, ...).
    InvalidMetadata(String),
    /// Tokenizer model or pre-tokenizer that this crate does not implement.
    Unsupported(String),
    /// Token ID outside `0..vocab_size`.
    InvalidTokenId { id: i64, vocab_size: usize },
    /// Input decodes to a code point above U+10FFFF; the reference throws.
    InvalidCodepoint { codepoint: u32 },
    /// `add_special` requested BOS/EOS but the vocabulary has none.
    MissingSpecialToken(&'static str),
    /// Bytes are not valid UTF-8 where text was requested.
    InvalidUtf8 { valid_up_to: usize, bytes: Vec<u8> },
    /// End of stream reached with an incomplete UTF-8 sequence buffered.
    IncompleteUtf8 { pending: Vec<u8> },
    /// Chat template could not be compiled or rendered.
    Template(String),
    /// Bad caller input (for example malformed chat settings).
    InvalidInput(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use TokenizerError::*;
        match self {
            Gguf(m) => write!(f, "{m}"),
            MissingKey(k) => write!(f, "missing metadata key {k}"),
            WrongType { key, expected, found } => {
                write!(f, "metadata key {key} has type {found}, expected {expected}")
            }
            InvalidMetadata(m) => write!(f, "invalid tokenizer metadata: {m}"),
            Unsupported(m) => write!(f, "unsupported: {m}"),
            InvalidTokenId { id, vocab_size } => {
                write!(f, "token id {id} is outside the vocabulary (size {vocab_size})")
            }
            InvalidCodepoint { codepoint } => {
                write!(f, "input contains code point 0x{codepoint:X}, above U+10FFFF")
            }
            MissingSpecialToken(t) => write!(f, "vocabulary has no {t} token"),
            InvalidUtf8 { valid_up_to, .. } => write!(f, "invalid UTF-8 after byte {valid_up_to}"),
            IncompleteUtf8 { pending } => {
                write!(f, "stream ended inside a UTF-8 sequence ({} bytes pending)", pending.len())
            }
            Template(m) => write!(f, "chat template: {m}"),
            InvalidInput(m) => write!(f, "invalid input: {m}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<crate::gguf::GgufError> for TokenizerError {
    fn from(e: crate::gguf::GgufError) -> Self {
        TokenizerError::Gguf(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, TokenizerError>;
