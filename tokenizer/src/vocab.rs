//! Tokenizer vocabulary, encoder and decoder.
//!
//! This is a Rust port of the BPE path of llama.cpp `src/llama-vocab.cpp`
//! (pinned commit c061df19838ff60970faf54fd7e414953590125d, MIT license) for
//! the `gpt2` tokenizer model with the LLAMA3 pre-tokenizer family, which is
//! the family that `tokenizer.ggml.pre = "lfm2"` maps to. Comments name the
//! reference function that each part follows.

use crate::error::{Result, TokenizerError};
use crate::gguf::{self, MetaArray, MetaValue, Metadata};
use crate::unicode;
use std::collections::{BTreeSet, BinaryHeap, HashMap};
use std::path::Path;

/// Token ID (index into the vocabulary).
pub type TokenId = u32;

/// Token attribute bits (`llama_token_attr`).
pub mod attr {
    pub const UNDEFINED: u32 = 0;
    pub const UNKNOWN: u32 = 1 << 0;
    pub const UNUSED: u32 = 1 << 1;
    pub const NORMAL: u32 = 1 << 2;
    pub const CONTROL: u32 = 1 << 3;
    pub const USER_DEFINED: u32 = 1 << 4;
    pub const BYTE: u32 = 1 << 5;
    pub const NORMALIZED: u32 = 1 << 6;
    pub const LSTRIP: u32 = 1 << 7;
    pub const RSTRIP: u32 = 1 << 8;
    pub const SINGLE_WORD: u32 = 1 << 9;
}

/// Pre-tokenizer names that the reference maps to `LLAMA_VOCAB_PRE_TYPE_LLAMA3`
/// with `ignore_merges = true` and `add_bos = true` (llama-vocab.cpp:2173-2185).
pub const LLAMA3_FAMILY_PRE: &[&str] = &[
    "llama3", "llama-v3", "llama-bpe", "falcon3", "falcon-h1", "pixtral", "midm-2.0", "lfm2", "jina-v5-nano",
];

/// Encode options. The two flags are independent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncodeOptions {
    /// Add BOS/EOS as configured by the vocabulary (`add_special`).
    pub add_special: bool,
    /// Match control/unknown token text in the input (`parse_special`).
    /// User-defined tokens are always matched, like the reference.
    pub parse_special: bool,
}

/// Decode options. The two flags are independent.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecodeOptions {
    /// Drop a leading BOS (when `add_bos`) and trailing EOS (when `add_eos`).
    pub remove_special: bool,
    /// Render control/unknown tokens as their text; otherwise they are empty.
    pub render_special: bool,
}

/// Special token IDs after metadata overrides.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpecialTokens {
    pub bos: Option<TokenId>,
    pub eos: Option<TokenId>,
    pub eot: Option<TokenId>,
    pub eom: Option<TokenId>,
    pub unk: Option<TokenId>,
    pub sep: Option<TokenId>,
    pub pad: Option<TokenId>,
    pub mask: Option<TokenId>,
    pub fim_pre: Option<TokenId>,
    pub fim_suf: Option<TokenId>,
    pub fim_mid: Option<TokenId>,
    pub fim_pad: Option<TokenId>,
    pub fim_rep: Option<TokenId>,
    pub fim_sep: Option<TokenId>,
    /// Newline token: first ID of `encode("\n")` (pad if empty).
    pub linefeed: Option<TokenId>,
}

#[derive(Clone, Debug)]
struct TokenData {
    text: Vec<u8>,
    attr: u32,
}

/// Facts about the load that differ from or cannot be fixed by the reference.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LoadNotes {
    /// Text-based special-token auto-detection had more than one candidate
    /// (the reference picks by hash-map order; this crate picks the first
    /// candidate in the reference's name-list order).
    pub ambiguous_autodetect: Vec<String>,
    /// Two equal-length special tokens can overlap. The reference uses an
    /// unstable sort for equal lengths; this crate sorts by ID. Empty for LFM2.
    pub ambiguous_special_order: Vec<(TokenId, TokenId)>,
    /// Special-token IDs in the metadata that are outside the vocabulary.
    /// Like the reference ("bad special token ... using default id"), the
    /// default ID is kept.
    pub bad_special_ids: Vec<String>,
}

/// An immutable, thread-safe tokenizer (`Send + Sync`).
#[derive(Clone, Debug)]
pub struct Tokenizer {
    model: String,
    pre: String,
    tokens: Vec<TokenData>,
    token_to_id: HashMap<Vec<u8>, TokenId>,
    bpe_ranks: HashMap<Vec<u8>, u32>,
    special: SpecialTokens,
    eog: BTreeSet<TokenId>,
    add_bos: bool,
    add_eos: bool,
    add_sep: bool,
    add_space_prefix: bool,
    clean_spaces: bool,
    ignore_merges: bool,
    special_cache: Vec<TokenId>,
    piece_cache: Vec<Vec<u8>>,
    chat_template: Option<String>,
    notes: LoadNotes,
}

// --- metadata accessors with the reference's strict typing -----------------

fn get_str<'a>(md: &'a Metadata, key: &'static str) -> Result<Option<&'a [u8]>> {
    match md.get(key) {
        None => Ok(None),
        Some(MetaValue::String(s)) => Ok(Some(s)),
        Some(v) => Err(TokenizerError::WrongType { key: key.into(), expected: "String", found: v.type_name() }),
    }
}

fn get_bool(md: &Metadata, key: &'static str) -> Result<Option<bool>> {
    match md.get(key) {
        None => Ok(None),
        Some(MetaValue::Bool(b)) => Ok(Some(*b)),
        Some(v) => Err(TokenizerError::WrongType { key: key.into(), expected: "Bool", found: v.type_name() }),
    }
}

fn get_u32(md: &Metadata, key: &'static str) -> Result<Option<u32>> {
    match md.get(key) {
        None => Ok(None),
        Some(MetaValue::U32(x)) => Ok(Some(*x)),
        Some(v) => Err(TokenizerError::WrongType { key: key.into(), expected: "U32", found: v.type_name() }),
    }
}

fn get_arr<'a>(md: &'a Metadata, key: &'static str) -> Result<Option<&'a MetaArray>> {
    match md.get(key) {
        None => Ok(None),
        Some(MetaValue::Array(a)) => Ok(Some(a)),
        Some(v) => Err(TokenizerError::WrongType { key: key.into(), expected: "Array", found: v.type_name() }),
    }
}

fn utf8_name(b: &[u8]) -> String {
    String::from_utf8_lossy(b).into_owned()
}

/// Key for the merge-rank map: length-prefixed first part, then second part.
fn rank_key(first: &[u8], second: &[u8]) -> Vec<u8> {
    let mut k = Vec::with_capacity(4 + first.len() + second.len());
    k.extend_from_slice(&(first.len() as u32).to_le_bytes());
    k.extend_from_slice(first);
    k.extend_from_slice(second);
    k
}

/// C `strtol(s, NULL, 16)` on the (at most 2 byte) slice used by
/// `token_to_byte` (`text.substr(3, 2)`).
fn strtol16(s: &[u8]) -> i64 {
    let mut i = 0;
    while i < s.len() && matches!(s[i], b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r') {
        i += 1;
    }
    let mut neg = false;
    if i < s.len() && (s[i] == b'+' || s[i] == b'-') {
        neg = s[i] == b'-';
        i += 1;
    }
    if s.len() >= i + 3 && s[i] == b'0' && (s[i + 1] | 0x20) == b'x' && s[i + 2].is_ascii_hexdigit() {
        i += 2;
    }
    let mut v: i64 = 0;
    while i < s.len() && s[i].is_ascii_hexdigit() {
        v = v * 16 + (s[i] as char).to_digit(16).unwrap_or(0) as i64;
        i += 1;
    }
    if neg {
        -v
    } else {
        v
    }
}

/// C `isspace` in the "C" locale.
#[inline]
fn c_isspace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

#[derive(Clone, Copy)]
enum Fragment {
    Raw { offset: usize, len: usize },
    Token(TokenId),
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Bigram {
    rank: u32,
    left: usize,
    right: usize,
    size: usize,
}

impl Ord for Bigram {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // BinaryHeap is a max-heap: the "greatest" element is the lowest rank,
        // then the lowest left index (llm_bigram_bpe::comparator).
        other.rank.cmp(&self.rank).then(other.left.cmp(&self.left)).then(other.right.cmp(&self.right))
    }
}
impl PartialOrd for Bigram {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy)]
struct Symbol {
    start: usize,
    n: usize,
    prev: isize,
    next: isize,
}

impl Tokenizer {
    /// Load from a GGUF file (metadata only; tensors are never read).
    pub fn from_gguf_file(path: impl AsRef<Path>) -> Result<Self> {
        let md = gguf::read_metadata_file(path)?;
        Self::from_metadata(&md)
    }

    /// Load from GGUF bytes held in memory.
    pub fn from_gguf_bytes(bytes: &[u8]) -> Result<Self> {
        let md = gguf::read_metadata(bytes, bytes.len() as u64)?;
        Self::from_metadata(&md)
    }

    /// Load from an already parsed metadata map. Follows
    /// `llama_vocab::impl::load` for the `gpt2` model type.
    pub fn from_metadata(md: &Metadata) -> Result<Self> {
        // --- vocab type ----------------------------------------------------
        let model = utf8_name(get_str(md, "tokenizer.ggml.model")?.ok_or(TokenizerError::MissingKey("tokenizer.ggml.model"))?);
        let pre = get_str(md, "tokenizer.ggml.pre")?.map(utf8_name).unwrap_or_default();
        if let Some(MetaValue::U32(_)) | None = md.get("tokenizer.ggml.token_type_count") {
        } else {
            return Err(TokenizerError::WrongType {
                key: "tokenizer.ggml.token_type_count".into(),
                expected: "U32",
                found: md["tokenizer.ggml.token_type_count"].type_name(),
            });
        }
        if model != "gpt2" {
            return Err(TokenizerError::Unsupported(format!("tokenizer model '{model}' (only 'gpt2' BPE is implemented)")));
        }

        // merges (bpe_ranks.emplace keeps the first rank of a duplicate pair)
        let merges = match get_arr(md, "tokenizer.ggml.merges")? {
            None => return Err(TokenizerError::MissingKey("tokenizer.ggml.merges")),
            Some(MetaArray::String(v)) => v,
            Some(a) => {
                return Err(TokenizerError::WrongType {
                    key: "tokenizer.ggml.merges".into(),
                    expected: "Array<String>",
                    found: format!("Array<{}>", a.type_name()),
                })
            }
        };
        let mut bpe_ranks: HashMap<Vec<u8>, u32> = HashMap::with_capacity(merges.len());
        for (i, word) in merges.iter().enumerate() {
            let (first, second): (&[u8], &[u8]) = match word.iter().skip(1).position(|&b| b == b' ') {
                Some(p) => {
                    let pos = p + 1;
                    (&word[..pos], &word[pos + 1..])
                }
                None => (&[], &[]),
            };
            bpe_ranks.entry(rank_key(first, second)).or_insert(i as u32);
        }

        // pre-tokenizer (only the LLAMA3 family is implemented)
        let mut add_bos;
        let add_eos;
        let mut add_sep = false;
        if pre.is_empty() {
            return Err(TokenizerError::Unsupported(
                "missing tokenizer.ggml.pre (reference falls back to the 'default' pre-tokenizer, which is not implemented)".into(),
            ));
        } else if LLAMA3_FAMILY_PRE.contains(&pre.as_str()) {
            add_bos = true;
            add_eos = false;
        } else {
            return Err(TokenizerError::Unsupported(format!(
                "pre-tokenizer '{pre}' (implemented: LLAMA3 family {:?})",
                LLAMA3_FAMILY_PRE
            )));
        }
        let mut add_eos = add_eos;
        let ignore_merges = true;
        let clean_spaces = true;
        let mut add_space_prefix = false;
        if let Some(b) = get_bool(md, "tokenizer.ggml.add_space_prefix")? {
            add_space_prefix = b;
        }
        let _ = get_bool(md, "tokenizer.ggml.remove_extra_whitespaces")?;

        // --- token list ----------------------------------------------------
        let token_list = match get_arr(md, "tokenizer.ggml.tokens")? {
            None => return Err(TokenizerError::MissingKey("tokenizer.ggml.tokens")),
            Some(MetaArray::String(v)) => v,
            Some(a) => {
                return Err(TokenizerError::WrongType {
                    key: "tokenizer.ggml.tokens".into(),
                    expected: "Array<String>",
                    found: format!("Array<{}>", a.type_name()),
                })
            }
        };
        let n_tokens = token_list.len();
        if n_tokens == 0 {
            return Err(TokenizerError::InvalidMetadata("tokenizer.ggml.tokens is empty".into()));
        }
        if n_tokens > i32::MAX as usize {
            return Err(TokenizerError::InvalidMetadata("vocabulary too large".into()));
        }
        match get_arr(md, "tokenizer.ggml.scores")? {
            None => {}
            Some(MetaArray::I32(v)) if v.len() >= n_tokens => {}
            Some(MetaArray::F32(v)) if v.len() >= n_tokens => {}
            Some(a @ (MetaArray::I32(_) | MetaArray::F32(_))) => {
                return Err(TokenizerError::InvalidMetadata(format!(
                    "tokenizer.ggml.scores has {} entries, fewer than {} tokens",
                    a.len(),
                    n_tokens
                )))
            }
            Some(a) => {
                return Err(TokenizerError::WrongType {
                    key: "tokenizer.ggml.scores".into(),
                    expected: "Array<I32|F32>",
                    found: format!("Array<{}>", a.type_name()),
                })
            }
        }
        let toktypes = match get_arr(md, "tokenizer.ggml.token_type")? {
            None => None,
            Some(MetaArray::I32(v)) => {
                if v.len() < n_tokens {
                    return Err(TokenizerError::InvalidMetadata(format!(
                        "tokenizer.ggml.token_type has {} entries, fewer than {} tokens",
                        v.len(),
                        n_tokens
                    )));
                }
                Some(v)
            }
            Some(a) => {
                return Err(TokenizerError::WrongType {
                    key: "tokenizer.ggml.token_type".into(),
                    expected: "Array<I32>",
                    found: format!("Array<{}>", a.type_name()),
                })
            }
        };

        let mut tokens = Vec::with_capacity(n_tokens);
        let mut token_to_id: HashMap<Vec<u8>, TokenId> = HashMap::with_capacity(n_tokens);
        for (i, w) in token_list.iter().enumerate() {
            let text = if w.is_empty() { format!("[EMPTY_{i}]").into_bytes() } else { w.clone() };
            token_to_id.insert(text.clone(), i as TokenId);
            let attr = match toktypes {
                None => attr::NORMAL,
                Some(tt) => match tt[i] {
                    2 => attr::UNKNOWN,
                    5 => attr::UNUSED,
                    1 => attr::NORMAL,
                    3 => attr::CONTROL,
                    4 => attr::USER_DEFINED,
                    6 => attr::BYTE,
                    _ => attr::UNDEFINED,
                },
            };
            tokens.push(TokenData { text, attr });
        }
        if token_to_id.len() != tokens.len() {
            return Err(TokenizerError::InvalidMetadata(format!(
                "duplicate token text: {} unique of {} tokens (the reference aborts)",
                token_to_id.len(),
                tokens.len()
            )));
        }

        // --- special token IDs ----------------------------------------------
        let n = n_tokens as i64;
        let mut ids: [i64; 14] = [11, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
        // bos eos eot eom unk sep pad mask fim_pre fim_suf fim_mid fim_pad fim_rep fim_sep
        const KEYS: [(&str, usize); 17] = [
            ("tokenizer.ggml.bos_token_id", 0),
            ("tokenizer.ggml.eos_token_id", 1),
            ("tokenizer.ggml.eot_token_id", 2),
            ("tokenizer.ggml.eom_token_id", 3),
            ("tokenizer.ggml.unknown_token_id", 4),
            ("tokenizer.ggml.seperator_token_id", 5),
            ("tokenizer.ggml.padding_token_id", 6),
            ("tokenizer.ggml.mask_token_id", 7),
            ("tokenizer.ggml.fim_pre_token_id", 8),
            ("tokenizer.ggml.fim_suf_token_id", 9),
            ("tokenizer.ggml.fim_mid_token_id", 10),
            ("tokenizer.ggml.fim_pad_token_id", 11),
            ("tokenizer.ggml.fim_rep_token_id", 12),
            ("tokenizer.ggml.fim_sep_token_id", 13),
            ("tokenizer.ggml.prefix_token_id", 8),
            ("tokenizer.ggml.suffix_token_id", 9),
            ("tokenizer.ggml.middle_token_id", 10),
        ];
        let mut bad_special: Vec<String> = Vec::new();
        for (key, slot) in KEYS {
            if ids[slot] >= 0 && ids[slot] >= n {
                ids[slot] = -1;
            }
            if let Some(new_id) = get_u32(md, key)? {
                if (new_id as i64) < n {
                    ids[slot] = new_id as i64;
                } else {
                    bad_special.push(format!("{key} = {new_id} is outside the vocabulary; kept default {}", ids[slot]));
                }
            }
        }
        if let Some(b) = get_bool(md, "tokenizer.ggml.add_bos_token")? {
            add_bos = b;
        }
        if let Some(b) = get_bool(md, "tokenizer.ggml.add_eos_token")? {
            add_eos = b;
        }
        if let Some(b) = get_bool(md, "tokenizer.ggml.add_sep_token")? {
            add_sep = b;
        }
        let _ = get_bool(md, "tokenizer.ggml.normalizer.lowercase")?;
        let _ = get_bool(md, "tokenizer.ggml.normalizer.strip_accents")?;
        match get_arr(md, "tokenizer.ggml.suppress_tokens")? {
            None | Some(MetaArray::I32(_)) => {}
            Some(a) => {
                return Err(TokenizerError::WrongType {
                    key: "tokenizer.ggml.suppress_tokens".into(),
                    expected: "Array<I32>",
                    found: format!("Array<{}>", a.type_name()),
                })
            }
        }

        // --- text-based auto-detection ----------------------------------------
        let mut notes = LoadNotes { bad_special_ids: bad_special, ..LoadNotes::default() };
        let lookup = |name: &str| token_to_id.get(name.as_bytes()).copied();
        const AUTODETECT: [(usize, &str, &[&str]); 8] = [
            (2, "EOT", &["<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "<|end_of_text|>", "<EOT>", "_<EOT>", "[EOT]", "<｜end▁of▁sentence｜>", "<end_of_utterance>"]),
            (3, "EOM", &["<|eom_id|>"]),
            (8, "FIM_PRE", &["<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>", "<｜fim▁begin｜>", "<PRE>", "▁<PRE>", "<|code_prefix|>", "<|prefix|>"]),
            (9, "FIM_SUF", &["<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>", "<｜fim▁hole｜>", "<SUF>", "▁<SUF>", "<|code_suffix|>", "<|suffix|>"]),
            (10, "FIM_MID", &["<|fim_middle|>", "<fim-middle>", "<fim_middle>", "<｜fim▁end｜>", "<MID>", "▁<MID>", "<|code_middle|>", "<|middle|>"]),
            (11, "FIM_PAD", &["<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]"]),
            (12, "FIM_REP", &["<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>"]),
            (13, "FIM_SEP", &["<|file_sep|>"]),
        ];
        for (slot, label, names) in AUTODETECT {
            if ids[slot] != -1 {
                continue;
            }
            // Candidates in name-list order. The reference takes the first match in
            // hash-map iteration order; for LFM2 its EOT is 124900 (<|im_end|>),
            // which this list-order rule also selects (probe evidence).
            let cands: Vec<TokenId> = names.iter().filter_map(|nm| lookup(nm)).collect();
            if let Some(&first) = cands.first() {
                if cands.len() > 1 {
                    notes.ambiguous_autodetect.push(format!("{label}: candidates {cands:?}, chose {first}"));
                }
                ids[slot] = first as i64;
                tokens[first as usize].attr |= attr::CONTROL;
            }
        }
        // unused: control tokens whose text contains "unused" (strstr stops at NUL)
        for t in tokens.iter_mut() {
            if t.attr & attr::CONTROL == 0 {
                continue;
            }
            if t.attr & attr::UNUSED == 0 {
                let upto = t.text.iter().position(|&b| b == 0).unwrap_or(t.text.len());
                if t.text[..upto].windows(6).any(|w| w == b"unused") {
                    t.attr |= attr::UNUSED;
                }
            }
        }
        // end-of-generation set
        let mut eog: BTreeSet<TokenId> = BTreeSet::new();
        for slot in [11, 12, 13] {
            if ids[slot] >= 0 {
                eog.insert(ids[slot] as TokenId);
            }
        }
        const EOG_NAMES: &[&str] = &[
            "<|eot_id|>", "<|im_end|>", "<|end|>", "<|return|>", "<|call|>", "<|flush|>", "<|calls|>", "<end_of_turn>",
            "<|endoftext|>", "</s>", "<|eom_id|>", "<EOT>", "_<EOT>", "[EOT]", "[EOS]", "<|end_of_text|>",
            "<end_of_utterance>", "<eos>", "<turn|>", "<|tool_response>", "<｜end▁of▁sentence｜>", "[e~[",
        ];
        for nm in EOG_NAMES {
            if let Some(id) = lookup(nm) {
                eog.insert(id);
                tokens[id as usize].attr |= attr::CONTROL;
            }
        }
        for nm in ["<|channel|>", "<|message|>", "<|start|>", "<|constrain|>"] {
            if let Some(id) = lookup(nm) {
                tokens[id as usize].attr = attr::USER_DEFINED;
            }
        }
        for slot in [1, 2, 3] {
            if ids[slot] >= 0 {
                eog.insert(ids[slot] as TokenId);
            }
        }
        {
            let (mut has_return, mut has_call, mut has_end, mut has_flush) = (false, false, false, false);
            let mut end_id = None;
            for &tid in &eog {
                let text = tokens[tid as usize].text.as_slice();
                match text {
                    b"<|return|>" => has_return = true,
                    b"<|call|>" | b"<|calls|>" => has_call = true,
                    b"<|flush|>" => has_flush = true,
                    b"<|end|>" => {
                        has_end = true;
                        end_id = Some(tid);
                    }
                    _ => {}
                }
            }
            if (has_return && has_call && has_end) || (has_call && has_flush && has_end) {
                if let Some(e) = end_id {
                    eog.remove(&e);
                    tokens[e as usize].attr = attr::USER_DEFINED;
                }
            }
        }
        {
            let (mut marker, mut s_id) = (false, None);
            for &tid in &eog {
                let text = tokens[tid as usize].text.as_slice();
                if text == b"<|tool_response>" || text == b"<|plamo:eos|>" {
                    marker = true;
                } else if text == b"</s>" {
                    s_id = Some(tid);
                }
            }
            if let (true, Some(s)) = (marker, s_id) {
                eog.remove(&s);
                tokens[s as usize].attr = attr::NORMAL;
            }
        }

        let opt = |v: i64| if v >= 0 { Some(v as TokenId) } else { None };
        let mut special = SpecialTokens {
            bos: opt(ids[0]),
            eos: opt(ids[1]),
            eot: opt(ids[2]),
            eom: opt(ids[3]),
            unk: opt(ids[4]),
            sep: opt(ids[5]),
            pad: opt(ids[6]),
            mask: opt(ids[7]),
            fim_pre: opt(ids[8]),
            fim_suf: opt(ids[9]),
            fim_mid: opt(ids[10]),
            fim_pad: opt(ids[11]),
            fim_rep: opt(ids[12]),
            fim_sep: opt(ids[13]),
            linefeed: None,
        };

        // --- special token cache (longest text first) -----------------------
        let mut special_cache: Vec<TokenId> = (0..n_tokens as TokenId)
            .filter(|&id| tokens[id as usize].attr & (attr::CONTROL | attr::USER_DEFINED | attr::UNKNOWN) != 0)
            .collect();
        special_cache.sort_by(|&a, &b| tokens[b as usize].text.len().cmp(&tokens[a as usize].text.len()).then(a.cmp(&b)));
        notes.ambiguous_special_order = find_equal_length_overlaps(&tokens, &special_cache);

        let chat_template = match get_str(md, "tokenizer.chat_template")? {
            None => None,
            Some(b) => Some(String::from_utf8(b.to_vec()).map_err(|_| {
                TokenizerError::InvalidMetadata("tokenizer.chat_template is not UTF-8".into())
            })?),
        };

        let mut tok = Tokenizer {
            model,
            pre,
            tokens,
            token_to_id,
            bpe_ranks,
            special: special.clone(),
            eog,
            add_bos,
            add_eos,
            add_sep,
            add_space_prefix,
            clean_spaces,
            ignore_merges,
            special_cache: Vec::new(),
            piece_cache: Vec::new(),
            chat_template,
            notes,
        };

        // linefeed: computed before the special cache exists, like the reference
        let lf = tok.encode(b"\n", EncodeOptions { add_special: false, parse_special: false })?;
        special.linefeed = lf.first().copied().or(special.pad);
        tok.special = special;
        tok.special_cache = special_cache;

        // --- piece cache (token_to_piece with special = true) ---------------
        let mut cache = Vec::with_capacity(n_tokens);
        for id in 0..n_tokens {
            cache.push(tok.piece_uncached(id)?);
        }
        tok.piece_cache = cache;

        // --- per-model attribute fix-ups (after the caches, like the reference)
        tok.apply_model_attr_fixups(md)?;
        Ok(tok)
    }

    fn apply_model_attr_fixups(&mut self, md: &Metadata) -> Result<()> {
        let name = get_str(md, "general.name")?.map(utf8_name).unwrap_or_default().to_ascii_lowercase();
        let pre = get_str(md, "tokenizer.ggml.pre")?.map(utf8_name).unwrap_or_default();
        let arch = get_str(md, "general.architecture")?.map(utf8_name).unwrap_or_default();
        let any = |s: &str, subs: &[&str]| subs.iter().any(|x| s.contains(x));
        let set = |tok: &mut Tokenizer, text: &str, a: u32, v: bool| -> Result<()> {
            let id = *tok.token_to_id.get(text.as_bytes()).ok_or_else(|| {
                TokenizerError::InvalidMetadata(format!("token '{text}' required by the model attribute rules is missing"))
            })?;
            let t = &mut tok.tokens[id as usize];
            t.attr = if v { t.attr | a } else { t.attr & !a };
            Ok(())
        };
        if any(&pre, &["jina-v2-de", "jina-v2-es", "jina-v2-code"]) || any(&arch, &["nomic-bert-moe", "jina-bert-v3"]) {
            if self.token_to_id.contains_key(b"<mask>".as_slice()) {
                set(self, "<mask>", attr::LSTRIP, true)?;
            }
        } else if any(&name, &["phi-3", "phi3"]) {
            for id in self.special_cache.clone() {
                self.tokens[id as usize].attr |= attr::RSTRIP;
            }
            set(self, "</s>", attr::RSTRIP, true)?;
            for t in ["<unk>", "<s>", "<|endoftext|>"] {
                set(self, t, attr::RSTRIP, false)?;
            }
        } else if any(&name, &["modern-bert"]) && self.token_to_id.contains_key(b"[MASK]".as_slice()) {
            set(self, "[MASK]", attr::LSTRIP, true)?;
        }
        Ok(())
    }

    // --- accessors ------------------------------------------------------------

    /// Number of tokens in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }
    /// `tokenizer.ggml.model`.
    pub fn model(&self) -> &str {
        &self.model
    }
    /// `tokenizer.ggml.pre`.
    pub fn pre(&self) -> &str {
        &self.pre
    }
    /// Special token IDs.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }
    pub fn bos(&self) -> Option<TokenId> {
        self.special.bos
    }
    pub fn eos(&self) -> Option<TokenId> {
        self.special.eos
    }
    pub fn pad(&self) -> Option<TokenId> {
        self.special.pad
    }
    /// Effective `add_bos` (family default unless metadata overrides it).
    pub fn add_bos(&self) -> bool {
        self.add_bos
    }
    pub fn add_eos(&self) -> bool {
        self.add_eos
    }
    pub fn add_sep(&self) -> bool {
        self.add_sep
    }
    pub fn ignore_merges(&self) -> bool {
        self.ignore_merges
    }
    pub fn clean_spaces(&self) -> bool {
        self.clean_spaces
    }
    /// End-of-generation token set.
    pub fn eog_tokens(&self) -> &BTreeSet<TokenId> {
        &self.eog
    }
    pub fn is_eog(&self, id: TokenId) -> bool {
        self.eog.contains(&id)
    }
    /// Raw vocabulary text of a token (GPT-2 byte-level form for normal tokens).
    pub fn token_text(&self, id: TokenId) -> Result<&[u8]> {
        Ok(&self.token(id)?.text)
    }
    /// Token attribute bits (see [`attr`]).
    pub fn token_attr(&self, id: TokenId) -> Result<u32> {
        Ok(self.token(id)?.attr)
    }
    /// Look up a token by its exact vocabulary text.
    pub fn text_to_token(&self, text: &[u8]) -> Option<TokenId> {
        self.token_to_id.get(text).copied()
    }
    /// The chat template stored in the metadata, if any.
    pub fn chat_template(&self) -> Option<&str> {
        self.chat_template.as_deref()
    }
    /// Load notes (ambiguities resolved deterministically).
    pub fn load_notes(&self) -> &LoadNotes {
        &self.notes
    }

    fn token(&self, id: TokenId) -> Result<&TokenData> {
        self.tokens
            .get(id as usize)
            .ok_or(TokenizerError::InvalidTokenId { id: id as i64, vocab_size: self.tokens.len() })
    }

    // --- pieces ---------------------------------------------------------------

    /// `token_to_piece` for the cache: special rendering on, no lstrip.
    fn piece_uncached(&self, id: usize) -> Result<Vec<u8>> {
        let t = &self.tokens[id];
        let a = t.attr;
        if a & (attr::UNKNOWN | attr::CONTROL | attr::USER_DEFINED) != 0 {
            return Ok(t.text.clone());
        }
        if a & attr::NORMAL != 0 {
            return decode_text(&t.text);
        }
        if a & attr::BYTE != 0 {
            if t.text.len() < 3 {
                return Err(TokenizerError::InvalidMetadata(format!(
                    "byte token {id} text is shorter than 3 bytes (the reference throws)"
                )));
            }
            let end = (3 + 2).min(t.text.len());
            return Ok(vec![strtol16(&t.text[3..end]) as u8]);
        }
        Ok(Vec::new())
    }

    /// Bytes for one token (`llama_token_to_piece` with `lstrip = 0`).
    /// With `render_special = false`, control and unknown tokens are empty.
    /// The bytes may be an incomplete part of a UTF-8 character.
    pub fn token_to_piece(&self, id: TokenId, render_special: bool) -> Result<&[u8]> {
        let t = self.token(id)?;
        if !render_special && t.attr & (attr::UNKNOWN | attr::CONTROL) != 0 {
            return Ok(&[]);
        }
        Ok(&self.piece_cache[id as usize])
    }

    // --- encode ---------------------------------------------------------------

    /// Encode raw bytes. Invalid UTF-8 is handled like the reference (each bad
    /// byte becomes U+FFFD before byte-level encoding).
    pub fn encode(&self, text: &[u8], opts: EncodeOptions) -> Result<Vec<TokenId>> {
        let mut frags: Vec<Fragment> = Vec::new();
        if !text.is_empty() {
            frags.push(Fragment::Raw { offset: 0, len: text.len() });
            frags = self.partition(text, frags, opts.parse_special);
        }
        let mut out = Vec::with_capacity(text.len() / 3 + 2);
        if opts.add_special && self.add_bos {
            out.push(self.special.bos.ok_or(TokenizerError::MissingSpecialToken("BOS"))?);
        }
        for f in frags {
            match f {
                Fragment::Raw { offset, len } => self.bpe(&text[offset..offset + len], &mut out)?,
                Fragment::Token(id) => out.push(id),
            }
        }
        if opts.add_special && self.add_eos {
            out.push(self.special.eos.ok_or(TokenizerError::MissingSpecialToken("EOS"))?);
        }
        Ok(out)
    }

    /// Encode a Rust string.
    pub fn encode_str(&self, text: &str, add_special: bool, parse_special: bool) -> Result<Vec<TokenId>> {
        self.encode(text.as_bytes(), EncodeOptions { add_special, parse_special })
    }

    /// `tokenizer_st_partition`.
    fn partition(&self, text: &[u8], mut buffer: Vec<Fragment>, parse_special: bool) -> Vec<Fragment> {
        for &special_id in &self.special_cache {
            let data = &self.tokens[special_id as usize];
            if !parse_special && data.attr & (attr::CONTROL | attr::UNKNOWN) != 0 {
                continue;
            }
            let needle = data.text.as_slice();
            let mut next: Vec<Fragment> = Vec::with_capacity(buffer.len() + 2);
            for frag in buffer {
                let (mut base_off, mut base_len) = match frag {
                    Fragment::Raw { offset, len } => (offset, len),
                    tok => {
                        next.push(tok);
                        continue;
                    }
                };
                loop {
                    let hay = &text[base_off..base_off + base_len];
                    let found = find_bytes(hay, needle);
                    let Some(rel) = found else {
                        if base_len > 0 {
                            next.push(Fragment::Raw { offset: base_off, len: base_len });
                        }
                        break;
                    };
                    let m = base_off + rel;
                    if m > base_off {
                        let left_off = base_off;
                        let mut left_len = m - base_off;
                        if data.attr & attr::LSTRIP != 0 {
                            while left_len > 0 && c_isspace(text[left_off + left_len - 1]) {
                                left_len -= 1;
                            }
                        }
                        if left_len > 0 {
                            next.push(Fragment::Raw { offset: left_off, len: left_len });
                        }
                    }
                    next.push(Fragment::Token(special_id));
                    if m + needle.len() < base_off + base_len {
                        let mut r_off = m + needle.len();
                        let mut r_len = base_len - ((m - base_off) + needle.len());
                        if data.attr & attr::RSTRIP != 0 {
                            while r_len > 0 && c_isspace(text[r_off]) {
                                r_off += 1;
                                r_len -= 1;
                            }
                        }
                        base_off = r_off;
                        base_len = r_len;
                        if r_len == 0 {
                            break;
                        }
                    } else {
                        break;
                    }
                }
            }
            buffer = next;
        }
        buffer
    }

    /// `llm_tokenizer_bpe_session::tokenize` for one raw fragment.
    fn bpe(&self, text: &[u8], out: &mut Vec<TokenId>) -> Result<()> {
        let cpts = unicode::cpts_from_utf8(text);
        let lens = unicode::split_llama3(&cpts);
        let byte_map = unicode::byte_to_utf8_table();
        let mut start = 0usize;
        let mut raw = Vec::new();
        let mut word = Vec::new();
        let mut symbols: Vec<Symbol> = Vec::new();
        let mut heap: BinaryHeap<Bigram> = BinaryHeap::new();
        for len in lens {
            raw.clear();
            for &c in &cpts[start..start + len] {
                unicode::cpt_to_utf8(c, &mut raw).ok_or(TokenizerError::InvalidCodepoint { codepoint: c })?;
            }
            start += len;
            word.clear();
            for &b in &raw {
                word.extend_from_slice(&byte_map[b as usize]);
            }

            symbols.clear();
            heap.clear();
            if self.ignore_merges && self.token_to_id.contains_key(word.as_slice()) {
                symbols.push(Symbol { start: 0, n: word.len(), prev: -1, next: -1 });
            } else {
                let mut offset = 0usize;
                let mut index = 0isize;
                while offset < word.len() {
                    let char_len = (word.len() - offset).min(unicode::len_utf8(word[offset]));
                    let s = offset;
                    offset += char_len;
                    symbols.push(Symbol {
                        start: s,
                        n: char_len,
                        prev: index - 1,
                        next: if offset == word.len() { -1 } else { index + 1 },
                    });
                    index += 1;
                }
                for i in 1..symbols.len() {
                    self.add_bigram(&word, &symbols, i as isize - 1, i as isize, &mut heap);
                }
                while let Some(bg) = heap.pop() {
                    let (l, r) = (symbols[bg.left], symbols[bg.right]);
                    if l.n == 0 || r.n == 0 {
                        continue;
                    }
                    // outdated if the concatenated text changed (same start, contiguous)
                    if l.n + r.n != bg.size {
                        continue;
                    }
                    symbols[bg.left].n += r.n;
                    symbols[bg.right].n = 0;
                    symbols[bg.left].next = r.next;
                    if r.next >= 0 {
                        symbols[r.next as usize].prev = bg.left as isize;
                    }
                    let lp = symbols[bg.left].prev;
                    let ln = symbols[bg.left].next;
                    self.add_bigram(&word, &symbols, lp, bg.left as isize, &mut heap);
                    self.add_bigram(&word, &symbols, bg.left as isize, ln, &mut heap);
                }
            }
            for sym in symbols.iter().filter(|s| s.n > 0) {
                let piece = &word[sym.start..sym.start + sym.n];
                match self.token_to_id.get(piece) {
                    Some(&id) => out.push(id),
                    None => {
                        for &b in piece {
                            if let Some(&id) = self.token_to_id.get([b].as_slice()) {
                                out.push(id);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn add_bigram(&self, word: &[u8], symbols: &[Symbol], left: isize, right: isize, heap: &mut BinaryHeap<Bigram>) {
        if left < 0 || right < 0 {
            return;
        }
        let (l, r) = (symbols[left as usize], symbols[right as usize]);
        let key = rank_key(&word[l.start..l.start + l.n], &word[r.start..r.start + r.n]);
        if let Some(&rank) = self.bpe_ranks.get(&key) {
            heap.push(Bigram { rank, left: left as usize, right: right as usize, size: l.n + r.n });
        }
    }

    // --- decode ---------------------------------------------------------------

    /// Whole-sequence decode (`llama_detokenize`), returning raw bytes. The
    /// bytes are not forced to be valid UTF-8 and nothing is replaced.
    pub fn decode(&self, ids: &[TokenId], opts: DecodeOptions) -> Result<Vec<u8>> {
        for &id in ids {
            self.token(id)?;
        }
        let mut ids = ids;
        let mut remove_space = self.add_space_prefix;
        if opts.remove_special && self.add_bos && !ids.is_empty() && Some(ids[0]) == self.special.bos {
            remove_space = false;
            ids = &ids[1..];
        }
        if opts.remove_special && self.add_eos && !ids.is_empty() && Some(ids[ids.len() - 1]) == self.special.eos {
            ids = &ids[..ids.len() - 1];
        }
        let mut text: Vec<u8> = Vec::new();
        for &id in ids {
            let mut piece = self.token_to_piece(id, opts.render_special)?;
            if remove_space && piece.first() == Some(&b' ') {
                piece = &piece[1..];
            }
            remove_space = false;
            text.extend_from_slice(piece);
        }
        if self.clean_spaces {
            clean_spaces(&mut text);
        }
        Ok(text)
    }

    /// Decode and require valid UTF-8 (explicit error otherwise).
    pub fn decode_to_string(&self, ids: &[TokenId], opts: DecodeOptions) -> Result<String> {
        let bytes = self.decode(ids, opts)?;
        String::from_utf8(bytes).map_err(|e| TokenizerError::InvalidUtf8 {
            valid_up_to: e.utf8_error().valid_up_to(),
            bytes: e.into_bytes(),
        })
    }

    /// Convert signed IDs (for example from JSON or a C API) with range checks.
    pub fn checked_ids(&self, ids: &[i64]) -> Result<Vec<TokenId>> {
        ids.iter()
            .map(|&id| {
                if id < 0 || id >= self.tokens.len() as i64 {
                    Err(TokenizerError::InvalidTokenId { id, vocab_size: self.tokens.len() })
                } else {
                    Ok(id as TokenId)
                }
            })
            .collect()
    }
}

/// `llama_decode_text`: GPT-2 byte-level text back to bytes.
fn decode_text(text: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len());
    let mut utf8 = Vec::with_capacity(4);
    for cpt in unicode::cpts_from_utf8(text) {
        utf8.clear();
        unicode::cpt_to_utf8(cpt, &mut utf8).ok_or(TokenizerError::InvalidCodepoint { codepoint: cpt })?;
        match unicode::cpt_to_byte(cpt) {
            Some(b) => out.push(b),
            None => {
                out.extend_from_slice(b"[UNK_BYTE_0x");
                for c in &utf8 {
                    out.extend_from_slice(format!("{c:02x}").as_bytes());
                }
                out.extend_from_slice(text);
                out.push(b']');
            }
        }
    }
    Ok(out)
}

/// The three `clean_spaces` passes of `llama_vocab::impl::detokenize`,
/// emulated in place with the same index arithmetic.
fn clean_spaces(text: &mut Vec<u8>) {
    let mut total = text.len();
    // first pass: " ?", " !", " .", " ,"
    let total1 = total;
    total = usize::from(total > 0);
    for i in 1..total1 {
        let x = text[i];
        if text[i - 1] == b' ' && matches!(x, b'?' | b'!' | b'.' | b',') {
            total -= 1;
        }
        text[total] = x;
        total += 1;
    }
    // second pass: " ' " -> "'"
    let total2 = total;
    total = usize::from(total > 0);
    let mut i = 1;
    while i < total2 {
        let x = text[i];
        if x == b'\'' && i + 1 < total2 && text[i - 1] == b' ' && text[i + 1] == b' ' {
            total -= 1;
            i += 1;
            text[i] = 0;
        }
        text[total] = x;
        total += 1;
        i += 1;
    }
    // third pass: " 's", " 'm", " 're", " 've"
    let total3 = total;
    total = usize::from(total > 0);
    for i in 1..total3 {
        let x = text[i];
        if text[i - 1] == b' ' && x == b'\'' && i + 1 < total3 {
            let x1 = text[i + 1];
            if x1 == b't' || x1 == b'd' {
            } else if x1 == b's' || x1 == b'm' {
                total -= 1;
            } else if i + 2 < total3 {
                let x2 = text[i + 2];
                if x1 == b'l' && x2 == b'l' {
                } else if (x1 == b'r' && x2 == b'e') || (x1 == b'v' && x2 == b'e') {
                    total -= 1;
                }
            }
        }
        text[total] = x;
        total += 1;
    }
    text.truncate(total);
}

fn find_bytes(hay: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > hay.len() {
        return None;
    }
    let first = needle[0];
    let last_start = hay.len() - needle.len();
    let mut i = 0;
    while i <= last_start {
        match hay[i..=last_start].iter().position(|&b| b == first) {
            None => return None,
            Some(p) => {
                i += p;
                if &hay[i..i + needle.len()] == needle {
                    return Some(i);
                }
                i += 1;
            }
        }
    }
    None
}

fn find_equal_length_overlaps(tokens: &[TokenData], cache: &[TokenId]) -> Vec<(TokenId, TokenId)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < cache.len() {
        let len = tokens[cache[i] as usize].text.len();
        let mut j = i;
        while j < cache.len() && tokens[cache[j] as usize].text.len() == len {
            j += 1;
        }
        let group = &cache[i..j];
        if group.len() <= 256 {
            for &a in group {
                for &b in group {
                    if a == b {
                        continue;
                    }
                    let (ta, tb) = (&tokens[a as usize].text, &tokens[b as usize].text);
                    if (1..len).any(|k| ta[len - k..] == tb[..k]) {
                        out.push((a, b));
                    }
                }
            }
        }
        i = j;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strtol_matches_c() {
        assert_eq!(strtol16(b"0A"), 10);
        assert_eq!(strtol16(b"ff"), 255);
        assert_eq!(strtol16(b""), 0);
        assert_eq!(strtol16(b"0x"), 0);
        assert_eq!(strtol16(b" 7"), 7);
        assert_eq!(strtol16(b"g1"), 0);
    }

    #[test]
    fn clean_spaces_passes() {
        let run = |s: &str| {
            let mut v = s.as_bytes().to_vec();
            clean_spaces(&mut v);
            String::from_utf8_lossy(&v).into_owned()
        };
        assert_eq!(run("Hello , world !"), "Hello, world!");
        assert_eq!(run("a ' b"), "a'b");
        assert_eq!(run("it 's I 'm we 're they 've I 'll it 'd don 't"), "it's I'm we're they've I 'll it 'd don 't");
        assert_eq!(run(""), "");
        assert_eq!(run(" ."), ".");
    }

    #[test]
    fn find_bytes_works() {
        assert_eq!(find_bytes(b"abcabc", b"ca"), Some(2));
        assert_eq!(find_bytes(b"aaab", b"aab"), Some(1));
        assert_eq!(find_bytes(b"ab", b"abc"), None);
    }
}
