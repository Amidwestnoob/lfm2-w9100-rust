//! Chat-template rendering for the model's `tokenizer.chat_template`.
//!
//! The template is rendered with `minijinja` (pure Rust). The test oracle is
//! Jinja2 (`chat_reference.py`, test-only). Settings that make the outputs
//! identical for the LFM2.5 template:
//!
//! - `trim_blocks` and `lstrip_blocks` on, trailing newline of the source
//!   dropped, auto-escaping off (Jinja2 `Environment` defaults used by the
//!   oracle).
//! - `{% generation %}` / `{% endgeneration %}` (a Hugging Face extension for
//!   assistant-span masks) are rewritten to `{% with %}` / `{% endwith %}`,
//!   which keeps the body and its variable scope, like the oracle's
//!   extension. Whitespace-control markers are kept. Span masks are not
//!   produced.
//! - Python-compatible replacements: `tojson` (`json.dumps(ensure_ascii=False)`:
//!   insertion order, `", "`/`": "` separators, no HTML escaping, Python float
//!   form), `string` and printed values (Python `str()`: `True`, `None`, list
//!   and dict `repr`), `trim` (Python `str.strip()` whitespace set), `join`
//!   (items through `str()`), and the methods `dict.get/items/keys/values`
//!   and `str.split/startswith/endswith/strip/lstrip/rstrip`.
//!
//! Known differences (documented, not approximated silently): any other
//! Python method is an error (`UnknownMethod`); Python `str.isprintable` for
//! `repr` uses the vendored llama.cpp Unicode table, which can differ from
//! the Python Unicode version for rare code points (the table matches Python
//! 3.13.5 / Unicode 15.1); integers beyond i128 support exact rendering, not
//! arbitrary-precision template arithmetic; JSON with lone surrogate
//! escapes is rejected; error messages differ from Jinja2 (both fail);
//! JSON `null` is a Python-like `None` object (see `PyNone`), so `x is none`
//! works but `x == none` is false for it (Jinja2: true).

use crate::error::{Result, TokenizerError};
use crate::vocab::{EncodeOptions, TokenId, Tokenizer};
use minijinja::value::{Object, ObjectRepr, Value, ValueKind};
use minijinja::{AutoEscape, Environment, Error, ErrorKind, State};
use std::sync::Arc;

/// JSON `null` in the template context. minijinja treats its own `none` as
/// an empty iterable; Python/Jinja2 raise `TypeError` when iterating `None`.
/// This object keeps Python behavior: falsy, defined, prints `None`,
/// serializes as `null`, matches `is none`, and is not iterable.
#[derive(Debug)]
struct PyNone;

impl Object for PyNone {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Plain
    }
    fn is_true(self: &Arc<Self>) -> bool {
        false
    }
    fn render(self: &Arc<Self>, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("None")
    }
}

fn py_none() -> Value {
    Value::from_object(PyNone)
}

fn is_py_none(v: &Value) -> bool {
    v.is_none() || v.downcast_object_ref::<PyNone>().is_some()
}

/// Integers beyond MiniJinja's i128 range remain exact for LFM2's string and
/// JSON rendering paths. They are not strings (which the template quotes).
#[derive(Debug)]
struct PyBigInt(String);

impl Object for PyBigInt {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Plain
    }
    fn render(self: &Arc<Self>, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Convert JSON to template values, keeping key order and Python `None`.
fn json_to_value(v: &serde_json::Value) -> Value {
    match v {
        serde_json::Value::Null => py_none(),
        serde_json::Value::Bool(b) => Value::from(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::from(i)
            } else if let Some(u) = n.as_u64() {
                Value::from(u)
            } else if !n.as_str().contains(['.', 'e', 'E']) {
                match n.as_str().parse::<i128>() {
                    Ok(i) => Value::from(i),
                    Err(_) => Value::from_object(PyBigInt(n.to_string())),
                }
            } else {
                Value::from(n.as_str().parse::<f64>().unwrap_or(f64::NAN))
            }
        }
        serde_json::Value::String(s) => Value::from(s.as_str()),
        serde_json::Value::Array(a) => Value::from(a.iter().map(json_to_value).collect::<Vec<_>>()),
        serde_json::Value::Object(o) => Value::from_iter(o.iter().map(|(k, v)| (Value::from(k.as_str()), json_to_value(v)))),
    }
}

const TEMPLATE_NAME: &str = "chat_template";

/// A compiled chat template.
pub struct ChatTemplate {
    env: Environment<'static>,
    source: String,
    rewritten: String,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate").field("source_len", &self.source.len()).finish()
    }
}

/// Structured chat request. Field names match the template variables.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ChatRequest {
    /// Array of message objects (`role`, `content`, optional `thinking`,
    /// `tool_calls`). Content may be a string or a list of typed items.
    pub messages: serde_json::Value,
    /// Tool definitions (formatting only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<serde_json::Value>,
    #[serde(default)]
    pub add_generation_prompt: bool,
    /// `None` leaves the template default (`false`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preserve_thinking: Option<bool>,
    pub bos_token: String,
    pub eos_token: String,
}

impl ChatRequest {
    /// Request with BOS/EOS text taken from the tokenizer's vocabulary.
    pub fn for_tokenizer(tok: &Tokenizer, messages: serde_json::Value) -> Result<Self> {
        let text = |id: Option<TokenId>, what: &'static str| -> Result<String> {
            let id = id.ok_or(TokenizerError::MissingSpecialToken(what))?;
            String::from_utf8(tok.token_text(id)?.to_vec())
                .map_err(|_| TokenizerError::InvalidMetadata(format!("{what} token text is not UTF-8")))
        };
        Ok(ChatRequest {
            messages,
            tools: None,
            add_generation_prompt: false,
            preserve_thinking: None,
            bos_token: text(tok.bos(), "BOS")?,
            eos_token: text(tok.eos(), "EOS")?,
        })
    }
}

/// Rewrite actual generation statements, not text in strings, comments or raw blocks.
fn rewrite_generation_tags(src: &str) -> String {
    let b = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let (mut i, mut last, mut raw) = (0, 0, false);
    while i + 1 < b.len() {
        if !raw && &b[i..i + 2] == b"{#" {
            i = src[i + 2..].find("#}").map_or(b.len(), |n| i + n + 4);
            continue;
        }
        let statement = &b[i..i + 2] == b"{%";
        if !statement && (raw || &b[i..i + 2] != b"{{") {
            i += 1;
            continue;
        }
        let mut start = i + 2;
        if start < b.len() && matches!(b[start], b'-' | b'+') { start += 1; }
        while start < b.len() && b[start].is_ascii_whitespace() { start += 1; }
        let mut word_end = start;
        while word_end < b.len() && (b[word_end].is_ascii_alphanumeric() || b[word_end] == b'_') {
            word_end += 1;
        }
        let word = &src[start..word_end];
        // Raw text has no quoting rules. Only an exact endraw statement matters.
        if raw {
            let mut end = word_end;
            while end < b.len() && b[end].is_ascii_whitespace() { end += 1; }
            if end < b.len() && matches!(b[end], b'-' | b'+') { end += 1; }
            if word == "endraw" && b.get(end..end + 2) == Some(b"%}") {
                raw = false;
                i = end + 2;
            } else {
                i += 2;
            }
            continue;
        }
        let close = if statement { b"%}" } else { b"}}" };
        let (mut end, mut quote, mut depth) = (i + 2, None, 0usize);
        while end + 1 < b.len() {
            let c = b[end];
            if let Some(q) = quote {
                if c == b'\\' { end += 2; continue; }
                if c == q { quote = None; }
            } else if matches!(c, b'\'' | b'"') {
                quote = Some(c);
            } else if depth == 0 && &b[end..end + 2] == close {
                break;
            } else if matches!(c, b'(' | b'[' | b'{') {
                depth += 1;
            } else if matches!(c, b')' | b']' | b'}') {
                depth = depth.saturating_sub(1);
            }
            end += 1;
        }
        if end + 1 >= b.len() { break; } // compiler reports malformed source
        let rest = src[word_end..end].trim().trim_end_matches(['-', '+']).trim();
        if statement && rest.is_empty() {
            if word == "raw" {
                raw = true;
            } else if word == "generation" || word == "endgeneration" {
                out.push_str(&src[last..start]);
                out.push_str(if word == "generation" { "with" } else { "endwith" });
                last = word_end;
            }
        }
        i = end + 2;
    }
    out.push_str(&src[last..]);
    out
}

impl ChatTemplate {
    /// Compile a template source (for example `chat_template.jinja`).
    pub fn new(source: &str) -> Result<Self> {
        let rewritten = rewrite_generation_tags(source);
        let mut env = Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_keep_trailing_newline(false);
        env.set_auto_escape_callback(|_| AutoEscape::None);
        env.set_formatter(|out, _state, value| {
            let s = py_str(value)?;
            out.write_str(&s).map_err(|_| Error::new(ErrorKind::WriteFailure, "write failed"))
        });
        env.add_filter("tojson", filter_tojson);
        env.add_filter("string", |v: Value| -> std::result::Result<String, Error> { py_str(&v) });
        env.add_filter("trim", filter_trim);
        env.add_filter("join", filter_join);
        env.set_unknown_method_callback(py_methods);
        env.add_test("none", |v: Value| is_py_none(&v));
        env.add_template_owned(TEMPLATE_NAME, rewritten.clone())
            .map_err(|e| TokenizerError::Template(format!("compile: {e:#}")))?;
        Ok(ChatTemplate { env, source: source.to_string(), rewritten })
    }

    /// Compile the template stored in the tokenizer metadata.
    pub fn from_tokenizer(tok: &Tokenizer) -> Result<Self> {
        let src = tok
            .chat_template()
            .ok_or_else(|| TokenizerError::InvalidMetadata("no tokenizer.chat_template in metadata".into()))?;
        Self::new(src)
    }

    /// Original template source.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Source after the `generation` tag rewrite.
    pub fn rewritten_source(&self) -> &str {
        &self.rewritten
    }

    /// Render with an arbitrary JSON object as the template context.
    pub fn render_json(&self, context: &serde_json::Value) -> Result<String> {
        if !context.is_object() {
            return Err(TokenizerError::InvalidInput("chat context must be a JSON object".into()));
        }
        let tmpl = self.env.get_template(TEMPLATE_NAME).map_err(|e| TokenizerError::Template(e.to_string()))?;
        tmpl.render(json_to_value(context)).map_err(|e| TokenizerError::Template(format!("render: {e:#}")))
    }

    /// Render a structured request.
    pub fn render(&self, req: &ChatRequest) -> Result<String> {
        if !req.messages.is_array() {
            return Err(TokenizerError::InvalidInput("messages must be an array".into()));
        }
        let ctx = serde_json::to_value(req).map_err(|e| TokenizerError::InvalidInput(e.to_string()))?;
        self.render_json(&ctx)
    }
}

impl Tokenizer {
    /// Render a chat request and tokenize the result. The template already
    /// emits BOS, so this uses `add_special = false, parse_special = true`
    /// (no duplicated BOS).
    pub fn encode_chat(&self, template: &ChatTemplate, req: &ChatRequest) -> Result<Vec<TokenId>> {
        let text = template.render(req)?;
        self.encode(text.as_bytes(), EncodeOptions { add_special: false, parse_special: true })
    }
}

// --- Python compatibility ----------------------------------------------------

fn tmpl_err(msg: impl Into<String>) -> Error {
    Error::new(ErrorKind::InvalidOperation, msg.into())
}

/// Python `repr(float)`.
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".into() } else { "0.0".into() };
    }
    let sci = format!("{:e}", x.abs()); // shortest round-trip digits
    let (mant, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mant.chars().filter(|c| c.is_ascii_digit()).collect();
    let decpt = exp + 1;
    let sign = if x < 0.0 { "-" } else { "" };
    let nd = digits.len() as i32;
    let body = if decpt <= -4 || decpt > 16 {
        let mut s = String::new();
        s.push_str(&digits[..1]);
        if nd > 1 {
            s.push('.');
            s.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        s.push_str(&format!("e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs()));
        s
    } else if decpt <= 0 {
        format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
    } else if decpt >= nd {
        format!("{}{}.0", digits, "0".repeat((decpt - nd) as usize))
    } else {
        format!("{}.{}", &digits[..decpt as usize], &digits[decpt as usize..])
    };
    format!("{sign}{body}")
}

fn number_repr(v: &Value) -> std::result::Result<String, Error> {
    if v.is_integer() {
        let i: i128 = i128::try_from(v.clone()).map_err(|_| tmpl_err("integer out of range"))?;
        Ok(i.to_string())
    } else {
        let f: f64 = f64::try_from(v.clone()).map_err(|_| tmpl_err("bad number"))?;
        Ok(py_float_repr(f))
    }
}

/// Python `repr(str)`.
pub fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for ch in s.chars() {
        let c = ch as u32;
        if ch == quote || ch == '\\' {
            out.push('\\');
            out.push(ch);
        } else if ch == '\t' {
            out.push_str("\\t");
        } else if ch == '\n' {
            out.push_str("\\n");
        } else if ch == '\r' {
            out.push_str("\\r");
        } else if c < 0x20 || c == 0x7f {
            out.push_str(&format!("\\x{c:02x}"));
        } else if c < 0x7f {
            out.push(ch);
        } else if crate::unicode::is_printable(c) {
            out.push(ch);
        } else if c <= 0xff {
            out.push_str(&format!("\\x{c:02x}"));
        } else if c <= 0xffff {
            out.push_str(&format!("\\u{c:04x}"));
        } else {
            out.push_str(&format!("\\U{c:08x}"));
        }
    }
    out.push(quote);
    out
}

fn map_items(v: &Value) -> std::result::Result<Vec<(Value, Value)>, Error> {
    let mut items = Vec::new();
    for k in v.try_iter()? {
        let val = v.get_item(&k)?;
        items.push((k, val));
    }
    Ok(items)
}

/// Python `repr()` for template values.
fn py_repr(v: &Value) -> std::result::Result<String, Error> {
    Ok(match v.kind() {
        ValueKind::String => py_str_repr(v.as_str().unwrap_or_default()),
        ValueKind::Seq | ValueKind::Iterable => {
            let parts: std::result::Result<Vec<String>, Error> = v.try_iter()?.map(|x| py_repr(&x)).collect();
            format!("[{}]", parts?.join(", "))
        }
        ValueKind::Map => {
            let mut parts = Vec::new();
            for (k, val) in map_items(v)? {
                parts.push(format!("{}: {}", py_repr(&k)?, py_repr(&val)?));
            }
            format!("{{{}}}", parts.join(", "))
        }
        _ => py_str(v)?,
    })
}

/// Python `str()` for template values (used for printing and `|string`).
fn py_str(v: &Value) -> std::result::Result<String, Error> {
    Ok(match v.kind() {
        ValueKind::Undefined => String::new(),
        ValueKind::None => "None".into(),
        ValueKind::Bool => if v.is_true() { "True".into() } else { "False".into() },
        ValueKind::Number => number_repr(v)?,
        ValueKind::String => v.as_str().unwrap_or_default().to_string(),
        ValueKind::Seq | ValueKind::Iterable | ValueKind::Map => py_repr(v)?,
        ValueKind::Bytes => return Err(tmpl_err("bytes values are not supported")),
        _ => v.to_string(),
    })
}

fn json_escape(s: &str, out: &mut String) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `json.dumps(v, ensure_ascii=False)`.
fn py_json(v: &Value, out: &mut String) -> std::result::Result<(), Error> {
    match v.kind() {
        ValueKind::Undefined => return Err(tmpl_err("Object of type Undefined is not JSON serializable")),
        ValueKind::None => out.push_str("null"),
        ValueKind::Plain if is_py_none(v) => out.push_str("null"),
        ValueKind::Plain if v.downcast_object_ref::<PyBigInt>().is_some() => {
            out.push_str(&v.downcast_object_ref::<PyBigInt>().unwrap().0);
        }
        ValueKind::Bool => out.push_str(if v.is_true() { "true" } else { "false" }),
        ValueKind::Number => {
            if v.is_integer() {
                out.push_str(&number_repr(v)?);
            } else {
                let f: f64 = f64::try_from(v.clone()).map_err(|_| tmpl_err("bad number"))?;
                if f.is_nan() {
                    out.push_str("NaN");
                } else if f.is_infinite() {
                    out.push_str(if f > 0.0 { "Infinity" } else { "-Infinity" });
                } else {
                    out.push_str(&py_float_repr(f));
                }
            }
        }
        ValueKind::String => json_escape(v.as_str().unwrap_or_default(), out),
        ValueKind::Seq | ValueKind::Iterable => {
            out.push('[');
            for (i, x) in v.try_iter()?.enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json(&x, out)?;
            }
            out.push(']');
        }
        ValueKind::Map => {
            out.push('{');
            for (i, (k, val)) in map_items(v)?.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                let key = match k.kind() {
                    ValueKind::String => k.as_str().unwrap_or_default().to_string(),
                    ValueKind::None => "null".into(),
                    ValueKind::Bool => if k.is_true() { "true".into() } else { "false".into() },
                    ValueKind::Number => {
                        let mut s = String::new();
                        py_json(&k, &mut s)?;
                        s
                    }
                    _ => return Err(tmpl_err("keys must be str, int, float, bool or None")),
                };
                json_escape(&key, out);
                out.push_str(": ");
                py_json(&val, out)?;
            }
            out.push('}');
        }
        _ => return Err(tmpl_err(format!("Object of type {} is not JSON serializable", v.kind()))),
    }
    Ok(())
}

fn filter_tojson(v: Value, kwargs: minijinja::value::Kwargs) -> std::result::Result<String, Error> {
    kwargs.assert_all_used()?; // indent/sort_keys etc. are not supported: explicit error
    let mut s = String::new();
    py_json(&v, &mut s)?;
    Ok(s)
}

/// Python whitespace (`str.isspace`) as used by `str.strip()`/`split()`.
fn py_isspace(c: char) -> bool {
    matches!(c as u32,
        0x09..=0x0D | 0x1C..=0x1F | 0x20 | 0x85 | 0xA0 | 0x1680 | 0x2000..=0x200A
        | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000)
}

fn py_strip<'a>(s: &'a str, chars: Option<&str>, left: bool, right: bool) -> &'a str {
    let pred = |c: char| match chars {
        None => py_isspace(c),
        Some(set) => set.contains(c),
    };
    let mut r = s;
    if left {
        r = r.trim_start_matches(pred);
    }
    if right {
        r = r.trim_end_matches(pred);
    }
    r
}

fn filter_trim(v: Value, chars: Option<Value>) -> std::result::Result<String, Error> {
    let s = py_str(&v)?;
    let chars = match &chars {
        None => None,
        Some(c) if is_py_none(c) || c.is_undefined() => None,
        Some(c) => Some(c.as_str().ok_or_else(|| tmpl_err("trim chars must be a string"))?.to_string()),
    };
    Ok(py_strip(&s, chars.as_deref(), true, true).to_string())
}

fn filter_join(v: Value, d: Option<String>) -> std::result::Result<String, Error> {
    let d = d.unwrap_or_default();
    if let Some(s) = v.as_str() {
        // Jinja2 joins the characters of a string
        return Ok(s.chars().map(|c| c.to_string()).collect::<Vec<_>>().join(&d));
    }
    let parts: std::result::Result<Vec<String>, Error> = v.try_iter()?.map(|x| py_str(&x)).collect();
    Ok(parts?.join(&d))
}

fn py_split(s: &str, sep: Option<&str>, maxsplit: i64) -> std::result::Result<Vec<String>, Error> {
    let mut out = Vec::new();
    match sep {
        Some("") => return Err(tmpl_err("empty separator")),
        Some(sep) => {
            let mut rest = s;
            let mut n = 0i64;
            while maxsplit < 0 || n < maxsplit {
                match rest.find(sep) {
                    Some(p) => {
                        out.push(rest[..p].to_string());
                        rest = &rest[p + sep.len()..];
                        n += 1;
                    }
                    None => break,
                }
            }
            out.push(rest.to_string());
        }
        None => {
            let mut rest = py_strip(s, None, true, false);
            let mut n = 0i64;
            while !rest.is_empty() {
                if maxsplit >= 0 && n >= maxsplit {
                    out.push(rest.to_string());
                    break;
                }
                let end = rest.find(py_isspace).unwrap_or(rest.len());
                out.push(rest[..end].to_string());
                rest = py_strip(&rest[end..], None, true, false);
                n += 1;
            }
        }
    }
    Ok(out)
}

fn py_methods(_state: &State, value: &Value, method: &str, args: &[Value]) -> std::result::Result<Value, Error> {
    let nargs = |lo: usize, hi: usize| -> std::result::Result<(), Error> {
        if args.len() < lo || args.len() > hi {
            Err(Error::new(ErrorKind::InvalidOperation, format!("{method}() takes {lo}..{hi} arguments")))
        } else {
            Ok(())
        }
    };
    match (value.kind(), method) {
        (ValueKind::Map, "get") => {
            nargs(1, 2)?;
            let v = value.get_item(&args[0])?;
            Ok(if v.is_undefined() { args.get(1).cloned().unwrap_or_else(py_none) } else { v })
        }
        (ValueKind::Map, "items") => {
            nargs(0, 0)?;
            Ok(Value::from(
                map_items(value)?.into_iter().map(|(k, v)| Value::from(vec![k, v])).collect::<Vec<_>>(),
            ))
        }
        (ValueKind::Map, "keys") => {
            nargs(0, 0)?;
            Ok(Value::from(value.try_iter()?.collect::<Vec<_>>()))
        }
        (ValueKind::Map, "values") => {
            nargs(0, 0)?;
            Ok(Value::from(map_items(value)?.into_iter().map(|(_, v)| v).collect::<Vec<_>>()))
        }
        (ValueKind::String, m @ ("startswith" | "endswith")) => {
            nargs(1, 1)?;
            let s = value.as_str().unwrap_or_default();
            let pat = args[0].as_str().ok_or_else(|| tmpl_err(format!("{m}() argument must be str")))?;
            Ok(Value::from(if m == "startswith" { s.starts_with(pat) } else { s.ends_with(pat) }))
        }
        (ValueKind::String, m @ ("strip" | "lstrip" | "rstrip")) => {
            nargs(0, 1)?;
            let s = value.as_str().unwrap_or_default();
            let chars = match args.first() {
                None => None,
                Some(c) if is_py_none(c) => None,
                Some(c) => Some(c.as_str().ok_or_else(|| tmpl_err("strip arg must be str or None"))?.to_string()),
            };
            Ok(Value::from(py_strip(s, chars.as_deref(), m != "rstrip", m != "lstrip").to_string()))
        }
        (ValueKind::String, "split") => {
            nargs(0, 2)?;
            let s = value.as_str().unwrap_or_default();
            let sep = match args.first() {
                None => None,
                Some(v) if is_py_none(v) => None,
                Some(v) => Some(v.as_str().ok_or_else(|| tmpl_err("must be str or None"))?.to_string()),
            };
            let maxsplit = match args.get(1) {
                None => -1,
                Some(v) => i64::try_from(v.clone()).map_err(|_| tmpl_err("maxsplit must be int"))?,
            };
            Ok(Value::from(py_split(s, sep.as_deref(), maxsplit)?.into_iter().map(Value::from).collect::<Vec<_>>()))
        }
        _ => Err(Error::from(ErrorKind::UnknownMethod)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn float_repr_matches_python() {
        let cases = [
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123456789012345680.0, "1.2345678901234568e+17"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-2.5, "-2.5"),
            (1.5e-7, "1.5e-07"),
            (3.14159, "3.14159"),
            (-0.0, "-0.0"),
        ];
        for (x, want) in cases {
            assert_eq!(py_float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn str_repr_matches_python() {
        assert_eq!(py_str_repr("abc"), "'abc'");
        assert_eq!(py_str_repr("it's"), "\"it's\"");
        assert_eq!(py_str_repr("a'b\"c"), "'a\\'b\"c'");
        assert_eq!(py_str_repr("x\ny\t\u{1}"), "'x\\ny\\t\\x01'");
        assert_eq!(py_str_repr("café 中"), "'café 中'");
        assert_eq!(py_str_repr("\u{a0}"), "'\\xa0'");
    }

    #[test]
    fn generation_rewrite() {
        let s = "a{%- generation -%}b{% endgeneration %}c{{ generation }}";
        assert_eq!(rewrite_generation_tags(s), "a{%- with -%}b{% endwith %}c{{ generation }}");
    }

    #[test]
    fn split_matches_python() {
        assert_eq!(py_split("a</think>b</think>c", Some("</think>"), -1).unwrap(), vec!["a", "b", "c"]);
        assert_eq!(py_split("  a  b ", None, -1).unwrap(), vec!["a", "b"]);
        assert_eq!(py_split("a,b,c", Some(","), 1).unwrap(), vec!["a", "b,c"]);
        assert!(py_split("a", Some(""), -1).is_err());
    }
}
