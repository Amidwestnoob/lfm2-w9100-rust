//! Metadata-only GGUF v3 reader.
//!
//! This follows the approach of the repaired `context/parser` crate (same
//! value-type ids, array layout `(element_type: u32, count: u64)`, little
//! endian, bounded allocations, explicit errors and no panics on malformed
//! input). It stops after the key/value section: tensor infos and tensor
//! data are never read, so the same code accepts the metadata-only GGUF from
//! `write_metadata_gguf.py` and the full model file.
//!
//! Differences from `context/parser`: strings are kept as raw bytes (GGUF does
//! not require UTF-8 and the reference loader does not validate it), and
//! arrays are stored in typed vectors to keep the 128k-entry vocabulary and
//! 293k-entry merge list compact.

use std::collections::BTreeMap;
use std::fmt;
use std::io::{BufReader, Read};
use std::path::Path;

/// GGUF value type ids.
pub mod ty {
    pub const U8: u32 = 0;
    pub const I8: u32 = 1;
    pub const U16: u32 = 2;
    pub const I16: u32 = 3;
    pub const U32: u32 = 4;
    pub const I32: u32 = 5;
    pub const F32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const U64: u32 = 10;
    pub const I64: u32 = 11;
    pub const F64: u32 = 12;
}

/// A typed GGUF array.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaArray {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    U32(Vec<u32>),
    I32(Vec<i32>),
    F32(Vec<f32>),
    Bool(Vec<bool>),
    String(Vec<Vec<u8>>),
    U64(Vec<u64>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl MetaArray {
    pub fn len(&self) -> usize {
        match self {
            MetaArray::U8(v) => v.len(),
            MetaArray::I8(v) => v.len(),
            MetaArray::U16(v) => v.len(),
            MetaArray::I16(v) => v.len(),
            MetaArray::U32(v) => v.len(),
            MetaArray::I32(v) => v.len(),
            MetaArray::F32(v) => v.len(),
            MetaArray::Bool(v) => v.len(),
            MetaArray::String(v) => v.len(),
            MetaArray::U64(v) => v.len(),
            MetaArray::I64(v) => v.len(),
            MetaArray::F64(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn type_name(&self) -> &'static str {
        match self {
            MetaArray::U8(_) => "U8",
            MetaArray::I8(_) => "I8",
            MetaArray::U16(_) => "U16",
            MetaArray::I16(_) => "I16",
            MetaArray::U32(_) => "U32",
            MetaArray::I32(_) => "I32",
            MetaArray::F32(_) => "F32",
            MetaArray::Bool(_) => "Bool",
            MetaArray::String(_) => "String",
            MetaArray::U64(_) => "U64",
            MetaArray::I64(_) => "I64",
            MetaArray::F64(_) => "F64",
        }
    }
}

/// A GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(Vec<u8>),
    Array(MetaArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl MetaValue {
    pub fn type_name(&self) -> String {
        match self {
            MetaValue::U8(_) => "U8".into(),
            MetaValue::I8(_) => "I8".into(),
            MetaValue::U16(_) => "U16".into(),
            MetaValue::I16(_) => "I16".into(),
            MetaValue::U32(_) => "U32".into(),
            MetaValue::I32(_) => "I32".into(),
            MetaValue::F32(_) => "F32".into(),
            MetaValue::Bool(_) => "Bool".into(),
            MetaValue::String(_) => "String".into(),
            MetaValue::Array(a) => format!("Array<{}>", a.type_name()),
            MetaValue::U64(_) => "U64".into(),
            MetaValue::I64(_) => "I64".into(),
            MetaValue::F64(_) => "F64".into(),
        }
    }
}

/// Metadata map, keyed by GGUF key.
pub type Metadata = BTreeMap<String, MetaValue>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufError(pub String);

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GGUF: {}", self.0)
    }
}
impl std::error::Error for GgufError {}

fn err<T>(msg: impl Into<String>) -> Result<T, GgufError> {
    Err(GgufError(msg.into()))
}

struct Reader<R: Read> {
    r: R,
    pos: u64,
    limit: u64,
}

impl<R: Read> Reader<R> {
    fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.pos)
    }
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, GgufError> {
        if n as u64 > self.remaining() {
            return err(format!("truncated input: need {} bytes at offset {}, {} left", n, self.pos, self.remaining()));
        }
        // The bound may describe a huge/sparse file or a truncated stream.
        // Grow only as bytes are actually read, never from a wire length alone.
        let mut buf = Vec::new();
        while buf.len() < n {
            let chunk = (n - buf.len()).min(64 * 1024);
            buf.try_reserve(chunk).map_err(|e| GgufError(format!("string allocation: {e}")))?;
            let start = buf.len();
            buf.resize(start + chunk, 0);
            self.fill(&mut buf[start..])?;
        }
        Ok(buf)
    }
    fn fill(&mut self, buf: &mut [u8]) -> Result<(), GgufError> {
        if buf.len() as u64 > self.remaining() {
            return err(format!("truncated input at offset {}", self.pos));
        }
        match self.r.read_exact(buf) {
            Ok(()) => {
                self.pos += buf.len() as u64;
                Ok(())
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => err(format!("truncated input at offset {}", self.pos)),
            Err(e) => err(format!("I/O error: {e}")),
        }
    }
    fn arr<const N: usize>(&mut self) -> Result<[u8; N], GgufError> {
        let mut b = [0u8; N];
        self.fill(&mut b)?;
        Ok(b)
    }
    fn u8(&mut self) -> Result<u8, GgufError> {
        Ok(self.arr::<1>()?[0])
    }
    fn u16(&mut self) -> Result<u16, GgufError> {
        Ok(u16::from_le_bytes(self.arr()?))
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn string(&mut self) -> Result<Vec<u8>, GgufError> {
        let len = self.u64()?;
        if len > self.remaining() {
            return err(format!("string length {} exceeds remaining {} bytes at offset {}", len, self.remaining(), self.pos));
        }
        let len = usize::try_from(len).map_err(|_| GgufError("string length exceeds address space".into()))?;
        self.bytes(len)
    }
    fn bool(&mut self) -> Result<bool, GgufError> {
        // gguf.cpp reads an int8 and treats every non-zero value as true.
        Ok(self.u8()? != 0)
    }
}

fn elem_size(t: u32) -> Option<u64> {
    Some(match t {
        ty::U8 | ty::I8 | ty::BOOL => 1,
        ty::U16 | ty::I16 => 2,
        ty::U32 | ty::I32 | ty::F32 => 4,
        ty::U64 | ty::I64 | ty::F64 => 8,
        ty::STRING => 8, // minimum: the length prefix
        _ => return None,
    })
}

fn read_array<R: Read>(rd: &mut Reader<R>) -> Result<MetaArray, GgufError> {
    let et = rd.u32()?;
    let n = rd.u64()?;
    let min = match elem_size(et) {
        Some(s) => s,
        None if et == ty::ARRAY => return err("nested arrays are not supported"),
        None => return err(format!("unknown array element type {et}")),
    };
    // Bound the allocation against the bytes that can still follow.
    if n.checked_mul(min).map_or(true, |b| b > rd.remaining()) {
        return err(format!("array count {} cannot fit in remaining {} bytes", n, rd.remaining()));
    }
    let n = usize::try_from(n).map_err(|_| GgufError("array count exceeds address space".into()))?;
    macro_rules! prim {
        ($variant:ident, $t:ty, $read:expr) => {{
            let mut v: Vec<$t> = Vec::new();
            for _ in 0..n {
                let value = $read(rd)?;
                v.try_reserve(1).map_err(|e| GgufError(format!("array allocation: {e}")))?;
                v.push(value);
            }
            MetaArray::$variant(v)
        }};
    }
    Ok(match et {
        ty::U8 => prim!(U8, u8, |r: &mut Reader<R>| r.u8()),
        ty::I8 => prim!(I8, i8, |r: &mut Reader<R>| r.u8().map(|x| x as i8)),
        ty::U16 => prim!(U16, u16, |r: &mut Reader<R>| r.u16()),
        ty::I16 => prim!(I16, i16, |r: &mut Reader<R>| r.u16().map(|x| x as i16)),
        ty::U32 => prim!(U32, u32, |r: &mut Reader<R>| r.u32()),
        ty::I32 => prim!(I32, i32, |r: &mut Reader<R>| r.u32().map(|x| x as i32)),
        ty::F32 => prim!(F32, f32, |r: &mut Reader<R>| r.u32().map(f32::from_bits)),
        ty::BOOL => prim!(Bool, bool, |r: &mut Reader<R>| r.bool()),
        ty::STRING => prim!(String, Vec<u8>, |r: &mut Reader<R>| r.string()),
        ty::U64 => prim!(U64, u64, |r: &mut Reader<R>| r.u64()),
        ty::I64 => prim!(I64, i64, |r: &mut Reader<R>| r.u64().map(|x| x as i64)),
        ty::F64 => prim!(F64, f64, |r: &mut Reader<R>| r.u64().map(f64::from_bits)),
        _ => unreachable!("checked by elem_size"),
    })
}

fn read_value<R: Read>(rd: &mut Reader<R>, t: u32) -> Result<MetaValue, GgufError> {
    Ok(match t {
        ty::U8 => MetaValue::U8(rd.u8()?),
        ty::I8 => MetaValue::I8(rd.u8()? as i8),
        ty::U16 => MetaValue::U16(rd.u16()?),
        ty::I16 => MetaValue::I16(rd.u16()? as i16),
        ty::U32 => MetaValue::U32(rd.u32()?),
        ty::I32 => MetaValue::I32(rd.u32()? as i32),
        ty::F32 => MetaValue::F32(f32::from_bits(rd.u32()?)),
        ty::BOOL => MetaValue::Bool(rd.bool()?),
        ty::STRING => MetaValue::String(rd.string()?),
        ty::ARRAY => MetaValue::Array(read_array(rd)?),
        ty::U64 => MetaValue::U64(rd.u64()?),
        ty::I64 => MetaValue::I64(rd.u64()? as i64),
        ty::F64 => MetaValue::F64(f64::from_bits(rd.u64()?)),
        _ => return err(format!("unknown metadata value type {t}")),
    })
}

/// Parse the GGUF header and metadata section from `reader`.
/// `limit` is the total number of readable bytes (the file length).
pub fn read_metadata<R: Read>(reader: R, limit: u64) -> Result<Metadata, GgufError> {
    let mut rd = Reader { r: reader, pos: 0, limit };
    let magic = rd.arr::<4>()?;
    if &magic != b"GGUF" {
        return err(format!("invalid magic {:?}", magic));
    }
    let version = rd.u32()?;
    if version != 3 {
        return err(format!("unsupported GGUF version {version} (expected 3)"));
    }
    let _tensor_count = rd.u64()?;
    let kv_count = rd.u64()?;
    // Each key/value needs at least 8 (key length) + 4 (type) + 1 bytes.
    if kv_count.checked_mul(13).map_or(true, |b| b > rd.remaining()) {
        return err(format!("metadata count {kv_count} cannot fit in the file"));
    }
    let mut md = Metadata::new();
    for _ in 0..kv_count {
        let key = rd.string()?;
        let key = String::from_utf8(key).map_err(|_| GgufError("metadata key is not UTF-8".into()))?;
        let t = rd.u32()?;
        let v = read_value(&mut rd, t)?;
        if md.insert(key.clone(), v).is_some() {
            return err(format!("duplicate metadata key {key}"));
        }
    }
    Ok(md)
}

/// Open a GGUF file and read its metadata only.
pub fn read_metadata_file(path: impl AsRef<Path>) -> Result<Metadata, GgufError> {
    let path = path.as_ref();
    let f = std::fs::File::open(path).map_err(|e| GgufError(format!("open {}: {e}", path.display())))?;
    let len = f.metadata().map_err(|e| GgufError(format!("stat {}: {e}", path.display())))?.len();
    read_metadata(BufReader::with_capacity(1 << 20, f), len)
}

/// Serialize metadata as a GGUF v3 file with zero tensors (test utility and
/// a Rust port of `write_metadata_gguf.py`). Keys are written in the given
/// order; padding follows `general.alignment` (default 32).
pub fn write_metadata_only(entries: &[(String, MetaValue)]) -> Vec<u8> {
    fn s(out: &mut Vec<u8>, b: &[u8]) {
        out.extend_from_slice(&(b.len() as u64).to_le_bytes());
        out.extend_from_slice(b);
    }
    fn tid(v: &MetaValue) -> u32 {
        match v {
            MetaValue::U8(_) => ty::U8,
            MetaValue::I8(_) => ty::I8,
            MetaValue::U16(_) => ty::U16,
            MetaValue::I16(_) => ty::I16,
            MetaValue::U32(_) => ty::U32,
            MetaValue::I32(_) => ty::I32,
            MetaValue::F32(_) => ty::F32,
            MetaValue::Bool(_) => ty::BOOL,
            MetaValue::String(_) => ty::STRING,
            MetaValue::Array(_) => ty::ARRAY,
            MetaValue::U64(_) => ty::U64,
            MetaValue::I64(_) => ty::I64,
            MetaValue::F64(_) => ty::F64,
        }
    }
    fn val(out: &mut Vec<u8>, v: &MetaValue) {
        match v {
            MetaValue::U8(x) => out.push(*x),
            MetaValue::I8(x) => out.push(*x as u8),
            MetaValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::Bool(x) => out.push(*x as u8),
            MetaValue::String(b) => s(out, b),
            MetaValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
            MetaValue::Array(a) => {
                macro_rules! w {
                    ($t:expr, $v:expr, $f:expr) => {{
                        out.extend_from_slice(&$t.to_le_bytes());
                        out.extend_from_slice(&($v.len() as u64).to_le_bytes());
                        for x in $v.iter() {
                            $f(out, x);
                        }
                    }};
                }
                match a {
                    MetaArray::U8(v) => w!(ty::U8, v, |o: &mut Vec<u8>, x: &u8| o.push(*x)),
                    MetaArray::I8(v) => w!(ty::I8, v, |o: &mut Vec<u8>, x: &i8| o.push(*x as u8)),
                    MetaArray::U16(v) => w!(ty::U16, v, |o: &mut Vec<u8>, x: &u16| o.extend_from_slice(&x.to_le_bytes())),
                    MetaArray::I16(v) => w!(ty::I16, v, |o: &mut Vec<u8>, x: &i16| o.extend_from_slice(&x.to_le_bytes())),
                    MetaArray::U32(v) => w!(ty::U32, v, |o: &mut Vec<u8>, x: &u32| o.extend_from_slice(&x.to_le_bytes())),
                    MetaArray::I32(v) => w!(ty::I32, v, |o: &mut Vec<u8>, x: &i32| o.extend_from_slice(&x.to_le_bytes())),
                    MetaArray::F32(v) => w!(ty::F32, v, |o: &mut Vec<u8>, x: &f32| o.extend_from_slice(&x.to_le_bytes())),
                    MetaArray::Bool(v) => w!(ty::BOOL, v, |o: &mut Vec<u8>, x: &bool| o.push(*x as u8)),
                    MetaArray::String(v) => w!(ty::STRING, v, |o: &mut Vec<u8>, x: &Vec<u8>| s(o, x)),
                    MetaArray::U64(v) => w!(ty::U64, v, |o: &mut Vec<u8>, x: &u64| o.extend_from_slice(&x.to_le_bytes())),
                    MetaArray::I64(v) => w!(ty::I64, v, |o: &mut Vec<u8>, x: &i64| o.extend_from_slice(&x.to_le_bytes())),
                    MetaArray::F64(v) => w!(ty::F64, v, |o: &mut Vec<u8>, x: &f64| o.extend_from_slice(&x.to_le_bytes())),
                }
            }
        }
    }
    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    let mut align = 32u64;
    for (k, v) in entries {
        if k == "general.alignment" {
            if let MetaValue::U32(a) = v {
                align = *a as u64;
            }
        }
        s(&mut out, k.as_bytes());
        out.extend_from_slice(&tid(v).to_le_bytes());
        val(&mut out, v);
    }
    if align > 0 {
        let pad = (align - (out.len() as u64 % align)) % align;
        out.extend(std::iter::repeat(0u8).take(pad as usize));
    }
    out
}

/// Parse the typed metadata JSON export used by the handoff packet
/// (`{"key": {"type": "U32", "value": ...}, ...}`), keeping key order.
pub fn metadata_from_typed_json(text: &str) -> Result<Vec<(String, MetaValue)>, GgufError> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| GgufError(format!("metadata JSON: {e}")))?;
    let obj = v.as_object().ok_or_else(|| GgufError("metadata JSON must be an object".into()))?;
    let mut out = Vec::with_capacity(obj.len());
    for (k, tv) in obj {
        out.push((k.clone(), typed_json_value(k, tv)?));
    }
    Ok(out)
}

fn typed_json_value(key: &str, tv: &serde_json::Value) -> Result<MetaValue, GgufError> {
    let t = tv.get("type").and_then(|t| t.as_str()).ok_or_else(|| GgufError(format!("{key}: missing type")))?;
    let v = tv.get("value").ok_or_else(|| GgufError(format!("{key}: missing value")))?;
    let bad = || GgufError(format!("{key}: value does not match type {t}"));
    let int = |v: &serde_json::Value| v.as_i64().ok_or_else(bad);
    let uint = |v: &serde_json::Value| v.as_u64().ok_or_else(bad);
    let flt = |v: &serde_json::Value| v.as_f64().ok_or_else(bad);
    Ok(match t {
        "U8" => MetaValue::U8(u8::try_from(uint(v)?).map_err(|_| bad())?),
        "I8" => MetaValue::I8(i8::try_from(int(v)?).map_err(|_| bad())?),
        "U16" => MetaValue::U16(u16::try_from(uint(v)?).map_err(|_| bad())?),
        "I16" => MetaValue::I16(i16::try_from(int(v)?).map_err(|_| bad())?),
        "U32" => MetaValue::U32(u32::try_from(uint(v)?).map_err(|_| bad())?),
        "I32" => MetaValue::I32(i32::try_from(int(v)?).map_err(|_| bad())?),
        "U64" => MetaValue::U64(uint(v)?),
        "I64" => MetaValue::I64(int(v)?),
        "F32" => MetaValue::F32(flt(v)? as f32),
        "F64" => MetaValue::F64(flt(v)?),
        "Bool" => MetaValue::Bool(v.as_bool().ok_or_else(bad)?),
        "String" => MetaValue::String(v.as_str().ok_or_else(bad)?.as_bytes().to_vec()),
        "Array" => {
            let items = v.as_array().ok_or_else(bad)?;
            if items.is_empty() {
                return Err(GgufError(format!("{key}: empty array has no element type")));
            }
            let mut vals = Vec::with_capacity(items.len());
            for it in items {
                vals.push(typed_json_value(key, it)?);
            }
            MetaValue::Array(array_from_values(key, vals)?)
        }
        other => return Err(GgufError(format!("{key}: unknown type {other}"))),
    })
}

fn array_from_values(key: &str, vals: Vec<MetaValue>) -> Result<MetaArray, GgufError> {
    macro_rules! collect {
        ($variant:ident) => {{
            let mut out = Vec::with_capacity(vals.len());
            for v in vals {
                match v {
                    MetaValue::$variant(x) => out.push(x),
                    _ => return Err(GgufError(format!("{key}: mixed array element types"))),
                }
            }
            MetaArray::$variant(out)
        }};
    }
    Ok(match &vals[0] {
        MetaValue::U8(_) => collect!(U8),
        MetaValue::I8(_) => collect!(I8),
        MetaValue::U16(_) => collect!(U16),
        MetaValue::I16(_) => collect!(I16),
        MetaValue::U32(_) => collect!(U32),
        MetaValue::I32(_) => collect!(I32),
        MetaValue::F32(_) => collect!(F32),
        MetaValue::Bool(_) => collect!(Bool),
        MetaValue::String(_) => collect!(String),
        MetaValue::U64(_) => collect!(U64),
        MetaValue::I64(_) => collect!(I64),
        MetaValue::F64(_) => collect!(F64),
        MetaValue::Array(_) => return Err(GgufError(format!("{key}: nested arrays are not supported"))),
    })
}
