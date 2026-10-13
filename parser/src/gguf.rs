//! GGUF v3 parser for the W9100 Rust port (piece 1).
//!
//! Parses the header, metadata key-value pairs, and tensor infos from a
//! GGUF file. Does not decode tensor data or allocate whole-model buffers.
//! All multi-byte integers are little-endian per the GGUF specification.

use std::collections::BTreeMap;
use std::io::{Read, Seek};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GgufError {
    message: String,
}

impl GgufError {
    fn new(msg: impl Into<String>) -> Self {
        GgufError {
            message: msg.into(),
        }
    }
}

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for GgufError {}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Metadata value variants matching the GGUF value-type enum.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    U64(u64),
    I64(i64),
    F64(f64),
}

/// Tensor information extracted from the GGUF header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGUF order (ne[0], ne[1], …).
    pub dimensions: Vec<u64>,
    /// GGML quantization type (e.g. 0 = F32, 14 = Q6_K).
    pub ggml_type: u32,
    /// Offset relative to `data_offset`, exactly as stored in the file.
    pub offset: u64,
    /// Total byte size of the tensor data.
    pub nbytes: u64,
}

/// Parsed GGUF file.
#[derive(Debug)]
pub struct Gguf {
    pub version: u32,
    pub alignment: u32,
    pub data_offset: u64,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
}

// ---------------------------------------------------------------------------
// Low-level read helpers (all little-endian)
// ---------------------------------------------------------------------------

fn read_exact<R: Read>(reader: &mut R, buf: &mut [u8]) -> Result<(), GgufError> {
    use std::io::ErrorKind;
    match reader.read_exact(buf) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::UnexpectedEof => {
            Err(GgufError::new("Unexpected end of file (truncated input)"))
        }
        Err(e) => Err(GgufError::new(format!("IO error: {}", e))),
    }
}

fn read_u8<R: Read>(reader: &mut R) -> Result<u8, GgufError> {
    let mut b = [0u8; 1];
    read_exact(reader, &mut b)?;
    Ok(b[0])
}

fn read_i8<R: Read>(reader: &mut R) -> Result<i8, GgufError> {
    Ok(read_u8(reader)? as i8)
}

fn read_u16<R: Read>(reader: &mut R) -> Result<u16, GgufError> {
    let mut b = [0u8; 2];
    read_exact(reader, &mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_i16<R: Read>(reader: &mut R) -> Result<i16, GgufError> {
    let mut b = [0u8; 2];
    read_exact(reader, &mut b)?;
    Ok(i16::from_le_bytes(b))
}

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, GgufError> {
    let mut b = [0u8; 4];
    read_exact(reader, &mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_i32<R: Read>(reader: &mut R) -> Result<i32, GgufError> {
    let mut b = [0u8; 4];
    read_exact(reader, &mut b)?;
    Ok(i32::from_le_bytes(b))
}

fn read_u64<R: Read>(reader: &mut R) -> Result<u64, GgufError> {
    let mut b = [0u8; 8];
    read_exact(reader, &mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_i64<R: Read>(reader: &mut R) -> Result<i64, GgufError> {
    let mut b = [0u8; 8];
    read_exact(reader, &mut b)?;
    Ok(i64::from_le_bytes(b))
}

fn read_f32<R: Read>(reader: &mut R) -> Result<f32, GgufError> {
    let mut b = [0u8; 4];
    read_exact(reader, &mut b)?;
    Ok(f32::from_le_bytes(b))
}

fn read_f64<R: Read>(reader: &mut R) -> Result<f64, GgufError> {
    let mut b = [0u8; 8];
    read_exact(reader, &mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_string<R: Read>(reader: &mut R) -> Result<String, GgufError> {
    let len = read_u64(reader)?;
    // Bound allocation: a single string > 1 GiB is almost certainly corrupt.
    if len > 1_073_741_824 {
        return Err(GgufError::new(format!(
            "String length {} exceeds maximum (1 GiB)",
            len
        )));
    }
    let mut buf = vec![0u8; len as usize];
    read_exact(reader, &mut buf)?;
    String::from_utf8(buf).map_err(|e| GgufError::new(format!("Invalid UTF-8 in string: {}", e)))
}

// ---------------------------------------------------------------------------
// GGML type helpers
// ---------------------------------------------------------------------------

/// Returns `(block_size, bytes_per_block)` for a GGML quantization type.
/// Returns `None` for unknown / unsupported types.
fn ggml_type_block_info(ggml_type: u32) -> Option<(u64, u64)> {
    match ggml_type {
        0 => Some((1, 4)),      // F32
        1 => Some((1, 2)),      // F16
        2 => Some((32, 18)),    // Q4_0
        3 => Some((32, 20)),    // Q4_1
        6 => Some((32, 22)),    // Q5_0
        7 => Some((32, 24)),    // Q5_1
        8 => Some((32, 34)),    // Q8_0
        9 => Some((32, 36)),    // Q8_1
        10 => Some((256, 84)),  // Q2_K
        11 => Some((256, 112)), // Q3_K
        12 => Some((256, 144)), // Q4_K
        13 => Some((256, 176)), // Q5_K
        14 => Some((256, 210)), // Q6_K
        15 => Some((256, 288)), // Q8_K
        _ => None,
    }
}

/// Compute the total byte size of a tensor from its GGML type and dimensions.
fn compute_tensor_nbytes(ggml_type: u32, dimensions: &[u64]) -> Result<u64, GgufError> {
    let (block_size, bytes_per_block) = ggml_type_block_info(ggml_type)
        .ok_or_else(|| GgufError::new(format!("Unknown GGML type: {}", ggml_type)))?;

    // Product of all dimensions = total element count.
    let mut num_elements: u64 = 1;
    for &d in dimensions {
        if d == 0 {
            return Ok(0); // empty tensor
        }
        num_elements = num_elements
            .checked_mul(d)
            .ok_or_else(|| GgufError::new("Dimension product overflow"))?;
    }

    if num_elements == 0 {
        return Ok(0);
    }

    // Quantized types require the element count to be a multiple of block_size.
    if block_size > 1 && num_elements % block_size != 0 {
        return Err(GgufError::new(format!(
            "Tensor has {} elements, not a multiple of block size {} for GGML type {}",
            num_elements, block_size, ggml_type
        )));
    }

    let num_blocks = num_elements / block_size;
    num_blocks
        .checked_mul(bytes_per_block)
        .ok_or_else(|| GgufError::new("Tensor size overflow"))
}

// ---------------------------------------------------------------------------
// Alignment helper
// ---------------------------------------------------------------------------

fn align_up(offset: u64, alignment: u32) -> u64 {
    if alignment == 0 {
        return offset;
    }
    let mask = (alignment as u64).wrapping_sub(1);
    (offset.wrapping_add(mask)) & !mask
}

fn is_power_of_two(n: u32) -> bool {
    n != 0 && (n & (n.wrapping_sub(1))) == 0
}

// ---------------------------------------------------------------------------
// Metadata value reader
// ---------------------------------------------------------------------------

fn read_metadata_value<R: Read>(reader: &mut R, value_type: u32) -> Result<Value, GgufError> {
    match value_type {
        0 => Ok(Value::U8(read_u8(reader)?)),
        1 => Ok(Value::I8(read_i8(reader)?)),
        2 => Ok(Value::U16(read_u16(reader)?)),
        3 => Ok(Value::I16(read_i16(reader)?)),
        4 => Ok(Value::U32(read_u32(reader)?)),
        5 => Ok(Value::I32(read_i32(reader)?)),
        6 => Ok(Value::F32(read_f32(reader)?)),
        7 => {
            let v = read_u8(reader)?;
            if v > 1 {
                return Err(GgufError::new(format!("Invalid BOOL value: {}", v)));
            }
            Ok(Value::Bool(v != 0))
        }
        8 => Ok(Value::String(read_string(reader)?)),
        9 => {
            // ARRAY: u32 element_type, then u64 length, then elements.
            // On-disk GGUF order is (element_type, count), per the GGUF spec
            // (see gguf-py Reader._get_field_parts). The earlier parser and its
            // fixtures had these two reversed.
            let elem_type = read_u32(reader)?;
            let len = read_u64(reader)?;
            // Bound allocation: > 10 M elements is almost certainly corrupt.
            if len > 10_000_000 {
                return Err(GgufError::new(format!(
                    "Array length {} exceeds maximum (10 M)",
                    len
                )));
            }
            let mut elems = Vec::with_capacity(len.min(4096) as usize);
            for _ in 0..len {
                elems.push(read_metadata_value(reader, elem_type)?);
            }
            Ok(Value::Array(elems))
        }
        10 => Ok(Value::U64(read_u64(reader)?)),
        11 => Ok(Value::I64(read_i64(reader)?)),
        12 => Ok(Value::F64(read_f64(reader)?)),
        _ => Err(GgufError::new(format!(
            "Unknown metadata value type: {}",
            value_type
        ))),
    }
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

/// Parse a GGUF v3 file from `reader`, which must be positioned at the start
/// of the GGUF data. `file_len` is the total file size in bytes.
///
/// Returns a [`Gguf`] with metadata and tensor infos, or a [`GgufError`]
/// describing the first problem encountered.
pub fn parse<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<Gguf, GgufError> {
    // --- Header ---
    let mut magic = [0u8; 4];
    read_exact(reader, &mut magic)?;
    if &magic != b"GGUF" {
        return Err(GgufError::new(format!(
            "Invalid magic: {:?} (expected \"GGUF\")",
            String::from_utf8_lossy(&magic)
        )));
    }

    let version = read_u32(reader)?;
    if version != 3 {
        return Err(GgufError::new(format!(
            "Unsupported GGUF version: {} (expected 3)",
            version
        )));
    }

    let tensor_count = read_u64(reader)?;
    let kv_count = read_u64(reader)?;

    // Bound counts: > 100 K tensors or KVs is almost certainly corrupt.
    const MAX_COUNT: u64 = 100_000;
    if tensor_count > MAX_COUNT {
        return Err(GgufError::new(format!(
            "Tensor count {} exceeds maximum ({})",
            tensor_count, MAX_COUNT
        )));
    }
    if kv_count > MAX_COUNT {
        return Err(GgufError::new(format!(
            "KV count {} exceeds maximum ({})",
            kv_count, MAX_COUNT
        )));
    }

    // --- Metadata key-value pairs ---
    let mut metadata: BTreeMap<String, Value> = BTreeMap::new();
    for _ in 0..kv_count {
        let key = read_string(reader)?;
        let value_type = read_u32(reader)?;
        let value = read_metadata_value(reader, value_type)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(GgufError::new(format!("Duplicate metadata key: {}", key)));
        }
    }

    // --- Alignment ---
    // The alignment is stored as "general.alignment" (U32) in the metadata.
    // Default to 32 if absent.
    let alignment = match metadata.get("general.alignment") {
        Some(Value::U32(v)) => *v,
        _ => 32,
    };
    if !is_power_of_two(alignment) {
        return Err(GgufError::new(format!(
            "Invalid alignment: {} (must be a power of two)",
            alignment
        )));
    }

    // --- Tensor infos ---
    let mut tensors: Vec<TensorInfo> = Vec::with_capacity(tensor_count.min(4096) as usize);
    for _ in 0..tensor_count {
        let name = read_string(reader)?;
        let n_dims = read_u32(reader)?;
        if n_dims > 8 {
            return Err(GgufError::new(format!(
                "Too many dimensions: {} (max 8)",
                n_dims
            )));
        }
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dimensions.push(read_u64(reader)?);
        }
        let ggml_type = read_u32(reader)?;
        let offset = read_u64(reader)?;

        let nbytes = compute_tensor_nbytes(ggml_type, &dimensions)?;

        // Check for duplicate tensor names.
        if tensors.iter().any(|t: &TensorInfo| t.name == name) {
            return Err(GgufError::new(format!("Duplicate tensor name: {}", name)));
        }

        tensors.push(TensorInfo {
            name,
            dimensions,
            ggml_type,
            offset,
            nbytes,
        });
    }

    // --- Compute data_offset ---
    let current_pos = reader
        .stream_position()
        .map_err(|e| GgufError::new(format!("IO error: {}", e)))?;
    let data_offset = align_up(current_pos, alignment);

    if data_offset > file_len {
        return Err(GgufError::new(format!(
            "Data offset {} exceeds file length {}",
            data_offset, file_len
        )));
    }

    // --- Validate tensor data fits within the file ---
    let data_size = file_len - data_offset;
    for t in &tensors {
        if t.offset > data_size {
            return Err(GgufError::new(format!(
                "Tensor \"{}\" offset {} exceeds data size {}",
                t.name, t.offset, data_size
            )));
        }
        let end = t.offset.checked_add(t.nbytes).ok_or_else(|| {
            GgufError::new(format!("Tensor \"{}\" offset+nbytes overflow", t.name))
        })?;
        if end > data_size {
            return Err(GgufError::new(format!(
                "Tensor \"{}\" end {} exceeds data size {}",
                t.name, end, data_size
            )));
        }
    }

    Ok(Gguf {
        version,
        alignment,
        data_offset,
        metadata,
        tensors,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // -- helpers --

    fn u32(v: u32) -> Value {
        Value::U32(v)
    }
    fn string(s: &str) -> Value {
        Value::String(s.to_string())
    }

    /// Build a minimal valid GGUF byte buffer.
    fn build_gguf(
        version: u32,
        alignment: u32,
        metadata: &[(String, Value)],
        tensors: &[(String, Vec<u64>, u32, u64)],
    ) -> Vec<u8> {
        let mut buf = Vec::new();
        // magic
        buf.extend_from_slice(b"GGUF");
        // version
        buf.extend_from_slice(&version.to_le_bytes());
        // tensor_count
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        // kv_count
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        // metadata
        for (k, v) in metadata {
            buf.extend_from_slice(&(k.len() as u64).to_le_bytes());
            buf.extend_from_slice(k.as_bytes());
            let (type_id, val_bytes) = value_to_bytes(v);
            buf.extend_from_slice(&type_id.to_le_bytes());
            buf.extend_from_slice(&val_bytes);
        }
        // tensor infos
        for (name, dims, ggml_type, offset) in tensors {
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&ggml_type.to_le_bytes());
            buf.extend_from_slice(&offset.to_le_bytes());
        }
        // pad to alignment
        let pos = buf.len();
        let padded = align_up(pos as u64, alignment) as usize;
        buf.resize(padded, 0);
        let data_offset = padded as u64;
        // append enough zero bytes to cover all tensor data
        let max_tensor_end = tensors
            .iter()
            .map(|(_, dims, ggml_type, offset)| {
                let nbytes = compute_tensor_nbytes(*ggml_type, dims).unwrap_or(0);
                *offset + nbytes
            })
            .max()
            .unwrap_or(0);
        let total_needed = (data_offset + max_tensor_end) as usize;
        if buf.len() < total_needed {
            buf.resize(total_needed, 0);
        }
        buf
    }

    fn value_to_bytes(v: &Value) -> (u32, Vec<u8>) {
        match v {
            Value::U8(x) => (0, vec![*x]),
            Value::I8(x) => (1, vec![*x as u8]),
            Value::U16(x) => (2, x.to_le_bytes().to_vec()),
            Value::I16(x) => (3, x.to_le_bytes().to_vec()),
            Value::U32(x) => (4, x.to_le_bytes().to_vec()),
            Value::I32(x) => (5, x.to_le_bytes().to_vec()),
            Value::F32(x) => (6, x.to_le_bytes().to_vec()),
            Value::Bool(x) => (7, vec![*x as u8]),
            Value::String(s) => {
                let mut b = Vec::new();
                b.extend_from_slice(&(s.len() as u64).to_le_bytes());
                b.extend_from_slice(s.as_bytes());
                (8, b)
            }
            Value::Array(vals) => {
                let mut b = Vec::new();
                // element type (u32) comes FIRST, then count (u64).
                let et = if let Some(v0) = vals.first() {
                    value_type_id(v0)
                } else {
                    0
                };
                b.extend_from_slice(&et.to_le_bytes());
                b.extend_from_slice(&(vals.len() as u64).to_le_bytes());
                for v in vals {
                    b.extend_from_slice(&value_to_bytes(v).1);
                }
                (9, b)
            }
            Value::U64(x) => (10, x.to_le_bytes().to_vec()),
            Value::I64(x) => (11, x.to_le_bytes().to_vec()),
            Value::F64(x) => (12, x.to_le_bytes().to_vec()),
        }
    }

    fn value_type_id(v: &Value) -> u32 {
        match v {
            Value::U8(_) => 0,
            Value::I8(_) => 1,
            Value::U16(_) => 2,
            Value::I16(_) => 3,
            Value::U32(_) => 4,
            Value::I32(_) => 5,
            Value::F32(_) => 6,
            Value::Bool(_) => 7,
            Value::String(_) => 8,
            Value::Array(_) => 9,
            Value::U64(_) => 10,
            Value::I64(_) => 11,
            Value::F64(_) => 12,
        }
    }

    // -- tests --

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(31, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(0, 1), 0);
        assert_eq!(align_up(5, 1), 5);
    }

    #[test]
    fn test_is_power_of_two() {
        assert!(is_power_of_two(1));
        assert!(is_power_of_two(2));
        assert!(is_power_of_two(32));
        assert!(is_power_of_two(1024));
        assert!(!is_power_of_two(0));
        assert!(!is_power_of_two(3));
        assert!(!is_power_of_two(100));
    }

    #[test]
    fn test_compute_nbytes_f32() {
        // F32: 4 bytes/element, block_size=1
        assert_eq!(compute_tensor_nbytes(0, &[2048]).unwrap(), 8192);
        assert_eq!(
            compute_tensor_nbytes(0, &[2048, 128000]).unwrap(),
            1048576000
        );
        assert_eq!(compute_tensor_nbytes(0, &[0]).unwrap(), 0);
        assert_eq!(compute_tensor_nbytes(0, &[10, 0, 5]).unwrap(), 0);
    }

    #[test]
    fn test_compute_nbytes_f16() {
        // F16: 2 bytes/element, block_size=1
        assert_eq!(compute_tensor_nbytes(1, &[2048]).unwrap(), 4096);
    }

    #[test]
    fn test_compute_nbytes_q6_k() {
        // Q6_K: 256 elements/block, 210 bytes/block
        // 2048 elements = 8 blocks * 210 = 1680
        assert_eq!(compute_tensor_nbytes(14, &[2048]).unwrap(), 1680);
        // 2048 * 128000 = 262144000 elements / 256 = 1024000 blocks * 210 = 215040000
        assert_eq!(
            compute_tensor_nbytes(14, &[2048, 128000]).unwrap(),
            215040000
        );
    }

    #[test]
    fn test_compute_nbytes_q6_k_bad_multiple() {
        // 100 is not a multiple of 256
        assert!(compute_tensor_nbytes(14, &[100]).is_err());
    }

    #[test]
    fn test_compute_nbytes_unknown_type() {
        assert!(compute_tensor_nbytes(999, &[10]).is_err());
    }

    #[test]
    fn test_parse_minimal() {
        let metadata = vec![
            ("general.architecture".to_string(), string("test")),
            ("general.alignment".to_string(), u32(32)),
        ];
        let tensors = vec![
            ("tensor_a".to_string(), vec![2048], 0, 0),
            ("tensor_b".to_string(), vec![2048, 128000], 14, 8192),
        ];
        let buf = build_gguf(3, 32, &metadata, &tensors);
        let mut cursor = Cursor::new(&buf);
        let gguf = parse(&mut cursor, buf.len() as u64).unwrap();

        assert_eq!(gguf.version, 3);
        assert_eq!(gguf.alignment, 32);
        assert_eq!(gguf.metadata.len(), 2);
        assert_eq!(gguf.tensors.len(), 2);

        assert_eq!(gguf.tensors[0].name, "tensor_a");
        assert_eq!(gguf.tensors[0].dimensions, vec![2048]);
        assert_eq!(gguf.tensors[0].ggml_type, 0);
        assert_eq!(gguf.tensors[0].nbytes, 8192);

        assert_eq!(gguf.tensors[1].name, "tensor_b");
        assert_eq!(gguf.tensors[1].dimensions, vec![2048, 128000]);
        assert_eq!(gguf.tensors[1].ggml_type, 14);
        assert_eq!(gguf.tensors[1].nbytes, 215040000);
    }

    #[test]
    fn test_parse_invalid_magic() {
        let buf = b"XXXX".to_vec();
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }

    #[test]
    fn test_parse_invalid_version() {
        let buf = build_gguf(2, 32, &[], &[]);
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }

    #[test]
    fn test_parse_invalid_alignment() {
        let metadata = vec![("general.alignment".to_string(), u32(33))];
        let buf = build_gguf(3, 33, &metadata, &[]);
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }

    #[test]
    fn test_parse_duplicate_metadata_key() {
        let metadata = vec![
            ("key".to_string(), string("a")),
            ("key".to_string(), string("b")),
        ];
        let buf = build_gguf(3, 32, &metadata, &[]);
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }

    #[test]
    fn test_parse_duplicate_tensor_name() {
        let tensors = vec![
            ("t".to_string(), vec![10], 0, 0),
            ("t".to_string(), vec![20], 0, 40),
        ];
        let buf = build_gguf(3, 32, &[], &tensors);
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }

    #[test]
    fn test_parse_unknown_ggml_type() {
        let tensors = vec![("t".to_string(), vec![10], 999, 0)];
        let buf = build_gguf(3, 32, &[], &tensors);
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }

    #[test]
    fn test_parse_truncated_header() {
        let buf = b"GGUF\x03\x00\x00\x00".to_vec();
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }

    #[test]
    fn test_parse_empty_tensor() {
        let tensors = vec![("empty".to_string(), vec![0, 2048], 0, 0)];
        let buf = build_gguf(3, 32, &[], &tensors);
        let mut cursor = Cursor::new(&buf);
        let gguf = parse(&mut cursor, buf.len() as u64).unwrap();
        assert_eq!(gguf.tensors[0].nbytes, 0);
    }

    #[test]
    fn test_parse_array_metadata() {
        let metadata = vec![(
            "test.array".to_string(),
            Value::Array(vec![u32(1), u32(2), u32(3)]),
        )];
        let buf = build_gguf(3, 32, &metadata, &[]);
        let mut cursor = Cursor::new(&buf);
        let gguf = parse(&mut cursor, buf.len() as u64).unwrap();
        match gguf.metadata.get("test.array") {
            Some(Value::Array(vals)) => {
                assert_eq!(vals.len(), 3);
                assert_eq!(vals[0], u32(1));
                assert_eq!(vals[1], u32(2));
                assert_eq!(vals[2], u32(3));
            }
            _ => panic!("Expected Array value"),
        }
    }

    #[test]
    fn test_parse_bool_metadata() {
        let metadata = vec![("test.bool".to_string(), Value::Bool(true))];
        let buf = build_gguf(3, 32, &metadata, &[]);
        let mut cursor = Cursor::new(&buf);
        let gguf = parse(&mut cursor, buf.len() as u64).unwrap();
        assert_eq!(gguf.metadata.get("test.bool"), Some(&Value::Bool(true)));
    }

    #[test]
    fn test_parse_default_alignment() {
        // No "general.alignment" in metadata -> default 32
        let metadata: Vec<(String, Value)> = vec![];
        let buf = build_gguf(3, 32, &metadata, &[]);
        let mut cursor = Cursor::new(&buf);
        let gguf = parse(&mut cursor, buf.len() as u64).unwrap();
        assert_eq!(gguf.alignment, 32);
    }

    #[test]
    fn test_parse_nondefault_alignment() {
        let metadata = vec![("general.alignment".to_string(), u32(64))];
        let buf = build_gguf(3, 64, &metadata, &[]);
        let mut cursor = Cursor::new(&buf);
        let gguf = parse(&mut cursor, buf.len() as u64).unwrap();
        assert_eq!(gguf.alignment, 64);
    }

    #[test]
    fn test_parse_out_of_file_storage() {
        // Build a minimal GGUF with a tensor that claims more data than the file contains.
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count
                                                    // tensor: name="big", dims=[1000000], type=F32, offset=0
        buf.extend_from_slice(&3u64.to_le_bytes()); // name length
        buf.extend_from_slice(b"big");
        buf.extend_from_slice(&1u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&1000000u64.to_le_bytes()); // dim[0]
        buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type = F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset
                                                    // pad to alignment (32)
        let pos = buf.len();
        let padded = align_up(pos as u64, 32) as usize;
        buf.resize(padded, 0);
        // file_len is just the header size, no room for tensor data
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }

    #[test]
    fn test_parse_overflowing_dimensions() {
        // u64::MAX * 2 overflows
        let tensors = vec![("huge".to_string(), vec![u64::MAX, 2], 0, 0)];
        let buf = build_gguf(3, 32, &[], &tensors);
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }

    #[test]
    fn test_parse_unknown_metadata_value_type() {
        // Manually craft a buffer with an unknown metadata value type (99)
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // kv_count
                                                    // one KV: key="k", value_type=99
        buf.extend_from_slice(&1u64.to_le_bytes()); // key length
        buf.extend_from_slice(b"k");
        buf.extend_from_slice(&99u32.to_le_bytes()); // unknown value type
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }

    #[test]
    fn test_parse_invalid_bool_value() {
        // BOOL with value 2 is invalid
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes());
        buf.extend_from_slice(b"k");
        buf.extend_from_slice(&7u32.to_le_bytes()); // BOOL type
        buf.push(2); // invalid bool value
        let mut cursor = Cursor::new(&buf);
        assert!(parse(&mut cursor, buf.len() as u64).is_err());
    }
}
