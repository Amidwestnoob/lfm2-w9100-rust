//! Read-only GGUF tensor storage and shape/type access.
//!
//! Reuses the `parser` crate (piece 1) to decode the GGUF header/metadata and
//! tensor infos, then exposes byte-level access to tensor payloads by name so
//! the rest of the engine can dequantize/upload them without copying the whole
//! model into memory at once.

use anyhow::{anyhow, Context, Result};
use parser::gguf::{self, Value};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

/// A loaded tensor: name, shape (ne[0]..ne[3]), ggml type, and byte range.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// Dimensions in GGUF order: ne[0] is the contiguous (fastest) axis.
    pub ne: Vec<u64>,
    /// GGML quantization type id (0 = F32, 14 = Q6_K, …).
    pub ggml_type: u32,
    /// Absolute byte offset of the payload within the file.
    pub offset: u64,
    /// Payload size in bytes.
    pub nbytes: u64,
}

/// GGML type ids used by this model.
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_Q6_K: u32 = 14;

pub struct ModelFile {
    path: String,
    file: File,
    pub metadata: BTreeMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    pub data_offset: u64,
    pub alignment: u32,
    by_name: BTreeMap<String, usize>,
}

impl ModelFile {
    pub fn open(path: &str) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open model file {path}"))?;
        let file_len = file.metadata().with_context(|| "stat model file")?.len();
        let mut reader = BufReader::new(&mut file);
        let gguf = gguf::parse(&mut reader, file_len).with_context(|| "parse GGUF header")?;

        let mut by_name = BTreeMap::new();
        let mut tensors = Vec::with_capacity(gguf.tensors.len());
        for (i, t) in gguf.tensors.into_iter().enumerate() {
            by_name.insert(t.name.clone(), i);
            tensors.push(TensorInfo {
                name: t.name,
                ne: t.dimensions,
                ggml_type: t.ggml_type,
                offset: gguf.data_offset + t.offset,
                nbytes: t.nbytes,
            });
        }

        Ok(ModelFile {
            path: path.to_string(),
            file,
            metadata: gguf.metadata,
            by_name,
            tensors,
            data_offset: gguf.data_offset,
            alignment: gguf.alignment,
        })
    }

    pub fn tensor(&self, name: &str) -> Result<&TensorInfo> {
        self.by_name
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| anyhow!("missing tensor {name}"))
    }

    /// Number of elements in a tensor (product of dims).
    pub fn tensor_nelements(&self, name: &str) -> Result<u64> {
        let t = self.tensor(name)?;
        Ok(t.ne.iter().product())
    }

    /// Read the full byte payload of a tensor.
    pub fn read_tensor_bytes(&self, name: &str) -> Result<Vec<u8>> {
        let t = self.tensor(name)?;
        let mut buf = vec![0u8; t.nbytes as usize];
        let mut file = &self.file;
        file.seek(SeekFrom::Start(t.offset))
            .with_context(|| format!("seek tensor {name}"))?;
        file.read_exact(&mut buf)
            .with_context(|| format!("read tensor {name}"))?;
        Ok(buf)
    }

    /// Read directly into reusable mapped staging; GGUF offsets and payloads
    /// are unchanged. This avoids a zeroed host Vec and an additional copy.
    pub fn read_tensor_into(&self, name: &str, destination: &mut [u8]) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let t = self.tensor(name)?;
        anyhow::ensure!(
            destination.len() == t.nbytes as usize,
            "tensor destination length"
        );
        self.file.read_exact_at(destination, t.offset)?;
        Ok(())
    }

    /// Read a slice of a tensor's payload (offset + len in bytes).
    pub fn read_tensor_range(&self, name: &str, byte_off: u64, len: u64) -> Result<Vec<u8>> {
        let t = self.tensor(name)?;
        if byte_off + len > t.nbytes {
            return Err(anyhow!("range exceeds tensor {name}"));
        }
        let mut buf = vec![0u8; len as usize];
        let mut file = &self.file;
        file.seek(SeekFrom::Start(t.offset + byte_off))?;
        file.read_exact(&mut buf)
            .with_context(|| format!("read range of {name}"))?;
        Ok(buf)
    }

    pub fn file_path(&self) -> &str {
        &self.path
    }

    // -- typed metadata accessors -------------------------------------------

    pub fn meta_u32(&self, key: &str) -> Result<u32> {
        match self.metadata.get(key) {
            Some(Value::U32(v)) => Ok(*v),
            Some(Value::U64(v)) => Ok(*v as u32),
            Some(Value::I32(v)) => Ok(*v as u32),
            Some(Value::I64(v)) => Ok(*v as u32),
            Some(Value::U8(v)) => Ok(*v as u32),
            other => Err(anyhow!("metadata {key} not a u32: {other:?}")),
        }
    }

    pub fn meta_u32_default(&self, key: &str, def: u32) -> u32 {
        self.meta_u32(key).unwrap_or(def)
    }

    pub fn meta_f32(&self, key: &str) -> Result<f32> {
        match self.metadata.get(key) {
            Some(Value::F32(v)) => Ok(*v),
            Some(Value::F64(v)) => Ok(*v as f32),
            other => Err(anyhow!("metadata {key} not an f32: {other:?}")),
        }
    }

    pub fn meta_f32_default(&self, key: &str, def: f32) -> f32 {
        self.meta_f32(key).unwrap_or(def)
    }

    pub fn meta_str(&self, key: &str) -> Result<String> {
        match self.metadata.get(key) {
            Some(Value::String(v)) => Ok(v.clone()),
            other => Err(anyhow!("metadata {key} not a string: {other:?}")),
        }
    }

    /// A u32 array (e.g. `lfm2moe.attention.head_count_kv`).
    pub fn meta_u32_array(&self, key: &str) -> Result<Vec<u32>> {
        match self.metadata.get(key) {
            Some(Value::Array(vals)) => vals
                .iter()
                .map(|v| match v {
                    Value::U32(x) => Ok(*x),
                    Value::I32(x) => Ok(*x as u32),
                    Value::U16(x) => Ok(*x as u32),
                    Value::I16(x) => Ok(*x as u32),
                    Value::U8(x) => Ok(*x as u32),
                    Value::I8(x) => Ok(*x as u32),
                    other => Err(anyhow!("array element not int: {other:?}")),
                })
                .collect(),
            other => Err(anyhow!("metadata {key} not an array: {other:?}")),
        }
    }

    /// Whether a tensor payload actually lives on disk (vs zero-size).
    pub fn has_tensor(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }
}
