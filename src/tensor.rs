use anyhow::{Error, Ok, anyhow};
use half::{bf16, f16};
use std::str::FromStr;

#[derive(Debug)]
pub enum LayoutType {
    Strided,
}

#[derive(Debug)]
pub enum DataType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    BOOL,
}

impl FromStr for DataType {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "F64" => Ok(DataType::F64),
            "F32" => Ok(DataType::F32),
            "F16" => Ok(DataType::F16),
            "BF16" => Ok(DataType::BF16),
            "I64" => Ok(DataType::I64),
            "I32" => Ok(DataType::I32),
            "I16" => Ok(DataType::I16),
            "I8" => Ok(DataType::I8),
            "U8" => Ok(DataType::U8),
            "BOOL" => Ok(DataType::BOOL),
            _ => Err(anyhow::anyhow!("Unknown dtype: {}", s)),
        }
    }
}

/// Enum store for the different possible datatypes
/// that can occur in
#[derive(Debug)]
pub enum TensorStorage {
    F64(Vec<f64>),
    F32(Vec<f32>),
    F16(Vec<f16>),
    BF16(Vec<bf16>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    I16(Vec<i16>),
    I8(Vec<i8>),
    U8(Vec<u8>),
    BOOL(Vec<bool>),
}

impl TensorStorage {
    fn dtype(&self) -> DataType {
        match self {
            TensorStorage::F64(_) => DataType::F64,
            TensorStorage::F32(_) => DataType::F32,
            TensorStorage::F16(_) => DataType::F16,
            TensorStorage::BF16(_) => DataType::BF16,
            TensorStorage::I64(_) => DataType::I64,
            TensorStorage::I32(_) => DataType::I32,
            TensorStorage::I16(_) => DataType::I16,
            TensorStorage::I8(_) => DataType::I8,
            TensorStorage::U8(_) => DataType::U8,
            TensorStorage::BOOL(_) => DataType::BOOL,
        }
    }

    fn from_bytes(dtype: &DataType, bytes: &[u8]) -> Result<Self, Error> {
        match dtype {
            DataType::F32 => {
                let data: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                Ok(TensorStorage::F32(data))
            }
            _ => Err(anyhow::anyhow!("Not implemented for dtype: {:?}", dtype)),
        }
    }
}

/// Tensor is a view of flat buffer
/// Tensor contains the actual data and some metadata
/// associated with that representation of the data
/// The size and strides can help you access the data
#[derive(Debug)]
pub struct Tensor {
    /// Tensor can be 2D / 3D, therefore, using vector to represent the size
    sizes: Vec<usize>,
    /// Stride is the map of the tensor representation in the physical space
    /// It helps you actually address the data in the tensors
    strides: Vec<usize>,
    layout: LayoutType,
    /// Model can have weights of different data types
    /// hence using an enum based pattern matching
    storage: TensorStorage,
}
