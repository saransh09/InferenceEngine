use half::{bf16, f16};

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
