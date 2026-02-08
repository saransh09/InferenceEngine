use anyhow::{Error, Ok};
use half::{bf16, f16};
use std::str::FromStr;

#[derive(Debug, Clone, Copy)]
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

macro_rules! from_bytes_impl {
    ($dtype: expr, $bytes: expr, $($variant:ident => $rust_type:ty),*) => {
        match $dtype {
            $(
                DataType::$variant => {
                    let data: Vec<$rust_type> = $bytes
                        .chunks_exact(std::mem::size_of::<$rust_type>())
                        .map(|c| <$rust_type>::from_le_bytes(c.try_into().unwrap()))
                        .collect();
                        Ok(TensorStorage::$variant(data))
                }
            )*
            _ => Err(anyhow::anyhow!("Not implemented for dtype: {:?}", $dtype)),
        }
    };
}

macro_rules! zeros_impl {
    ($dtype: expr, $count: expr, $($variant:ident => $rust_type:ty),*) => {
        match $dtype {
            $(
                DataType::$variant => Ok(TensorStorage::$variant(vec![<$rust_type>::default(); $count])),
            )*
            _ => Err(anyhow::anyhow!("Not implemented for dtype: {:?}", $dtype)),
        }
    };
}

impl TensorStorage {
    pub fn dtype(&self) -> DataType {
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

    pub fn from_bytes(dtype: &DataType, bytes: &[u8]) -> Result<Self, Error> {
        if matches!(dtype, DataType::BOOL) {
            let data: Vec<bool> = bytes.iter().map(|&b| b != 0).collect();
            return Ok(TensorStorage::BOOL(data));
        }
        from_bytes_impl!(dtype, bytes,
            F64 => f64,
            F32 => f32,
            F16 => f16,
            BF16 => bf16,
            I64 => i64,
            I32 => i32,
            I16 => i16,
            I8 => i8,
            U8 => u8
        )
    }

    pub fn zeros(dtype: &DataType, count: usize) -> Result<Self, Error> {
        if matches!(dtype, DataType::BOOL) {
            return Ok(TensorStorage::BOOL(vec![false; count]));
        }
        zeros_impl!(dtype, count,
            F64 => f64,
            F32 => f32,
            F16 => f16,
            BF16 => bf16,
            I64 => i64,
            I32 => i32,
            I16 => i16,
            I8 => i8,
            U8 => u8
        )
    }

    pub fn len(&self) -> usize {
        match self {
            TensorStorage::F64(v) => v.len(),
            TensorStorage::F32(v) => v.len(),
            TensorStorage::F16(v) => v.len(),
            TensorStorage::BF16(v) => v.len(),
            TensorStorage::I64(v) => v.len(),
            TensorStorage::I32(v) => v.len(),
            TensorStorage::I16(v) => v.len(),
            TensorStorage::I8(v) => v.len(),
            TensorStorage::U8(v) => v.len(),
            TensorStorage::BOOL(v) => v.len(),
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
    shape: Vec<usize>,
    /// Stride is the map of the tensor representation in the physical space
    /// It helps you actually address the data in the tensors
    strides: Vec<usize>,
    layout: LayoutType,
    /// Model can have weights of different data types
    /// hence using an enum based pattern matching
    storage: TensorStorage,
}

/// It was getting too repetitive, so I asked Claude what to do
/// I was suggested this macro, so I added it here! will use it to learn
/// in the future, will try to write my own macro!
macro_rules! elementwise_op {
    ($a: expr, $b: expr, $op:tt, $($variant:ident),*) => {
        match ($a, $b) {
            $( (TensorStorage::$variant(a), TensorStorage::$variant(b)) => {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x $op *y;
                }
            } )*
            _ => return Err(anyhow::anyhow!("dtype mismatch or unsupported")),
        }
    };
}

macro_rules! elementwise_binary {
    ($a:expr, $b:expr, $op:tt, $($variant:ident),*) => {
        match ($a, $b) {
            $( (TensorStorage::$variant(a), TensorStorage::$variant(b)) => {
                TensorStorage::$variant(a.iter().zip(b.iter()).map(|(x,y)| x $op y).collect())
            } )*
            _ => return Err(anyhow::anyhow!("dtype mismatch or unsupported")),
        }
    };
}

impl Tensor {
    pub fn new(shape: Vec<usize>, layout: LayoutType, storage: TensorStorage) -> Self {
        let strides = Tensor::compute_strides(&shape);
        Self {
            shape,
            strides,
            layout,
            storage,
        }
    }

    fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
        shape
            .iter()
            .rev()
            .scan(1, |acc, &dim| {
                let stride = *acc;
                *acc *= dim;
                Some(stride)
            })
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }

    pub fn add_assign(&mut self, other: &Tensor) -> Result<(), Error> {
        if self.shape != other.shape {
            return Err(anyhow::anyhow!("Shape mismatch"));
        }
        match (&mut self.storage, &other.storage) {
            (TensorStorage::F16(a), TensorStorage::F16(b)) => {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x = f16::from_f32(x.to_f32() + y.to_f32());
                }
            }
            (TensorStorage::BF16(a), TensorStorage::BF16(b)) => {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x = bf16::from_f32(x.to_f32() + y.to_f32());
                }
            }
            _ => {
                elementwise_op!(&mut self.storage, &other.storage, +=, F64, F32, I64, I32, I16, I8, U8)
            }
        }
        Ok(())
    }

    pub fn add(&self, other: &Tensor) -> Result<Tensor, Error> {
        if self.shape != other.shape {
            return Err(anyhow::anyhow!("Shape mismatch"));
        }
        let storage = match (&self.storage, &other.storage) {
            (TensorStorage::F16(a), TensorStorage::F16(b)) => TensorStorage::F16(
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| f16::from_f32(x.to_f32() + y.to_f32()))
                    .collect(),
            ),
            (TensorStorage::BF16(a), TensorStorage::BF16(b)) => TensorStorage::BF16(
                a.iter()
                    .zip(b.iter())
                    .map(|(x, y)| bf16::from_f32(x.to_f32() + y.to_f32()))
                    .collect(),
            ),
            _ => {
                elementwise_binary!(&self.storage, &other.storage, +, F64, F32, I64, I32, I16, I8, U8)
            }
        };
        Ok(Tensor::new(
            self.shape.clone(),
            self.layout.clone(),
            storage,
        ))
    }
}
