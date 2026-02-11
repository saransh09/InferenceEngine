use anyhow::{Error, Ok};
use half::{bf16, f16};
use std::{
    ops::{Add, AddAssign},
    str::FromStr,
    usize,
};

/// This is added based on the reference article
/// Apparently there can be different layout strategies
/// For now, only Strided will be a valid layout type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LayoutType {
    Strided,
}

#[derive(Clone, Debug, PartialEq)]
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
#[derive(Debug, PartialEq)]
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

macro_rules! matmul_impl {
    (
        $a:expr, $b:expr, $c:expr,
        $a_strides:expr, $b_strides:expr, $c_strides:expr,
        $batch_dims:expr, $total_batches:expr,
        $m:expr, $n:expr, $k:expr,
        $($variant:ident),*
    ) => {
        match ($a, $b, $c) {
            $( (TensorStorage::$variant(a), TensorStorage::$variant(b), TensorStorage::$variant(c)) => {
                for batch_idx in 0..$total_batches {
                    let batch_coords = Tensor::index_to_coords(batch_idx, $batch_dims);
                    let a_offset = Tensor::flat_index(&batch_coords, $a_strides).unwrap();
                    let b_offset = Tensor::flat_index(&batch_coords, $b_strides).unwrap();
                    let c_offset = Tensor::flat_index(&batch_coords, $c_strides).unwrap();
                    for i in 0..$m {
                        for j in 0..$n {
                            for p in 0..$k {
                                c[c_offset + i * $n + j] +=
                                    a[a_offset + i * $k + p] * b[b_offset + p * $n + j];
                            }
                        }
                    }
                }
            } )*
            _ => return Err(anyhow::anyhow!("dtype mismatch or unsupported")),
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

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub enum Scalar {
    Float(f64),
    Int(i64),
}

impl From<f64> for Scalar {
    fn from(v: f64) -> Self {
        Scalar::Float(v)
    }
}

impl From<f32> for Scalar {
    fn from(v: f32) -> Self {
        Scalar::Float(v as f64)
    }
}
impl From<f16> for Scalar {
    fn from(v: f16) -> Self {
        Scalar::Float(v.to_f64())
    }
}

impl From<bf16> for Scalar {
    fn from(v: bf16) -> Self {
        Scalar::Float(v.to_f64())
    }
}

impl From<i64> for Scalar {
    fn from(v: i64) -> Self {
        Scalar::Int(v)
    }
}

impl From<i32> for Scalar {
    fn from(v: i32) -> Self {
        Scalar::Int(v as i64)
    }
}

impl From<i16> for Scalar {
    fn from(v: i16) -> Self {
        Scalar::Int(v as i64)
    }
}

impl From<i8> for Scalar {
    fn from(v: i8) -> Self {
        Scalar::Int(v as i64)
    }
}

impl From<u8> for Scalar {
    fn from(v: u8) -> Self {
        Scalar::Int(v as i64)
    }
}

/// Tensor is a view of flat buffer
/// Tensor contains the actual data and some metadata
/// associated with that representation of the data
/// The size and strides can help you access the data
#[derive(Debug, PartialEq)]
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

    pub fn compute_strides(shape: &Vec<usize>) -> Vec<usize> {
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

    fn scalar_mul_f64(&self, scalar: f64) -> Result<Tensor, Error> {
        let storage = match &self.storage {
            TensorStorage::F64(v) => TensorStorage::F64(v.iter().map(|x| x * scalar).collect()),
            TensorStorage::F32(v) => {
                TensorStorage::F32(v.iter().map(|x| x * scalar as f32).collect())
            }
            TensorStorage::F16(v) => TensorStorage::F16(
                v.iter()
                    .map(|x| f16::from_f64(x.to_f64() * scalar))
                    .collect(),
            ),
            TensorStorage::BF16(v) => TensorStorage::BF16(
                v.iter()
                    .map(|x| bf16::from_f64(x.to_f64() * scalar))
                    .collect(),
            ),
            _ => {
                return Err(anyhow::anyhow!(
                    "Float scalar mul not supported for integer tensors"
                ));
            }
        };
        Ok(Tensor::new(self.shape.clone(), self.layout, storage))
    }

    fn scalar_mul_i64(&self, scalar: i64) -> Result<Tensor, Error> {
        let storage = match &self.storage {
            TensorStorage::I64(v) => TensorStorage::I64(v.iter().map(|x| x * scalar).collect()),
            TensorStorage::I32(v) => {
                TensorStorage::I32(v.iter().map(|x| x * scalar as i32).collect())
            }
            TensorStorage::I16(v) => {
                TensorStorage::I16(v.iter().map(|x| x * scalar as i16).collect())
            }
            TensorStorage::I8(v) => TensorStorage::I8(v.iter().map(|x| x * scalar as i8).collect()),
            TensorStorage::U8(v) => TensorStorage::U8(v.iter().map(|x| x * scalar as u8).collect()),
            _ => {
                return Err(anyhow::anyhow!(
                    "Int scalar mul not supported for float tensors"
                ));
            }
        };
        Ok(Tensor::new(self.shape.clone(), self.layout, storage))
    }

    pub fn scalar_mul(&self, scalar: impl Into<Scalar>) -> Result<Tensor, Error> {
        match scalar.into() {
            Scalar::Float(s) => self.scalar_mul_f64(s),
            Scalar::Int(s) => self.scalar_mul_i64(s),
        }
    }

    pub fn flat_index(coords: &Vec<usize>, strides: &[usize]) -> Result<usize, Error> {
        if strides.len() != coords.len() {
            return Err(anyhow::anyhow!(
                "Coords should have the same dimensions as the strides"
            ));
        }
        Ok(strides.iter().zip(coords.iter()).map(|(s, c)| s * c).sum())
    }

    pub fn index_to_coords(idx: usize, batch_dims: &[usize]) -> Vec<usize> {
        let mut flat_idx = idx;
        let mut batch_coords = Vec::<usize>::new();
        for i in (0..batch_dims.len()).rev() {
            batch_coords.push(flat_idx % batch_dims[i]);
            flat_idx /= batch_dims[i];
        }
        batch_coords.reverse();
        batch_coords
    }

    /// The tensor multiplication needed here is not the generalised
    /// tensor mutliplication (but in general Batch Matrix Multiplication)
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, Error> {
        // we say that, first we ensure that the shape of the tensor is of the same lengths
        if self.shape.len() != other.shape.len() {
            return Err(anyhow::anyhow!("Matrix shape should be in same dimension"));
        }
        // in reality (we just perform a 2D matrix operation but we loop over)
        // [B,H,M,K] @ [B,H,K,N] = [B,H,M,N]
        // for now not supporting broadcasting operation
        // In the next iteration, this has to be relaxed as broadcasting operation is required
        // in GPT models  Linear layers | [B,S,D] @ [D,D']
        for i in 0..self.shape.len() - 2 {
            if self.shape[i] != other.shape[i] {
                return Err(anyhow::anyhow!("The batch dimensions should be equal"));
            }
        }

        // The condition for 2D matrix multiplication should be met
        if self.shape[self.shape.len() - 1] != other.shape[self.shape.len() - 2] {
            return Err(anyhow::anyhow!("Tensor multiplication incompatability"));
        }

        if self.storage.dtype() != other.storage.dtype() {
            return Err(anyhow::anyhow!("The types of tensors should be the same"));
        }

        let m = self.shape[self.shape.len() - 2];
        let k = self.shape[self.shape.len() - 1];
        let n = other.shape[other.shape.len() - 1];
        let new_shape: Vec<usize> = self.shape[..self.shape.len() - 2]
            .iter()
            .copied()
            .chain([
                self.shape[self.shape.len() - 2],
                other.shape[other.shape.len() - 1],
            ])
            .collect();
        let new_storage_size = new_shape.iter().product::<usize>();
        let mut res = Tensor::new(
            new_shape,
            LayoutType::Strided,
            TensorStorage::zeros(&self.storage.dtype(), new_storage_size)?,
        );
        let batch_dims = &self.shape[..&self.shape.len() - 2];
        let total_batches = batch_dims.iter().product::<usize>();

        match (&self.storage, &other.storage, &mut res.storage) {
            (TensorStorage::F16(a), TensorStorage::F16(b), TensorStorage::F16(c)) => {
                for batch_idx in 0..total_batches {
                    let batch_coords = Tensor::index_to_coords(batch_idx, batch_dims);
                    let a_offset =
                        Tensor::flat_index(&batch_coords, &self.strides[..batch_dims.len()])?;
                    let b_offset =
                        Tensor::flat_index(&batch_coords, &other.strides[..batch_dims.len()])?;
                    let c_offet =
                        Tensor::flat_index(&batch_coords, &res.strides[..batch_dims.len()])?;

                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0_f32;
                            for p in 0..k {
                                sum += a[a_offset + i * k + p].to_f32()
                                    * b[b_offset + p * n + j].to_f32();
                            }
                            c[c_offet + i * n + j] = f16::from_f32(sum);
                        }
                    }
                }
            }
            (TensorStorage::BF16(a), TensorStorage::BF16(b), TensorStorage::BF16(c)) => {
                for batch_idx in 0..total_batches {
                    let batch_coords = Tensor::index_to_coords(batch_idx, batch_dims);
                    let a_offset =
                        Tensor::flat_index(&batch_coords, &self.strides[..batch_dims.len()])?;
                    let b_offset =
                        Tensor::flat_index(&batch_coords, &other.strides[..batch_dims.len()])?;
                    let c_offet =
                        Tensor::flat_index(&batch_coords, &res.strides[..batch_dims.len()])?;

                    for i in 0..m {
                        for j in 0..n {
                            let mut sum = 0.0_f32;
                            for p in 0..k {
                                sum += a[a_offset + i * k + p].to_f32()
                                    * b[b_offset + p * n + j].to_f32();
                            }
                            c[c_offet + i * n + j] = bf16::from_f32(sum);
                        }
                    }
                }
            }
            _ => matmul_impl!(
                &self.storage,
                &other.storage,
                &mut res.storage,
                &self.strides[..batch_dims.len()],
                &other.strides[..batch_dims.len()],
                &res.strides[..batch_dims.len()],
                batch_dims,
                total_batches,
                m,
                n,
                k,
                F64,
                F32,
                I64,
                I32,
                I16,
                I8
            ),
        }
        Ok(res)
    }

    /// For the Transpose, we just need to reverse the strides
    /// No need for data copying, this becomes an O(1) operation
    /// This is a zero cost view of the Tensor, which is alright
    /// However, during computations it might be worthwhile to
    /// recompute
    pub fn T(&mut self) {
        self.strides.reverse();
        self.shape.reverse();
    }

    pub fn transpose(&mut self) {
        self.T();
    }
}

impl AddAssign<&Tensor> for Tensor {
    fn add_assign(&mut self, rhs: &Tensor) {
        self.add_assign(rhs)
            .expect("Tensor add_assign failed: shape or dtype unmatch")
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: &Tensor) -> Tensor {
        Tensor::add(self, rhs).expect("Tensor add failed: shape of dtype unmatch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_strides_1d() {
        let shape = vec![5];
        assert_eq!(Tensor::compute_strides(&shape), vec![1]);
    }

    #[test]
    fn test_compute_stride_2d() {
        let shape = vec![3, 4];
        assert_eq!(Tensor::compute_strides(&shape), vec![4, 1]);
    }

    #[test]
    fn test_compute_stride_3d() {
        let shape = vec![2, 3, 4];
        assert_eq!(Tensor::compute_strides(&shape), vec![12, 4, 1]);
    }

    #[test]
    fn test_compute_stride_4d() {
        let shape = vec![2, 3, 4, 5];
        assert_eq!(Tensor::compute_strides(&shape), [60, 20, 5, 1]);
    }

    #[test]
    fn test_storage_len_f32() {
        assert_eq!(
            TensorStorage::F32(vec![
                1 as f32, 2 as f32, 3 as f32, 4 as f32, 5 as f32, 6 as f32
            ])
            .len(),
            6
        );
    }

    #[test]
    fn test_storage_dtype_f32() {
        assert_eq!(
            TensorStorage::F32(vec![
                1 as f32, 2 as f32, 3 as f32, 4 as f32, 5 as f32, 6 as f32
            ])
            .dtype(),
            DataType::F32
        )
    }

    #[test]
    fn test_from_bytes_f32() {
        let bytes: [u8; 12] = [
            0x00, 0x00, 0x80, 0x3F, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40,
        ];
        let v = TensorStorage::from_bytes(&DataType::F32, &bytes).unwrap();
        assert_eq!(v, TensorStorage::F32(vec![1.0_f32, 2.0_f32, 3.0_f32]));
    }

    #[test]
    fn test_from_bytes_i32() {
        let bytes: [u8; 12] = [
            0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        ];
        let v = TensorStorage::from_bytes(&DataType::I32, &bytes).unwrap();
        assert_eq!(v, TensorStorage::I32(vec![1_i32, 2_i32, 3_i32]));
    }

    #[test]
    fn test_zeros_f32() {
        let zero = TensorStorage::zeros(&DataType::F32, 10).unwrap();
        assert_eq!(zero, TensorStorage::F32(vec![0 as f32; 10]));
    }

    #[test]
    fn test_tensor_new() {
        let t = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::zeros(&DataType::F32, 24).unwrap(),
        );
        assert_eq!(t.shape, vec![2, 3, 4]);
        assert_eq!(t.layout, LayoutType::Strided);
        assert_eq!(t.strides, vec![12, 4, 1]);
        assert_eq!(t.storage, TensorStorage::F32(vec![0_f32; 24]));
    }

    #[test]
    fn test_is_empty() {
        let ts1 = TensorStorage::F32(vec![]);
        assert!(ts1.is_empty() == true);
        let ts2 = TensorStorage::F32(vec![1_f32; 10]);
        assert!(ts2.is_empty() == false);
    }

    #[test]
    fn test_add_assign_f32_same_shape() {
        let mut t1 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        let t2 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        t1.add_assign(&t2);
        assert_eq!(
            t1,
            Tensor::new(
                vec![2, 3, 4],
                LayoutType::Strided,
                TensorStorage::F32(vec![2_f32; 2 * 3 * 4]),
            )
        )
    }

    #[test]
    fn test_add_f32_same_shape() {
        let t1 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        let t2 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        let t3 = t1.add(&t2).unwrap();
        // original tensor unchanged
        assert_eq!(
            t1,
            Tensor::new(
                vec![2, 3, 4],
                LayoutType::Strided,
                TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
            )
        );
        // new tensor
        assert_eq!(
            t3,
            Tensor::new(
                vec![2, 3, 4],
                LayoutType::Strided,
                TensorStorage::F32(vec![2_f32; 2 * 3 * 4]),
            )
        )
    }

    #[test]
    fn test_add_bf16_same_shape() {
        let t1 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::BF16(vec![bf16::from_f32(1_f32); 2 * 3 * 4]),
        );
        let t2 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::BF16(vec![bf16::from_f32(1_f32); 2 * 3 * 4]),
        );
        let t3 = t1.add(&t2).unwrap();
        // original tensor unchanged
        assert_eq!(
            t1,
            Tensor::new(
                vec![2, 3, 4],
                LayoutType::Strided,
                TensorStorage::BF16(vec![bf16::from_f32(1_f32); 2 * 3 * 4]),
            )
        );
        // new tensor
        assert_eq!(
            t3,
            Tensor::new(
                vec![2, 3, 4],
                LayoutType::Strided,
                TensorStorage::BF16(vec![bf16::from_f32(2_f32); 2 * 3 * 4]),
            )
        )
    }

    #[test]
    fn test_add_shape_mismatch_error() {
        let t1 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        let t2 = Tensor::new(
            vec![2, 3, 5],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 5]),
        );
        let t3 = t1.add(&t2);
        assert!(t3.is_err());
        let err = t3.expect_err("Expected shape mismatch error");
        assert!(err.to_string().contains("Shape mismatch"));
    }

    #[test]
    fn test_add_dtype_mismatch_error() {
        let t1 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        let t2 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::I32(vec![1 as i32; 2 * 3 * 4]),
        );
        let t3 = t1.add(&t2);
        assert!(t3.is_err());
        let err = t3.expect_err("dtype mismatch");
        assert!(err.to_string().contains("dtype mismatch"));
    }

    #[test]
    fn test_add_assign_f32_same_shape_opertor() {
        let mut t1 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        let t2 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        t1 += &t2;
        assert_eq!(
            t1,
            Tensor::new(
                vec![2, 3, 4],
                LayoutType::Strided,
                TensorStorage::F32(vec![2_f32; 2 * 3 * 4]),
            )
        )
    }

    #[test]
    fn test_add_f32_same_shape_operator() {
        let t1 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        let t2 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        let t3 = &t1 + &t2;
        // original tensor unchanged
        assert_eq!(
            t1,
            Tensor::new(
                vec![2, 3, 4],
                LayoutType::Strided,
                TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
            )
        );
        // new tensor
        assert_eq!(
            t3,
            Tensor::new(
                vec![2, 3, 4],
                LayoutType::Strided,
                TensorStorage::F32(vec![2_f32; 2 * 3 * 4]),
            )
        )
    }

    #[test]
    fn test_mul_scalar_f32() {
        let t1 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        let t = t1.scalar_mul(4_f32).unwrap();
        assert_eq!(
            t,
            Tensor::new(
                vec![2, 3, 4],
                LayoutType::Strided,
                TensorStorage::F32(vec![4_f32; 2 * 3 * 4])
            )
        );
    }

    #[test]
    fn test_mul_scalar_i32() {
        let t1 = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::I32(vec![1_i32; 2 * 3 * 4]),
        );
        let t = t1.scalar_mul(4_i32).unwrap();
        assert_eq!(
            t,
            Tensor::new(
                vec![2, 3, 4],
                LayoutType::Strided,
                TensorStorage::I32(vec![4_i32; 2 * 3 * 4])
            )
        );
    }

    #[test]
    fn test_matrix_mul_2_d_compatible_shapes() {
        let t1 = Tensor::new(
            vec![2, 3],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64; 2 * 3]),
        );
        let t2 = Tensor::new(
            vec![3, 2],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64; 2 * 3]),
        );
        let matrix_product = t1.mul(&t2).unwrap();
        assert_eq!(matrix_product.shape, vec![2, 2]);
        assert_eq!(matrix_product.storage, TensorStorage::F64(vec![3_f64; 4]));

        let t3 = Tensor::new(
            vec![2, 2],
            LayoutType::Strided,
            TensorStorage::I64(vec![1_i64, 2_i64, 3_i64, 4_i64]),
        );
        let t4 = Tensor::new(
            vec![2, 2],
            LayoutType::Strided,
            TensorStorage::I64(vec![5_i64, 6_i64, 7_i64, 8_i64]),
        );
        let matrix_product_2 = t3.mul(&t4).unwrap();
        assert_eq!(matrix_product_2.shape, vec![2, 2]);
        assert_eq!(
            matrix_product_2.storage,
            TensorStorage::I64(vec![19_i64, 22_i64, 43_i64, 50_i64])
        );

        let t5 = Tensor::new(
            vec![2, 3],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64]),
        );
        let mut t6 = Tensor::new(
            vec![2, 3],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64]),
        );
        t6.T();
        let matrix_product_3 = t5.mul(&t6).unwrap();
        assert_eq!(matrix_product_3.shape, vec![2, 2]);
        assert_eq!(
            matrix_product_3.storage,
            TensorStorage::F64(vec![22_f64, 28_f64, 49_f64, 64_f64])
        );
    }

    #[test]
    fn test_matrix_mul_with_transpose() {
        let t1 = Tensor::new(
            vec![2, 3],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64, 6_f64]),
        );
        let mut t2 = Tensor::new(
            vec![2, 3],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64; 2 * 3]),
        );
        t2.T();
        let t3 = t1.mul(&t2).unwrap();
        assert_eq!(t3.shape, vec![2, 2]);
        assert_eq!(t3.storage, TensorStorage::F64(vec![3_f64; 4]));
    }

    #[test]
    fn test_matrix_mul_3_d_compatible_shapes() {
        let t1 = Tensor::new(
            vec![2, 2, 3],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64; 2 * 2 * 3]),
        );
        let t2 = Tensor::new(
            vec![2, 3, 2],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64; 2 * 2 * 3]),
        );
        let matrix_product = t1.mul(&t2).unwrap();
        assert_eq!(matrix_product.shape, vec![2, 2, 2]);
        assert_eq!(matrix_product.storage, TensorStorage::F64(vec![3_f64; 8]));
    }

    #[test]
    fn test_matrix_mul_4_d_compatible_shapes() {
        let t1 = Tensor::new(
            vec![3, 2, 2, 3],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64; 3 * 2 * 2 * 3]),
        );
        let t2 = Tensor::new(
            vec![3, 2, 3, 2],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64; 3 * 2 * 2 * 3]),
        );
        let matrix_product = t1.mul(&t2).unwrap();
        assert_eq!(matrix_product.shape, vec![3, 2, 2, 2]);
        assert_eq!(matrix_product.storage, TensorStorage::F64(vec![3_f64; 24]));
    }

    #[test]
    fn test_matrix_mul_incompatible_shape() {
        let t1 = Tensor::new(
            vec![2, 3],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64; 2 * 3]),
        );
        let t2 = Tensor::new(
            vec![2, 3],
            LayoutType::Strided,
            TensorStorage::F64(vec![1_f64; 2 * 3]),
        );
        let matrix_product = t1.mul(&t2);
        assert!(matrix_product.is_err());
        let err = matrix_product.expect_err("incompatible shape");
        assert!(
            err.to_string()
                .contains("Tensor multiplication incompatability")
        );
    }

    #[test]
    fn test_transpose_tensor() {
        let mut t = Tensor::new(
            vec![2, 3, 4],
            LayoutType::Strided,
            TensorStorage::F32(vec![1_f32; 2 * 3 * 4]),
        );
        t.T();
        assert_eq!(t.shape, vec![4, 3, 2]);
        assert_eq!(t.strides, vec![1, 4, 12]);
    }
}
