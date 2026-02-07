#[derive(Clone, Copy)]
pub enum LayoutType {
    Strided,
}

#[derive(Clone, Copy)]
pub enum DataType {
    F64,
    F32,
    F16,
    BF16,
    Q8,
    Q4,
}

/// Tensor is a view of flat buffer
/// Tensor contains the actual data and some metadata
/// associated with that representation of the data
/// The size and strides can help you access the data
pub struct Tensor<T> {
    /// Tensor can be 2D / 3D, therefore, using vector to represent the size
    sizes: Vec<usize>,
    /// Stride is the map of the tensor representation in the physical space
    /// It helps you actually address the data in the tensors
    strides: Vec<usize>,
    dtype: DataType,
    layout: LayoutType,
    data: Vec<T>,
}
