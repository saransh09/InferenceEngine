use anyhow::{Error, Ok};
/// SafeTensor format
/// |--8 bytes---|----------N Bytes------|------XYZXYZXYZXYZ--------|
///  (N = u64 int)         |                    ^_______    ^---|
///  size of header        |                            |       |
///                        |                 offsets : [BEGIN, END]<-----------|
///                        V                                                   |
///                 |--------------------------------------------------------| |
///                 |JSON utf- string represneting                           | |
///                 |the header                                              | |
///                 | {                                                      | |
///                 |   "Tensor_NAME_1" : {                                  | |
///                 |          "dtype": DATA_TYPE  // ex. "F16"              | |
///                 |          "shape": List<Integer> // ex [1, 16, 256]     | |
///                 |          "offsets": [BEGIN, END] // ex [457, 8576]-------|
///                 |   },                                                   |
///                 |   "Tensor_NAME_2" : (...),                             |
///                 |   ....                                                 |
///                 |   "__metadata__" : (...), // special key for storing   |
///                 |                             free form text to text map |
///                 | }                                                      |
///                 | // DATA_TYPE can be one of ["F64", "F32", "F16, "BF16  |
///                 |    "I64", "I32", "I16", "I8", "U8", "BOOL"]            |
///                 |________________________________________________________|
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
};

use crate::tensor::Tensor;

pub enum ModelType {
    SafeTensor,
}

// pub trait ModelFormat {
//     fn parse(path: &str) -> Result<HashMap<String, Tensor>, Error>;
// }

// pub struct ModelParser {
//     model_type: ModelType,
//     model_path: String,
// }

pub struct Config {}

pub struct Model {
    tensors: HashMap<String, Tensor>,
    config: Option<Config>,
}

impl Model {
    pub fn from_safetensor(model_path: &str) -> Result<Self, Error> {
        let tensors = Self::parse_model(model_path)?;
        Ok(Self {
            tensors,
            config: None,
        })
    }

    fn parse_model(model_path: &str) -> Result<HashMap<String, Tensor>, Error> {
        let mut f = BufReader::new(File::open(model_path)?);
        let mut header_size_buf = [0u8; 8];
        f.read_exact(&mut header_size_buf)?;
        let header_size = u64::from_le_bytes(header_size_buf);
        // for byte in f.bytes() {
        //     let header_size = byte?;
        // }
        println!("Header size is {}", header_size);
        Ok(HashMap::new())
    }
}
