use anyhow::{Error, Ok};
use memmap2::Mmap;
use std::{collections::HashMap, fs::File, str::FromStr};
use tensor::{DataType, Tensor, TensorStorage};

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

pub enum ModelType {
    SafeTensor,
}

#[derive(Debug)]
pub struct Config {}

#[derive(Debug)]
pub struct Model {
    tensors: HashMap<String, Tensor>,
    config: Option<Config>,
}

impl Model {
    pub fn new() -> Self {
        Model {
            tensors: HashMap::new(),
            config: None,
        }
    }

    pub fn from_safetensor(&mut self, model_path: &str) -> Result<(), Error> {
        self.parse_model(model_path)?;
        // self.config = self.parse_config();
        Ok(())
    }

    fn parse_model(&mut self, model_path: &str) -> Result<(), Error> {
        let file = File::open(model_path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        // reading header_size
        let header_size = u64::from_le_bytes(mmap[0..8].try_into()?) as usize;

        // read header
        let header_str = std::str::from_utf8(&mmap[8..8 + header_size])?;
        let header_json: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)?;

        let data_offset = 8 + header_size;
        for (name, val) in header_json.into_iter() {
            if name == "__metadata__" {
                continue;
            }
            let shape: Vec<usize> = val
                .get("shape")
                .unwrap()
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s.as_u64().unwrap() as usize)
                .collect();

            let dtype = DataType::from_str(val.get("dtype").unwrap().as_str().unwrap())?;

            let offsets = val.get("data_offsets").unwrap().as_array().unwrap();
            let begin = offsets[0].as_u64().unwrap() as usize;
            let end = offsets[1].as_u64().unwrap() as usize;

            let tensor_bytes = &mmap[data_offset + begin..data_offset + end];
            let storage = TensorStorage::from_bytes(&dtype, tensor_bytes)?;

            self.tensors.insert(
                name,
                Tensor::new(shape, tensor::LayoutType::Strided, storage),
            );
        }
        Ok(())
    }
}
