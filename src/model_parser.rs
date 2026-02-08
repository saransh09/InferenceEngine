use crate::tensor::{Tensor, TensorStorage};
use anyhow::{Error, Ok};
use std::{
    collections::HashMap,
    fs::File,
    io::{BufReader, Read},
};

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

pub struct Config {}

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
        self.tensors = self.parse_model(model_path)?;
        // self.config = self.parse_config();
        Ok(())
    }

    fn parse_model(&mut self, model_path: &str) -> Result<HashMap<String, Tensor>, Error> {
        let mut f = BufReader::new(File::open(model_path)?);

        // reading header_size
        let mut header_size_buf = [0u8; 8];
        f.read_exact(&mut header_size_buf)?;
        let header_size = u64::from_le_bytes(header_size_buf);

        // read header
        let mut header = vec![0u8; header_size as usize];
        f.read_exact(&mut header)?;
        let header_str = std::str::from_utf8(&header)?;
        let header_json: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)?;
        // println!("{:?}", header_json);

        for (name, val) in header_json.into_iter() {
            // Create the shape vector for the Tensor
            let json_shape = val.get("shape").unwrap().as_array().unwrap();
            let mut tensor_shape = Vec::new();
            for s in json_shape.into_iter() {
                tensor_shape.push(s.as_u64().unwrap() as usize);
            }

            let json_dtype = val.get("dtype").unwrap().as_str().unwrap();
            // let mut storage = TensorStorage

            break;
        }
        Ok(HashMap::new())
    }
}
