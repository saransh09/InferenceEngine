pub mod model_parser;
pub mod tensor;

use std::io::Error;

use model_parser::*;

fn main() -> Result<(), Error> {
    let model = Model::from_safetensor("models/gpt_test_mini/model.safetensors");

    Ok(())
}
