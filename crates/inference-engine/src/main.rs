use std::io::Error;

use model_parser::*;

fn main() -> Result<(), Error> {
    let model = Model::load_model("models/gpt_test_mini/model.safetensors");
    println!("{:?}", model);
    Ok(())
}
