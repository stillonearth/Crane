use anyhow::Result;
use tokenizers::Tokenizer;

pub trait TokenStreamer {
    fn append(&mut self, token_id: u32) -> Result<()>;
    fn finalize(&mut self) -> Result<()>;
}

pub struct TextStreamer {
    pub tokenizer: Tokenizer,
    pub buffer: String,
}

impl TokenStreamer for TextStreamer {
    fn append(&mut self, token_id: u32) -> Result<()> {
        let token = self
            .tokenizer
            .decode(&[token_id], true)
            .expect("decode failed");
        self.buffer.push_str(&token);
        print!("{}", token);
        Ok(())
    }

    fn finalize(&mut self) -> Result<()> {
        println!();
        Ok(())
    }
}
