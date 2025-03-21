pub mod based;
pub mod streamer;

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub repetition_penalty: f32,
    pub repeat_last_n: usize,
    pub do_sample: bool,
    pub pad_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 120,
            temperature: Some(1.0),
            top_p: Some(1.0),
            repetition_penalty: 1.0,
            repeat_last_n: 5,
            do_sample: false,
            pad_token_id: None,
            eos_token_id: None,
        }
    }
}

impl GenerationConfig {
    pub fn with_max_tokens(max: usize) -> Self {
        Self {
            max_new_tokens: max,
            ..Default::default()
        }
    }
}
