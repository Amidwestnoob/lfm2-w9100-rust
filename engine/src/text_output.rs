//! Append-only cleanup of StreamDecoder's UTF-8 chunks for LFM2 detokenization.
//! The accepted tokenizer's whole-sequence decoder applies three space-cleanup
//! passes; StreamDecoder deliberately emits raw pieces. Recompute those passes
//! on completed text and withhold a short suffix that future input can change.
//! Each pass has at most two bytes of lookahead and can change only its trailing
//! few bytes. Sixteen bytes also cover the UTF-8 boundary. No token is replayed
//! through inference, and no complete response is held until generation ends.
#[derive(Default)]
pub struct DecodedText {
    raw: String,
    emitted: String,
    clean_spaces: bool,
}
impl DecodedText {
    pub fn new(clean_spaces: bool) -> Self {
        Self {
            clean_spaces,
            ..Self::default()
        }
    }
    pub fn push(&mut self, chunk: &str) -> String {
        self.raw.push_str(chunk);
        self.take(false)
    }
    pub fn finish(&mut self, chunk: &str) -> String {
        self.raw.push_str(chunk);
        self.take(true)
    }
    fn take(&mut self, final_chunk: bool) -> String {
        let mut bytes = self.raw.as_bytes().to_vec();
        if self.clean_spaces {
            clean_spaces(&mut bytes);
        }
        // Cleanup only removes ASCII, so it preserves valid UTF-8.
        let text = String::from_utf8(bytes).expect("ASCII cleanup preserves UTF-8");
        let mut end = if final_chunk || !self.clean_spaces {
            text.len()
        } else {
            text.len().saturating_sub(16)
        };
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        assert!(
            text.starts_with(&self.emitted),
            "cleanup changed an already-emitted prefix"
        );
        end = end.max(self.emitted.len());
        let delta = text[self.emitted.len()..end].to_owned();
        self.emitted.push_str(&delta);
        delta
    }
}

// Same index arithmetic as the accepted tokenizer/src/vocab.rs clean_spaces,
// itself following llama.cpp's MIT detokenizer (tokenizer/LICENSE-llama.cpp).
// In-place behavior is intentional: replacing this with generic string replace
// would differ on overlapping punctuation/apostrophe inputs.
fn clean_spaces(text: &mut Vec<u8>) {
    let mut total = text.len();
    let total1 = total;
    total = usize::from(total > 0);
    for i in 1..total1 {
        let x = text[i];
        if text[i - 1] == b' ' && matches!(x, b'?' | b'!' | b'.' | b',') {
            total -= 1;
        }
        text[total] = x;
        total += 1;
    }
    let total2 = total;
    total = usize::from(total > 0);
    let mut i = 1;
    while i < total2 {
        let x = text[i];
        if x == b'\'' && i + 1 < total2 && text[i - 1] == b' ' && text[i + 1] == b' ' {
            total -= 1;
            i += 1;
            text[i] = 0;
        }
        text[total] = x;
        total += 1;
        i += 1;
    }
    let total3 = total;
    total = usize::from(total > 0);
    for i in 1..total3 {
        let x = text[i];
        if text[i - 1] == b' ' && x == b'\'' && i + 1 < total3 {
            let x1 = text[i + 1];
            if x1 == b't' || x1 == b'd' {
            } else if x1 == b's' || x1 == b'm' {
                total -= 1;
            } else if i + 2 < total3 {
                let x2 = text[i + 2];
                if x1 == b'l' && x2 == b'l' {
                } else if (x1 == b'r' && x2 == b'e') || (x1 == b'v' && x2 == b'e') {
                    total -= 1;
                }
            }
        }
        text[total] = x;
        total += 1;
    }
    text.truncate(total);
}
