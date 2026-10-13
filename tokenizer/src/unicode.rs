//! Unicode helpers ported from llama.cpp `src/unicode.cpp` (pinned commit
//! c061df19838ff60970faf54fd7e414953590125d, MIT license).
//!
//! The functions keep the reference behavior byte for byte, including its
//! permissive UTF-8 decoder: invalid bytes become U+FFFD one byte at a time,
//! overlong forms and surrogate code points are decoded as they are, and
//! 4-byte forms can produce values above U+10FFFF. Text is handled as raw
//! bytes and `u32` code points, not as Rust `str`/`char`, for this reason.

use crate::unicode_data::{MAP_LOWERCASE, MAP_UPPERCASE, RANGES_FLAGS, RANGES_NFD, WHITESPACE};
use std::sync::OnceLock;

pub const MAX_CODEPOINTS: u32 = 0x110000;

pub const FLAG_UNDEFINED: u16 = 0x0001;
pub const FLAG_NUMBER: u16 = 0x0002;
pub const FLAG_LETTER: u16 = 0x0004;
pub const FLAG_SEPARATOR: u16 = 0x0008;
pub const FLAG_ACCENT_MARK: u16 = 0x0010;
pub const FLAG_PUNCTUATION: u16 = 0x0020;
pub const FLAG_SYMBOL: u16 = 0x0040;
pub const FLAG_CONTROL: u16 = 0x0080;
pub const FLAG_WHITESPACE: u16 = 0x0100;
pub const FLAG_LOWERCASE: u16 = 0x0200;
pub const FLAG_UPPERCASE: u16 = 0x0400;
pub const FLAG_NFD: u16 = 0x0800;

/// Code point flags, same bit layout as `unicode_cpt_flags::as_uint()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CptFlags(pub u16);

impl CptFlags {
    #[inline]
    pub fn is_number(self) -> bool {
        self.0 & FLAG_NUMBER != 0
    }
    #[inline]
    pub fn is_letter(self) -> bool {
        self.0 & FLAG_LETTER != 0
    }
    #[inline]
    pub fn is_whitespace(self) -> bool {
        self.0 & FLAG_WHITESPACE != 0
    }
    #[inline]
    pub fn is_separator(self) -> bool {
        self.0 & FLAG_SEPARATOR != 0
    }
    #[inline]
    pub fn is_control(self) -> bool {
        self.0 & FLAG_CONTROL != 0
    }
    #[inline]
    pub fn as_uint(self) -> u16 {
        self.0
    }
}

fn flags_table() -> &'static [u16] {
    static TABLE: OnceLock<Vec<u16>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = vec![FLAG_UNDEFINED; MAX_CODEPOINTS as usize];
        for w in RANGES_FLAGS.windows(2) {
            let (ini, flags) = w[0];
            let (end, _) = w[1];
            for cpt in ini..end {
                t[cpt as usize] = flags;
            }
        }
        for &cpt in WHITESPACE.iter() {
            t[cpt as usize] |= FLAG_WHITESPACE;
        }
        for &(_, lower) in MAP_LOWERCASE.iter() {
            t[lower as usize] |= FLAG_LOWERCASE;
        }
        for &(_, upper) in MAP_UPPERCASE.iter() {
            t[upper as usize] |= FLAG_UPPERCASE;
        }
        for &(_, _, nfd) in RANGES_NFD.iter() {
            t[nfd as usize] |= FLAG_NFD;
        }
        t
    })
}

/// `unicode_cpt_flags_from_cpt`: values outside the table are UNDEFINED.
#[inline]
pub fn cpt_flags(cpt: u32) -> CptFlags {
    let t = flags_table();
    if (cpt as usize) < t.len() {
        CptFlags(t[cpt as usize])
    } else {
        CptFlags(FLAG_UNDEFINED)
    }
}

/// `unicode_tolower`.
pub fn tolower(cpt: u32) -> u32 {
    match MAP_LOWERCASE.binary_search_by(|p| p.0.cmp(&cpt)) {
        Ok(i) => MAP_LOWERCASE[i].1,
        Err(_) => cpt,
    }
}

/// `unicode_len_utf8`: length implied by the lead byte's high nibble.
#[inline]
pub fn len_utf8(src: u8) -> usize {
    const LOOKUP: [usize; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LOOKUP[(src >> 4) as usize]
}

/// `unicode_cpt_from_utf8`: returns `None` where the reference throws.
fn cpt_from_utf8(utf8: &[u8], offset: &mut usize) -> Option<u32> {
    let o = *offset;
    let b0 = utf8[o] as u32;
    let cont = |i: usize| -> bool { o + i < utf8.len() && (utf8[o + i] & 0xc0) == 0x80 };
    if b0 & 0x80 == 0 {
        *offset += 1;
        return Some(b0);
    }
    if b0 & 0x40 == 0 {
        return None;
    }
    if b0 & 0x20 == 0 {
        if !cont(1) {
            return None;
        }
        *offset += 2;
        return Some(((b0 & 0x1f) << 6) | (utf8[o + 1] as u32 & 0x3f));
    }
    if b0 & 0x10 == 0 {
        if !(cont(1) && cont(2)) {
            return None;
        }
        *offset += 3;
        return Some(
            ((b0 & 0x0f) << 12) | ((utf8[o + 1] as u32 & 0x3f) << 6) | (utf8[o + 2] as u32 & 0x3f),
        );
    }
    if b0 & 0x08 == 0 {
        if !(cont(1) && cont(2) && cont(3)) {
            return None;
        }
        *offset += 4;
        return Some(
            ((b0 & 0x07) << 18)
                | ((utf8[o + 1] as u32 & 0x3f) << 12)
                | ((utf8[o + 2] as u32 & 0x3f) << 6)
                | (utf8[o + 3] as u32 & 0x3f),
        );
    }
    None
}

/// `unicode_cpts_from_utf8`: invalid input becomes U+FFFD, one byte at a time.
pub fn cpts_from_utf8(utf8: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(utf8.len());
    let mut offset = 0;
    while offset < utf8.len() {
        match cpt_from_utf8(utf8, &mut offset) {
            Some(c) => out.push(c),
            None => {
                offset += 1;
                out.push(0xFFFD);
            }
        }
    }
    out
}

/// `unicode_cpt_to_utf8`: `None` where the reference throws (above U+10FFFF).
/// Surrogate code points are encoded as 3 bytes, like the reference.
pub fn cpt_to_utf8(cpt: u32, out: &mut Vec<u8>) -> Option<()> {
    if cpt <= 0x7f {
        out.push(cpt as u8);
    } else if cpt <= 0x7ff {
        out.push(0xc0 | ((cpt >> 6) & 0x1f) as u8);
        out.push(0x80 | (cpt & 0x3f) as u8);
    } else if cpt <= 0xffff {
        out.push(0xe0 | ((cpt >> 12) & 0x0f) as u8);
        out.push(0x80 | ((cpt >> 6) & 0x3f) as u8);
        out.push(0x80 | (cpt & 0x3f) as u8);
    } else if cpt <= 0x10ffff {
        out.push(0xf0 | ((cpt >> 18) & 0x07) as u8);
        out.push(0x80 | ((cpt >> 12) & 0x3f) as u8);
        out.push(0x80 | ((cpt >> 6) & 0x3f) as u8);
        out.push(0x80 | (cpt & 0x3f) as u8);
    } else {
        return None;
    }
    Some(())
}

/// GPT-2 byte-to-unicode table: byte -> code point (`unicode_byte_to_utf8_map`).
pub fn byte_to_cpt_table() -> &'static [u32; 256] {
    static T: OnceLock<[u32; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let mut map = [u32::MAX; 256];
        for ch in 0x21..=0x7Eu32 {
            map[ch as usize] = ch;
        }
        for ch in 0xA1..=0xACu32 {
            map[ch as usize] = ch;
        }
        for ch in 0xAE..=0xFFu32 {
            map[ch as usize] = ch;
        }
        let mut n = 0;
        for ch in 0..256usize {
            if map[ch] == u32::MAX {
                map[ch] = 256 + n;
                n += 1;
            }
        }
        map
    })
}

/// Byte -> UTF-8 bytes of its GPT-2 code point (precomputed).
pub fn byte_to_utf8_table() -> &'static [Vec<u8>; 256] {
    static T: OnceLock<[Vec<u8>; 256]> = OnceLock::new();
    T.get_or_init(|| {
        let cpts = byte_to_cpt_table();
        std::array::from_fn(|b| {
            let mut v = Vec::with_capacity(2);
            cpt_to_utf8(cpts[b], &mut v).expect("table code points are valid");
            v
        })
    })
}

/// Inverse map: code point -> byte (`unicode_utf8_to_byte`), `None` if absent.
pub fn cpt_to_byte(cpt: u32) -> Option<u8> {
    static T: OnceLock<Vec<i16>> = OnceLock::new();
    let t = T.get_or_init(|| {
        let mut inv = vec![-1i16; 512];
        for (b, &c) in byte_to_cpt_table().iter().enumerate() {
            inv[c as usize] = b as i16;
        }
        inv
    });
    match t.get(cpt as usize) {
        Some(&b) if b >= 0 => Some(b as u8),
        _ => None,
    }
}

/// `unicode_regex_split_custom_llama3` over code points: returns word lengths
/// (in code points). Implements
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
/// with the exact control flow of the reference.
pub fn split_llama3(cpts: &[u32]) -> Vec<usize> {
    let mut out = Vec::with_capacity(cpts.len() / 3 + 1);
    let offset_ini = 0usize;
    let offset_end = cpts.len();
    const OUT_OF_RANGE: u32 = 0xFFFF_FFFF;
    let get_cpt = |pos: usize| -> u32 {
        if offset_ini <= pos && pos < offset_end {
            cpts[pos]
        } else {
            OUT_OF_RANGE
        }
    };
    let get_flags = |pos: usize| -> CptFlags {
        if offset_ini <= pos && pos < offset_end {
            cpt_flags(cpts[pos])
        } else {
            CptFlags(0)
        }
    };
    let mut prev_end = offset_ini;
    let mut add_token = |end: usize, out: &mut Vec<usize>| -> usize {
        let len = end - prev_end;
        if len > 0 {
            out.push(len);
        }
        prev_end = end;
        len
    };

    let mut pos = offset_ini;
    while pos < offset_end {
        let cpt = get_cpt(pos);
        let flags = get_flags(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if cpt == '\'' as u32 && pos + 1 < offset_end {
            let next = tolower(get_cpt(pos + 1));
            if next == 's' as u32 || next == 't' as u32 || next == 'm' as u32 || next == 'd' as u32 {
                pos += add_token(pos + 2, &mut out);
                continue;
            }
            if pos + 2 < offset_end {
                let next_next = tolower(get_cpt(pos + 2));
                if (next == 'r' as u32 && next_next == 'e' as u32)
                    || (next == 'v' as u32 && next_next == 'e' as u32)
                    || (next == 'l' as u32 && next_next == 'l' as u32)
                {
                    pos += add_token(pos + 3, &mut out);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?\p{L}+
        if !(cpt == '\r' as u32 || cpt == '\n' as u32 || flags.is_number())
            && (flags.is_letter() || get_flags(pos + 1).is_letter())
        {
            pos += 1;
            while get_flags(pos).is_letter() {
                pos += 1;
            }
            add_token(pos, &mut out);
            continue;
        }

        // \p{N}{1,3}
        if flags.is_number() {
            let mut ini = pos;
            while get_flags(pos).is_number() {
                pos += 1;
                if pos - ini >= 3 {
                    add_token(pos, &mut out);
                    ini = pos;
                }
            }
            add_token(pos, &mut out);
            continue;
        }

        // <space>?[^\s\p{L}\p{N}]+[\r\n]*
        let mut flags2 = if cpt == ' ' as u32 { get_flags(pos + 1) } else { flags };
        if !(flags2.is_whitespace() || flags2.is_letter() || flags2.is_number()) && flags.as_uint() != 0 {
            if cpt == ' ' as u32 {
                pos += 1;
            }
            while !(flags2.is_whitespace() || flags2.is_letter() || flags2.is_number())
                && flags2.as_uint() != 0
            {
                pos += 1;
                flags2 = get_flags(pos);
            }
            let mut cpt2 = get_cpt(pos);
            while cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                pos += 1;
                cpt2 = get_cpt(pos);
            }
            add_token(pos, &mut out);
            continue;
        }

        let mut num_whitespaces = 0usize;
        let mut last_end_r_or_n = 0usize;
        while get_flags(pos + num_whitespaces).is_whitespace() {
            let cpt2 = get_cpt(pos + num_whitespaces);
            if cpt2 == '\r' as u32 || cpt2 == '\n' as u32 {
                last_end_r_or_n = pos + num_whitespaces + 1;
            }
            num_whitespaces += 1;
        }

        // \s*[\r\n]+
        if last_end_r_or_n > 0 {
            pos = last_end_r_or_n;
            add_token(pos, &mut out);
            continue;
        }

        // \s+(?!\S)
        if num_whitespaces > 1 && get_cpt(pos + num_whitespaces) != OUT_OF_RANGE {
            pos += num_whitespaces - 1;
            add_token(pos, &mut out);
            continue;
        }

        // \s+
        if num_whitespaces > 0 {
            pos += num_whitespaces;
            add_token(pos, &mut out);
            continue;
        }

        // no matches
        pos += 1;
        add_token(pos, &mut out);
    }
    out
}

/// Python `str.isprintable()` approximation using the same Unicode table:
/// printable unless the code point is in \p{C} or \p{Z} (space excepted).
pub fn is_printable(cpt: u32) -> bool {
    if cpt == ' ' as u32 {
        return true;
    }
    let f = cpt_flags(cpt);
    !(f.is_control() || f.is_separator() || f.0 & FLAG_UNDEFINED != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_map_matches_gpt2_layout() {
        let t = byte_to_cpt_table();
        assert_eq!(t[b' ' as usize], 0x120); // 'Ġ'
        assert_eq!(t[b'\n' as usize], 0x10A); // 'Ċ'
        assert_eq!(t[b'A' as usize], b'A' as u32);
        for b in 0..=255u8 {
            assert_eq!(cpt_to_byte(t[b as usize]), Some(b));
        }
    }

    #[test]
    fn invalid_utf8_becomes_replacement_per_byte() {
        assert_eq!(cpts_from_utf8(&[0xff, b'a']), vec![0xFFFD, 'a' as u32]);
        assert_eq!(cpts_from_utf8(&[0xe2, 0x82]), vec![0xFFFD, 0xFFFD]);
        // overlong and surrogate forms are accepted, like the reference
        assert_eq!(cpts_from_utf8(&[0xc0, 0x80]), vec![0]);
        assert_eq!(cpts_from_utf8(&[0xed, 0xa0, 0x80]), vec![0xD800]);
    }

    #[test]
    fn llama3_split_basic() {
        let text = "Hello world's 12345  \n\nok";
        let cpts = cpts_from_utf8(text.as_bytes());
        let lens = split_llama3(&cpts);
        let mut words = vec![];
        let mut s = 0;
        for l in lens {
            words.push(cpts[s..s + l].iter().map(|&c| char::from_u32(c).unwrap()).collect::<String>());
            s += l;
        }
        assert_eq!(words, vec!["Hello", " world", "'s", " ", "123", "45", "  \n\n", "ok"]);
    }
}
