//! Linear-feedback shift register (LFSR) sequences — the backbone of the
//! absolute position code (André et al. 2020 §II, 2021 §II-C).
//!
//! ## Why an LFSR
//!
//! A maximal-length LFSR of order `n` produces a sequence of `2ⁿ − 1` bits in
//! which **every window of `n` consecutive bits is unique**. That uniqueness is
//! exactly what makes position *absolute*: from any local view that captures `n`
//! bits, you can identify where in the whole sequence you are — i.e. which
//! period order `k` you are looking at. The pattern materializes this single
//! bit track along each axis.
//!
//! ## The "remove the all-ones word" rule
//!
//! A maximal LFSR never produces the all-zero state (it would be a fixed point),
//! so its sequence naturally contains every n-bit word *except* all-zeros. The
//! Vernier design additionally removes the all-**ones** word `2ⁿ − 1`, so that
//! every n-bit window contains at least one `0`. The missing periods of those
//! zeros act as an **embedded clock** that lets the decoder find coding-cell
//! boundaries. This module exposes both the raw maximal sequence and the
//! position→bit lookup the renderer needs.

/// A maximal-length LFSR over `n` bits with a given tap mask (feedback
/// polynomial). Generates the `2ⁿ − 1` bit sequence used as the absolute code
/// along one axis.
#[derive(Clone, Debug)]
pub struct Lfsr {
    /// Register order (bits per unique window).
    pub order: u32,
    /// The generated bit sequence, length `2^order - 1`.
    bits: Vec<u8>,
}

impl Lfsr {
    /// Builds a maximal-length sequence of the given `order` using a known
    /// primitive polynomial (tap set) for that order.
    ///
    /// Returns `None` if `order` is outside the small supported range (the
    /// taps table). Orders 3..=16 cover every pattern size of interest (a
    /// 12-bit code already yields an 11 cm target at 9 µm period).
    pub fn maximal(order: u32) -> Option<Self> {
        if (4..=12).contains(&order) {
            // C++ MegarenaBitSequence left-shift Fibonacci LFSR (orders 4-12).
            // Matches vernier/src/MegarenaBitSequence.cpp::generate() exactly.
            let code_max = 1u32 << order;
            let code_count = code_max - 1;
            let mut bits = Vec::with_capacity(code_count as usize);
            let mut code = code_count; // initial state: codeCount (all ones)
            bits.push(1u8); // bit[0] = 1 (hardcoded from all-ones initial state)
            for _ in 1..code_count {
                let nb = cpp_next_bit(order, code);
                code = (code * 2) % code_max + nb as u32;
                bits.push(nb);
            }
            Some(Self { order, bits })
        } else {
            // Galois LFSR fallback for orders outside the C++ megarena range.
            let taps = primitive_taps(order)?;
            let mask = (1u32 << order) - 1;
            let mut state = mask;
            let len = (1u32 << order) - 1;
            let mut bits = Vec::with_capacity(len as usize);
            for _ in 0..len {
                let out = (state & 1) as u8;
                bits.push(out);
                state >>= 1;
                if out == 1 {
                    state ^= taps;
                }
                state &= mask;
            }
            Some(Self { order, bits })
        }
    }

    /// The full bit sequence (length `2^order - 1`).
    pub fn bits(&self) -> &[u8] {
        &self.bits
    }

    /// Sequence length, `2^order - 1`.
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Whether the sequence is empty (never, for valid orders; for completeness).
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// The bit at code position `k`, wrapping if `k` exceeds the sequence
    /// length. Used by the renderer: cell index `k` along an axis selects bit
    /// `bit_at(k)`, which then decides present/absent central period.
    pub fn bit_at(&self, k: usize) -> u8 {
        self.bits[k % self.bits.len()]
    }

    /// Locates a window of `order` consecutive bits in the sequence, returning
    /// the starting position (the absolute order `k`) — or `None` if the window
    /// does not occur.
    ///
    /// This is the inverse of [`bit_at`](Lfsr::bit_at) over a window: the decoder
    /// reads `order` bits out of the image and asks "where in the sequence am I?"
    /// The maximal-LFSR uniqueness property guarantees the answer, if any, is
    /// unique. `window` must have length `order`.
    ///
    /// Implemented by scanning the cyclic sequence once per call. For repeated
    /// decoding, build a [`WindowIndex`] instead, which precomputes the lookup.
    pub fn locate(&self, window: &[u8]) -> Option<usize> {
        let n = self.order as usize;
        if window.len() != n {
            return None;
        }
        let len = self.bits.len();
        for start in 0..len {
            let matches = (0..n).all(|j| self.bits[(start + j) % len] == window[j]);
            if matches {
                return Some(start);
            }
        }
        None
    }

    /// Builds a precomputed window→position index for fast repeated localization.
    pub fn window_index(&self) -> WindowIndex {
        let n = self.order as usize;
        let len = self.bits.len();
        let mut map = std::collections::HashMap::with_capacity(len);
        for start in 0..len {
            let mut word = 0u32;
            for j in 0..n {
                word = (word << 1) | self.bits[(start + j) % len] as u32;
            }
            map.insert(word, start);
        }
        WindowIndex {
            order: self.order,
            map,
        }
    }
}

/// A precomputed map from `order`-bit window (packed MSB-first) to its starting
/// position in the LFSR sequence. Lets the decoder localize a window in O(1).
#[derive(Clone, Debug)]
pub struct WindowIndex {
    order: u32,
    map: std::collections::HashMap<u32, usize>,
}

impl WindowIndex {
    /// Locates a decoded bit window, returning its absolute position `k`.
    ///
    /// `window` must have length `order`; returns `None` if its length is wrong
    /// or the window is not a valid sequence window (e.g. decode errors produced
    /// a never-occurring pattern).
    pub fn locate(&self, window: &[u8]) -> Option<usize> {
        let n = self.order as usize;
        if window.len() != n {
            return None;
        }
        let mut word = 0u32;
        for &b in window {
            word = (word << 1) | (b & 1) as u32;
        }
        self.map.get(&word).copied()
    }
}

/// Feedback bit for the C++ Fibonacci left-shift LFSR at the given order.
/// Matches vernier/src/MegarenaBitSequence.cpp::nextBit().
fn cpp_next_bit(order: u32, state: u32) -> u8 {
    let b = |pos: u32| ((state >> pos) & 1) as u8;
    match order {
        4 => b(0) ^ b(3),
        5 => b(1) ^ b(4),
        6 => b(0) ^ b(5),
        7 => b(2) ^ b(6),
        8 => b(0) ^ b(1) ^ b(6) ^ b(7),
        9 => b(3) ^ b(8),
        10 => b(6) ^ b(9),
        11 => b(1) ^ b(10),
        12 => b(0) ^ b(1) ^ b(7) ^ b(11),
        _ => 0,
    }
}

/// Primitive-polynomial tap masks for small LFSR orders (Galois form).
///
/// Each value is the XOR mask applied on a 1-output. These are standard
/// maximal-length polynomials; the exact choice does not matter for the pattern
/// as long as it is primitive (maximal period), since the decoder is built from
/// the *same* sequence this generator produces.
fn primitive_taps(order: u32) -> Option<u32> {
    // Masks correspond to well-known primitive polynomials. Bit positions are
    // 0-indexed from the LSB end (Galois feedback into the high bit after shift).
    let taps = match order {
        3 => 0b110,               // x^3 + x^2 + 1
        4 => 0b1100,              // x^4 + x^3 + 1
        5 => 0b10100,             // x^5 + x^3 + 1
        6 => 0b110000,            // x^6 + x^5 + 1
        7 => 0b1100000,           // x^7 + x^6 + 1
        8 => 0b10111000,          // x^8 + x^6 + x^5 + x^4 + 1
        9 => 0b100010000,         // x^9 + x^5 + 1
        10 => 0b1001000000,       // x^10 + x^7 + 1
        11 => 0b10100000000,      // x^11 + x^9 + 1
        12 => 0b100000101001,     // x^12 + x^11 + x^8 + x^6 + 1
        13 => 0b1000000001101,    // x^13 + x^4 + x^3 + x^1 + 1
        14 => 0b10000000010101,   // x^14 + x^5 + x^3 + x^1 + 1
        15 => 0b110000000000000,  // x^15 + x^14 + 1
        16 => 0b1101000000001000, // x^16 + x^15 + x^13 + x^4 + 1
        _ => return None,
    };
    Some(taps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maximal_length_is_correct() {
        for order in 3..=12 {
            let l = Lfsr::maximal(order).unwrap();
            assert_eq!(l.len(), (1usize << order) - 1, "order {order}");
        }
    }

    #[test]
    fn every_n_window_is_unique() {
        // The defining property: all n-bit windows distinct (over the cyclic
        // sequence). This is what makes position absolute.
        let order = 6u32;
        let l = Lfsr::maximal(order).unwrap();
        let bits = l.bits();
        let n = order as usize;
        let mut seen = std::collections::HashSet::new();
        for start in 0..bits.len() {
            let mut word = 0u32;
            for j in 0..n {
                word = (word << 1) | bits[(start + j) % bits.len()] as u32;
            }
            assert!(seen.insert(word), "duplicate window at {start}");
        }
        // 2^n - 1 distinct nonzero windows expected.
        assert_eq!(seen.len(), (1usize << n) - 1);
    }

    #[test]
    fn sequence_excludes_all_zero_window() {
        // A maximal LFSR never yields the all-zero state, so no n-bit window is
        // all zeros — the complement of the embedded-clock property.
        let order = 5u32;
        let l = Lfsr::maximal(order).unwrap();
        let bits = l.bits();
        let n = order as usize;
        for start in 0..bits.len() {
            let all_zero = (0..n).all(|j| bits[(start + j) % bits.len()] == 0);
            assert!(!all_zero, "found all-zero window at {start}");
        }
    }

    #[test]
    fn locate_inverts_bit_at() {
        // The keystone decoder property: read a window starting at any position,
        // and `locate` recovers that position. This is exactly what the megarena
        // decoder does to turn decoded bits into an absolute order k.
        let order = 8u32;
        let l = Lfsr::maximal(order).unwrap();
        let n = order as usize;
        for start in 0..l.len() {
            let window: Vec<u8> = (0..n).map(|j| l.bit_at(start + j)).collect();
            assert_eq!(l.locate(&window), Some(start), "scan locate at {start}");
        }
    }

    #[test]
    fn window_index_matches_scan() {
        let order = 9u32;
        let l = Lfsr::maximal(order).unwrap();
        let idx = l.window_index();
        let n = order as usize;
        for start in 0..l.len() {
            let window: Vec<u8> = (0..n).map(|j| l.bit_at(start + j)).collect();
            assert_eq!(idx.locate(&window), Some(start), "index locate at {start}");
        }
    }

    #[test]
    fn locate_rejects_invalid_window() {
        let l = Lfsr::maximal(6).unwrap();
        // All-zero window never occurs in a maximal sequence.
        assert_eq!(l.locate(&[0, 0, 0, 0, 0, 0]), None);
        // Wrong length.
        assert_eq!(l.locate(&[1, 0, 1]), None);
    }
}
