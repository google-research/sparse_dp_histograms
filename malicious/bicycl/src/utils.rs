// Copyright 2023 Google LLC
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

use rand::{thread_rng, Rng};
use rug::integer::Order;
use rug::Integer;

/// Generate a uniformly random integer with a given number of bits.
pub fn random_integer_bits(num_bits: u32) -> Integer {
    let num_bytes = ((num_bits + 7) / 8) as usize;
    let mut buf = vec![0u8; num_bytes];
    thread_rng().fill(buf.as_mut_slice());

    // If num_bits is not a multiple of 8, count how many significant bits there are
    // in the last byte.
    let num_bits_in_incomplete_byte = num_bits % 8;
    if num_bits_in_incomplete_byte > 0 {
        let last_byte_mask = (1 << num_bits_in_incomplete_byte) - 1;
        buf[num_bytes - 1] &= last_byte_mask;
    }

    Integer::from_digits(&buf, Order::Lsf)
}

/// Generate a uniformly random integer modulo a given modulus with a given
/// statistical security parameter.
pub fn random_integer_below(modulus: &Integer, statistical_security: u32) -> Integer {
    let mut r = random_integer_bits(modulus.significant_bits() + statistical_security);
    r.modulo_mut(modulus);
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::ops::Pow;

    #[test]
    fn test_random_integer_bits() {
        for num_bits in 1..=32 {
            let r = random_integer_bits(num_bits);
            assert!(r < Integer::from(2).pow(num_bits));
            assert!(r >= 0);
        }

        {
            let num_bits = 1337;
            let r = random_integer_bits(num_bits);
            assert!(r < Integer::from(2).pow(num_bits));
            assert!(r >= 0);
            // Should hold except with probability 2^-80.
            assert!(r >= Integer::from(2).pow(num_bits - 80));
        }
    }

    #[test]
    fn test_random_integer_below() {
        let modulus: Integer = Integer::from(2).pow(127) - 1;
        let r = random_integer_below(&modulus, 40);
        assert!(r < modulus);
        assert!(r >= 0);
    }
}
