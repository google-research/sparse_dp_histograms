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

use curve25519_dalek::ristretto::RistrettoPoint;
use curve25519_dalek::Scalar;
use rand::thread_rng;

/// Implementation of the Dodis-Yampolskiy PRF over the prime order group of
/// Curve25519 with Fq as message and key space where q is the group order.
pub struct DYPRF {}

impl DYPRF {
    /// Generate a key, i.e., a uniformly random group element.
    pub fn generate_key() -> Scalar {
        Scalar::random(&mut thread_rng())
    }

    /// Evaluate the PRF on the given `key` and `input`.
    pub fn evaluate(key: &Scalar, input: &Scalar) -> RistrettoPoint {
        let exponent = key + input;
        debug_assert_ne!(
            exponent,
            Scalar::ZERO,
            "happens only with negligible probability"
        );
        let exponent = exponent.invert();
        RistrettoPoint::mul_base(&exponent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prf() {
        let k = DYPRF::generate_key();
        // Check that the PRF is not trivially broken.
        let x1 = Scalar::from(42u32);
        let x2 = Scalar::from(1337u32);
        let y1 = DYPRF::evaluate(&k, &x1);
        let y2 = DYPRF::evaluate(&k, &x2);
        assert_ne!(y1, y2);
    }
}
