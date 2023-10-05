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

use bicycl::cl_context::CLContext;
use curve25519_dalek::Scalar;
use rug::integer::Order;
use rug::Integer;

/// Returns the order of the Ristretto group as a `rug::Integer`.
fn q_as_integer() -> Integer {
    Integer::from_str_radix(
        "1000000000000000000000000000000014def9dea2f79cd65812631a5cf5d3ed",
        16,
    )
    .expect("constant string should be parsed successfully")
}

/// Creates a `CLContext` with `M` being equal to the order of the Ristretto group.
pub fn make_cl_context() -> CLContext {
    CLContext::new(&q_as_integer(), 1, 128).expect("CL setup should not fail with these parameters")
}

/// Converts a `curve25519::Scalar` `x` into a `rug::Integer`.
pub fn scalar_to_rug_integer(x: &Scalar) -> Integer {
    Integer::from_digits(x.as_bytes(), Order::Lsf)
}

/// Converts a `rug::Integer` `x` into a `curve25519::Scalar`.
/// `x` must be in the range [0, q).
pub fn rug_integer_to_scalar(x: &Integer) -> Scalar {
    debug_assert!(!x.is_negative() && *x < q_as_integer());
    let mut bytes = [0u8; 32];
    x.write_digits(&mut bytes, Order::Lsf);
    Scalar::from_canonical_bytes(bytes).expect("bytes should be in the canonical encoding")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_cl_context() {
        let q = q_as_integer();
        let clctx = make_cl_context();
        assert_eq!(*clctx.get_M(), q);
        assert_eq!(*clctx.get_q(), q);
    }

    #[test]
    fn test_integer_scalar_conversion() {
        let q = q_as_integer();
        for x in [
            Integer::ZERO,
            Integer::ONE.clone(),
            Integer::from(42),
            Integer::from(23),
            Integer::from(1337),
            Integer::from(&q - 4747),
            Integer::from(&q - 1),
        ] {
            assert_eq!(scalar_to_rug_integer(&rug_integer_to_scalar(&x)), x);
        }
    }
}
