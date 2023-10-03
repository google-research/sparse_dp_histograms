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

//! Providing a Rust representation of class group elements used by BICYCL.
//! These are represented by binary quadratic forms with negative discriminant.

use crate::error::Error;
use crate::mpz;
use rug::integer::BorrowInteger;
use rug::Integer;
use std::cmp::PartialEq;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Deref;

/// Represents a class group element as a binary quadratic form (a, b, c),
/// wrapping BICYCL's `QFI` class.
pub struct GroupElement {
    qfi: cxx::UniquePtr<ffi::QFI>,
}

impl GroupElement {
    pub(crate) fn from_raw(qfi: cxx::UniquePtr<ffi::QFI>) -> Self {
        Self { qfi }
    }

    pub(crate) fn as_raw(&self) -> &cxx::UniquePtr<ffi::QFI> {
        &self.qfi
    }

    pub(crate) fn as_mut_raw(&mut self) -> &mut cxx::UniquePtr<ffi::QFI> {
        &mut self.qfi
    }

    /// Create a new binary quadratic form from the given integers a, b, c.
    pub fn new_with_value(a: &Integer, b: &Integer, c: &Integer) -> Result<Self, Error> {
        unsafe {
            let mpz_a = mpz::rug_integer_to_bicycl_mpz_ref(a);
            let mpz_b = mpz::rug_integer_to_bicycl_mpz_ref(b);
            let mpz_c = mpz::rug_integer_to_bicycl_mpz_ref(c);
            Ok(Self {
                qfi: ffi::qfi_new_with_value(mpz_a.as_ref(), mpz_b.as_ref(), mpz_c.as_ref())?,
            })
        }
    }

    /// Check if this is an identity element (i.e., if it equals (1, b, c)).
    pub fn is_one(&self) -> bool {
        self.qfi.is_one()
    }

    /// Get the first coefficient a of the binary quadratic form.
    pub fn get_a(&self) -> BorrowInteger {
        unsafe { mpz::bicycl_mpz_to_rug_integer_ref(self.qfi.a()) }
    }

    /// Get the second coefficient b of the binary quadratic form.
    pub fn get_b(&self) -> BorrowInteger {
        unsafe { mpz::bicycl_mpz_to_rug_integer_ref(self.qfi.b()) }
    }

    /// Get the third coefficient c of the binary quadratic form.
    pub fn get_c(&self) -> BorrowInteger {
        unsafe { mpz::bicycl_mpz_to_rug_integer_ref(self.qfi.c()) }
    }

    /// Compute the discriminant of the binary quadratic form.
    pub fn compute_discriminant(&self) -> Integer {
        unsafe { mpz::bicycl_mpz_to_rug_integer(ffi::qfi_get_discriminant(&self.qfi)) }
    }

    pub fn eval(&self, x: &Integer, y: &Integer) -> Integer {
        unsafe {
            let mpz_x = mpz::rug_integer_to_bicycl_mpz_ref(x);
            let mpz_y = mpz::rug_integer_to_bicycl_mpz_ref(y);
            mpz::bicycl_mpz_to_rug_integer(ffi::qfi_eval(&self.qfi, mpz_x.as_ref(), mpz_y.as_ref()))
        }
    }
    /// Compute the inverse, i.e., negate the binary quadratic form.
    pub fn invert(&mut self) {
        ffi::qfi_neg(&self.qfi);
    }

    /// Compute a compressed representation.
    pub fn compress(&self) -> CompressedGroupElement {
        CompressedGroupElement::from_raw(ffi::qfi_to_compressed_repr(&self.qfi))
    }
}

impl fmt::Debug for GroupElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GE")
            .field("a", &self.get_a())
            .field("b", &self.get_b())
            .field("c", &self.get_c())
            .finish()
    }
}

impl Clone for GroupElement {
    fn clone(&self) -> Self {
        Self {
            qfi: ffi::qfi_clone(&self.qfi),
        }
    }
}

impl PartialEq for GroupElement {
    fn eq(&self, other: &Self) -> bool {
        ffi::qfi_eq(&self.qfi, &other.qfi)
    }
}

/// Struct acting as a reference to a `GroupElement`.
#[derive(Debug)]
pub struct GroupElementRef<'a> {
    ge: GroupElement,
    phantom_: PhantomData<&'a ()>,
}

impl<'a> From<&'a ffi::QFI> for GroupElementRef<'a> {
    fn from(qfi: &QFI) -> Self {
        Self {
            // Create a new group element that holds a UniquePtr to the QFI.
            // Since we do not own the QFI, we must make sure that we do not call free on this
            // pointer.
            ge: unsafe {
                GroupElement::from_raw(cxx::UniquePtr::from_raw(qfi as *const QFI as *mut QFI))
            },
            phantom_: PhantomData,
        }
    }
}

impl<'a> Drop for GroupElementRef<'a> {
    fn drop(&mut self) {
        // Exchange the QFI pointer with a null pointer.
        let mut p = cxx::UniquePtr::null();
        std::mem::swap(self.ge.as_mut_raw(), &mut p);
        // Release the original QFI raw pointer and forget it without freeing.
        p.into_raw();
    }
}

impl<'a> Deref for GroupElementRef<'a> {
    type Target = GroupElement;

    fn deref(&self) -> &Self::Target {
        &self.ge
    }
}

impl<'a> AsRef<GroupElement> for GroupElementRef<'a> {
    fn as_ref(&self) -> &GroupElement {
        &self.ge
    }
}

impl<'a> PartialEq for GroupElementRef<'a> {
    fn eq(&self, other: &Self) -> bool {
        ffi::qfi_eq(self.ge.as_raw(), other.ge.as_raw())
    }
}

/// Compressed representation of a binary quadratic form.  Can be decompressed
/// given the discriminant.
pub struct CompressedGroupElement {
    comp_qfi: cxx::UniquePtr<ffi::QFICompressedRepresentation>,
}

impl CompressedGroupElement {
    fn from_raw(comp_qfi: cxx::UniquePtr<ffi::QFICompressedRepresentation>) -> Self {
        Self { comp_qfi }
    }

    /// Reconstruct the binary quadratic form given the discriminant.
    pub fn decompress(&self, discriminant: &Integer) -> GroupElement {
        unsafe {
            let mpz_discriminant = mpz::rug_integer_to_bicycl_mpz_ref(discriminant);
            GroupElement::from_raw(ffi::qfi_from_compressed_repr(
                &self.comp_qfi,
                mpz_discriminant.as_ref(),
            ))
        }
    }
}

#[cxx::bridge]
mod ffi {

    unsafe extern "C++" {
        include!("bicycl/cxx/helpers.h");

        #[namespace = "BICYCL"]
        type Mpz = crate::mpz::Mpz;

        #[namespace = "BICYCL"]
        type QFI;
        fn is_one(self: &QFI) -> bool;
        fn a(self: &QFI) -> &Mpz;
        fn b(self: &QFI) -> &Mpz;
        fn c(self: &QFI) -> &Mpz;

        #[namespace = "bicycl_rs_helpers"]
        fn qfi_new_with_value(a: &Mpz, b: &Mpz, c: &Mpz) -> Result<UniquePtr<QFI>>;
        #[namespace = "bicycl_rs_helpers"]
        fn qfi_clone(v: &QFI) -> UniquePtr<QFI>;
        #[namespace = "bicycl_rs_helpers"]
        fn qfi_get_discriminant(qfi: &QFI) -> UniquePtr<Mpz>;
        #[namespace = "bicycl_rs_helpers"]
        fn qfi_eval(qfi: &QFI, x: &Mpz, y: &Mpz) -> UniquePtr<Mpz>;
        #[namespace = "bicycl_rs_helpers"]
        fn qfi_neg(qfi: &UniquePtr<QFI>);
        #[namespace = "bicycl_rs_helpers"]
        fn qfi_to_compressed_repr(qfi: &QFI) -> UniquePtr<QFICompressedRepresentation>;
        #[namespace = "bicycl_rs_helpers"]
        fn qfi_from_compressed_repr(
            qfi: &QFICompressedRepresentation,
            discriminant: &Mpz,
        ) -> UniquePtr<QFI>;
        #[namespace = "bicycl_rs_helpers"]
        fn qfi_eq(rhs: &QFI, lhs: &QFI) -> bool;

        #[namespace = "BICYCL"]
        type QFICompressedRepresentation;
        fn nbits(self: &QFICompressedRepresentation) -> usize;
    }
}

pub use ffi::QFI;

#[cfg(test)]
mod tests {
    use super::*;

    fn compute_discriminant(a: &Integer, b: &Integer, c: &Integer) -> Integer {
        b * b - Integer::from(4) * a * c
    }

    fn compute_evaluation(
        a: &Integer,
        b: &Integer,
        c: &Integer,
        x: &Integer,
        y: &Integer,
    ) -> Integer {
        // We need `Integer::from` here, since the multiplication of two `Integer`s
        // returns a not-yet-evaluated result, which must first be computed
        // (which happens in the `Integer::from` call) before we can multiply it
        // again.
        a * Integer::from(x * x) + b * Integer::from(x * y) + c * Integer::from(y * y)
    }

    fn test_ge_new(a: i32, b: i32, c: i32, one: bool) {
        let [a, b, c] = [Integer::from(a), Integer::from(b), Integer::from(c)];
        let [x, y] = [Integer::from(13), Integer::from(37)];
        let g = GroupElement::new_with_value(&a, &b, &c);
        assert!(g.is_ok());
        let g = g.unwrap();
        assert_eq!(g.is_one(), one);
        assert_eq!(*g.get_a(), a);
        assert_eq!(*g.get_b(), b);
        assert_eq!(*g.get_c(), c);
        assert_eq!(g.compute_discriminant(), compute_discriminant(&a, &b, &c));
        assert_eq!(g.eval(&x, &y), compute_evaluation(&a, &b, &c, &x, &y));

        let h = g.clone();
        assert_eq!(h, g);
    }

    #[test]
    fn test_ge_new_with_value_valid() {
        test_ge_new(1, 1, 1, true);
        test_ge_new(1, 0, 2, true);
        test_ge_new(5, -1, 27, false);
    }

    #[test]
    fn test_ge_new_with_value_invalid() {
        let [a, b, c] = [Integer::from(1), Integer::from(2), Integer::from(3)];
        let g = GroupElement::new_with_value(&a, &b, &c);
        assert!(g.is_err());
    }

    #[test]
    fn test_ge_ref() {
        let [a, b, c] = [Integer::from(5), Integer::from(-1), Integer::from(27)];
        let g = GroupElement::new_with_value(&a, &b, &c).unwrap();
        let g_ref = GroupElementRef::from(&**g.as_raw());
        assert_eq!(&g, g_ref.as_ref());
    }

    #[test]
    fn test_ge_compress() {
        let [a, b, c] = [Integer::from(5), Integer::from(-1), Integer::from(27)];
        let g = GroupElement::new_with_value(&a, &b, &c).unwrap();
        let c = g.compress();
        let h = c.decompress(&g.compute_discriminant());
        assert_eq!(h, g);
    }
}
