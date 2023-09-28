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

//! BICYCL includes a custom class `Mpz` that wraps the GMP `mpz_t` type.  This
//! module provides interoperatbility between `Mpz` and `Integer`s from the rug
//! library which are also based on the GMP library.

use cxx::{type_id, ExternType};
use gmp_mpfr_sys::gmp;
use rug::integer::BorrowInteger;
use rug::Integer;
use std::marker::PhantomData;

/// Wrapper around `gmp::mpz_ptr` allowing us to implement the `ExternType`
/// trait to map the `gmp::mpz_ptr` directly to the GMP `mpz_ptr` used in the
/// BICYCL C++ library.
#[allow(non_camel_case_types)]
pub struct MpzPtrWrapper(pub gmp::mpz_ptr);

unsafe impl ExternType for MpzPtrWrapper {
    type Id = type_id!("mpz_ptr");
    type Kind = cxx::kind::Trivial;
}

/// Wrapper around `gmp::mpz_srcptr` allowing us to implement the `ExternType`
/// trait to map the `gmp::mpz_srcptr` directly to the GMP `mpz_srcptr` used in
/// the BICYCL C++ library.
#[allow(non_camel_case_types)]
pub struct MpzSrcPtrWrapper(pub gmp::mpz_srcptr);

unsafe impl ExternType for MpzSrcPtrWrapper {
    type Id = type_id!("mpz_srcptr");
    type Kind = cxx::kind::Trivial;
}

/// Create initialized GMP `mpz_t`.
fn create_initialized_mpz_t() -> gmp::mpz_t {
    use core::mem::MaybeUninit;
    unsafe {
        let mut z = MaybeUninit::uninit();
        gmp::mpz_init(z.as_mut_ptr());
        z.assume_init()
    }
}

/// Convert a BICYCL `Mpz` into a rug `Integer` holding the same underlying
/// `mpz_t`.
pub unsafe fn bicycl_mpz_to_rug_integer(v: cxx::UniquePtr<ffi::Mpz>) -> Integer {
    // Create a new rug Interger containing a default-initialized mpz_t.
    let mut out = Integer::new();
    // Convert the UniqePtr into a raw pointer s.t. we can pass it to the FFI
    // interface.
    let raw_v = v.into_raw();
    // Swap the new mpz_t from the Integer with the one inside the Mpz.
    gmp::mpz_swap(out.as_raw_mut(), ffi::mpz_get_raw_mpz_mut(raw_v).0);
    // Put the raw pointer back into a UniqePtr s.t. it gets freed and the new mpz_t
    // gets cleared.
    cxx::UniquePtr::from_raw(raw_v);
    // Return the rug::Integer that now contains the mpz_t that was previously in
    // the Mpz.
    out
}

/// Convert a rug `Integer` into a BICYCL `Mpz` holding the same underlying
/// `mpz_t`.
pub unsafe fn rug_integer_to_bicycl_mpz(mut v: Integer) -> cxx::UniquePtr<ffi::Mpz> {
    // Create a new `Mpz` (containing a default-initialized `mpz_t`) and transform
    // it into a raw pointer.
    let raw_ptr_to_new_mpz = ffi::mpz_new().into_raw();
    // Swap the new mpz_t from the Mpz with the one inside the Integer.
    gmp::mpz_swap(
        v.as_raw_mut(),
        ffi::mpz_get_raw_mpz_mut(raw_ptr_to_new_mpz).0,
    );
    // Put the raw pointer to the new Mpz back into a UniqePtr.
    unsafe { cxx::UniquePtr::from_raw(raw_ptr_to_new_mpz) }
}

/// Create a rug `BorrowInteger` that references the same underlying `mpz_t` as
/// the given reference to a BICYCL `Mpz`.
pub unsafe fn bicycl_mpz_to_rug_integer_ref(v: &ffi::Mpz) -> BorrowInteger<'_> {
    // Create the `BorrowInteger` by extracting the pointer to the `mpz_t` contained
    // in the `Mpz`.
    BorrowInteger::from_raw(*ffi::mpz_get_raw_mpz(v).0)
}

/// Struct that acts as a reference to an `Mpz` instance.
///
/// Actually contains an owned `Mpz` instance, but makes sure not to clear the
/// underlying `mpz_t` when dropped.
pub struct MpzRef<'a> {
    mpz_: cxx::UniquePtr<ffi::Mpz>,
    phantom_: PhantomData<&'a ()>,
}

impl<'a> Drop for MpzRef<'a> {
    fn drop(&mut self) {
        // Release the `mpz_t` contained in the `Mpz` and reset the latter such that it
        // is still valid. Hence, the original `mpz_t` will not be deallocated
        // when the `Mpz` is destroyed.

        // Create a null pointer to temporarily put into `self`.
        let mut tmp_ptr = cxx::UniquePtr::null();
        std::mem::swap(&mut self.mpz_, &mut tmp_ptr);
        // Extract the raw pointer from the UniquePtr.
        let raw_ptr_to_mpz = tmp_ptr.into_raw();
        // Create a new mpz_t.
        let mut fresh_raw_mpz = create_initialized_mpz_t();
        unsafe {
            // Swap that mpz_t with the one in the Mpz.
            gmp::mpz_swap(
                &mut fresh_raw_mpz,
                ffi::mpz_get_raw_mpz_mut(raw_ptr_to_mpz).0,
            );
            // Put the raw pointer back into to UniquePtr.
            self.mpz_ = cxx::UniquePtr::from_raw(raw_ptr_to_mpz);
        }
    }
}

impl<'a> AsRef<ffi::Mpz> for MpzRef<'a> {
    fn as_ref(&self) -> &ffi::Mpz {
        self.mpz_.as_ref().unwrap()
    }
}

/// Create an `MpzRef` that references the same underlying `mpz_t` as a given
/// reference to a rug `Integer`.
pub unsafe fn rug_integer_to_bicycl_mpz_ref(v: &Integer) -> MpzRef<'_> {
    // We need to create an Mpz which contains an mpz_t pointing to the same data as
    // the mpz_t inside the Integer.

    // Create a shallow copy of the `mpz_t` inside the `Integer`.
    let mut raw_mpz_cpy = *v.as_raw();

    // We will exchange this copy with the `mpz_t` inside a newly created `Mpz`
    // and release the other one.

    // Convert the UniqePtr into a raw pointer s.t. we can pass it to the FFI
    // interface.
    let raw_ptr_to_new_mpz = ffi::mpz_new().into_raw();

    // Perform the swap.
    gmp::mpz_swap(
        &mut raw_mpz_cpy,
        ffi::mpz_get_raw_mpz_mut(raw_ptr_to_new_mpz).0,
    );
    // Release the other one.
    gmp::mpz_clear(&mut raw_mpz_cpy);

    // Put the Mpz into an MpzRef wrapper which makes sure that the `mpz_t` will not
    // be cleared upon destruction.
    MpzRef {
        mpz_: unsafe { cxx::UniquePtr::from_raw(raw_ptr_to_new_mpz) },
        phantom_: PhantomData,
    }
}

#[cxx::bridge]
mod ffi {

    unsafe extern "C++" {
        include!("bicycl/cxx/helpers.h");

        /// GMP's `mpz_ptr` type corresponding to the `mpz_t` used in the BICYCL
        /// library.
        type mpz_ptr = super::MpzPtrWrapper;

        /// GMP's `mpz_srcptr` type corresponding to the `mpz_t` used in the
        /// BICYCL library.
        type mpz_srcptr = super::MpzSrcPtrWrapper;

        /// BICYCL's C++ wrapper class `Mpz` around around GMP's C API.
        #[namespace = "BICYCL"]
        type Mpz;

        #[namespace = "bicycl_rs_helpers"]
        fn mpz_new() -> UniquePtr<Mpz>;
        #[namespace = "bicycl_rs_helpers"]
        #[cxx_name = "mpz_get_raw_mpz"]
        unsafe fn mpz_get_raw_mpz_mut(mpz: *mut Mpz) -> mpz_ptr;
        #[namespace = "bicycl_rs_helpers"]
        unsafe fn mpz_get_raw_mpz(mpz: *const Mpz) -> mpz_srcptr;
    }
}

pub use ffi::Mpz;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rug_mpz_conversion() {
        let rug_a = Integer::from(42);
        let rug_a_new = unsafe {
            let bicycl_a = rug_integer_to_bicycl_mpz(rug_a.clone());
            bicycl_mpz_to_rug_integer(bicycl_a)
        };
        assert_eq!(rug_a_new, rug_a);
    }

    #[test]
    fn test_rug_mpz_conversion_ref() {
        let rug_a = Integer::from(42);
        unsafe {
            let bicycl_a_ref = rug_integer_to_bicycl_mpz_ref(&rug_a);
            let rug_a_new_ref = bicycl_mpz_to_rug_integer_ref(bicycl_a_ref.as_ref());
            assert_eq!(*rug_a_new_ref, rug_a);
        };
    }
}
