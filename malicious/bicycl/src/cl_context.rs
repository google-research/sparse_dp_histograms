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

//! Contains tools to generate the public parameters of the CL framework and
//! operate on the group elements.

use crate::error::Error;
use crate::group_element::{GroupElement, GroupElementRef};
use crate::mpz;
use rug::integer::BorrowInteger;
use rug::Integer;

/// Data structure containing CL public parameters as well as precomputation
/// data.  Can be used to perform group operation in this setup.
pub struct CLContext {
    cl_hsmqk: cxx::UniquePtr<ffi::CL_HSMqk>,
}

impl CLContext {
    /// Create CL public parameters with an easy-dlog subgroup of order M = q^k
    /// and the given security level.
    pub fn new(q: &Integer, k: usize, seclevel: u32) -> Result<Self, Error> {
        unsafe {
            let mpz_q = mpz::rug_integer_to_bicycl_mpz_ref(q);

            Ok(Self {
                cl_hsmqk: ffi::cl_hsmqk_new(mpz_q.as_ref(), k, seclevel)?,
            })
        }
    }

    /// Return the prime q.
    pub fn get_q(&self) -> BorrowInteger {
        unsafe { mpz::bicycl_mpz_to_rug_integer_ref(self.cl_hsmqk.q()) }
    }

    /// Return the exponent k.
    pub fn get_k(&self) -> usize {
        self.cl_hsmqk.k()
    }

    /// Return M = q^k.
    #[allow(non_snake_case)]
    pub fn get_M(&self) -> BorrowInteger {
        unsafe { mpz::bicycl_mpz_to_rug_integer_ref(self.cl_hsmqk.M()) }
    }

    /// Get h, the generator of the unknown order subgroup.
    pub fn get_h(&self) -> GroupElementRef {
        GroupElementRef::from(self.cl_hsmqk.h())
    }

    /// Compute g1 * g2.
    pub fn mul(&self, g1: &GroupElement, g2: &GroupElement) -> GroupElement {
        unimplemented!()
    }

    /// Compute g^e.
    pub fn exp(&self, base: &GroupElement, e: &Integer) -> GroupElement {
        unimplemented!()
    }

    /// Compute g1^e1 * g2^e2.
    pub fn double_exp(
        &self,
        base1: &GroupElement,
        base2: &GroupElement,
        e1: &Integer,
        e2: &Integer,
    ) -> GroupElement {
        unimplemented!()
    }

    /// Compute h^e, where h is the generator of the unknown order subgroup.
    pub fn power_of_h(&self, e: &Integer) -> GroupElement {
        unsafe {
            let mpz_e = mpz::rug_integer_to_bicycl_mpz_ref(e);
            GroupElement::from_raw(ffi::cl_hsmqk_power_of_h(&self.cl_hsmqk, mpz_e.as_ref()))
        }
    }

    /// Compute f^e, where f is the generator of F.
    pub fn power_of_f(&self, e: &Integer) -> GroupElement {
        unsafe {
            let mpz_e = mpz::rug_integer_to_bicycl_mpz_ref(e);
            GroupElement::from_raw(ffi::cl_hsmqk_power_of_f(&self.cl_hsmqk, mpz_e.as_ref()))
        }
    }

    /// Compute the discrete logarithm in the subgroup F to base f.
    #[allow(non_snake_case)]
    pub fn dlog_in_F(&self, fm: &GroupElement) -> Result<Integer, Error> {
        unsafe {
            Ok(mpz::bicycl_mpz_to_rug_integer(ffi::cl_hsmqk_dlog_in_F(
                &self.cl_hsmqk,
                fm.as_raw(),
            )?))
        }
    }

    /// Check if a group element is in the subgroup F.
    #[allow(non_snake_case)]
    pub fn is_in_F(&self, g: &GroupElement) -> bool {
        self.exp(g, &self.get_M()).is_one()
    }

    /// Return bound on secret keys (i.e., large enough to induce an
    /// almost-uniform distribution in H).
    pub fn secretkey_bound(&self) -> BorrowInteger {
        unsafe { mpz::bicycl_mpz_to_rug_integer_ref(self.cl_hsmqk.secretkey_bound()) }
    }

    /// Return bound on messages (same as M).
    pub fn cleartext_bound(&self) -> BorrowInteger {
        unsafe { mpz::bicycl_mpz_to_rug_integer_ref(self.cl_hsmqk.cleartext_bound()) }
    }

    /// Return bound on the randomness used for encryption (same as for secret
    /// keys).
    pub fn encrypt_randomness_bound(&self) -> BorrowInteger {
        unsafe { mpz::bicycl_mpz_to_rug_integer_ref(self.cl_hsmqk.encrypt_randomness_bound()) }
    }
}

#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("bicycl/cxx/helpers.h");

        #[namespace = "BICYCL"]
        type Mpz = crate::mpz::Mpz;
        #[namespace = "BICYCL"]
        type QFI = crate::group_element::QFI;

        #[namespace = "BICYCL"]
        type CL_HSMqk;
        fn k(self: &CL_HSMqk) -> usize;
        fn q(self: &CL_HSMqk) -> &Mpz;
        fn M(self: &CL_HSMqk) -> &Mpz;
        fn h(self: &CL_HSMqk) -> &QFI;
        fn compact_variant(self: &CL_HSMqk) -> bool;
        fn large_message_variant(self: &CL_HSMqk) -> bool;
        fn secretkey_bound(self: &CL_HSMqk) -> &Mpz;
        fn cleartext_bound(self: &CL_HSMqk) -> &Mpz;
        fn encrypt_randomness_bound(self: &CL_HSMqk) -> &Mpz;

        #[namespace = "bicycl_rs_helpers"]
        fn cl_hsmqk_new(q: &Mpz, k: usize, seclevel: u32) -> Result<UniquePtr<CL_HSMqk>>;
        #[namespace = "bicycl_rs_helpers"]
        fn cl_hsmqk_power_of_h(cl_hsmqk: &CL_HSMqk, e: &Mpz) -> UniquePtr<QFI>;
        #[namespace = "bicycl_rs_helpers"]
        fn cl_hsmqk_power_of_f(cl_hsmqk: &CL_HSMqk, e: &Mpz) -> UniquePtr<QFI>;
        #[namespace = "bicycl_rs_helpers"]
        fn cl_hsmqk_dlog_in_F(cl_hsmqk: &CL_HSMqk, fm: &QFI) -> Result<UniquePtr<Mpz>>;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::ops::Pow;

    #[test]
    fn test_constructor_fail() {
        // q needs to have at least seclevel bits
        let cl_ctx = CLContext::new(&Integer::from(47), 1, 112);
        assert!(cl_ctx.is_err());

        // q needs to be prime
        let cl_ctx = CLContext::new(&Integer::from(Integer::from(1) << 129), 1, 112);
        assert!(cl_ctx.is_err());

        // k needs to be positive
        let cl_ctx = CLContext::new(
            &Integer::from(Integer::from(Integer::from(1) << 127) - 1),
            0,
            112,
        );
        assert!(cl_ctx.is_err());

        // unsupported security level
        let cl_ctx = CLContext::new(
            &Integer::from(Integer::from(Integer::from(1) << 127) - 1),
            1,
            100,
        );
        assert!(cl_ctx.is_err());
    }

    #[test]
    fn test_constructor_success() {
        let q = Integer::from(Integer::from(Integer::from(1) << 127) - 1);
        let k = 1;
        let seclevel = 112;
        let cl_ctx = CLContext::new(&q, k, seclevel);
        assert!(cl_ctx.is_ok());
        let cl_ctx = cl_ctx.unwrap();
        assert_eq!(*cl_ctx.get_q(), q);
        assert_eq!(cl_ctx.get_k(), k);
        assert_eq!(*cl_ctx.get_M(), q.pow(k as u32));
    }

    fn make_cl_context() -> CLContext {
        let q = Integer::from(Integer::from(Integer::from(1) << 127) - 1);
        let k = 1;
        let seclevel = 112;
        let cl_ctx = CLContext::new(&q, k, seclevel);
        assert!(cl_ctx.is_ok());
        cl_ctx.unwrap()
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_easy_dlog_subgroup() {
        let cl_ctx = make_cl_context();
        let M = cl_ctx.get_M();
        assert!(cl_ctx.power_of_f(&M).is_one());
        for e in [0, 1, 2, 3, 4, 5, 42, 47, 1337] {
            let fe = cl_ctx.power_of_f(&Integer::from(e));
            let e_prime = cl_ctx.dlog_in_F(&fe);
            assert!(e_prime.is_ok());
            let e_prime = e_prime.unwrap();
            assert_eq!(e_prime, Integer::from(e));
            if e == 0 {
                continue;
            }
            let Mme = Integer::from((*M).clone() - e);
            let fMme = cl_ctx.power_of_f(&Mme);
            let Mme_prime = cl_ctx.dlog_in_F(&fMme);
            assert!(Mme_prime.is_ok());
            let Mme_prime = Mme_prime.unwrap();
            assert_eq!(Mme_prime, Mme);
        }
    }

    #[test]
    fn test_h() {
        let cl_ctx = make_cl_context();
        let h = cl_ctx.get_h();
        assert!(cl_ctx.power_of_h(&Integer::from(0)).is_one());
        assert_eq!(*h, cl_ctx.power_of_h(&Integer::from(1)));
    }
}
