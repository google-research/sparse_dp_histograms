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
use crate::group_element::GroupElement;
use crate::mpz;
use rug::integer::BorrowInteger;
use rug::ops::Pow;
use rug::Integer;

/// Data structure containing CL public parameters as well as precomputation
/// data.  Can be used to perform group operation in this setup.
///
/// We use the following notation and refer to BICYCL Section 3 for more
/// details:
/// - M = q^k is the message space, where q is an odd prime larger than
///   2^secparam and k is positive
/// - \Delta_K is the fundamental discriminant of the number field
/// - \Delta_(q^k) is a non-fundamental discriminant
/// - \hat{G} = Cl(\Delta_(q^k))^2
/// - \hat{\Gamma} = Cl(\Delta_K)^2
/// - G = F x H is a cyclic subgroup of \hat{G}, where F = <f> is of order q^k
///   and H = <h>
/// - \pi: hat{G} -> hat{\Gamma}
///     - surjective
///     - ker(\pi) = F
///     - \pi|H is injective s.t. \Gamma = <\gamma> = \pi(H)
///     - \pi(h)^M = \pi(h^M) = \gamma
///     - can efficiently compute elements of \pi^-1
/// - \psi: hat{\Gamma} -> hat{G}
///     - x |-> x'^M where x' \in \pi^-1(x)
///     - \psi(\gamma) = h^(M^2)
pub struct CLContext {
    cl_hsmqk: cxx::UniquePtr<ffi::CL_HSMqk>,
    // We keep copies of the generators around, so that we can return references to them in a
    // unified way (`cl_hsmqk` only contains a copy of either h or gamma).
    f: GroupElement,
    h: GroupElement,
    g: GroupElement,
    gamma: GroupElement,
}

impl CLContext {
    /// Create CL public parameters with an easy-dlog subgroup of order M = q^k
    /// and the given security level.
    pub fn new(q: &Integer, k: usize, seclevel: u32) -> Result<Self, Error> {
        #[allow(non_snake_case)]
        let M = Integer::from(q.pow(k as u32));
        unsafe {
            let mpz_q = mpz::rug_integer_to_bicycl_mpz_ref(q);
            let mpz_one = mpz::rug_integer_to_bicycl_mpz_ref(Integer::ONE);

            let cl_hsmqk = ffi::cl_hsmqk_new(mpz_q.as_ref(), k, seclevel)?;
            let f =
                { GroupElement::from_raw(ffi::cl_hsmqk_power_of_f(&cl_hsmqk, mpz_one.as_ref())) };
            let h =
                { GroupElement::from_raw(ffi::cl_hsmqk_power_of_h(&cl_hsmqk, mpz_one.as_ref())) };
            let g = Self::mul_in_class_group(cl_hsmqk.Cl_Delta(), &f, &h);
            let gamma = {
                let mut t = h.clone();
                t.to_maximal_order(&M, &mpz::bicycl_mpz_to_rug_integer_ref(cl_hsmqk.DeltaK()));
                Self::exp_in_class_group(cl_hsmqk.Cl_Delta(), &t, &M)
            };

            Ok(Self {
                cl_hsmqk,
                f,
                h,
                g,
                gamma,
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

    /// Get the discriminant \Delta_(q^k).
    #[allow(non_snake_case)]
    pub fn get_Deltaqk(&self) -> BorrowInteger {
        unsafe { mpz::bicycl_mpz_to_rug_integer_ref(self.cl_hsmqk.Delta()) }
    }

    /// Get the fundamental discriminant \Delta_K.
    #[allow(non_snake_case)]
    pub fn get_DeltaK(&self) -> BorrowInteger {
        unsafe { mpz::bicycl_mpz_to_rug_integer_ref(self.cl_hsmqk.DeltaK()) }
    }

    /// Get f, the generator of the subgroup of order M = q^k.
    pub fn get_f(&self) -> &GroupElement {
        &self.f
    }

    /// Get h, the generator of the unknown order subgroup H.
    pub fn get_h(&self) -> &GroupElement {
        &self.h
    }

    /// Get g, the generator of the group G.
    pub fn get_g(&self) -> &GroupElement {
        &self.g
    }

    /// Get \gamma, the generator of the group \Gamma.
    pub fn get_gamma(&self) -> &GroupElement {
        &self.gamma
    }

    /// Compute g1 * g2 in \hat{G} = Cl(\Delta_(q^k)).
    #[allow(non_snake_case)]
    pub fn mul_in_G_hat(&self, g1: &GroupElement, g2: &GroupElement) -> GroupElement {
        debug_assert!(self.is_in_G_hat(g1));
        debug_assert!(self.is_in_G_hat(g2));
        Self::mul_in_class_group(self.cl_hsmqk.Cl_Delta(), g1, g2)
    }

    /// Compute g1 * g2 in \hat{\Gamma} = Cl(\Delta_K).
    #[allow(non_snake_case)]
    pub fn mul_in_Gamma_hat(&self, g1: &GroupElement, g2: &GroupElement) -> GroupElement {
        debug_assert!(self.is_in_Gamma_hat(g1));
        debug_assert!(self.is_in_Gamma_hat(g2));
        Self::mul_in_class_group(self.cl_hsmqk.Cl_DeltaK(), g1, g2)
    }

    #[allow(non_snake_case)]
    fn mul_in_class_group(
        cl: &ffi::ClassGroup,
        g1: &GroupElement,
        g2: &GroupElement,
    ) -> GroupElement {
        GroupElement::from_raw(ffi::class_group_nucomp(cl, g1.as_raw(), g2.as_raw()))
    }

    /// Compute g^e in \hat{G} = Cl(\Delta_(q^k)).
    #[allow(non_snake_case)]
    pub fn exp_in_G_hat(&self, base: &GroupElement, exp: &Integer) -> GroupElement {
        debug_assert!(self.is_in_G_hat(base));
        Self::exp_in_class_group(self.cl_hsmqk.Cl_Delta(), base, exp)
    }

    /// Compute g^e in \hat{Gamma} = Cl(\Delta_K).
    #[allow(non_snake_case)]
    pub fn exp_in_Gamma_hat(&self, base: &GroupElement, exp: &Integer) -> GroupElement {
        debug_assert!(self.is_in_Gamma_hat(base));
        Self::exp_in_class_group(self.cl_hsmqk.Cl_DeltaK(), base, exp)
    }

    #[allow(non_snake_case)]
    fn exp_in_class_group(
        cl: &ffi::ClassGroup,
        base: &GroupElement,
        exp: &Integer,
    ) -> GroupElement {
        GroupElement::from_raw(ffi::class_group_nupow(cl, base.as_raw(), unsafe {
            mpz::rug_integer_to_bicycl_mpz_ref(exp).as_ref()
        }))
    }

    /// Compute h^e, where h is the generator of the unknown order subgroup.
    pub fn power_of_h(&self, e: &Integer) -> GroupElement {
        unsafe {
            let mpz_e = mpz::rug_integer_to_bicycl_mpz_ref(e);
            GroupElement::from_raw(ffi::cl_hsmqk_power_of_h(&self.cl_hsmqk, mpz_e.as_ref()))
        }
    }

    /// Compute \gamma^e, where \gamma is the generator of \Gamma
    pub fn power_of_gamma(&self, e: &Integer) -> GroupElement {
        self.exp_in_Gamma_hat(&self.gamma, e)
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
        debug_assert!(self.is_in_F(fm));
        unsafe {
            Ok(mpz::bicycl_mpz_to_rug_integer(ffi::cl_hsmqk_dlog_in_F(
                &self.cl_hsmqk,
                fm.as_raw(),
            )?))
        }
    }

    /// Check if a group element is in the subgroup F.
    #[allow(non_snake_case)]
    pub fn is_in_F(&self, v: &GroupElement) -> bool {
        self.cl_hsmqk.in_F(v.as_raw())
    }

    /// Check if a group element is in the group \hat{G} = Cl(\Delta_(q^k)).
    #[allow(non_snake_case)]
    pub fn is_in_G_hat(&self, v: &GroupElement) -> bool {
        self.cl_hsmqk.in_Cl_DeltaSq(v.as_raw())
    }

    /// Check if a group element is in the group \hat{Gamma} = Cl(\Delta_K).
    #[allow(non_snake_case)]
    pub fn is_in_Gamma_hat(&self, v: &GroupElement) -> bool {
        self.cl_hsmqk.in_Cl_DeltaKSq(v.as_raw())
    }

    /// Map \pi: \hat{G} -> \hat{\Gamma}
    #[allow(non_snake_case)]
    pub fn map_pi_G_to_Gamma(&self, v: &GroupElement) -> GroupElement {
        debug_assert!(self.is_in_G_hat(v));
        let mut t = v.clone();
        t.to_maximal_order(&self.get_M(), &self.get_DeltaK());
        t
    }

    /// Map \psi: \hat{\Gamma} -> \hat{G}
    #[allow(non_snake_case)]
    pub fn map_psi_Gamma_to_G(&self, v: &GroupElement) -> GroupElement {
        debug_assert!(self.is_in_Gamma_hat(v));
        let t = v.clone();
        unsafe { ffi::cl_hsmqk_from_Cl_DeltaK_to_Cl_Delta(&self.cl_hsmqk, t.as_raw()) };
        t
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
        type ClassGroup;

        #[namespace = "bicycl_rs_helpers"]
        fn class_group_nucomp(cl: &ClassGroup, lhs: &QFI, rhs: &QFI) -> UniquePtr<QFI>;
        #[namespace = "bicycl_rs_helpers"]
        fn class_group_nupow(cl: &ClassGroup, base: &QFI, exp: &Mpz) -> UniquePtr<QFI>;

        #[namespace = "BICYCL"]
        type CL_HSMqk;
        fn k(self: &CL_HSMqk) -> usize;
        fn q(self: &CL_HSMqk) -> &Mpz;
        fn M(self: &CL_HSMqk) -> &Mpz;
        fn h(self: &CL_HSMqk) -> &QFI;
        fn Delta(self: &CL_HSMqk) -> &Mpz;
        fn DeltaK(self: &CL_HSMqk) -> &Mpz;
        fn Cl_Delta(self: &CL_HSMqk) -> &ClassGroup;
        fn Cl_DeltaK(self: &CL_HSMqk) -> &ClassGroup;
        fn compact_variant(self: &CL_HSMqk) -> bool;
        fn large_message_variant(self: &CL_HSMqk) -> bool;
        fn secretkey_bound(self: &CL_HSMqk) -> &Mpz;
        fn cleartext_bound(self: &CL_HSMqk) -> &Mpz;
        fn encrypt_randomness_bound(self: &CL_HSMqk) -> &Mpz;
        fn in_F(self: &CL_HSMqk, v: &QFI) -> bool;
        fn in_Cl_DeltaSq(self: &CL_HSMqk, v: &QFI) -> bool;
        fn in_Cl_DeltaKSq(self: &CL_HSMqk, v: &QFI) -> bool;

        #[namespace = "bicycl_rs_helpers"]
        fn cl_hsmqk_new(q: &Mpz, k: usize, seclevel: u32) -> Result<UniquePtr<CL_HSMqk>>;
        #[namespace = "bicycl_rs_helpers"]
        fn cl_hsmqk_power_of_h(cl_hsmqk: &CL_HSMqk, e: &Mpz) -> UniquePtr<QFI>;
        #[namespace = "bicycl_rs_helpers"]
        fn cl_hsmqk_power_of_f(cl_hsmqk: &CL_HSMqk, e: &Mpz) -> UniquePtr<QFI>;
        #[namespace = "bicycl_rs_helpers"]
        fn cl_hsmqk_dlog_in_F(cl_hsmqk: &CL_HSMqk, fm: &QFI) -> Result<UniquePtr<Mpz>>;
        #[namespace = "bicycl_rs_helpers"]
        unsafe fn cl_hsmqk_from_Cl_DeltaK_to_Cl_Delta(cl_hsmqk: &CL_HSMqk, v: &UniquePtr<QFI>);
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
        let cl_ctx = CLContext::new(&(Integer::from(1) << 129), 1, 112);
        assert!(cl_ctx.is_err());

        // k needs to be positive
        let cl_ctx = CLContext::new(&((Integer::from(1) << 127) - 1), 0, 112);
        assert!(cl_ctx.is_err());

        // unsupported security level
        let cl_ctx = CLContext::new(&((Integer::from(1) << 127) - 1), 1, 100);
        assert!(cl_ctx.is_err());
    }

    #[test]
    fn test_constructor_success() {
        let q = (Integer::from(1) << 127) - 1;
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
        let q = (Integer::from(1) << 127) - 1;
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
        assert!(cl_ctx.is_in_F(cl_ctx.get_f()));
        for e in [0, 1, 2, 3, 4, 5, 42, 47, 1337] {
            let fe = cl_ctx.power_of_f(&Integer::from(e));
            assert!(cl_ctx.is_in_F(&fe));
            let e_prime = cl_ctx.dlog_in_F(&fe);
            assert!(e_prime.is_ok());
            let e_prime = e_prime.unwrap();
            assert_eq!(e_prime, Integer::from(e));
            if e == 0 {
                continue;
            }
            let Mme = (*M).clone() - e;
            let fMme = cl_ctx.power_of_f(&Mme);
            assert!(cl_ctx.is_in_F(&fMme));
            let Mme_prime = cl_ctx.dlog_in_F(&fMme);
            assert!(Mme_prime.is_ok());
            let Mme_prime = Mme_prime.unwrap();
            assert_eq!(Mme_prime, Mme);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_h_and_gamma() {
        let cl_ctx = make_cl_context();
        let h = cl_ctx.get_h();
        let gamma = cl_ctx.get_gamma();
        let M = cl_ctx.get_M();

        assert!(!cl_ctx.is_in_F(h));
        assert!(cl_ctx.is_in_G_hat(h));
        assert!(!cl_ctx.is_in_Gamma_hat(h));
        assert!(cl_ctx.power_of_h(&Integer::from(0)).is_one());
        assert_eq!(*h, cl_ctx.power_of_h(&Integer::from(1)));

        assert!(!cl_ctx.is_in_F(gamma));
        assert!(!cl_ctx.is_in_G_hat(gamma));
        assert!(cl_ctx.is_in_Gamma_hat(gamma));
        assert!(cl_ctx.power_of_gamma(&Integer::from(0)).is_one());
        assert_eq!(*gamma, cl_ctx.power_of_gamma(&Integer::from(1)));

        let pi_h = &cl_ctx.map_pi_G_to_Gamma(h);
        assert_eq!(cl_ctx.exp_in_Gamma_hat(pi_h, &M), *gamma);
        let psi_gamma = cl_ctx.map_psi_Gamma_to_G(gamma);
        assert_eq!(psi_gamma, cl_ctx.power_of_h(&Integer::from(M.square_ref())));
    }

    #[test]
    fn test_mul() {
        let cl_ctx = make_cl_context();

        let h = cl_ctx.get_h();
        let h_sq = cl_ctx.power_of_h(&Integer::from(2));
        let h_sq_prime = cl_ctx.mul_in_G_hat(h, h);
        assert_eq!(h_sq_prime, h_sq);

        let f = cl_ctx.get_f();
        let f_sq = cl_ctx.power_of_f(&Integer::from(2));
        let f_sq_prime = cl_ctx.mul_in_G_hat(f, f);
        assert_eq!(f_sq_prime, f_sq);
    }

    #[test]
    fn test_exp() {
        let cl_ctx = make_cl_context();
        let exp = Integer::from(1337);

        let h = cl_ctx.get_h();
        let he = cl_ctx.power_of_h(&exp);
        let he_prime = cl_ctx.exp_in_G_hat(h, &exp);
        assert_eq!(he_prime, he);

        let f = cl_ctx.get_f();
        let fe = cl_ctx.power_of_f(&exp);
        let fe_prime = cl_ctx.exp_in_G_hat(f, &exp);
        assert_eq!(fe_prime, fe);

        let g = cl_ctx.get_g();
        let ge = cl_ctx.mul_in_G_hat(&cl_ctx.power_of_f(&exp), &cl_ctx.power_of_h(&exp));
        let ge_prime = cl_ctx.exp_in_G_hat(g, &exp);
        assert_eq!(ge_prime, ge);
    }
}
