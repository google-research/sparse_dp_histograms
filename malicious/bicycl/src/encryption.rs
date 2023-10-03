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

use crate::cl_context::CLContext;
use crate::error::Error;
use crate::group_element::GroupElement;
use crate::traits::LHEScheme;
use crate::utils::random_integer_bits;
use rug::Integer;

/// The HSM-CLqk encryption scheme.
pub struct HSMCLqk<'a> {
    cl_ctx: &'a CLContext,
    compact: bool,
}

impl<'a> HSMCLqk<'a> {
    /// Create a new instance of the encryption scheme with normal or compact
    /// ciphertexts.
    pub fn new(cl_ctx: &'a CLContext, compact: bool) -> Self {
        Self { cl_ctx, compact }
    }

    /// Return whether the normal or the compact version is used.
    pub fn is_compact(&self) -> bool {
        self.compact
    }
}

/// Implementation of the HSM-CLqk encryption scheme.
impl<'a> LHEScheme for HSMCLqk<'a> {
    type SecretKey = Integer;
    type PublicKey = GroupElement;
    type Message = Integer;
    type Ciphertext = (GroupElement, GroupElement);
    type Randomness = Integer;

    fn generate_secret_key(&self) -> Self::SecretKey {
        random_integer_bits(self.cl_ctx.secretkey_bound().significant_bits())
    }

    fn compute_public_key(&self, sk: &Self::SecretKey) -> Self::PublicKey {
        if self.compact {
            self.cl_ctx.power_of_gamma(sk)
        } else {
            self.cl_ctx.power_of_h(sk)
        }
    }

    fn encrypt(
        &self,
        pk: &Self::PublicKey,
        m: &Self::Message,
    ) -> Result<(Self::Ciphertext, Self::Randomness), Error> {
        let r = random_integer_bits(self.cl_ctx.encrypt_randomness_bound().significant_bits());
        let c = self.encrypt_with_randomness(pk, m, &r)?;
        Ok((c, r))
    }

    fn encrypt_with_randomness(
        &self,
        pk: &Self::PublicKey,
        m: &Self::Message,
        r: &Self::Randomness,
    ) -> Result<Self::Ciphertext, Error> {
        if (self.compact && !self.cl_ctx.is_in_Gamma_hat(pk))
            || (!self.compact && !self.cl_ctx.is_in_G_hat(pk))
        {
            return Err(Error::InvalidArgument(
                "pk is in the wrong group".to_owned(),
            ));
        }
        if !self.cl_ctx.is_in_ZM(m) {
            return Err(Error::InvalidArgument(
                "m lies not in the message space".to_owned(),
            ));
        }
        let c1 = if self.compact {
            self.cl_ctx.power_of_gamma(r)
        } else {
            self.cl_ctx.power_of_h(r)
        };
        let c2 = self.cl_ctx.mul_in_G_hat(
            &if self.compact {
                self.cl_ctx
                    .map_psi_Gamma_to_G(&self.cl_ctx.exp_in_Gamma_hat(pk, r))
            } else {
                self.cl_ctx.exp_in_G_hat(pk, r)
            },
            &self.cl_ctx.power_of_f(m),
        );
        Ok((c1, c2))
    }

    fn decrypt(&self, sk: &Self::SecretKey, c: &Self::Ciphertext) -> Option<Self::Message> {
        let (c1, c2) = c;
        let fm = {
            let t = if self.compact {
                let mut t = self.cl_ctx.exp_in_Gamma_hat(c1, sk);
                t = self.cl_ctx.map_psi_Gamma_to_G(&t);
                t.invert();
                t
            } else {
                let mut t = self.cl_ctx.exp_in_G_hat(c1, sk);
                t.invert();
                t
            };
            self.cl_ctx.mul_in_G_hat(c2, &t)
        };
        if self.cl_ctx.is_in_F(&fm) {
            Some(
                self.cl_ctx
                    .dlog_in_F(&fm)
                    .expect("we have already checked that fm is in F"),
            )
        } else {
            None
        }
    }

    /// Rerandomize a ciphertext.
    fn rerandomize(
        &self,
        pk: &Self::PublicKey,
        c: &Self::Ciphertext,
    ) -> (Self::Ciphertext, Self::Randomness) {
        let r = random_integer_bits(self.cl_ctx.encrypt_randomness_bound().significant_bits());
        let c_new = self.rerandomize_with_randomness(pk, c, &r);
        (c_new, r)
    }

    fn rerandomize_with_randomness(
        &self,
        pk: &Self::PublicKey,
        c: &Self::Ciphertext,
        r: &Self::Randomness,
    ) -> Self::Ciphertext {
        let (c1, c2) = c;
        (
            if self.compact {
                self.cl_ctx
                    .mul_in_Gamma_hat(c1, &self.cl_ctx.power_of_gamma(r))
            } else {
                self.cl_ctx.mul_in_G_hat(c1, &self.cl_ctx.power_of_h(r))
            },
            self.cl_ctx.mul_in_G_hat(
                c2,
                &if self.compact {
                    self.cl_ctx
                        .map_psi_Gamma_to_G(&self.cl_ctx.exp_in_Gamma_hat(pk, r))
                } else {
                    self.cl_ctx.exp_in_G_hat(pk, r)
                },
            ),
        )
    }

    /// Perform homomorphic addition of two ciphertexts.
    fn add_ciphertexts(&self, c1: &Self::Ciphertext, c2: &Self::Ciphertext) -> Self::Ciphertext {
        let ((c1_1, c1_2), (c2_1, c2_2)) = (c1, c2);
        (
            if self.compact {
                self.cl_ctx.mul_in_Gamma_hat(c1_1, c2_1)
            } else {
                self.cl_ctx.mul_in_G_hat(c1_1, c2_1)
            },
            self.cl_ctx.mul_in_G_hat(c1_2, c2_2),
        )
    }

    /// Perform homomorphic addition of a ciphertext and a message.
    fn add_ciphertext_and_message(
        &self,
        c: &Self::Ciphertext,
        m: &Self::Message,
    ) -> Self::Ciphertext {
        let (c1, c2) = c;
        (
            c1.clone(),
            self.cl_ctx.mul_in_G_hat(c2, &self.cl_ctx.power_of_f(m)),
        )
    }

    /// Perform homomorphic multiplication of a ciphertext and a message.
    fn mul_ciphertext_and_message(
        &self,
        c: &Self::Ciphertext,
        m: &Self::Message,
    ) -> Self::Ciphertext {
        let (c1, c2) = c;
        (
            if self.compact {
                self.cl_ctx.exp_in_Gamma_hat(c1, m)
            } else {
                self.cl_ctx.exp_in_G_hat(c1, m)
            },
            self.cl_ctx.exp_in_G_hat(c2, m),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cl_context::tests::make_test_cl_context;
    use itertools::iproduct;

    #[allow(non_snake_case)]
    fn make_test_messages(M: &Integer) -> Vec<Integer> {
        vec![
            Integer::ZERO,
            Integer::ONE.clone(),
            Integer::from(42),
            Integer::from(23),
            Integer::from(1337),
            Integer::from(M - 4747),
            Integer::from(M - 1),
        ]
    }

    fn test_encryption_key_generation_w_param(compact: bool) {
        let cl_ctx = make_test_cl_context();
        let hsmcl = HSMCLqk::new(&cl_ctx, compact);
        let (sk, pk) = hsmcl.generate_key_pair();
        if compact {
            assert!(cl_ctx.is_in_Gamma_hat(&pk));
            assert_eq!(pk, cl_ctx.power_of_gamma(&sk));
        } else {
            assert!(cl_ctx.is_in_G_hat(&pk));
            assert_eq!(pk, cl_ctx.power_of_h(&sk));
        }
    }

    #[test]
    fn test_encryption_key_generation() {
        test_encryption_key_generation_w_param(false);
    }

    #[test]
    fn test_encryption_key_generation_compact() {
        test_encryption_key_generation_w_param(true);
    }

    fn test_encryption_correctness_w_param(compact: bool) {
        let cl_ctx = make_test_cl_context();
        let hsmcl = HSMCLqk::new(&cl_ctx, compact);
        assert_eq!(hsmcl.is_compact(), compact);
        let (sk, pk) = hsmcl.generate_key_pair();
        for m in make_test_messages(&cl_ctx.get_M()) {
            let (c, r) = hsmcl.encrypt(&pk, &m).expect("encryption should succeed");

            if compact {
                assert!(cl_ctx.is_in_Gamma_hat(&c.0));
            } else {
                assert!(cl_ctx.is_in_G_hat(&c.0));
            }
            assert!(cl_ctx.is_in_G_hat(&c.1));
            let decrypted_m = hsmcl.decrypt(&sk, &c);
            assert!(decrypted_m.is_some());
            let decrypted_m = decrypted_m.unwrap();
            assert_eq!(decrypted_m, m);

            assert_eq!(hsmcl.encrypt_with_randomness(&pk, &m, &r).unwrap(), c);
        }
    }

    #[test]
    fn test_encryption_correctness() {
        test_encryption_correctness_w_param(false);
    }

    #[test]
    fn test_encryption_correctness_compact() {
        test_encryption_correctness_w_param(true);
    }

    fn test_encryption_rerandomization_w_param(compact: bool) {
        let cl_ctx = make_test_cl_context();
        let hsmcl = HSMCLqk::new(&cl_ctx, compact);
        let (sk, pk) = hsmcl.generate_key_pair();
        for m in make_test_messages(&cl_ctx.get_M()) {
            let (c, _) = hsmcl.encrypt(&pk, &m).expect("encryption should succeed");
            let (d, r) = hsmcl.rerandomize(&pk, &c);
            assert_ne!(c, d);
            let decrypted_m = hsmcl.decrypt(&sk, &d);
            assert!(decrypted_m.is_some());
            let decrypted_m = decrypted_m.unwrap();
            assert_eq!(decrypted_m, m);

            assert_eq!(hsmcl.rerandomize_with_randomness(&pk, &c, &r), d);
        }
    }

    #[test]
    fn test_encryption_rerandomization() {
        test_encryption_rerandomization_w_param(false);
    }

    #[test]
    fn test_encryption_rerandomization_compact() {
        test_encryption_rerandomization_w_param(true);
    }

    fn test_encryption_addition_w_param(compact: bool) {
        let cl_ctx = make_test_cl_context();
        let hsmcl = HSMCLqk::new(&cl_ctx, compact);
        let (sk, pk) = hsmcl.generate_key_pair();
        for (m1, m2) in iproduct!(
            make_test_messages(&cl_ctx.get_M()),
            make_test_messages(&cl_ctx.get_M())
        ) {
            let (c1, _) = hsmcl.encrypt(&pk, &m1).expect("encryption should succeed");
            let (c2, _) = hsmcl.encrypt(&pk, &m2).expect("encryption should succeed");
            let d = hsmcl.add_ciphertexts(&c1, &c2);
            let decrypted_m = hsmcl.decrypt(&sk, &d);
            assert!(decrypted_m.is_some());
            let decrypted_m = decrypted_m.unwrap();
            let expected_m = {
                let mut t = m1 + m2;
                t.modulo_mut(&cl_ctx.get_M());
                t
            };
            assert_eq!(decrypted_m, expected_m);
        }
    }

    #[test]
    fn test_encryption_addition() {
        test_encryption_addition_w_param(false);
    }

    #[test]
    fn test_encryption_addition_compact() {
        test_encryption_addition_w_param(true);
    }

    fn test_encryption_addition_with_message_w_param(compact: bool) {
        let cl_ctx = make_test_cl_context();
        let hsmcl = HSMCLqk::new(&cl_ctx, compact);
        let (sk, pk) = hsmcl.generate_key_pair();
        for (m1, m2) in iproduct!(
            make_test_messages(&cl_ctx.get_M()),
            make_test_messages(&cl_ctx.get_M())
        ) {
            let (c1, _) = hsmcl.encrypt(&pk, &m1).expect("encryption should succeed");
            let d = hsmcl.add_ciphertext_and_message(&c1, &m2);
            let decrypted_m = hsmcl.decrypt(&sk, &d);
            assert!(decrypted_m.is_some());
            let decrypted_m = decrypted_m.unwrap();
            let expected_m = {
                let mut t = m1 + m2;
                t.modulo_mut(&cl_ctx.get_M());
                t
            };
            assert_eq!(decrypted_m, expected_m);
        }
    }

    #[test]
    fn test_encryption_addition_with_message() {
        test_encryption_addition_with_message_w_param(false);
    }

    #[test]
    fn test_encryption_addition_with_message_compact() {
        test_encryption_addition_with_message_w_param(true);
    }

    fn test_encryption_multiplication_with_message_w_param(compact: bool) {
        let cl_ctx = make_test_cl_context();
        let hsmcl = HSMCLqk::new(&cl_ctx, compact);
        let (sk, pk) = hsmcl.generate_key_pair();
        for (m1, m2) in iproduct!(
            make_test_messages(&cl_ctx.get_M()),
            make_test_messages(&cl_ctx.get_M())
        ) {
            let (c1, _) = hsmcl.encrypt(&pk, &m1).expect("encryption should succeed");
            let d = hsmcl.mul_ciphertext_and_message(&c1, &m2);
            let decrypted_m = hsmcl.decrypt(&sk, &d);
            assert!(decrypted_m.is_some());
            let decrypted_m = decrypted_m.unwrap();
            let expected_m = {
                let mut t = m1 * m2;
                t.modulo_mut(&cl_ctx.get_M());
                t
            };
            assert_eq!(decrypted_m, expected_m);
        }
    }

    #[test]
    fn test_encryption_multiplication_with_message() {
        test_encryption_multiplication_with_message_w_param(false);
    }

    #[test]
    fn test_encryption_multiplication_with_message_compact() {
        test_encryption_multiplication_with_message_w_param(true);
    }
}
