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
use crate::traits::{LHEScheme, ThresholdLHEScheme};
use crate::utils::random_integer_bits;
use rug::Integer;
use std::borrow::Borrow;

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
        self.cl_ctx.power_of_generator(self.compact, sk)
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
        if !self.cl_ctx.is_in_group(self.compact, pk) {
            return Err(Error::InvalidArgument(
                "pk is in the wrong group".to_owned(),
            ));
        }
        if !self.cl_ctx.is_in_ZM(m) {
            return Err(Error::InvalidArgument(
                "m lies not in the message space".to_owned(),
            ));
        }
        let c1 = self.cl_ctx.power_of_generator(self.compact, r);
        let c2 = self.cl_ctx.mul_in_G_hat(
            &self.cl_ctx.cond_map_psi_Gamma_to_G(
                self.compact,
                &self.cl_ctx.exp_in_group(self.compact, pk, r),
            ),
            &self.cl_ctx.power_of_f(m),
        );
        Ok((c1, c2))
    }

    fn decrypt(&self, sk: &Self::SecretKey, c: &Self::Ciphertext) -> Result<Self::Message, Error> {
        let (c1, c2) = c;
        let fm = {
            let mut t = self.cl_ctx.exp_in_group(self.compact, c1, sk);
            t.invert();
            self.cl_ctx
                .mul_in_G_hat(c2, &self.cl_ctx.cond_map_psi_Gamma_to_G(self.compact, &t))
        };

        self.cl_ctx
            .is_in_F(&fm)
            .then(|| {
                self.cl_ctx
                    .dlog_in_F(&fm)
                    .expect("we have already checked that fm is in F")
            })
            .ok_or(Error::DecryptionError("decryption failed".to_owned()))
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
            self.cl_ctx.mul_in_group(
                self.compact,
                c1,
                &self.cl_ctx.power_of_generator(self.compact, r),
            ),
            self.cl_ctx.mul_in_G_hat(
                c2,
                &self.cl_ctx.cond_map_psi_Gamma_to_G(
                    self.compact,
                    &self.cl_ctx.exp_in_group(self.compact, pk, r),
                ),
            ),
        )
    }

    /// Perform homomorphic addition of two ciphertexts.
    fn add_ciphertexts(
        &self,
        (c1_1, c1_2): &Self::Ciphertext,
        (c2_1, c2_2): &Self::Ciphertext,
    ) -> Self::Ciphertext {
        (
            self.cl_ctx.mul_in_group(self.compact, c1_1, c2_1),
            self.cl_ctx.mul_in_G_hat(c1_2, c2_2),
        )
    }

    /// Perform homomorphic addition of a ciphertext and a message.
    fn add_ciphertext_and_message(
        &self,
        (c1, c2): &Self::Ciphertext,
        m: &Self::Message,
    ) -> Self::Ciphertext {
        (
            c1.clone(),
            self.cl_ctx.mul_in_G_hat(c2, &self.cl_ctx.power_of_f(m)),
        )
    }

    /// Perform homomorphic multiplication of a ciphertext and a message.
    fn mul_ciphertext_and_message(
        &self,
        (c1, c2): &Self::Ciphertext,
        m: &Self::Message,
    ) -> Self::Ciphertext {
        (
            self.cl_ctx.exp_in_group(self.compact, c1, m),
            self.cl_ctx.exp_in_G_hat(c2, m),
        )
    }
}

impl<'a> ThresholdLHEScheme for HSMCLqk<'a> {
    type SecretKeyShare = Self::SecretKey;
    type PublicKeyShare = Self::PublicKey;
    type PartialDecryption = GroupElement;

    /// Generate shares of the secret and the public key.
    fn generate_key_share(&self) -> (Self::SecretKeyShare, Self::PublicKeyShare) {
        self.generate_key_pair()
    }

    fn combine_public_key_shares<'b>(
        &self,
        pks: impl IntoIterator<Item = impl Borrow<Self::PublicKeyShare>>,
    ) -> Result<Self::PublicKey, Error> {
        let mut pks = pks.into_iter().peekable();
        let mut pk = pks
            .next()
            .ok_or_else(|| Error::InvalidArgument("iterator is empty".to_owned()))?
            .borrow()
            .clone();
        for pk_i in pks {
            pk = self.cl_ctx.mul_in_group(self.compact, &pk, pk_i.borrow());
        }
        Ok(pk)
    }

    fn decrypt_to_ciphertext(
        &self,
        sk_i: &Self::SecretKeyShare,
        (c1, c2): &Self::Ciphertext,
    ) -> Self::Ciphertext {
        (c1.clone(), {
            let mut t = self.cl_ctx.exp_in_group(self.compact, c1, sk_i);
            t.invert();
            self.cl_ctx
                .mul_in_G_hat(c2, &self.cl_ctx.cond_map_psi_Gamma_to_G(self.compact, &t))
        })
    }

    fn partially_decrypt(
        &self,
        sk_i: &Self::SecretKeyShare,
        (c1, _): &Self::Ciphertext,
    ) -> Self::PartialDecryption {
        self.cl_ctx.exp_in_group(self.compact, c1, sk_i)
    }

    fn complete_decryption(
        &self,
        (_, c2): &Self::Ciphertext,
        pds: impl IntoIterator<Item = impl Borrow<Self::PartialDecryption>>,
    ) -> Result<Self::Message, Error> {
        let mut pds = pds.into_iter();
        let mut fm = pds
            .next()
            .ok_or_else(|| Error::InvalidArgument("iterator is non-empty".to_owned()))?
            .borrow()
            .clone();
        for pd_i in pds {
            fm = self.cl_ctx.mul_in_group(self.compact, &fm, pd_i.borrow());
        }
        fm.invert();
        fm = self
            .cl_ctx
            .mul_in_G_hat(&self.cl_ctx.cond_map_psi_Gamma_to_G(self.compact, &fm), c2);
        self.cl_ctx
            .is_in_F(&fm)
            .then(|| {
                self.cl_ctx
                    .dlog_in_F(&fm)
                    .expect("we have already checked that fm is in F")
            })
            .ok_or_else(|| Error::DecryptionError("decryption failed".to_owned()))
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
            assert!(decrypted_m.is_ok());
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
            assert!(decrypted_m.is_ok());
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
            assert!(decrypted_m.is_ok());
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
            assert!(decrypted_m.is_ok());
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
            assert!(decrypted_m.is_ok());
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

    fn test_threshold_encryption_correctness_w_param(compact: bool) {
        let cl_ctx = make_test_cl_context();
        let hsmcl = HSMCLqk::new(&cl_ctx, compact);
        assert_eq!(hsmcl.is_compact(), compact);
        let num_parties = 3;
        let key_pair_shares: Vec<_> = (0..num_parties)
            .map(|_| hsmcl.generate_key_share())
            .collect();
        assert!(hsmcl.combine_public_key_shares(&[]).is_err());
        let pk = hsmcl.combine_public_key_shares(key_pair_shares.iter().map(|(_, pk_i)| pk_i));
        assert!(pk.is_ok());
        let pk = pk.unwrap();
        for m in make_test_messages(&cl_ctx.get_M()) {
            let (c, _) = hsmcl.encrypt(&pk, &m).expect("encryption should succeed");
            assert!(cl_ctx.is_in_group(compact, &c.0));
            assert!(cl_ctx.is_in_G_hat(&c.1));
            {
                let pds: Vec<_> = key_pair_shares
                    .iter()
                    .map(|(sk_i, _)| hsmcl.partially_decrypt(sk_i, &c))
                    .collect();
                assert!(hsmcl.complete_decryption(&c, &[]).is_err());
                let decrypted_m = hsmcl.complete_decryption(&c, &pds);
                assert!(decrypted_m.is_ok());
                let decrypted_m = decrypted_m.unwrap();
                assert_eq!(decrypted_m, m);
            }
            {
                let mut c_new = c.clone();
                for (sk_i, _) in key_pair_shares.iter().take(num_parties - 1) {
                    c_new = hsmcl.decrypt_to_ciphertext(sk_i, &c_new);
                    assert!(cl_ctx.is_in_group(compact, &c_new.0));
                    assert!(cl_ctx.is_in_G_hat(&c_new.1));
                }
                let decrypted_m = hsmcl.decrypt(&key_pair_shares[num_parties - 1].0, &c_new);
                assert!(decrypted_m.is_ok());
                let decrypted_m = decrypted_m.unwrap();
                assert_eq!(decrypted_m, m);
            }
        }
    }

    #[test]
    fn test_threshold_encryption_correctness() {
        test_threshold_encryption_correctness_w_param(false);
    }

    #[test]
    fn test_threshold_encryption_correctness_compact() {
        test_threshold_encryption_correctness_w_param(true);
    }
}
