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

//! Some basic \Sigma protocols for the CL Framework and the HSM-CL encryption
//! schemes.

use crate::cl_context::CLContext;
use crate::encryption::HSMCLqk;
use crate::error::Error;
use crate::group_element::GroupElement;
use crate::traits::{LHEScheme, SigmaProtocol};
use crate::utils::{random_integer_below, random_integer_bits};
use rug::Integer;

/// Common parameters for the \Sigma protocols in this module.
struct ProofParams<'a> {
    /// CL Context to perform group operations.
    pub cl_ctx: &'a CLContext,
    /// Bound on integer witnesses (e.g., exponents of unknown order group
    /// elements).
    pub integer_witness_size: u32,
    /// Statistical security parameter for the honest-verifier zero-knowledge
    /// property.
    pub zk_parameter: u32,
    /// Size of the challenge, determines the (knowledge) soundness error.
    pub challenge_size: u32,
    /// Size of random masks to hide integer witnesses.
    pub integer_randomness_size: u32,
    /// Size of random masks to hide message witnesses in case modular reduction
    /// cannot be used.
    pub message_randomness_size: u32,
}

impl<'a> ProofParams<'a> {
    /// Create parameter struct (see struct description).
    fn new(
        cl_ctx: &'a CLContext,
        zk_parameter: u32,
        soundness_parameter: u32,
        integer_witness_size: u32,
    ) -> Self {
        Self {
            cl_ctx,
            zk_parameter,
            challenge_size: soundness_parameter,
            integer_witness_size,
            integer_randomness_size: integer_witness_size + soundness_parameter + zk_parameter,
            message_randomness_size: cl_ctx.get_M().significant_bits()
                + soundness_parameter
                + zk_parameter,
        }
    }
}

pub trait CLProof<'a> {
    /// - `cl_context` CL setup
    /// - `zk_parameters` is the statistical security parameter for HVZK
    /// - `soundness_parameters` is the challenge size
    /// - `log_witness_bound` is a public bound on the size of the witness
    fn new(
        cl_ctx: &'a CLContext,
        zk_parameter: u32,
        soundness_parameter: u32,
        witness_size: u32,
    ) -> Self;
}

pub trait HSMCLProof<'a> {
    /// - `hsmcl` HSM-CL instance
    /// - other parameters: same as in `CLProof`
    fn new(
        cl_ctx: &'a CLContext,
        hsmcl: &'a HSMCLqk,
        zk_parameter: u32,
        soundness_parameter: u32,
        witness_size: u32,
    ) -> Self;
}

/// \Sigma protocol for the discrete logarithm aka Schnorr's protocol in the
/// unknown order group H to base h or in \Gamma to base \gamma.
#[allow(non_snake_case)]
pub struct CLFixedBaseDLog<'a> {
    params: ProofParams<'a>,
    in_Gamma: bool,
}

impl<'a> CLFixedBaseDLog<'a> {
    #[allow(non_snake_case)]
    fn new(
        cl_ctx: &'a CLContext,
        in_Gamma: bool,
        zk_parameter: u32,
        soundness_parameter: u32,
        witness_size: u32,
    ) -> Self {
        Self {
            params: ProofParams::new(cl_ctx, zk_parameter, soundness_parameter, witness_size),
            in_Gamma,
        }
    }
}

impl<'a> SigmaProtocol for CLFixedBaseDLog<'a> {
    type Statement = GroupElement;
    type Witness = Integer;
    type Commitment = GroupElement;
    type Challenge = Integer;
    type Response = Integer;
    type ProverState = Integer;
    type Error = Error;

    fn check_witness(&self, y: &Self::Statement, x: &Self::Witness) -> Result<(), Self::Error> {
        (!x.is_negative()
            && x.significant_bits() <= self.params.integer_witness_size
            && (!self.in_Gamma || self.params.cl_ctx.power_of_gamma(x) == *y)
            && (self.in_Gamma || self.params.cl_ctx.power_of_h(x) == *y))
            .then_some(())
            .ok_or(Error::InvalidArgument("invalid witness".to_owned()))
    }

    fn commit(
        &self,
        y: &Self::Statement,
        x: &Self::Witness,
    ) -> Result<(Self::Commitment, Self::ProverState), Self::Error> {
        self.check_witness(y, x)?;
        let r = random_integer_bits(self.params.integer_randomness_size);
        let a = if self.in_Gamma {
            self.params.cl_ctx.power_of_gamma(&r)
        } else {
            self.params.cl_ctx.power_of_h(&r)
        };
        Ok((a, r))
    }

    fn challenge(&self) -> Self::Challenge {
        random_integer_bits(self.params.challenge_size)
    }

    fn respond(
        &self,
        r: &Self::ProverState,
        y: &Self::Statement,
        x: &Self::Witness,
        k: &Self::Challenge,
    ) -> Result<Self::Response, Self::Error> {
        self.check_witness(y, x)?;
        if k.significant_bits() > self.params.challenge_size {
            return Err(Error::ProtocolError("challenge too large".to_owned()));
        }
        Ok(Integer::from(k * x + r))
    }

    fn verify(
        &self,
        y: &Self::Statement,
        a: &Self::Commitment,
        k: &Self::Challenge,
        u: &Self::Response,
    ) -> bool {
        if self.in_Gamma {
            let lhs = self.params.cl_ctx.power_of_gamma(u);
            let rhs = self
                .params
                .cl_ctx
                .mul_in_Gamma_hat(&self.params.cl_ctx.exp_in_Gamma_hat(y, k), a);
            self.params.cl_ctx.is_in_Gamma_hat(y) && lhs == rhs
        } else {
            let lhs = self.params.cl_ctx.power_of_h(u);
            let rhs = self
                .params
                .cl_ctx
                .mul_in_G_hat(&self.params.cl_ctx.exp_in_G_hat(y, k), a);
            self.params.cl_ctx.is_in_G_hat(y) && lhs == rhs
        }
    }
}

/// \Sigma protocol for a proof of plaintext knowledge (PoPK) of the HSM-CL
/// encryption scheme.
///
/// "I know message and randomness that has been used to create this ciphertext
/// under the given public key."
pub struct PoPK<'a> {
    params: ProofParams<'a>,
    hsmcl: &'a HSMCLqk<'a>,
}

impl<'a> HSMCLProof<'a> for PoPK<'a> {
    fn new(
        cl_ctx: &'a CLContext,
        hsmcl: &'a HSMCLqk,
        zk_parameter: u32,
        soundness_parameter: u32,
        integer_witness_size: u32,
    ) -> Self {
        Self {
            params: ProofParams::new(
                cl_ctx,
                zk_parameter,
                soundness_parameter,
                integer_witness_size,
            ),
            hsmcl,
        }
    }
}

impl<'a> SigmaProtocol for PoPK<'a> {
    type Statement = (
        <HSMCLqk<'a> as LHEScheme>::PublicKey,
        <HSMCLqk<'a> as LHEScheme>::Ciphertext,
    );
    type Witness = (
        <HSMCLqk<'a> as LHEScheme>::Message,
        <HSMCLqk<'a> as LHEScheme>::Randomness,
    );
    type Commitment = <HSMCLqk<'a> as LHEScheme>::Ciphertext;
    type Challenge = Integer;
    type Response = (Integer, Integer);
    type ProverState = (Integer, Integer);
    type Error = Error;

    fn check_witness(
        &self,
        (pk, c): &Self::Statement,
        (m, r): &Self::Witness,
    ) -> Result<(), Self::Error> {
        (self.params.cl_ctx.is_in_ZM(m)
            && 0 <= *r
            && r.significant_bits() <= self.params.integer_witness_size
            && self
                .hsmcl
                .encrypt_with_randomness(pk, m, r)
                .expect("encryption should succeed")
                == *c)
            .then_some(())
            .ok_or(Error::InvalidArgument("invalid witness".to_owned()))
    }

    fn commit(
        &self,
        statement: &Self::Statement,
        witness: &Self::Witness,
    ) -> Result<(Self::Commitment, Self::ProverState), Self::Error> {
        self.check_witness(statement, witness)?;
        let (pk, _) = statement;
        let r_m = random_integer_below(&self.params.cl_ctx.get_M(), self.params.zk_parameter);
        let (c_r, r_r) = self
            .hsmcl
            .encrypt(pk, &r_m)
            .expect("encryption should succeed");
        Ok((c_r, (r_m, r_r)))
    }

    fn challenge(&self) -> Self::Challenge {
        random_integer_bits(self.params.challenge_size)
    }

    fn respond(
        &self,
        (r_m, r_r): &Self::ProverState,
        statement: &Self::Statement,
        witness: &Self::Witness,
        k: &Self::Challenge,
    ) -> Result<Self::Response, Self::Error> {
        self.check_witness(statement, witness)?;
        let (m, r) = witness;
        if k.significant_bits() > self.params.challenge_size {
            return Err(Error::ProtocolError("challenge too large".to_owned()));
        }
        let u_m = {
            let mut t = Integer::from(k * m + r_m);
            t.modulo_mut(&self.params.cl_ctx.get_M());
            t
        };
        let u_r = Integer::from(k * r + r_r);
        Ok((u_m, u_r))
    }

    fn verify(
        &self,
        (pk, c): &Self::Statement,
        c_r: &Self::Commitment,
        k: &Self::Challenge,
        (u_m, u_r): &Self::Response,
    ) -> bool {
        let lhs = self
            .hsmcl
            .encrypt_with_randomness(pk, u_m, u_r)
            .expect("encryption should succeed");
        let rhs = self
            .hsmcl
            .add_ciphertexts(&self.hsmcl.mul_ciphertext_and_message(c, k), c_r);
        lhs == rhs
    }
}

/// \Sigma protocol for a proof of multiplier knowledge (PoMK) of the HSM-CL
/// encryption scheme.
///
/// "I know message and randomness that has been used to create this ciphertext
/// by using the homomorphic multiplication operation and rerandomization of the
/// given ciphertext under the given public key."
pub struct PoMK<'a> {
    params: ProofParams<'a>,
    hsmcl: &'a HSMCLqk<'a>,
}

impl<'a> HSMCLProof<'a> for PoMK<'a> {
    fn new(
        cl_ctx: &'a CLContext,
        hsmcl: &'a HSMCLqk,
        zk_parameter: u32,
        soundness_parameter: u32,
        integer_witness_size: u32,
    ) -> Self {
        Self {
            params: ProofParams::new(
                cl_ctx,
                zk_parameter,
                soundness_parameter,
                integer_witness_size,
            ),
            hsmcl,
        }
    }
}

impl<'a> SigmaProtocol for PoMK<'a> {
    type Statement = (
        <HSMCLqk<'a> as LHEScheme>::PublicKey,
        <HSMCLqk<'a> as LHEScheme>::Ciphertext,
        <HSMCLqk<'a> as LHEScheme>::Ciphertext,
    );
    type Witness = (
        <HSMCLqk<'a> as LHEScheme>::Message,
        <HSMCLqk<'a> as LHEScheme>::Randomness,
    );
    type Commitment = <HSMCLqk<'a> as LHEScheme>::Ciphertext;
    type Challenge = Integer;
    type Response = (Integer, Integer);
    type ProverState = (Integer, Integer);
    type Error = Error;

    fn check_witness(
        &self,
        (pk, c_old, c_new): &Self::Statement,
        (m, r): &Self::Witness,
    ) -> Result<(), Self::Error> {
        (self.params.cl_ctx.is_in_ZM(m)
            && !r.is_negative()
            && r.significant_bits() <= self.params.integer_witness_size
            && self.hsmcl.rerandomize_with_randomness(
                pk,
                &self.hsmcl.mul_ciphertext_and_message(c_old, m),
                r,
            ) == *c_new)
            .then_some(())
            .ok_or(Error::InvalidArgument("invalid witness".to_owned()))
    }

    fn commit(
        &self,
        statement: &Self::Statement,
        witness: &Self::Witness,
    ) -> Result<(Self::Commitment, Self::ProverState), Self::Error> {
        self.check_witness(statement, witness)?;
        let (pk, c_old, _) = statement;
        let r_m = random_integer_bits(self.params.message_randomness_size);
        let (c_r, r_r) = self
            .hsmcl
            .rerandomize(pk, &self.hsmcl.mul_ciphertext_and_message(c_old, &r_m));
        Ok((c_r, (r_m, r_r)))
    }

    fn challenge(&self) -> Self::Challenge {
        random_integer_bits(self.params.challenge_size)
    }

    fn respond(
        &self,
        (r_m, r_r): &Self::ProverState,
        statement: &Self::Statement,
        witness: &Self::Witness,
        k: &Self::Challenge,
    ) -> Result<Self::Response, Self::Error> {
        self.check_witness(statement, witness)?;
        let (m, r) = witness;
        if k.significant_bits() > self.params.challenge_size {
            return Err(Error::ProtocolError("challenge too large".to_owned()));
        }
        let u_m = Integer::from(k * m + r_m);
        let u_r = Integer::from(k * r + r_r);
        Ok((u_m, u_r))
    }

    fn verify(
        &self,
        (pk, c_old, c_new): &Self::Statement,
        c_r: &Self::Commitment,
        k: &Self::Challenge,
        (u_m, u_r): &Self::Response,
    ) -> bool {
        let lhs = self.hsmcl.rerandomize_with_randomness(
            pk,
            &self.hsmcl.mul_ciphertext_and_message(c_old, u_m),
            u_r,
        );
        let rhs = self
            .hsmcl
            .add_ciphertexts(&self.hsmcl.mul_ciphertext_and_message(c_new, k), c_r);
        lhs == rhs
    }
}

/// \Sigma protocol for a proof of decryption (PoDec) of the HSM-CL
/// encryption scheme.
///
/// "I know a secret key corresponding to the given public key and a message
/// that has been obtained by decrypting this ciphertext."
pub struct PoDec<'a> {
    params: ProofParams<'a>,
    hsmcl: &'a HSMCLqk<'a>,
    dlog_proof: CLFixedBaseDLog<'a>,
}

impl<'a> HSMCLProof<'a> for PoDec<'a> {
    fn new(
        cl_ctx: &'a CLContext,
        hsmcl: &'a HSMCLqk,
        zk_parameter: u32,
        soundness_parameter: u32,
        integer_witness_size: u32,
    ) -> Self {
        Self {
            params: ProofParams::new(
                cl_ctx,
                zk_parameter,
                soundness_parameter,
                integer_witness_size,
            ),
            hsmcl,
            dlog_proof: CLFixedBaseDLog::new(
                cl_ctx,
                hsmcl.is_compact(),
                zk_parameter,
                soundness_parameter,
                integer_witness_size,
            ),
        }
    }
}

impl<'a> SigmaProtocol for PoDec<'a> {
    type Statement = (
        <HSMCLqk<'a> as LHEScheme>::PublicKey,
        <HSMCLqk<'a> as LHEScheme>::Ciphertext,
    );
    type Witness = (
        <HSMCLqk<'a> as LHEScheme>::SecretKey,
        <HSMCLqk<'a> as LHEScheme>::Message,
    );
    type Commitment = (GroupElement, GroupElement);
    type Challenge = Integer;
    type Response = (Integer, Integer);
    type ProverState = (Integer, Integer);
    type Error = Error;

    fn check_witness(
        &self,
        (pk, c): &Self::Statement,
        (sk, m): &Self::Witness,
    ) -> Result<(), Self::Error> {
        self.dlog_proof.check_witness(pk, sk)?;
        let decrypted_m = self
            .hsmcl
            .decrypt(sk, c)
            .map_err(|_| Error::InvalidArgument("invalid witness".to_owned()))?;
        (self.params.cl_ctx.is_in_ZM(m) && decrypted_m == *m)
            .then_some(())
            .ok_or(Error::InvalidArgument("invalid witness".to_owned()))
    }

    fn commit(
        &self,
        statement: &Self::Statement,
        witness: &Self::Witness,
    ) -> Result<(Self::Commitment, Self::ProverState), Self::Error> {
        self.check_witness(statement, witness)?;
        let (pk, (c1, _)) = statement;
        let (sk, _) = witness;
        let (pk_r, r_sk) = self.dlog_proof.commit(pk, sk)?;
        let r_m = random_integer_below(&self.params.cl_ctx.get_M(), self.params.zk_parameter);
        let c2_r = self.params.cl_ctx.mul_in_G_hat(
            &self.params.cl_ctx.power_of_f(&r_m),
            &if self.hsmcl.is_compact() {
                self.params
                    .cl_ctx
                    .map_psi_Gamma_to_G(&self.params.cl_ctx.exp_in_Gamma_hat(c1, &r_sk))
            } else {
                self.params.cl_ctx.exp_in_G_hat(c1, &r_sk)
            },
        );
        Ok(((pk_r, c2_r), (r_sk, r_m)))
    }
    fn challenge(&self) -> Self::Challenge {
        random_integer_bits(self.params.challenge_size)
    }

    fn respond(
        &self,
        (r_sk, r_m): &Self::ProverState,
        statement: &Self::Statement,
        witness: &Self::Witness,
        k: &Self::Challenge,
    ) -> Result<Self::Response, Self::Error> {
        self.check_witness(statement, witness)?;
        if k.significant_bits() > self.params.challenge_size {
            return Err(Error::ProtocolError("challenge too large".to_owned()));
        }
        let (pk, _) = statement;
        let (sk, m) = witness;
        let u_sk = self.dlog_proof.respond(r_sk, pk, sk, k)?;
        let u_m = {
            let mut t = Integer::from(k * m + r_m);
            t.modulo_mut(&self.params.cl_ctx.get_M());
            t
        };
        Ok((u_sk, u_m))
    }

    fn verify(
        &self,
        (pk, (c1, c2)): &Self::Statement,
        (pk_r, c2_r): &Self::Commitment,
        k: &Self::Challenge,
        (u_sk, u_m): &Self::Response,
    ) -> bool {
        let lhs = self.params.cl_ctx.mul_in_G_hat(
            &self.params.cl_ctx.power_of_f(u_m),
            &if self.hsmcl.is_compact() {
                self.params
                    .cl_ctx
                    .map_psi_Gamma_to_G(&self.params.cl_ctx.exp_in_Gamma_hat(c1, u_sk))
            } else {
                self.params.cl_ctx.exp_in_G_hat(c1, u_sk)
            },
        );
        let rhs = self
            .params
            .cl_ctx
            .mul_in_G_hat(&self.params.cl_ctx.exp_in_G_hat(c2, k), c2_r);

        self.dlog_proof.verify(pk, pk_r, k, u_sk) && lhs == rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cl_context::tests::make_test_cl_context;

    fn test_sigma<P: SigmaProtocol>(sigma: &P, statement: &P::Statement, witness: &P::Witness)
    where
        <P as SigmaProtocol>::Challenge: PartialEq,
    {
        assert!(sigma.check_witness(statement, witness).is_ok());
        let (commitment, prover_state) = sigma
            .commit(statement, witness)
            .expect("commit should work");
        let challenge = sigma.challenge();
        let response = sigma
            .respond(&prover_state, statement, witness, &challenge)
            .expect("respond should work");
        assert!(sigma.verify(statement, &commitment, &challenge, &response));

        // Check that the verifier does not trivially accept by testing that it fails
        // with a different challenge.
        let wrong_challenge = {
            let mut k = sigma.challenge();
            while k == challenge {
                k = sigma.challenge();
            }
            k
        };
        assert!(!sigma.verify(statement, &commitment, &wrong_challenge, &response));
    }

    fn test_dlog(in_gamma: bool) {
        let zk_parameter = 40;
        let soundness_parameter = 112;

        let cl_ctx = make_test_cl_context();
        let integer_witness_size = cl_ctx.secretkey_bound().significant_bits();
        let x = random_integer_bits(integer_witness_size);
        let y = if in_gamma {
            cl_ctx.power_of_gamma(&x)
        } else {
            cl_ctx.power_of_h(&x)
        };

        let sigma = CLFixedBaseDLog::new(
            &cl_ctx,
            in_gamma,
            zk_parameter,
            soundness_parameter,
            integer_witness_size,
        );
        let (statement, witness) = (y, x);
        test_sigma(&sigma, &statement, &witness);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dlog_in_H() {
        test_dlog(false);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_dlog_in_Gamma() {
        test_dlog(true);
    }

    fn test_popk_w_param(compact: bool) {
        let zk_parameter = 40;
        let soundness_parameter = 112;

        let cl_ctx = make_test_cl_context();
        let hsmcl = HSMCLqk::new(&cl_ctx, compact);
        let integer_witness_size = cl_ctx.secretkey_bound().significant_bits();
        let (_, pk) = hsmcl.generate_key_pair();
        let m = random_integer_below(&cl_ctx.get_M(), 40);
        let (c, r) = hsmcl.encrypt(&pk, &m).expect("encryption should succeed");

        let sigma = PoPK::new(
            &cl_ctx,
            &hsmcl,
            zk_parameter,
            soundness_parameter,
            integer_witness_size,
        );
        let (statement, witness) = ((pk, c), (m, r));
        test_sigma(&sigma, &statement, &witness);
    }

    #[test]
    fn test_popk() {
        test_popk_w_param(false);
    }

    #[test]
    fn test_popk_compact() {
        test_popk_w_param(true);
    }

    fn test_pomk_w_param(compact: bool) {
        let zk_parameter = 40;
        let soundness_parameter = 112;

        let cl_ctx = make_test_cl_context();
        let hsmcl = HSMCLqk::new(&cl_ctx, compact);
        let integer_witness_size = cl_ctx.secretkey_bound().significant_bits();
        let (_, pk) = hsmcl.generate_key_pair();
        let m1 = random_integer_below(&cl_ctx.get_M(), 40);
        let m2 = random_integer_below(&cl_ctx.get_M(), 40);
        let (c1, _) = hsmcl.encrypt(&pk, &m1).expect("encryption should succeed");
        let (c2, r2) = hsmcl.rerandomize(&pk, &hsmcl.mul_ciphertext_and_message(&c1, &m2));

        let sigma = PoMK::new(
            &cl_ctx,
            &hsmcl,
            zk_parameter,
            soundness_parameter,
            integer_witness_size,
        );
        let (statement, witness) = ((pk, c1, c2), (m2, r2));
        test_sigma(&sigma, &statement, &witness);
    }

    #[test]
    fn test_pomk() {
        test_pomk_w_param(false);
    }

    #[test]
    fn test_pomk_compact() {
        test_pomk_w_param(true);
    }

    fn test_podec_w_param(compact: bool) {
        let zk_parameter = 40;
        let soundness_parameter = 112;

        let cl_ctx = make_test_cl_context();
        let hsmcl = HSMCLqk::new(&cl_ctx, compact);
        let integer_witness_size = cl_ctx.secretkey_bound().significant_bits();
        let (sk, pk) = hsmcl.generate_key_pair();
        let m = random_integer_below(&cl_ctx.get_M(), 40);
        let (c, _) = hsmcl.encrypt(&pk, &m).expect("encryption should succeed");
        let decrypted_m = hsmcl.decrypt(&sk, &c).expect("decryption should succeed");
        assert_eq!(decrypted_m, m);

        let sigma = PoDec::new(
            &cl_ctx,
            &hsmcl,
            zk_parameter,
            soundness_parameter,
            integer_witness_size,
        );
        let (statement, witness) = ((pk, c), (sk, m));
        test_sigma(&sigma, &statement, &witness);
    }

    #[test]
    fn test_podec() {
        test_podec_w_param(false);
    }

    #[test]
    fn test_podec_compact() {
        test_podec_w_param(true);
    }
}
