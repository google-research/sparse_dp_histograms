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

use crate::error::Error;
use std::borrow::Borrow;

/// Interface for a linearly homomorphic encryption scheme.
pub trait LHEScheme {
    /// Type of a secret key.
    type SecretKey: Clone;
    /// Type of a public key.
    type PublicKey: Clone;
    /// Type of a message.
    type Message: Clone;
    /// Type of a ciphertext.
    type Ciphertext: Clone;
    /// Type of randomness used in encryption.
    type Randomness: Clone;

    /// Generate a new secret key.
    fn generate_secret_key(&self) -> Self::SecretKey;

    /// Compute the public key from a secret key.
    fn compute_public_key(&self, sk: &Self::SecretKey) -> Self::PublicKey;

    /// Generate a new key pair.
    fn generate_key_pair(&self) -> (Self::SecretKey, Self::PublicKey) {
        let sk = self.generate_secret_key();
        let pk = self.compute_public_key(&sk);
        (sk, pk)
    }

    /// Encrypt a message `m` under the public key `pk`.  Obtain the resulting
    /// ciphertext and the randomness used for the encryption.  Returns an
    /// error if `m` is not in the plaintext space.
    ///
    /// Keeping the randomness secret is essential for security. It can be used
    /// to prove knowledge of the plaintext in zero-knowledge.
    fn encrypt(
        &self,
        pk: &Self::PublicKey,
        m: &Self::Message,
    ) -> Result<(Self::Ciphertext, Self::Randomness), Error>;

    /// Encrypt a message with the given randomness.  Encrypt a message `m`
    /// under the public key `pk` with the given randomness `r`.  Obtain the
    /// resulting ciphertext.  Returns an error if `m` is not in the
    /// plaintext space.
    ///
    /// The randomness needs to be sampled from a distribution that induces
    /// almost uniform distributions in H and \Gamma.  Keeping the
    /// randomness secret is essential for security. It can be used to prove
    /// knowledge of the plaintext in zero-knowledge.
    fn encrypt_with_randomness(
        &self,
        pk: &Self::PublicKey,
        m: &Self::Message,
        r: &Self::Randomness,
    ) -> Result<Self::Ciphertext, Error>;

    /// Decrypt a ciphertext `c` using the secret key `sk` and obtain the
    /// encrypted message if successful.
    fn decrypt(&self, sk: &Self::SecretKey, c: &Self::Ciphertext) -> Result<Self::Message, Error>;

    /// Rerandomize a ciphertext `c` under the public key `pk`. Obtain a new
    /// ciphertext containing the same message and the randomness used for
    /// rerandomization.
    fn rerandomize(
        &self,
        pk: &Self::PublicKey,
        c: &Self::Ciphertext,
    ) -> (Self::Ciphertext, Self::Randomness);

    /// Rerandomize a ciphertext `c` under the public key `pk` using the
    /// randomness `r`. Obtain a new ciphertext containing the same message.
    fn rerandomize_with_randomness(
        &self,
        pk: &Self::PublicKey,
        c: &Self::Ciphertext,
        r: &Self::Randomness,
    ) -> Self::Ciphertext;

    /// Perform homomorphic addition of two ciphertexts `c1` and `c2`. The
    /// resulting ciphertext contains the sum of the messages encrypted in
    /// `c1` and `c2`.
    fn add_ciphertexts(&self, c1: &Self::Ciphertext, c2: &Self::Ciphertext) -> Self::Ciphertext;

    /// Perform homomorphic addition of a ciphertext `c` and a message `m`. The
    /// resulting ciphertext contains the sum of the message encrypted in
    /// `c` and `m`.
    fn add_ciphertext_and_message(
        &self,
        c: &Self::Ciphertext,
        m: &Self::Message,
    ) -> Self::Ciphertext;

    /// Perform homomorphic multiplication of a ciphertext `c` and a message
    /// `m`. The resulting ciphertext contains the product of the message
    /// encrypted in `c` and `m`.
    fn mul_ciphertext_and_message(
        &self,
        c: &Self::Ciphertext,
        m: &Self::Message,
    ) -> Self::Ciphertext;
}

/// Interface that enables an `LHEScheme` to be used as a threshold enryption
/// scheme.
pub trait ThresholdLHEScheme: LHEScheme {
    /// Type of a share of a secret key.
    type SecretKeyShare: Clone;
    /// Type of a share of a public key.
    type PublicKeyShare: Clone;
    /// Type of a share of a partial decryption.
    type PartialDecryption: Clone;

    /// Generate shares of the secret and the public key.
    fn generate_key_share(&self) -> (Self::SecretKeyShare, Self::PublicKeyShare);

    /// Combined compute the public key from a non-empty collection of shares.
    fn combine_public_key_shares(
        &self,
        pks: impl IntoIterator<Item = impl Borrow<Self::PublicKeyShare>>,
    ) -> Result<Self::PublicKey, Error>;

    /// Partially decrypt a ciphertext `c` using the a secret key share `sk_i`
    /// and obtain a new ciphertext which is an encryption under the other
    /// parties public keys, i.e., the used key share is no longer needed to
    /// decrypt the resulting ciphertext.
    fn decrypt_to_ciphertext(
        &self,
        sk_i: &Self::SecretKeyShare,
        c: &Self::Ciphertext,
    ) -> Self::Ciphertext;

    /// Partially decrypt a ciphertext `c` using the a secret key share `sk_i`
    /// and obtain a partial decryption.
    fn partially_decrypt(
        &self,
        sk_i: &Self::SecretKeyShare,
        c: &Self::Ciphertext,
    ) -> Self::PartialDecryption;

    /// Combine the partial decryptions `pds` to complete the decryption of a
    /// ciphertext `c`.
    fn complete_decryption(
        &self,
        c: &Self::Ciphertext,
        pds: impl IntoIterator<Item = impl Borrow<Self::PartialDecryption>>,
    ) -> Result<Self::Message, Error>;
}

/// Trait for \Sigma protocols, which are public-coin three-round protocols
/// between a prover and a verifier of the following form:
///
/// 1. The prover sends a commitment.
/// 2. The verifier sends a randomly sampled challenge.
/// 3. The prover sends a response.
///
/// Finally, the verifier accepts or reject depending on the transcript.
pub trait SigmaProtocol {
    /// Type of the public statement to prove.
    type Statement;
    /// Type of the secret witness to prove.
    type Witness;
    /// Type of the first message from the prover to the verifier.
    type Commitment;
    /// Type of the second (randomly sampled) message from the verifier to the
    /// prover.
    type Challenge;
    /// Type of the third message from the prover to the verifier.
    type Response;
    /// State that the prover needs to keep around between commit and response.
    type ProverState;
    /// `Error` type that is returned in the `Result`s.
    type Error;

    /// Test if the provided witness is actually valid for the statement.
    fn check_witness(
        &self,
        statement: &Self::Statement,
        witness: &Self::Witness,
    ) -> Result<(), Error>;

    /// Compute the first (commitment) message of the prover for a given
    /// `statement` and `witness` and output `state` needed to respond to
    /// challenges.
    fn commit(
        &self,
        statement: &Self::Statement,
        witness: &Self::Witness,
    ) -> Result<(Self::Commitment, Self::ProverState), Error>;

    /// Sample a random challenge.
    fn challenge(&self) -> Self::Challenge;

    /// Compute the third message (response) of the prover for a given
    /// `statement`, `witness`, `state`.
    fn respond(
        &self,
        state: &Self::ProverState,
        y: &Self::Statement,
        x: &Self::Witness,
        challenge: &Self::Challenge,
    ) -> Result<Self::Response, Error>;

    /// Verify the transcript (`commitment`, `challenge`, `response`) for a
    /// given `statement`.
    fn verify(
        &self,
        statement: &Self::Statement,
        commitment: &Self::Commitment,
        challenge: &Self::Challenge,
        response: &Self::Response,
    ) -> bool;
}
