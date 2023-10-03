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

/// Interface for a linearly homomorphic encryption scheme.
pub trait LHEScheme {
    /// Type of a secret key.
    type SecretKey;
    /// Type of a public key.
    type PublicKey;
    /// Type of a message.
    type Message;
    /// Type of a ciphertext.
    type Ciphertext;
    /// Type of randomness used in encryption.
    type Randomness;

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
    fn decrypt(&self, sk: &Self::SecretKey, c: &Self::Ciphertext) -> Option<Self::Message>;

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
