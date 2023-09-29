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

#include "rand.h"

namespace bicycl_rs_helpers {

BICYCL::Mpz SecureRandGen::random_mpz(const BICYCL::Mpz& modulus) {
  BICYCL::Mpz r =
      random_mpz_2exp(modulus.nbits() + statistical_security_parameter_);
  BICYCL::Mpz output;
  BICYCL::Mpz::mod(output, r, modulus);
  return output;
}

BICYCL::Mpz SecureRandGen::random_mpz_2exp(mp_bitcnt_t n_bits) {
  if (!n_bits) {
    return BICYCL::Mpz();
  }
  size_t byte_size = n_bits / 8;
  size_t bits_in_last_byte = n_bits & 7;
  if (bits_in_last_byte) {
    byte_size += 1;
  }
  if (byte_size > INT_MAX) {
    throw std::invalid_argument("to many random bits");
  }
  std::vector<unsigned char> buf(byte_size);
  int ret = RAND_bytes(buf.data(), byte_size);
  if (ret != 1) {
    throw std::runtime_error("random number generation failed");
  }

  if (bits_in_last_byte) {
    buf[byte_size - 1] <<= 8 - bits_in_last_byte;
  }

  mpz_t r;
  mpz_init(r);
  mpz_import(r, byte_size, -1, 1, -1, 0, buf.data());

  return BICYCL::Mpz(r);
}

unsigned char SecureRandGen::random_uchar() {
  unsigned char output;
  int ret = RAND_bytes(&output, sizeof(output));
  if (ret != 1) {
    throw std::runtime_error("random number generation failed");
  }
  return output;
}

unsigned long SecureRandGen::random_ui(unsigned long mod) {
  BICYCL::Mpz modulus(mod);
  BICYCL::Mpz output;
  BICYCL::Mpz r =
      random_mpz_2exp(modulus.nbits() + statistical_security_parameter_);
  BICYCL::Mpz::mod(output, r, modulus);
  return output;
}

unsigned long SecureRandGen::random_ui_2exp(mp_bitcnt_t n_bits) {
  unsigned long output;
  size_t bit_size = 8 * sizeof(output);
  if (n_bits > bit_size) {
    throw std::invalid_argument("to many random bits for this datatype");
  }
  int ret =
      RAND_bytes(reinterpret_cast<unsigned char*>(&output), sizeof(output));
  if (ret != 1) {
    throw std::runtime_error("random number generation failed");
  }
  // Clear most significant bits such that we get an n_bits output.
  output >>= (bit_size - n_bits);
  return output;
}

BICYCL::Mpz SecureRandGen::random_prime(mp_bitcnt_t n_bits) {
  if (n_bits <= 1) {
    // There is no 1 bit prime, but this is what BICYCL does.
    return BICYCL::Mpz(2UL);
  } else if (n_bits == 2) {
    return BICYCL::Mpz(random_bool() ? 3UL : 2UL);
  } else {
    BICYCL::Mpz p;
    do {
      p = random_mpz_2exp(n_bits);  // Random number 0 <= p < 2^n_bits.
      p.setbit(n_bits - 1);         // Ensure that p is >= 2^(n_bits-1).
      p.nextprime();                // Ensure p is prime.
    } while (p.nbits() != n_bits);
    return p;
  }
}

}  // namespace bicycl_rs_helpers
