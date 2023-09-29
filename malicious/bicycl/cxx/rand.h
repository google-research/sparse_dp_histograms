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

#ifndef BICYCL_RUST_RAND_H
#define BICYCL_RUST_RAND_H

#include <openssl/rand.h>

#include "bicycl/gmp_extras.hpp"

namespace bicycl_rs_helpers {

// Reimplementation of BICYCL's random number generation class.  We replace the
// usage of GMPs insecure random number with calls to OpenSSL's `RAND_bytes`.
class SecureRandGen : public BICYCL::RandGen {
 public:
  // Create a new instance by either using the default statistical security
  // parameter or choosing a custom one.
  explicit SecureRandGen(size_t statistical_security_parameter =
                             DEFAULT_STATISTICAL_SECURITY_PARAMETER)
      : BICYCL::RandGen(),
        statistical_security_parameter_(statistical_security_parameter) {}

  // Sample a random big integer modulo a given number.
  BICYCL::Mpz random_mpz(const BICYCL::Mpz&) override;

  // Sample a random big integer with a given number of bits.
  BICYCL::Mpz random_mpz_2exp(mp_bitcnt_t) override;

  // Sample a random unsigned char.
  unsigned char random_uchar() override;
  // Sample a random unsigned integer modulo a given number.
  unsigned long random_ui(unsigned long) override;
  // Sample a random unsigned integer with a given number of bits.
  unsigned long random_ui_2exp(mp_bitcnt_t) override;

  // Sample a random prime number with a given number of bits.
  BICYCL::Mpz random_prime(mp_bitcnt_t) override;

 private:
  // Default value for the statistical security parameter.
  static const size_t DEFAULT_STATISTICAL_SECURITY_PARAMETER = 40;
  // Statistical security parameter used for this instance.
  const size_t statistical_security_parameter_;
};

}  // namespace bicycl_rs_helpers

#endif  // BICYCL_RUST_RAND_H
