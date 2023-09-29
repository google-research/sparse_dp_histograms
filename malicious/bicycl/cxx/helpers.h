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

#ifndef BICYCL_RUST_HELPERS_H
#define BICYCL_RUST_HELPERS_H

#include <memory>

#include "bicycl/gmp_extras.hpp"
#include "bicycl/qfi.hpp"
#include "trycatch.h"

namespace bicycl_rs_helpers {

// Wrapper around Mpz's default constructor.
inline std::unique_ptr<BICYCL::Mpz> mpz_new() {
  return std::make_unique<BICYCL::Mpz>();
}

// Wrapper around Mpz's `mpz_ptr` conversion operator.
// Used because cast operators seem unsupported by cxx.
inline mpz_ptr mpz_get_raw_mpz(BICYCL::Mpz* mpz) { return mpz_ptr(*mpz); }

// Wrapper around Mpz's `mpz_srcptr` conversion operator.
// Used because cast operators seem unsupported by cxx.
inline mpz_srcptr mpz_get_raw_mpz(const BICYCL::Mpz* mpz) {
  return mpz_srcptr(*mpz);
}

}  // namespace bicycl_rs_helpers

#endif  // BICYCL_RUST_HELPERS_H
