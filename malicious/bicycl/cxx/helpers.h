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

#include "bicycl/cxx/rand.h"
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

// Wrapper around QFI's constructor with explicitly given coefficients.
// Throws if the form is not primitive.
inline std::unique_ptr<BICYCL::QFI> qfi_new_with_value(const BICYCL::Mpz& a,
                                                       const BICYCL::Mpz& b,
                                                       const BICYCL::Mpz& c) {
  return std::make_unique<BICYCL::QFI>(a, b, c);
}

// Wrapper around QFI::discriminant, which returns an Mpz by value.
inline std::unique_ptr<BICYCL::Mpz> qfi_get_discriminant(
    const BICYCL::QFI& qfi) {
  return std::make_unique<BICYCL::Mpz>(qfi.discriminant());
}

// Wrapper around QFI::eval, which returns an Mpz by value.
inline std::unique_ptr<BICYCL::Mpz> qfi_eval(const BICYCL::QFI& qfi,
                                             const BICYCL::Mpz& x,
                                             const BICYCL::Mpz& y) {
  return std::make_unique<BICYCL::Mpz>(qfi.eval(x, y));
}

// Wrapper around QFI::neg.
inline void qfi_neg(const std::unique_ptr<BICYCL::QFI>& qfi) { qfi->neg(); }

// Wrapper around QFI::compressed_repr, which returns a
// QFICompressedRepresentation by value.
inline std::unique_ptr<BICYCL::QFICompressedRepresentation>
qfi_to_compressed_repr(const BICYCL::QFI& qfi) {
  return std::make_unique<BICYCL::QFICompressedRepresentation>(
      qfi.compressed_repr());
}

// Wrapper around QFI's constructor, which takes a QFICompressedRepresentation
// and the discriminant.
inline std::unique_ptr<BICYCL::QFI> qfi_from_compressed_repr(
    const BICYCL::QFICompressedRepresentation& comp_qfi,
    const BICYCL::Mpz& discriminant) {
  return std::make_unique<BICYCL::QFI>(comp_qfi, discriminant);
}

// Wrapper around QFI's operator== method.
inline bool qfi_eq(const BICYCL::QFI& lhs, const BICYCL::QFI& rhs) {
  return lhs == rhs;
}

}  // namespace bicycl_rs_helpers

#endif  // BICYCL_RUST_HELPERS_H
