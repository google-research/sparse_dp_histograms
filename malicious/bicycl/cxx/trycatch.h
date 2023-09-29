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

#ifndef BICYCL_RUST_TRYCATCH_H
#define BICYCL_RUST_TRYCATCH_H

#include <stdexcept>
#include <string>

namespace rust {
namespace behavior {

template <typename Try, typename Fail>
static void trycatch(Try &&func, Fail &&fail) noexcept {
  // We want let the Rust library know what kind of exception has occurred.
  // Since the `fail` function only takes a single string as argument, we prefix
  // the error description with the exception type, and then remove it again on
  // the Rust side.
  try {
    func();
  } catch (const std::range_error &e) {
    fail(std::string("range_error__") + e.what());
  } catch (const std::invalid_argument &e) {
    fail(std::string("invalid_argument__") + e.what());
  } catch (const std::runtime_error &e) {
    fail(std::string("runtime_error__") + e.what());
  } catch (const std::exception &e) {
    fail(std::string("exception__") + e.what());
  }
}

}  // namespace behavior
}  // namespace rust

#endif  // BICYCL_RUST_TRYCATCH_H
