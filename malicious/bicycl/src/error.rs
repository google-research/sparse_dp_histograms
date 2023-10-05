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

use cxx;

/// Error type for the `Result`s in this crate. The variants correspond to the
/// exceptions thrown by BICYCL.
#[derive(Clone, Debug)]
pub enum Error {
    RangeError(String),
    InvalidArgument(String),
    RuntimeError(String),
    DecryptionError(String),
    OtherException(String),
    ProtocolError(String),
}

impl From<cxx::Exception> for Error {
    fn from(e: cxx::Exception) -> Self {
        let what = e.what();
        let (etype, msg) = what.split_once("__").expect("TODO");
        match etype {
            "range_error" => Self::RangeError(msg.to_owned()),
            "invalid_argument" => Self::InvalidArgument(msg.to_owned()),
            "runtime_error" => Self::RuntimeError(msg.to_owned()),
            _ => Self::OtherException(msg.to_owned()),
        }
    }
}
