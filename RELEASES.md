# Releases

## 0.1.3 (2025-04-04)

- Added a `get_disjoint_mut` method to `RingMap`, matching Rust 1.86's
  `HashMap` method.
- Added a `get_disjoint_indices_mut` method to `RingMap` and `map::Slice`,
  matching Rust 1.86's `get_disjoint_mut` method on slices.

## 0.1.2 (2025-03-10)

- Added `ringmap_with_default!` and `ringset_with_default!` to be used with
  alternative hashers, especially when using the crate without `std`.
- Implemented `PartialEq` between each `Slice` and `[]`/arrays.

## 0.1.1 (2025-01-29)

- Optimized the branch behavior of the iterators.

## 0.1.0 (2025-01-21)

- Initial release, based on `indexmap v2.7.1`.
