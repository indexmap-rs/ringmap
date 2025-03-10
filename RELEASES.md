# Releases

## 0.1.2 (2025-03-10)

- Added `ringmap_with_default!` and `ringset_with_default!` to be used with
  alternative hashers, especially when using the crate without `std`.
- Implemented `PartialEq` between each `Slice` and `[]`/arrays.

## 0.1.1 (2025-01-29)

- Optimized the branch behavior of the iterators.

## 0.1.0 (2025-01-21)

- Initial release, based on `indexmap v2.7.1`.
