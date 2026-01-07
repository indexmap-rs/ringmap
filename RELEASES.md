# Releases

## 0.2.2 (2026-01-07)

- Added `map::Slice::split_at_checked` and `split_at_mut_checked`.
- Added `set::Slice::split_at_checked`.

## 0.2.1 (2025-11-20)

- Simplified a lot of internals using `hashbrown`'s new bucket API.

## 0.2.0 (2025-10-18)

- **MSRV**: Rust 1.82.0 or later is now required.
- Error types now implement `core::error::Error`.
- Added `pop_back_if` and `pop_front_if` methods to `RingMap` and
  `RingSet`.
- Deprecated map and set `insert_full` in favor of `push_back` and
  `push_front`.
- Added new `Entry` back/front methods and deprecated the old:
  - `insert` => `push_back`, `push_front`
  - `insert_entry` => `push_back_entry`, `push_front_entry`
  - `or_default` => `or_push_back_default`, `or_push_front_default`
  - `or_insert` => `or_push_back`, `or_push_front`
  - `or_insert_with` => `or_push_back_with`, `or_push_front_with`
  - `or_insert_with_key` => `or_push_back_with_key`, `or_push_front_with_key`

## 0.1.8 (2025-09-18)

- Updated the `hashbrown` dependency to version 0.16.

## 0.1.7 (2025-09-15)

- Switched the "serde" feature to depend on `serde_core`, improving build
  parallelism in cases where other dependents have enabled "serde/derive".

## 0.1.6 (2025-09-08)

- Added a `get_key_value_mut` method to `RingMap`.
- Removed the unnecessary `Ord` bound on `insert_sorted_by` methods.

## 0.1.5 (2025-08-22)

- Added `insert_sorted_by` and `insert_sorted_by_key` methods to `RingMap`,
  `RingSet`, and `VacantEntry`, like customizable versions of `insert_sorted`.
- Added `is_sorted`, `is_sorted_by`, and `is_sorted_by_key` methods to
  `RingMap` and `RingSet`, as well as their `Slice` counterparts.
- Added `sort_by_key` and `sort_unstable_by_key` methods to `RingMap` and
  `RingSet`, as well as parallel counterparts.
- Added `replace_index` methods to `RingMap`, `RingSet`, and `VacantEntry`
  to replace the key (or set value) at a given index.
- Added optional `sval` serialization support.

## 0.1.4 (2025-06-26)

- Added `extract_if` methods to `RingMap` and `RingSet`, similar to the
  methods for `HashMap` and `HashSet` with ranges like `Vec::extract_if`.
- Added more `#[track_caller]` annotations to functions that may panic.

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
