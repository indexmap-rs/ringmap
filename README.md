# ringmap

[![build status](https://github.com/indexmap-rs/ringmap/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/indexmap-rs/ringmap/actions)
[![crates.io](https://img.shields.io/crates/v/ringmap.svg)](https://crates.io/crates/ringmap)
[![docs](https://docs.rs/ringmap/badge.svg)](https://docs.rs/ringmap)
[![rustc](https://img.shields.io/badge/rust-1.68%2B-orange.svg)](https://img.shields.io/badge/rust-1.68%2B-orange.svg)

A pure-Rust hash table which preserves (in a limited sense) insertion order,
with efficient deque-like manipulation of both the front and back ends.

This crate implements compact map and set data-structures,
where the iteration order of the keys is independent from their hash or
value. It preserves insertion order in most mutating operations, and it
allows lookup of entries by either hash table key or numerical index.

# Background

This crate was forked from [`indexmap`](https://crates.io/crates/indexmap),
with the primary difference being a change from `Vec` to `VecDeque` for the
primary item storage. As a result, it has many of the same properties, as
well as a few new ones:

- Order is **independent of hash function** and hash values of keys.
- Fast to iterate.
- Indexed in compact space.
- Efficient pushing and popping from both the front and back.
- Preserves insertion order **as long** as you don't call `.swap_remove_back()`
  or other methods that explicitly change order.
  - In `ringmap`, the regular `.remove()` **does** preserve insertion order,
    equivalent to what `indexmap` calls `.shift_remove()`.
- Uses hashbrown for the inner table, just like Rust's libstd `HashMap` does.

`ringmap` also follows [`ordermap`](https://crates.io/crates/ordermap) in using
its entry order for `PartialEq` and `Eq`, whereas `indexmap` considers the same
entries in *any* order to be equal for drop-in compatibility with `HashMap`
semantics. Using the order is faster, and also allows `ringmap` to implement
`PartialOrd`, `Ord`, and `Hash`.

# Recent Changes

See [RELEASES.md](https://github.com/indexmap-rs/ringmap/blob/main/RELEASES.md).
