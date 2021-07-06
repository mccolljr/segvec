# segvec

[![docs.rs](https://docs.rs/segvec/badge.svg?version=latest)](https://docs.rs/segvec)
[![crates.io](https://img.shields.io/crates/v/segvec)](https://crates.io/crates/segvec)

This crate provides the `SegVec` data structure.

It is similar to `Vec`, but allocates memory in chunks of increasing size, referred to as
"segments". This involves a few trade-offs:

#### Pros:

- Element addresses are stable across `push` operations even if the `SegVec` must grow.
- Resizing only allocates the additional space needed, and doesn't require copying.

#### Cons:

- Operations are slower (some, like `insert`, `remove`, and `drain`, are much slower) for a `SegVec` than for a `Vec` (multiple pointer dereferences, mapping indices to `(segment, offset)` pairs)
- Direct slicing is unavailable (i.e. no `&[T]` or `&mut [T]`), though `slice` and `slice_mut` are available

## Use Cases

1. You have a long-lived `Vec` whose size fluctuates between very large and very small throughout the life of the program.
2. You have a large append-only `Vec` and would benefit from stable references to the elements

## Features

- `small-vec` - Uses [`SmallVec`](https://github.com/servo/rust-smallvec) instead of `Vec` to store the list of segments, allowing the first few segment headers to live on the stack. Can speed up access for small `SegVec` values.
- `thin-segments` - Uses [`ThinVec`](https://github.com/Gankra/thin-vec) instead of `Vec` to store the data for each segment, meaning that each segment header takes up the space of a single `usize`, rathern than 3 when using `Vec`.

License: MIT
