# segvec

![GitHub branch checks state](https://img.shields.io/github/checks-status/mccolljr/segvec/master)
![Documentation on docs.rs](https://docs.rs/segvec/badge.svg?version=latest)

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

License: MIT
