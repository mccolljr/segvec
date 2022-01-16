//! This crate provides the [`SegVec`][crate::SegVec] data structure.
//!
//! It is similar to [`Vec`][std::vec::Vec], but allocates memory in chunks of increasing size, referred to as
//! "segments". This involves a few trade-offs:
//!
//! #### Pros:
//!
//! - Element addresses are stable across [`push`][crate::SegVec::push] operations even if the `SegVec` must grow.
//! - Resizing only allocates the additional space needed, and doesn't require copying.
//!
//! #### Cons:
//!
//! - Operations are slower (some, like [`insert`][crate::SegVec::insert], [`remove`][crate::SegVec::remove], and [`drain`][crate::SegVec::drain], are much slower) than for a `Vec`
//!    due to the need for multiple pointer dereferences and conversion between linear indexes and `(segment, offset)` pairs
//! - Direct slicing is unavailable (i.e. no `&[T]` or `&mut [T]`), though `slice` and `slice_mut` are available
//!
//! ## Use Cases
//!
//! 1. You have a long-lived `Vec` whose size fluctuates between very large and very small throughout the life of the program.
//! 2. You have a large append-only `Vec` and would benefit from stable references to the elements
//!
//! ## Features
//!
//! - `small-vec` - Uses [`SmallVec`](https://github.com/servo/rust-smallvec) instead of `Vec` to store the list of segments, allowing the first few segment headers to live on the stack. Can speed up access for small `SegVec` values.
//! - `thin-segments` - Uses [`ThinVec`](https://github.com/Gankra/thin-vec) instead of `Vec` to store the data for each segment, meaning that each segment header takes up the space of a single `usize`, rathern than 3 when using `Vec`.

use std::{
    cmp,
    convert::TryFrom,
    fmt::Debug,
    hash::Hash,
    iter::{FromIterator, FusedIterator},
    mem,
    num::NonZeroUsize,
    ops::{Bound, Index, IndexMut, RangeBounds},
};

#[cfg(test)]
mod tests;

mod inner {
    #[cfg(feature = "thin-segments")]
    pub type Segment<T> = thin_vec::ThinVec<T>;
    #[cfg(not(feature = "thin-segments"))]
    pub type Segment<T> = Vec<T>;

    #[cfg(feature = "small-vec")]
    pub type Segments<T> = smallvec::SmallVec<[Segment<T>; 3]>;
    #[cfg(not(feature = "small-vec"))]
    pub type Segments<T> = Vec<Segment<T>>;
}

/// A data structure similar to [`Vec`][std::vec::Vec], but that does not copy on re-size and can
/// release memory when it is truncated.
///
/// Capacity is allocated in "segments". Assuming the default growth factor of 1:
/// - A `SegVec` with a capacity of 0 does not allocate.
/// - A `SegVec` with a capacity of 1 allocates a single segment of length 1.
/// - A `SegVec` with a capacity of 2 allocates two segments of length 1.
/// - A `SegVec` with a capacity of 3 or 4 elements allocates two segments of length one, and a segment of length 2.
///
/// Each subsequent segment is allocated with a capacity equal to the total capacity of the preceeding
/// segments. In other words, each segment after the first segment doubles the capacity of the `SegVec`.
/// If the growth factor is a power of two (such as the default growth factor of 1), the capacity of the
/// `SegVec` will always be a power of two.
///
/// It is possible to specify a growth factor using [`SegVec::with_factor`][crate::SegVec::with_factor].
/// By choosing an appropriate growth factor, allocation count and memory usage can be fine-tuned.
pub struct SegVec<T> {
    factor: NonZeroUsize,
    len: usize,
    capacity: usize,
    segments: inner::Segments<T>,
}

impl<T> SegVec<T> {
    /// Create a new [`SegVec`][crate::SegVec] with a length and capacity of 0 using the default growth factor of 1.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// assert_eq!(v.capacity(), 0);
    /// v.reserve(1);
    /// assert_eq!(v.capacity(), 1);
    /// ```
    pub fn new() -> Self {
        Self::with_factor(1)
    }

    /// Create a new [`SegVec`][crate::SegVec] with a length and capacity of 0, and the given growth factor.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::with_factor(4);
    /// assert_eq!(v.capacity(), 0);
    /// v.reserve(1);
    /// assert_eq!(v.capacity(), 4);
    /// ```
    ///
    /// # Panics
    /// - If `factor` is zero
    pub fn with_factor(factor: usize) -> Self {
        let factor = NonZeroUsize::new(factor).expect("non-zero factor");
        SegVec {
            factor,
            len: 0,
            capacity: 0,
            segments: inner::Segments::new(),
        }
    }

    /// Create a new [`SegVec`][crate::SegVec] with a length of 0 and a capacity large enough to
    /// hold the given number of elements, using the default growth factor of 1.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let v: SegVec<i32> = SegVec::with_capacity(5);
    /// assert_eq!(v.capacity(), 8);
    /// ```
    ///
    /// # Panics
    /// - If the required capacity overflows `usize`
    pub fn with_capacity(capacity_hint: usize) -> Self {
        let mut v = SegVec::new();
        v.reserve(capacity_hint);
        v
    }

    /// Create a new [`SegVec`][crate::SegVec] with a length of 0 and a capacity large enough to
    /// hold the given number of elements, using the provided growth factor.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let v: SegVec<i32> = SegVec::with_capacity_and_factor(5, 4);
    /// assert_eq!(v.capacity(), 8);
    /// ```
    ///
    /// # Panics
    /// - If the required capacity overflows `usize`
    pub fn with_capacity_and_factor(capacity_hint: usize, factor: usize) -> Self {
        let mut v = SegVec::with_factor(factor);
        v.reserve(capacity_hint);
        v
    }

    /// The number of elements in the [`SegVec`][crate::SegVec]
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// v.push(2);
    /// assert_eq!(v.len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// The capacity of the [`SegVec`][crate::SegVec]
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::with_capacity(3);
    /// assert_eq!(v.capacity(), 4);
    /// ```
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Reserve enough capacity to insert the given number of elements into the
    /// [`SegVec`][crate::SegVec] without allocating. If the capacity is already sufficient,
    /// nothing happens.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// assert_eq!(v.capacity(), 0);
    /// v.reserve(3);
    /// assert_eq!(v.capacity(), 4);
    /// ```
    ///
    /// # Panics
    /// - If the required capacity overflows `usize`
    pub fn reserve(&mut self, additional: usize) {
        let min_cap = match self.len().checked_add(additional) {
            Some(c) => c,
            None => capacity_overflow(),
        };
        if min_cap <= self.capacity() {
            return;
        }
        let (segment, _) = self.segment_and_offset(min_cap - 1);
        for i in self.segments.len()..=segment {
            let seg_size = self.segment_capacity(i);
            #[cfg(feature = "thin-segments")]
            self.segments.push(inner::Segment::with_capacity(seg_size));
            #[cfg(not(feature = "thin-segments"))]
            self.segments.push(Vec::with_capacity(seg_size));
            self.capacity += seg_size;
        }
    }

    /// Returns a reference to the data at the given index in the [`SegVec`][crate::SegVec], if it
    /// exists.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// assert_eq!(v.get(0), None);
    /// v.push(1);
    /// assert_eq!(*v.get(0).unwrap(), 1);
    /// ```
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len {
            let (seg, offset) = self.segment_and_offset(index);
            Some(&self.segments[seg][offset])
        } else {
            None
        }
    }

    /// Returns a mutable reference to the data at the given index in the [`SegVec`][crate::SegVec],
    /// if it exists.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// assert_eq!(v.get_mut(0), None);
    /// v.push(1);
    /// assert_eq!(*v.get_mut(0).unwrap(), 1);
    /// ```
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len {
            let (seg, offset) = self.segment_and_offset(index);
            Some(&mut self.segments[seg][offset])
        } else {
            None
        }
    }

    /// Pushes a new value onto the end of the [`SegVec`][crate::SegVec], resizing if necessary.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// assert_eq!(v[0], 1);
    /// ```
    ///
    /// # Panics
    /// - If the required capacity overflows `usize`
    pub fn push(&mut self, val: T) {
        self.reserve(1);
        let (seg, _) = self.segment_and_offset(self.len);
        self.segments[seg].push(val);
        self.len += 1;
    }

    /// Removes the last value from the [`SegVec`][crate::SegVec] and returns it, or returns `None`
    /// if it is empty.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// assert_eq!(v.pop().unwrap(), 1);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        match self.len {
            0 => None,
            size => {
                let (seg, offset) = self.segment_and_offset(size);
                self.len -= 1;
                match offset {
                    0 => self.segments[seg - 1].pop(),
                    _ => self.segments[seg].pop(),
                }
            }
        }
    }

    /// Truncates the [`SegVec`][crate::SegVec] to the given length.
    /// If the given length is larger than the current length, this is a no-op.
    /// Otherwise, the capacity is reduced and any excess elements are dropped.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// v.push(2);
    /// v.push(3);
    /// assert_eq!(v.len(), 3);
    /// assert_eq!(v.capacity(), 4);
    /// v.truncate(1);
    /// assert_eq!(v.len(), 1);
    /// assert_eq!(v.capacity(), 1);
    /// ```
    pub fn truncate(&mut self, len: usize) {
        if len < self.capacity {
            let (seg, offset) = self.segment_and_offset(len);
            if offset == 0 {
                self.segments.drain(seg..);
            } else {
                if len < self.len {
                    self.segments[seg].drain(offset..);
                }
                self.segments.drain(seg + 1..);
            }
            self.capacity = match self.segments.len() {
                0 => 0,
                n => 2usize.pow((n - 1) as u32),
            };
            self.len = len;
        }
    }

    /// Resizes the [`SegVec`][crate::SegVec] so that the length is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the `SegVec` is extended by the difference, with each additional slot filled with the result of calling the closure `f`.
    /// The return values from `f` will end up in the `SegVec` in the order they have been generated.
    /// If `new_len` is less than `len`, the `SegVec` is simply truncated.
    /// If `new_len` is equal to `len`, this is a no-op.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// let mut counter = 0i32;
    /// v.resize_with(4, || { counter += 1; counter });
    /// assert_eq!(counter, 4);
    /// assert_eq!(v.len(), 4);
    /// assert_eq!(v.pop().unwrap(), 4);
    /// ```
    pub fn resize_with<F>(&mut self, new_len: usize, f: F)
    where
        F: FnMut() -> T,
    {
        let cur_len = self.len();
        if new_len > cur_len {
            let to_add = new_len - cur_len;
            self.extend(std::iter::repeat_with(f).take(to_add));
        } else if new_len < cur_len {
            self.truncate(new_len);
        }
    }

    /// Resizes the [`SegVec`][crate::SegVec] so that the length is equal to `new_len`.
    ///
    /// If `new_len` is greater than `len`, the `SegVec` is extended by the difference, with each additional slot filled with the result of calling the `clone` on `val`.
    /// The cloned values will end up in the `SegVec` in the order they have been generated.
    /// If `new_len` is less than `len`, the `SegVec` is simply truncated.
    /// If `new_len` is equal to `len`, this is a no-op.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.resize(4, 100);
    /// assert_eq!(v.len(), 4);
    /// assert_eq!(v.pop().unwrap(), 100);
    /// ```
    pub fn resize(&mut self, new_len: usize, val: T)
    where
        T: Clone,
    {
        let cur_len = self.len();
        if new_len > cur_len {
            let to_add = new_len - cur_len;
            self.extend(std::iter::repeat(val).take(to_add));
        } else if new_len < cur_len {
            self.truncate(new_len);
        }
    }

    /// Returns an iterator over immutable references to the elements in the
    /// [`SegVec`][crate::SegVec].
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// v.push(2);
    /// v.push(3);
    /// let mut i = v.iter();
    /// assert_eq!(*i.next().unwrap(), 1);
    /// assert_eq!(*i.next().unwrap(), 2);
    /// assert_eq!(*i.next().unwrap(), 3);
    /// assert_eq!(i.next(), None);
    /// ```
    pub fn iter(&self) -> Iter<T> {
        Iter {
            size: self.len,
            iter: self.segments.iter().flatten(),
        }
    }

    /// Insert the given value at the given index in the [`SegVec`][crate::SegVec].
    /// This operation requires `O(N)` time due to the fact that the data is segmented -
    /// the new element is pushed onto the end and then shifted backwards into position.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// v.push(2);
    /// v.insert(0, 100);
    /// assert_eq!(v[0], 100);
    /// ```
    ///
    /// # Panics
    /// - If the given index is greater than `self.len()`
    /// - If the required capacity overflows `usize`
    pub fn insert(&mut self, index: usize, val: T) {
        if index > self.len {
            index_oob("SegVec::insert", index, self.len);
        }
        if mem::size_of::<T>() == 0 {
            self.push(val);
            return;
        }
        self.reserve(1);
        let (mut seg_idx, mut seg_offset) = self.segment_and_offset(index);
        let mut displaced = val;
        loop {
            let maybe_displaced = unsafe {
                let segment = &mut self.segments[seg_idx];
                let seg_len = segment.len();
                let seg_cap = segment.capacity();
                if seg_len == 0 {
                    debug_assert!(
                        seg_offset == 0,
                        "expected offset == 0 when inserting into an empty segment"
                    );
                    segment.push(displaced);
                    None
                } else if seg_len < seg_cap {
                    debug_assert!(
                        seg_offset <= seg_len,
                        "expected offset <= len when inserting into a partially full segment"
                    );
                    let src_ptr = segment.as_mut_ptr().add(seg_offset);
                    let dst_ptr = src_ptr.add(1);
                    std::ptr::copy(src_ptr, dst_ptr, seg_len - seg_offset);
                    std::ptr::write(src_ptr, displaced);
                    segment.set_len(seg_len + 1);
                    None
                } else {
                    debug_assert!(
                        seg_offset < seg_len,
                        "expected offset < len when inserting into a full segment"
                    );
                    let new_displaced = std::ptr::read(&mut segment[seg_len - 1]);
                    let src_ptr = segment.as_mut_ptr().add(seg_offset);
                    let dst_ptr = src_ptr.add(1);
                    std::ptr::copy(src_ptr, dst_ptr, seg_len - seg_offset - 1);
                    std::ptr::write(src_ptr, displaced);
                    Some(new_displaced)
                }
            };
            match maybe_displaced {
                Some(new_displaced) => {
                    displaced = new_displaced;
                    seg_idx += 1;
                    seg_offset = 0;
                }
                None => break,
            }
        }
        self.len += 1
    }

    /// Removes the value at the given index in the [`SegVec`][crate::SegVec] and returns it.
    /// This operation requires `O(N)` time due to the fact that the data is segmented -
    /// the element is shifted to the end and then popped.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// v.push(2);
    /// assert_eq!(v.remove(1), 2);
    /// assert_eq!(v.len(), 1);
    /// ```
    ///
    /// # Panics
    /// - If the given index is greater than or equal to `self.len()`
    pub fn remove(&mut self, index: usize) -> T {
        if index >= self.len {
            index_oob("SegVec::remove", index, self.len);
        }
        if mem::size_of::<T>() == 0 {
            return self.pop().unwrap();
        }
        let (mut seg_idx, seg_offset) = self.segment_and_offset(index);
        // SAFETY:
        // At this point, it is known that index points to a valid, non-zero-sized T in
        // the structure, and so it is safe to read a value of type T from this location
        let removed = unsafe { std::ptr::read(&self.segments[seg_idx][seg_offset]) };
        let mut orig_len = self.segments[seg_idx].len();
        let mut orig_cap = self.segments[seg_idx].capacity();
        // SAFETY:
        // 1. index is known to be strictly less than self.len
        // 2. from #1, seg_offset is known to be strictly less than orig_len
        // 3. from #2, seg_offset + 1 is known to be less than or equal to orig_len,
        //    and orig_len - 1 is known to be greater than or equal to seg_offset
        // 4. from #1-3, the pseudo-operation copy(segment[seg_offset+1..orig_len], segment[seg_offset..orig_len-1])
        //    is known to be valid
        // 5. all elements from 0 to orig_len are initialized, by definition of orig_len
        // 6. from #2 and #5, we know it is safe to set the length of the segment to orig_len-1
        unsafe {
            // copy segment[seg_offset+1..orig_len] to segment[seg_offset..orig_len-1], then reduce the length of the segment by 1:
            //  before copy: [_, X, a, b, c] (X is the value read into `removed`)
            //   after copy: [_, a, b, c, c]
            // after resize: [_, a, b, c]    (the second c is not dropped, per the implementation of `set_len`)
            let dst_ptr = &mut self.segments[seg_idx][seg_offset] as *mut T;
            let src_ptr = dst_ptr.add(1);
            std::ptr::copy(src_ptr, dst_ptr, orig_len - seg_offset - 1);
            self.segments[seg_idx].set_len(orig_len - 1);
        }
        // if the initial segment was full, it may be necessary to shift elements back from subsequent segments to keep continuity
        while orig_len == orig_cap {
            // advance to the next segment index (which may be out of bounds)
            seg_idx += 1;
            if seg_idx >= self.segments.len() {
                break;
            }
            // the segment at seg_idx is now known to be in-bounds
            orig_len = self.segments[seg_idx].len();
            orig_cap = self.segments[seg_idx].capacity();
            if orig_len > 0 {
                // SAFETY:
                // orig_len is known to be non-zero now, so reading from the 0th index in the segment at seg_idx is safe.
                let displaced = unsafe { std::ptr::read(&self.segments[seg_idx][0]) };
                // seg_idx-1 is known to exist, and to have exactly one empty slot to push into
                self.segments[seg_idx - 1].push(displaced);
                // SAFETY:
                // 1. orig_len is known to be non-zero now, so the head of the segment is known to be a valid pointer to a T
                // 2. from #1, 1 is known to be less than or equal to orig_len,
                //    and orig_len - 1 is known to be greater than or equal to 0
                // 3. from #1+2, the pseudo-operation copy(segment[1..orig_len], segment[0..orig_len-1]) is known to be valid
                // 4. all elements from 0 to orig_len are initialized, by definition of orig_len
                // 6. from #1 and #4, we know it is safe to set the length of the segment to orig_len-1
                unsafe {
                    let dst_ptr = self.segments[seg_idx].as_mut_ptr();
                    let src_ptr = dst_ptr.add(1);
                    std::ptr::copy(src_ptr, dst_ptr, orig_len - 1);
                    self.segments[seg_idx].set_len(orig_len - 1);
                }
            }
        }
        // total length has decreased by 1, reflect this
        self.len -= 1;
        return removed;
    }

    /// Returns an iterator that removes and returns values from within the given range of the
    /// [`SegVec`][crate::SegVec]. See [`Drain`][crate::Drain] for more information.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// v.push(2);
    /// v.drain(..).for_each(|v| println!("{}", v));
    /// assert_eq!(v.len(), 0);
    /// ```
    ///
    /// # Panics
    /// - If the end index is greater than `self.len()`
    /// - If the start index is greater than the end index.
    pub fn drain<R>(&mut self, range: R) -> Drain<T>
    where
        R: RangeBounds<usize>,
    {
        let (start, end) = self.bounds("SegVec::drain", range);
        Drain {
            inner: self,
            drained: 0,
            index: start,
            total: end - start,
        }
    }

    /// Returns a [`Slice`][crate::Slice] over the given range in the [`SegVec`][crate::SegVec].
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// v.push(2);
    /// let s = v.slice(1..);
    /// assert_eq!(s[0], 2);
    /// ```
    ///
    /// # Panics
    /// - If the end index is greater than `self.len()`
    /// - If the start index is greater than the end index.
    pub fn slice<R>(&self, range: R) -> Slice<'_, T>
    where
        R: RangeBounds<usize>,
    {
        let (start, end) = self.bounds("SegVec::slice", range);
        Slice {
            inner: self,
            start,
            len: end - start,
        }
    }

    /// Returns a [`SliceMut`][crate::SliceMut] over the given range in the
    /// [`SegVec`][crate::SegVec].
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// v.push(2);
    /// let mut s = v.slice_mut(..1);
    /// s[0] = 100;
    /// assert_eq!(v[0], 100);
    /// ```
    ///
    /// # Panics
    /// - If the end index is greater than `self.len()`
    /// - If the start index is greater than the end index.
    pub fn slice_mut<R>(&mut self, range: R) -> SliceMut<'_, T>
    where
        R: RangeBounds<usize>,
    {
        let (start, end) = self.bounds("SegVec::slice_mut", range);
        SliceMut {
            inner: self,
            start,
            len: end - start,
        }
    }

    /// Reverses the elements in the [`SegVec`][crate::SegVec].
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// v.push(2);
    /// v.push(3);
    /// v.push(4);
    /// v.push(5);
    /// v.push(6);
    /// v.reverse();
    /// assert_eq!(v.into_iter().collect::<Vec<_>>(), vec![6, 5, 4, 3, 2, 1]);
    /// ```
    pub fn reverse(&mut self) {
        if self.len() < 2 {
            return;
        }
        let mut left = 0;
        let mut right = self.len() - 1;
        while left < right {
            self.swap(left, right);
            left += 1;
            right -= 1;
        }
    }

    /// Sort the [`SegVec`][crate::SegVec] in ascending order (unstable)
    pub fn sort_unstable(&mut self)
    where
        T: Ord,
    {
        self.sort_unstable_by(Ord::cmp)
    }

    /// Sort the [`SegVec`][crate::SegVec] in ascending order (unstable) using the given comparison function
    pub fn sort_unstable_by<F>(&mut self, mut compare: F)
    where
        F: FnMut(&T, &T) -> cmp::Ordering,
    {
        fn partition<T, F>(v: &mut SegVec<T>, lo: usize, hi: usize, compare: &mut F) -> usize
        where
            F: FnMut(&T, &T) -> cmp::Ordering,
        {
            let pivot = lo;
            let mut left = lo;
            let mut right = hi + 1;
            while left < right {
                loop {
                    left += 1;
                    if left >= right || compare(&v[pivot], &v[left]).is_lt() {
                        break;
                    }
                }
                loop {
                    right -= 1;
                    if right <= left || compare(&v[right], &v[pivot]).is_lt() {
                        break;
                    }
                }
                if right > left {
                    v.swap(right, left);
                }
            }

            let final_pivot_location = if compare(&v[right], &v[pivot]).is_lt() {
                right
            } else {
                right - 1
            };
            v.swap(final_pivot_location, pivot);
            final_pivot_location
        }

        pub fn quicksort<T, F>(v: &mut SegVec<T>, lo: usize, hi: usize, compare: &mut F)
        where
            F: FnMut(&T, &T) -> cmp::Ordering,
        {
            if hi > lo {
                match hi - lo {
                    1 => {
                        if compare(&v[hi], &v[lo]).is_lt() {
                            v.swap(lo, hi);
                        }
                    }
                    _ => {
                        let mid = partition(v, lo, hi, compare);
                        if mid > lo {
                            quicksort(v, lo, mid - 1, compare);
                        }
                        if mid < hi {
                            quicksort(v, mid + 1, hi, compare);
                        }
                    }
                }
            }
        }

        match self.len() {
            0..=1 => {}
            len => quicksort(self, 0, len - 1, &mut compare),
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        if a != b {
            let a_ptr = &mut self[a] as *mut T;
            let b_ptr = &mut self[b] as *mut T;
            // SAFETY:
            // 1. a != b, so a_ptr and b_ptr cannot alias one another
            // 2. If either a or b are invalid as indexes into the structure, a panic
            //    will occur before we get here. Thus, a_ptr and b_ptr are both derived
            //    from valid references to some element T in the structure.
            unsafe { std::ptr::swap(a_ptr, b_ptr) };
        }
    }

    fn bounds<R>(&self, caller: &str, range: R) -> (usize, usize)
    where
        R: RangeBounds<usize>,
    {
        let size = self.len;
        let start = range.start_bound();
        let start = match start {
            Bound::Included(&start) => start,
            Bound::Excluded(start) => start.checked_add(1).expect("start bound fits into usize"),
            Bound::Unbounded => 0,
        };

        let end = range.end_bound();
        let end = match end {
            Bound::Included(end) => end.checked_add(1).expect("end bound fits into usize"),
            Bound::Excluded(&end) => end,
            Bound::Unbounded => size,
        };

        if start > end {
            panic!("{}: lower bound {} > upper bound {}", caller, start, end);
        }
        if end > size {
            index_oob(caller, end, size);
        }
        (start, end)
    }

    fn segment_capacity(&self, segment_index: usize) -> usize {
        match segment_index {
            0 => self.factor.get(),
            n => {
                let pow = u32::try_from(n - 1).expect("fewer than 64 segments");
                match 2usize
                    .checked_pow(pow)
                    .and_then(|n| n.checked_mul(self.factor.get()))
                {
                    Some(size) => size,
                    None => capacity_overflow(),
                }
            }
        }
    }

    fn segment_and_offset(&self, linear_index: usize) -> (usize, usize) {
        let normal = linear_index
            .checked_div(self.factor.get())
            .expect("non-zero growth factor");
        let (segment, pow) = match checked_log2_floor(normal) {
            None => (0usize, 0u32),
            Some(s) => (s as usize + 1, s),
        };
        match 2usize.pow(pow).checked_mul(self.factor.get()) {
            Some(mod_base) => {
                let offset = linear_index % mod_base;
                (segment, offset)
            }
            None => unreachable!(),
        }
    }
}

impl<T: Clone> Clone for SegVec<T> {
    fn clone(&self) -> Self {
        SegVec {
            len: self.len,
            capacity: self.capacity,
            segments: self.segments.clone(),
            factor: self.factor,
        }
    }
}

impl<T> Index<usize> for SegVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            Some(t) => t,
            None => index_oob("SegVec::index", index, self.len),
        }
    }
}

impl<T> IndexMut<usize> for SegVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let size = self.len;
        match self.get_mut(index) {
            Some(t) => t,
            None => index_oob("SegVec::index_mut", index, size),
        }
    }
}

impl<T: Debug> Debug for SegVec<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<T: Hash> Hash for SegVec<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.iter().for_each(|i| i.hash(state));
    }
}

impl<T> PartialEq for SegVec<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        if self.len() != other.len() {
            return false;
        }
        (0..self.len()).all(|i| &self[i] == &other[i])
    }
}

impl<T> Eq for SegVec<T> where T: Eq {}

impl<T> Extend<T> for SegVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (min_size, max_size) = iter.size_hint();
        let additional = std::cmp::max(max_size.unwrap_or(0), min_size);
        self.reserve(additional);
        for i in iter {
            self.push(i);
        }
    }
}

impl<'a, T: Copy + 'a> Extend<&'a T> for SegVec<T> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        <Self as Extend<T>>::extend(self, iter.into_iter().copied())
    }
}

impl<T> FromIterator<T> for SegVec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut v = SegVec::new();
        v.extend(iter);
        v
    }
}

impl<T> IntoIterator for SegVec<T> {
    type IntoIter = IntoIter<T>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            size: self.len,
            iter: self.segments.into_iter().flatten(),
        }
    }
}

/// Iterator over immutable references to items in a [`SegVec`][crate::SegVec].
pub struct Iter<'a, T> {
    size: usize,
    iter: std::iter::Flatten<std::slice::Iter<'a, inner::Segment<T>>>,
}

impl<'a, T: 'a> Iterator for Iter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(i) => {
                self.size -= 1;
                Some(i)
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<'a, T> DoubleEndedIterator for Iter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.iter.next_back() {
            Some(i) => {
                self.size -= 1;
                Some(i)
            }
            None => None,
        }
    }
}

impl<'a, T> FusedIterator for Iter<'a, T> {}
impl<'a, T> ExactSizeIterator for Iter<'a, T> {}

/// Consuming iterator over items in a [`SegVec`][crate::SegVec].
pub struct IntoIter<T> {
    size: usize,
    iter: std::iter::Flatten<<inner::Segments<T> as std::iter::IntoIterator>::IntoIter>,
}

impl<T> Iterator for IntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            Some(i) => {
                self.size -= 1;
                Some(i)
            }
            None => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.size, Some(self.size))
    }
}

impl<T> DoubleEndedIterator for IntoIter<T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        match self.iter.next_back() {
            Some(i) => {
                self.size -= 1;
                Some(i)
            }
            None => None,
        }
    }
}

impl<T> FusedIterator for IntoIter<T> {}
impl<T> ExactSizeIterator for IntoIter<T> {}

/// Removes and returns elements from a range in a [`SegVec`][crate::SegVec].
/// Any un-consumed elements are removed and dropped when a `Drain` is dropped.
/// If a `Drain` is forgotten (via [`std::mem::forget`]), it is unspecified how many elements are
/// removed. The current implementation calls `SegVec::remove` on a single element on each call to
/// `next`.
pub struct Drain<'a, T> {
    inner: &'a mut SegVec<T>,
    index: usize,
    total: usize,
    drained: usize,
}

impl<'a, T> Iterator for Drain<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.drained < self.total {
            let next = self.inner.remove(self.index);
            self.drained += 1;
            Some(next)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.total - self.drained;
        (left, Some(left))
    }
}

impl<'a, T> DoubleEndedIterator for Drain<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let left = self.total - self.drained;
        if left > 0 {
            let next = self.inner.remove(self.index + (left - 1));
            self.drained += 1;
            Some(next)
        } else {
            None
        }
    }
}

impl<'a, T> FusedIterator for Drain<'a, T> {}
impl<'a, T> ExactSizeIterator for Drain<'a, T> {}

impl<'a, T> Drop for Drain<'a, T> {
    fn drop(&mut self) {
        self.for_each(drop);
    }
}

/// Provides an immutable view of elements from a range in [`SegVec`][crate::SegVec].
pub struct Slice<'a, T: 'a> {
    inner: &'a SegVec<T>,
    start: usize,
    len: usize,
}

impl<'a, T: 'a> Copy for Slice<'a, T> {}

impl<'a, T: 'a> Clone for Slice<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: Debug + 'a> Debug for Slice<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T: 'a> Slice<'a, T> {
    /// Returns the number of elements in the [`Slice`][crate::Slice].
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns an iterator over immutable references to the elements of the
    /// [`Slice`][crate::Slice].
    pub fn iter(&self) -> SliceIter<'a, T> {
        SliceIter {
            slice: *self,
            index: 0,
        }
    }
}

impl<'a, T: 'a> Index<usize> for Slice<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &'a Self::Output {
        match slice_index_to_base_index(self.start, index, self.len) {
            Some(idx) => self.inner.index(idx),
            _ => index_oob("Slice::index", index, self.len),
        }
    }
}

impl<'a, T: 'a> IntoIterator for Slice<'a, T> {
    type IntoIter = SliceIter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Iterator over immutable references to the elements of a [`Slice`][crate::Slice].
pub struct SliceIter<'a, T: 'a> {
    slice: Slice<'a, T>,
    index: usize,
}

impl<'a, T: 'a> Iterator for SliceIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.slice.len {
            let index = self.index;
            self.index += 1;
            // SAFETY:
            // 1. index corresponds to a valid value in the slice
            // 2. the value at index must live for at least the lifetime 'a
            // 3. from #1+2, it is known that a taking an &'a to the value in the
            //    slice is safe
            Some(unsafe { &*(self.slice.index(index) as *const T) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.slice.len - self.index;
        (left, Some(left))
    }
}

impl<'a, T: 'a> FusedIterator for SliceIter<'a, T> {}
impl<'a, T: 'a> ExactSizeIterator for SliceIter<'a, T> {}

/// Provides a mutable view of elements from a range in [`SegVec`][crate::SegVec].
pub struct SliceMut<'a, T: 'a> {
    inner: &'a mut SegVec<T>,
    start: usize,
    len: usize,
}

impl<'a, T: 'a> SliceMut<'a, T> {
    /// Returns the number of elements in the [`SliceMut`][crate::SliceMut].
    pub fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T: 'a> Index<usize> for SliceMut<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match slice_index_to_base_index(self.start, index, self.len) {
            Some(idx) => self.inner.index(idx),
            _ => index_oob("SliceMut::index", index, self.len),
        }
    }
}

impl<'a, T: 'a> IndexMut<usize> for SliceMut<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match slice_index_to_base_index(self.start, index, self.len) {
            Some(idx) => self.inner.index_mut(idx),
            _ => index_oob("SliceMut::index_mut", index, self.len),
        }
    }
}

impl<'a, T: 'a> IntoIterator for SliceMut<'a, T> {
    type IntoIter = SliceMutIter<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> Self::IntoIter {
        SliceMutIter {
            slice: self,
            index: 0,
        }
    }
}

/// Iterator over mutable references to the elements of a [`SliceMut`][crate::SliceMut].
pub struct SliceMutIter<'a, T: 'a> {
    slice: SliceMut<'a, T>,
    index: usize,
}

impl<'a, T: 'a> Iterator for SliceMutIter<'a, T> {
    type Item = &'a mut T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.slice.len {
            let index = self.index;
            self.index += 1;
            // SAFETY:
            // 1. index corresponds to a valid value in the slice
            // 2. the value at index must live for at least the lifetime 'a
            // 3. from #1+2, it is known that a taking an &'a mut to the value in the
            //    slice is safe
            Some(unsafe { &mut *(self.slice.index_mut(index) as *mut T) })
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.slice.len - self.index;
        (left, Some(left))
    }
}

impl<'a, T: 'a> FusedIterator for SliceMutIter<'a, T> {}
impl<'a, T: 'a> ExactSizeIterator for SliceMutIter<'a, T> {}

/// Returns the highest power of 2 that is less than or equal to `v` when `v` is non-zero.
/// If `v` is zero, `None` is returned.
fn checked_log2_floor(v: usize) -> Option<u32> {
    if v > 0 {
        Some((usize::BITS - 1) - v.leading_zeros())
    } else {
        None
    }
}

fn slice_index_to_base_index(
    start_idx: usize,
    slice_idx: usize,
    slice_len: usize,
) -> Option<usize> {
    match start_idx.checked_add(slice_idx) {
        Some(idx) if idx - start_idx < slice_len => Some(idx),
        _ => None,
    }
}

#[cold]
fn capacity_overflow() -> ! {
    panic!("SegVec: capacity overflow")
}

#[cold]
fn index_oob(caller: &str, idx: usize, len: usize) -> ! {
    panic!(
        "{}: index out of bounds: index is {}, len is {}",
        caller, idx, len
    )
}
