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

use std::{
    cmp,
    convert::TryFrom,
    fmt::Debug,
    hash::Hash,
    iter::{FromIterator, FusedIterator},
    mem,
    ops::{Bound, Index, IndexMut, RangeBounds},
};

#[cfg(test)]
mod tests;

mod raw;

/// A data structure similar to [`Vec`][std::vec::Vec], but that does not copy on re-size and can
/// release memory when it is truncated.
///
/// - Capacity is allocated in "segments".
/// - For a `SegVec` with one element, a segment of length 1 is allocated.
/// - For a `SegVec` with 2 elements, two segments of length 1 are allocated.
/// - For a `SegVec` with 3 or 4 elements, two segments of length one, and a segment of length 2 are
///   allocated.
/// - Each allocated segment is the size of the entire capacity prior to the allocation of that
///   segment.
/// In other words, each segment allocated doubles the capacity of the `SegVec`. As a consequence, a
/// `SegVec` always has a capacity that is a power of 2, with the special case of the empty `SegVec`
/// with a capacity of 0.
pub struct SegVec<T> {
    cap: usize,
    size: usize,
    segments: Vec<Vec<T>>,
}

impl<T> SegVec<T> {
    /// Create a new [`SegVec`][crate::SegVec] with a length and capacity of 0
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let v: SegVec<i32> = SegVec::new();
    /// assert_eq!(v.capacity(), 0);
    /// ```
    pub fn new() -> Self {
        SegVec {
            cap: 0,
            size: 0,
            segments: Vec::with_capacity(0),
        }
    }

    /// Create a new [`SegVec`][crate::SegVec] with a length of 0 and a capacity large enough to
    /// hold the given number of elements.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let v: SegVec<i32> = SegVec::with_capacity(5);
    /// assert_eq!(v.capacity(), 8);
    /// ```
    ///
    /// # Panics
    /// - If the required capacity overflows `usize`
    pub fn with_capacity(hint: usize) -> Self {
        let mut v = SegVec::new();
        v.reserve(hint);
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
        self.size
    }

    /// The capacity of the [`SegVec`][crate::SegVec]
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::with_capacity(3);
    /// assert_eq!(v.capacity(), 4);
    /// ```
    pub fn capacity(&self) -> usize {
        self.cap
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
        let min_cap = match self.size.checked_add(additional) {
            Some(n) => n,
            None => capacity_overflow(),
        };
        if min_cap <= self.cap {
            return;
        }
        let new_cap = match min_cap.checked_next_power_of_two() {
            Some(v) => v,
            None => capacity_overflow(),
        };
        let current_slots = match u32::try_from(self.segments.len()) {
            Ok(v) => v,
            Err(_) => unreachable!(),
        };
        let required_slots = match checked_log2_ceil(new_cap) {
            Some(n) => n + 1,
            None => 1,
        };
        self.segments.extend(
            (current_slots..required_slots).map(|slot_idx| match slot_idx {
                0 => Vec::with_capacity(1),
                n => Vec::with_capacity(2usize.pow(n - 1)),
            }),
        );
        self.cap = new_cap;
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
        if index < self.size {
            let (seg, offset) = segment_and_offset(index);
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
        if index < self.size {
            let (seg, offset) = segment_and_offset(index);
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
        let (seg, _) = segment_and_offset(self.size);
        self.segments[seg].push(val);
        self.size += 1;
    }

    /// Removes the last value from the [`SegVec`][crate::SegVec] and returns it, or returns `None`
    /// if it is empty.
    ///
    /// ```
    /// # use segvec::SegVec;
    /// let mut v: SegVec<i32> = SegVec::new();
    /// v.push(1);
    /// assert_eq!(v[0], 1);
    /// ```
    pub fn pop(&mut self) -> Option<T> {
        match self.size {
            0 => None,
            size => {
                let (seg, offset) = segment_and_offset(size);
                let popped = match offset {
                    0 => self.segments[seg - 1].pop().unwrap(),
                    _ => self.segments[seg].pop().unwrap(),
                };
                self.size -= 1;
                Some(popped)
            }
        }
    }

    /// Truncates the [`SegVec`][crate::SegVec] to the given length.
    /// If the given length is larger than the current length, this is a no-op.
    /// Otherwise, the capacity is reduced and any excess elements are dropped.
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
        if len < self.cap {
            let (seg, offset) = segment_and_offset(len);
            if offset == 0 {
                self.segments.drain(seg..);
            } else {
                if len < self.size {
                    self.segments[seg].drain(offset..);
                }
                self.segments.drain(seg + 1..);
            }
            self.cap = match self.segments.len() {
                0 => 0,
                n => 2usize.pow((n - 1) as u32),
            };
            self.size = len;
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
            size: self.size,
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
        if index > self.size {
            index_oob("SegVec::insert", index, self.size);
        }
        // push val onto the end, then shift it to the correct location
        self.push(val);
        if mem::size_of::<T>() > 0 {
            let mut pos = self.size - 1;
            while index < pos {
                self.swap(pos, pos - 1);
                pos -= 1;
            }
        }
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
        if index >= self.size {
            index_oob("SegVec::remove", index, self.size);
        }
        // shift the value at index to the end, then pop it
        if mem::size_of::<T>() > 0 {
            let mut pos = index;
            while pos < self.size - 1 {
                self.swap(pos, pos + 1);
                pos += 1;
            }
        }
        self.pop().unwrap()
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
        self.sort_unstable_by(|a, b| a.cmp(b))
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
            let av = unsafe { &mut *(&mut self[a] as *mut T) };
            let bv = unsafe { &mut *(&mut self[b] as *mut T) };
            std::mem::swap(av, bv);
        }
    }

    fn bounds<R>(&self, caller: &str, range: R) -> (usize, usize)
    where
        R: RangeBounds<usize>,
    {
        let size = self.size;
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
}

impl<T: Clone> Clone for SegVec<T> {
    fn clone(&self) -> Self {
        SegVec {
            cap: self.cap,
            size: self.size,
            segments: self.segments.clone(),
        }
    }
}

impl<T> Index<usize> for SegVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match self.get(index) {
            Some(t) => t,
            None => index_oob("SegVec::index", index, self.size),
        }
    }
}

impl<T> IndexMut<usize> for SegVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let size = self.size;
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

impl<T> Extend<T> for SegVec<T> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        let iter = iter.into_iter();
        let (min_size, max_size) = iter.size_hint();
        let additional = max_size.unwrap_or(min_size);
        self.reserve(additional);
        for i in iter {
            self.push(i);
        }
    }
}

impl<T> Eq for SegVec<T> where T: Eq {}

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
            size: self.size,
            iter: self.segments.into_iter().flatten(),
        }
    }
}

/// Iterator over immutable references to items in a [`SegVec`][crate::SegVec].
pub struct Iter<'a, T> {
    size: usize,
    iter: std::iter::Flatten<std::slice::Iter<'a, Vec<T>>>,
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
    iter: std::iter::Flatten<std::vec::IntoIter<Vec<T>>>,
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
            self.index += 1;
            Some(unsafe { &*(self.slice.index(self.index - 1) as *const T) })
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
            self.index += 1;
            Some(unsafe { &mut *(self.slice.index_mut(self.index - 1) as *mut T) })
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

fn checked_log2_ceil(v: usize) -> Option<u32> {
    if v > 0 {
        Some((usize::BITS - 1) - v.leading_zeros())
    } else {
        None
    }
}

fn segment_and_offset(index: usize) -> (usize, usize) {
    match checked_log2_ceil(index) {
        None => (0, 0),
        Some(l) => {
            let b = l as usize + 1;
            let p = index % 2usize.pow(l as u32);
            (b, p)
        }
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
