use crate::*;
use either::Either;
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Provides an immutable view of elements from a range in [`SegVec`][crate::SegVec].
pub struct Slice<'a, T: 'a> {
    inner: &'a dyn SegmentIndex<T>,
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
    // private ctor
    #[inline]
    pub(crate) fn new(segvec: &'a dyn SegmentIndex<T>, start: usize, len: usize) -> Self {
        Slice {
            inner: segvec,
            start,
            len,
        }
    }

    /// Returns the number of elements in the [`Slice`][crate::Slice].
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true when a slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an iterator over immutable references to the elements of the
    /// [`Slice`][crate::Slice].
    pub fn iter(&self) -> SliceIter<'a, T> {
        SliceIter {
            iter: self.segmented_iter().flatten(),
            start: 0,
            end: self.len,
        }
    }

    /// Returns an iterator over immutable references of slices of elements of the
    /// [`Slice`][crate::Slice].
    pub fn segmented_iter(&self) -> SegmentedIter<'a, T> {
        let start = self.inner.segment_and_offset(self.start);
        // The 'end' is inclusive because we don't want to spill into the next segment. For an
        // empty slice we have to prevent integer underflow, we just store a (0,0), this will
        // not be used later since len is checked first to be not zero.
        let end = if self.len > 0 {
            self.inner.segment_and_offset(self.start + self.len - 1)
        } else {
            (0, 0)
        };

        SegmentedIter {
            slice: *self,
            start,
            end,
        }
    }

    /// Sub-slices an existing slice, returns a new [`Slice`][crate::Slice] covering the given
    /// `range`.
    ///
    /// # Panics
    /// - If the end index is greater than `self.len()`
    /// - If the start index is greater than the end index.
    pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> Self {
        let (start, end) = bounds(self.len, "Slice::slice", range);
        Slice {
            inner: self.inner,
            start: self.start + start,
            len: end - start,
        }
    }

    /// Tries to treat this SegVec [`Slice`] as a contiguous [`slice`]. This only
    /// succeeds if all the elements of this [`Slice`] are in the same segment.
    pub fn as_contiguous(&self) -> Option<&'a [T]> {
        let (start_segment, start_offset) = self.inner.segment_and_offset(self.start);

        if self.len == 0 {
            // this could just return Some(&[]), but the address of the slice
            // would be dangling - not sure if this matters much...
            let segment = unsafe { self.inner.segment_unchecked(start_segment) };
            return Some(&segment[start_offset..start_offset]);
        }

        let (end_segment, end_offset) = self.inner.segment_and_offset(self.start + self.len - 1);
        (start_segment == end_segment).then(|| {
            let segment = unsafe { self.inner.segment_unchecked(start_segment) };
            &segment[start_offset..end_offset + 1]
        })
    }
}

impl<'a, T: 'a> Index<usize> for Slice<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match slice_index_to_base_index(self.start, index, self.len) {
            Some(idx) => SegmentIndex::index(self.inner, idx),
            _ => index_oob("Slice::index", index, self.len),
        }
    }
}

impl<'a, T: 'a> IntoIterator for Slice<'a, T> {
    type IntoIter = SliceIter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        let start = self.inner.segment_and_offset(self.start);
        // The 'end' is inclusive because we don't want to spill into the next segment. For an
        // empty slice we have to prevent integer underflow, we just store a (0,0), this will
        // not be used later since len is checked first to be not zero.
        let end = if self.len > 0 {
            self.inner.segment_and_offset(self.start + self.len - 1)
        } else {
            (0, 0)
        };

        let seg_iter = SegmentedIter {
            slice: self,
            start,
            end,
        };

        SliceIter {
            iter: seg_iter.flatten(),
            start: 0,
            end: self.len,
        }
    }
}

impl<'a, T: 'a> IntoIterator for &Slice<'a, T> {
    type IntoIter = SliceIter<'a, T>;
    type Item = &'a T;

    fn into_iter(self) -> Self::IntoIter {
        let start = self.inner.segment_and_offset(self.start);
        // The 'end' is inclusive because we don't want to spill into the next segment. For an
        // empty slice we have to prevent integer underflow, we just store a (0,0), this will
        // not be used later since len is checked first to be not zero.
        let end = if self.len > 0 {
            self.inner.segment_and_offset(self.start + self.len - 1)
        } else {
            (0, 0)
        };

        let seg_iter = SegmentedIter {
            slice: *self,
            start,
            end,
        };

        SliceIter {
            iter: seg_iter.flatten(),
            start: 0,
            end: self.len,
        }
    }
}

/// Iterator over immutable references to the elements of a [`Slice`][crate::Slice].
pub struct SliceIter<'a, T: 'a> {
    iter: Flatten<SegmentedIter<'a, T>>,
    // Since Flatten's size_hint is not sufficient we have to do our own accounting here.
    start: usize,
    end: usize,
}

impl<'a, T: 'a> Iterator for SliceIter<'a, T> {
    type Item = &'a T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.end > self.start {
            self.start += 1;
            self.iter.next()
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.end - self.start;
        (left, Some(left))
    }
}

impl<'a, T: 'a> FusedIterator for SliceIter<'a, T> {}
impl<'a, T: 'a> ExactSizeIterator for SliceIter<'a, T> {}

impl<'a, T: 'a> DoubleEndedIterator for SliceIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.end > self.start {
            self.end -= 1;
            self.iter.next_back()
        } else {
            None
        }
    }
}

/// Iterator over immutable references to slices of the elements of a [`Slice`][crate::Slice].
pub struct SegmentedIter<'a, T: 'a> {
    slice: Slice<'a, T>,
    start: (usize, usize),
    end: (usize, usize),
}

impl<'a, T: 'a> Iterator for SegmentedIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        // We never return an empty slice
        if self.slice.len == 0 || self.start.0 > self.end.0 {
            return None;
        }

        // Safety: * Slices are always valid ranges
        //         * We returned 'None' above when done
        let ret = unsafe {
            if self.start.0 == self.end.0 {
                self.slice
                    .inner
                    .segment_unchecked(self.start.0)
                    .get_unchecked(self.start.1..=self.end.1)
            } else {
                self.slice
                    .inner
                    .segment_unchecked(self.start.0)
                    .get_unchecked(self.start.1..)
            }
        };
        self.start = (self.start.0 + 1, 0);
        Some(ret)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = 1 + self.end.0 - self.start.0;
        (left, Some(left))
    }
}

impl<'a, T: 'a> FusedIterator for SegmentedIter<'a, T> {}
impl<'a, T: 'a> ExactSizeIterator for SegmentedIter<'a, T> {}

impl<'a, T: 'a> DoubleEndedIterator for SegmentedIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // We never return an empty slice
        if self.slice.len == 0 || self.start.0 > self.end.0 {
            return None;
        }

        // Safety: * Slices are always valid ranges
        //         * We returned 'None' above when done
        let ret = unsafe {
            if self.start.0 == self.end.0 {
                self.slice
                    .inner
                    .segment_unchecked(self.end.0)
                    .get_unchecked(self.start.1..=self.end.1)
            } else {
                self.slice
                    .inner
                    .segment_unchecked(self.end.0)
                    .get_unchecked(..=self.end.1)
            }
        };
        // need to be careful not to underflow self.end.0 when done.
        if self.end.0 != 0 {
            self.end = (
                self.end.0 - 1,
                // Safety: self.end.0 is a valid index (slice invariant) and not zero as checked above
                unsafe { self.slice.inner.segment_unchecked(self.end.0 - 1).len() - 1 },
            );
        } else {
            // with start.0 > end.0 the iterator is flagged as 'done'
            self.start = (1, 0);
        }
        Some(ret)
    }
}

/// Provides a mutable view of elements from a range in [`SegVec`][crate::SegVec].
pub struct SliceMut<'a, T: 'a> {
    inner: &'a mut dyn SegmentIndexMut<T>,
    start: usize,
    len: usize,
}

impl<'a, T: Clone + Debug + 'a> Debug for SliceMut<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl<'a, T: 'a> SliceMut<'a, T> {
    // private ctor
    #[inline]
    pub(crate) fn new(segvec: &'a mut dyn SegmentIndexMut<T>, start: usize, len: usize) -> Self {
        SliceMut {
            inner: segvec,
            start,
            len,
        }
    }

    /// Returns the number of elements in the [`SliceMut`][crate::SliceMut].
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true when a slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an iterator over immutable references to the elements of the
    /// [`SliceMut`][crate::SliceMut].
    pub fn iter(&self) -> SliceIter<'_, T> {
        let end = self.len;

        SliceIter {
            iter: self.segmented_iter().flatten(),
            start: 0,
            end,
        }
    }

    /// Returns an iterator over mutable references to the elements of the
    /// [`SliceMut`][crate::SliceMut].
    pub fn iter_mut(&mut self) -> SliceMutIter<'a, T> {
        let end = self.len;
        SliceMutIter {
            iter: self.segmented_iter_mut().flatten(),
            start: 0,
            end,
        }
    }

    /// Returns an iterator over immutable references of slices of elements of the
    /// [`SliceMut`][crate::SliceMut].
    pub fn segmented_iter(&self) -> SegmentedIter<'_, T> {
        let start = self.inner.segment_and_offset(self.start);
        // The 'end' is inclusive because we don't want to spill into the next segment. For an
        // empty slice we have to prevent integer underflow, we just store a (0,0), this will
        // not be used later since len is checked first to be not zero.
        let end = if self.len > 0 {
            self.inner.segment_and_offset(self.start + self.len - 1)
        } else {
            (0, 0)
        };

        let slice = Slice {
            inner: self.inner.as_segment_index(),
            start: self.start,
            len: self.len,
        };

        SegmentedIter { slice, start, end }
    }

    /// Returns an iterator over immutable references of slices of elements of the
    /// [`SliceMut`][crate::SliceMut].
    pub fn segmented_iter_mut(&mut self) -> SegmentedMutIter<'a, T> {
        let start = self.inner.segment_and_offset(self.start);
        // The 'end' is inclusive because we don't want to spill into the next segment. For an
        // empty slice we have to prevent integer underflow, we just store a (0,0), this will
        // not be used later since len is checked first to be not zero.
        let end = if self.len > 0 {
            self.inner.segment_and_offset(self.start + self.len - 1)
        } else {
            (0, 0)
        };

        SegmentedMutIter {
            slice: Either::Left((self.into(), PhantomData)),
            start,
            end,
        }
    }

    /// Sub-slices an existing 'SliceMut', returns a new [`Slice`][crate::Slice] covering the given
    /// `range`.
    ///
    /// # Panics
    /// - If the end index is greater than `self.len()`
    /// - If the start index is greater than the end index.
    pub fn slice<R: RangeBounds<usize>>(&'a self, range: R) -> Slice<'a, T> {
        let (start, end) = bounds(self.len, "SliceMut::slice", range);
        Slice {
            inner: self.inner.as_segment_index(),
            start: self.start + start,
            len: end - start,
        }
    }

    /// Sub-slices an existing 'SliceMut', returns a new [`SliceMut`][crate::SliceMut] covering the given
    /// `range`.
    ///
    /// # Panics
    /// - If the end index is greater than `self.len()`
    /// - If the start index is greater than the end index.
    pub fn slice_mut<R: RangeBounds<usize>>(&'a mut self, range: R) -> SliceMut<'a, T> {
        let (start, end) = bounds(self.len, "SliceMut::slice_mut", range);
        SliceMut {
            inner: self.inner,
            start: self.start + start,
            len: end - start,
        }
    }

    // PLANNED: split_at_mut()
}

impl<'a, T: 'a> Index<usize> for SliceMut<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match slice_index_to_base_index(self.start, index, self.len) {
            Some(idx) => SegmentIndex::index(self.inner, idx),
            _ => index_oob("SliceMut::index", index, self.len),
        }
    }
}

impl<'a, T: 'a> IndexMut<usize> for SliceMut<'a, T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match slice_index_to_base_index(self.start, index, self.len) {
            Some(idx) => SegmentIndexMut::index_mut(self.inner, idx),
            _ => index_oob("SliceMut::index_mut", index, self.len),
        }
    }
}

/// Iterator over immutable references to the elements of a [`SliceMut`][crate::SliceMut].
pub struct SliceMutIter<'a, T: 'a> {
    iter: Flatten<SegmentedMutIter<'a, T>>,
    // Since Flatten's size_hint is not sufficient we have to do our own accounting here.
    start: usize,
    end: usize,
}

impl<'a, T: 'a> Iterator for SliceMutIter<'a, T> {
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.end > self.start {
            self.start += 1;
            self.iter.next()
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = self.end - self.start;
        (left, Some(left))
    }
}

impl<'a, T: 'a> FusedIterator for SliceMutIter<'a, T> {}
impl<'a, T: 'a> ExactSizeIterator for SliceMutIter<'a, T> {}

impl<'a, T: 'a> DoubleEndedIterator for SliceMutIter<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.end > self.start {
            self.end -= 1;
            self.iter.next_back()
        } else {
            None
        }
    }
}

impl<'a, T: 'a> IntoIterator for SliceMut<'a, T> {
    type IntoIter = SliceMutIter<'a, T>;
    type Item = &'a mut T;

    fn into_iter(self) -> Self::IntoIter {
        let start = self.inner.segment_and_offset(self.start);
        // The 'end' is inclusive because we don't want to spill into the next segment. For an
        // empty slice we have to prevent integer underflow, we just store a (0,0), this will
        // not be used later since len is checked first to be not zero.
        let end = if self.len > 0 {
            self.inner.segment_and_offset(self.start + self.len - 1)
        } else {
            (0, 0)
        };

        let self_end = self.len;

        let seg_iter = SegmentedMutIter {
            slice: Either::Right(self),
            start,
            end,
        };

        SliceMutIter {
            iter: seg_iter.flatten(),
            start: 0,
            end: self_end,
        }
    }
}

/// Iterator over mutable references to slices of the elements of a [`SliceMut`][crate::SliceMut].
pub struct SegmentedMutIter<'a, T: 'a> {
    // Safety:
    // We can not use a reference here because aliasing rules and `fn next(&self)` would
    // introduce a lifetime on 'self while we keep 'a here. Thus we `Either` store a lifetime
    // erased NonNull here with the original Lifetime as PhantomData. Or we store a owned
    // SliceMut in case of a `IntoIter`.
    //
    // Prior art: https://doc.rust-lang.org/std/slice/struct.ChunksMut.html
    #[allow(clippy::type_complexity)]
    slice: Either<(NonNull<SliceMut<'a, T>>, PhantomData<&'a mut T>), SliceMut<'a, T>>,
    start: (usize, usize),
    end: (usize, usize),
}

impl<'a, T: 'a> Iterator for SegmentedMutIter<'a, T> {
    type Item = &'a mut [T];

    fn next(&mut self) -> Option<Self::Item> {
        let slice = match self.slice.as_mut() {
            // SAFETY: this pointer is always initialized to a valid and non-aliased reference.
            Either::Left(s) => unsafe { s.0.as_mut() },
            // SAFETY: we need to get the lifetime of 'self' here
            Either::Right(s) => unsafe { (s as *mut SliceMut<_>).as_mut().unwrap_unchecked() },
        };

        // We never return an empty slice
        if slice.len == 0 || self.start.0 > self.end.0 {
            return None;
        }

        // Safety: * Slices are always valid ranges
        //         * We returned 'None' above when done
        let ret = unsafe {
            if self.start.0 == self.end.0 {
                slice
                    .inner
                    .segment_unchecked_mut(self.start.0)
                    .get_unchecked_mut(self.start.1..=self.end.1)
            } else {
                slice
                    .inner
                    .segment_unchecked_mut(self.start.0)
                    .get_unchecked_mut(self.start.1..)
            }
        };
        self.start = (self.start.0 + 1, 0);
        Some(ret)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = 1 + self.end.0 - self.start.0;
        (left, Some(left))
    }
}

impl<'a, T: 'a> FusedIterator for SegmentedMutIter<'a, T> {}
impl<'a, T: 'a> ExactSizeIterator for SegmentedMutIter<'a, T> {}

impl<'a, T: 'a> DoubleEndedIterator for SegmentedMutIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        let slice = match self.slice.as_mut() {
            // SAFETY: this pointer is always initialized to a valid and non-aliased reference.
            Either::Left(s) => unsafe { s.0.as_mut() },
            // SAFETY: we need to get the lifetime of 'self' here
            Either::Right(s) => unsafe { (s as *mut SliceMut<_>).as_mut().unwrap_unchecked() },
        };

        let start = (self.start.0, self.start.1);
        let end = (self.end.0, self.end.1);

        // need to be careful not to underflow self.end.0 when done.
        if end.0 != 0 {
            self.end = (
                end.0 - 1,
                // Safety: self.end.0 is a valid index (slice invariant) and not zero as checked above
                unsafe { slice.inner.segment_unchecked(end.0 - 1).len() - 1 },
            );
        } else {
            // with start.0 > end.0 the iterator is flagged as 'done'
            self.start = (1, 0);
        }

        // We never return an empty slice
        unsafe {
            if slice.len == 0 || start.0 > end.0 {
                None
            } else if start.0 == end.0 {
                // Safety: * Slices are always valid ranges
                //         * We returned 'None' above when done
                Some(
                    slice
                        .inner
                        .segment_unchecked_mut(end.0)
                        .get_unchecked_mut(self.start.1..=end.1),
                )
            } else {
                Some(
                    slice
                        .inner
                        .segment_unchecked_mut(end.0)
                        .get_unchecked_mut(..=end.1),
                )
            }
        }
    }
}

/// Extends Index<> with methods to get segments and (segment,offset) tuples
pub(crate) trait SegmentIndex<T>: Index<usize, Output = T> {
    fn index(&self, i: usize) -> &T;
    fn segment_and_offset(&self, i: usize) -> (usize, usize);
    unsafe fn segment_unchecked(&self, i: usize) -> &[T];
}

/// Extends SegmentIndex<> with methods for mutable access.
pub(crate) trait SegmentIndexMut<T>: SegmentIndex<T> + IndexMut<usize, Output = T> {
    fn index_mut(&mut self, i: usize) -> &mut T;
    unsafe fn segment_unchecked_mut(&mut self, i: usize) -> &mut [T];

    // Downcasts `&SegmentIndexMut<T>` to `&SegmentIndex<T>`
    fn as_segment_index(&self) -> &dyn SegmentIndex<T>;
}

impl<T, C: MemConfig> SegmentIndex<T> for SegVec<T, C> {
    #[inline]
    fn index(&self, i: usize) -> &T {
        Index::index(self, i)
    }

    #[inline]
    fn segment_and_offset(&self, i: usize) -> (usize, usize) {
        self.config.segment_and_offset(i)
    }

    #[inline]
    unsafe fn segment_unchecked(&self, i: usize) -> &[T] {
        self.segments.get_unchecked(i)
    }
}

impl<T, C: MemConfig> SegmentIndexMut<T> for SegVec<T, C>
where
    Self: IndexMut<usize, Output = T>,
{
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        IndexMut::index_mut(self, i)
    }

    #[inline]
    unsafe fn segment_unchecked_mut(&mut self, i: usize) -> &mut [T] {
        self.segments.get_unchecked_mut(i)
    }

    #[inline]
    fn as_segment_index(&self) -> &dyn SegmentIndex<T> {
        self
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
