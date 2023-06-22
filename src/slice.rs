use crate::*;

/// Extends Index<> with methods to get segments and (segment,offset) tuples
pub(crate) trait SegmentIndex<T>: Index<usize, Output = T> {
    fn index(&self, i: usize) -> &T;
    fn index_mut(&mut self, i: usize) -> &mut T;
    fn segment_and_offset(&self, i: usize) -> (usize, usize);
    fn segment(&self, i: usize) -> &[T];
    fn segment_mut(&mut self, i: usize) -> &mut [T];
}

impl<T, C: MemConfig> SegmentIndex<T> for SegVec<T, C> {
    #[inline]
    fn index(&self, i: usize) -> &T {
        Index::index(self, i)
    }

    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        IndexMut::index_mut(self, i)
    }

    #[inline]
    fn segment_and_offset(&self, i: usize) -> (usize, usize) {
        self.config.segment_and_offset(i)
    }

    #[inline]
    fn segment(&self, i: usize) -> &[T] {
        &self.segments[i]
    }

    #[inline]
    fn segment_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.segments[i]
    }
}

/// Provides an immutable view of elements from a range in [`SegVec`][crate::SegVec].
pub struct Slice<'a, T: 'a> {
    inner: &'a (dyn SegmentIndex<T>),
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
    // internal ctor
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
        let (start, end) = bounds(self.len, "Slice::subslice", range);
        Slice {
            inner: self.inner,
            start: self.start + start,
            len: end - start,
        }
    }
}

impl<'a, T: 'a> Index<usize> for Slice<'a, T> {
    type Output = T;

    fn index(&self, index: usize) -> &'a Self::Output {
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
        self.iter()
    }
}

/// Iterator over immutable references to the elements of a [`Slice`][crate::Slice].
pub struct SliceIter<'a, T: 'a> {
    iter: Flatten<SegmentedIter<'a, T>>,
    // Since Flatten is opaque we have to do our own accounting for size_hint here.
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

impl<'a, T> DoubleEndedIterator for SliceIter<'a, T> {
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

        let ret = if self.start.0 == self.end.0 {
            &self.slice.inner.segment(self.start.0)[self.start.1..=self.end.1]
        } else {
            &self.slice.inner.segment(self.start.0)[self.start.1..]
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

impl<'a, T> DoubleEndedIterator for SegmentedIter<'a, T> {
    fn next_back(&mut self) -> Option<Self::Item> {
        // We never return an empty slice
        if self.slice.len == 0 || self.start.0 > self.end.0 {
            return None;
        }

        let ret = if self.start.0 == self.end.0 {
            &self.slice.inner.segment(self.end.0)[self.start.1..=self.end.1]
        } else {
            &self.slice.inner.segment(self.end.0)[..=self.end.1]
        };
        // need to be careful not to underflow self.end.0 when done.
        if self.end.0 != 0 {
            self.end = (
                self.end.0 - 1,
                self.slice.inner.segment(self.end.0 - 1).len() - 1,
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
    inner: &'a mut dyn IndexMut<usize, Output = T>,
    start: usize,
    len: usize,
}

impl<'a, T: 'a> SliceMut<'a, T> {
    // internal ctor
    #[inline]
    pub(crate) fn new(
        segvec: &'a mut dyn IndexMut<usize, Output = T>,
        start: usize,
        len: usize,
    ) -> Self {
        SliceMut {
            inner: segvec,
            start,
            len,
        }
    }

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
