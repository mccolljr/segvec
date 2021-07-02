use std::{convert::TryFrom, fmt::Debug, mem::ManuallyDrop, num::NonZeroUsize, ptr::NonNull};

/// Pointer to the head of a segment. Does not store any information about capacity or
/// length, as this can be derived from information available in the parent data structure.
#[repr(transparent)]
struct SegPtr<T>(NonNull<T>);

impl<T> SegPtr<T> {
    /// Construct a new `SegPtr`.
    pub fn new(capacity: usize) -> Self {
        let mut vec = ManuallyDrop::new(Vec::with_capacity(capacity));
        // SAFETY:
        // vec will not be dropped, and the pointer returned by Vec::as_mut_ptr is aligned and non-null
        SegPtr(unsafe { NonNull::new_unchecked(vec.as_mut_ptr()) })
    }

    /// Returns an immutable reference to the data at the given index in the segment.
    ///
    /// # Safety
    /// 1. `index` MUST point to an undropped, live value in the segment.
    /// 2. `take_index` MUST NOT be called with an index equal to `index` while the returned reference is alive
    /// 3. `get_ref_mut` MUST NOT be called with an index equal to `index` while the returned reference is alive
    pub unsafe fn get_ref(&self, index: usize) -> &T {
        &*self
            .0
            .as_ptr()
            .offset(isize::try_from(index).expect("index fits into isize"))
    }

    /// Returns a mutable reference to the data at the given index in the segment.
    ///
    /// # Safety
    /// 1. `index` MUST point to an undropped, live value in the segment.
    /// 2. `take_index` MUST NOT be called with an index equal to `index` while the
    ///    the pointer or any references derived from it are alive
    /// 3. `get_ref_mut` MUST NOT be called with an index equal to `index` while the returned reference is alive
    pub unsafe fn get_ref_mut(&mut self, index: usize) -> &mut T {
        &mut *self
            .0
            .as_ptr()
            .offset(isize::try_from(index).expect("index fits into isize"))
    }

    /// Reads and returns the data at the given index in the segment.
    ///
    /// # Safety
    /// 1. `index` MUST point to an undropped, live value in the segment.
    /// 2. `take_index` MUST not be called with the same value for `index` again before
    ///     a call to `write_index` is made for that value of `index`.
    pub unsafe fn take_index(&mut self, index: usize) -> T {
        std::ptr::read(
            self.0
                .as_ptr()
                .offset(isize::try_from(index).expect("index fits into isize")),
        )
    }

    /// Writes the given value to the given index in the segment.
    ///
    /// # Safety
    /// 1. `index` MUST be the index immediately following the index of the last undropped, live element in the segment
    /// 2. `index` MUST be less than the capacity the segment was allocated with
    pub unsafe fn write_index(&mut self, index: usize, value: T) {
        std::ptr::write(
            self.0
                .as_ptr()
                .offset(isize::try_from(index).expect("index fits into isize")),
            value,
        )
    }

    /// Drops a range of values at the end of the segment. Equivalent to calling `SegPtr::take_index` for
    /// each index in the range (`len`, `start`]
    ///
    /// # Safety
    /// 1. `start` MUST point to an undropped, live element in the segment
    /// 2. `len` MUST accurately represent the number of undropped, live elements in the segment
    /// 3. `len` MUST be less than or equal to the capacity the segment was allocated with
    pub unsafe fn deallocate_tail(&mut self, start: usize, len: usize) {
        for index in (start..len).rev() {
            drop(self.take_index(index))
        }
    }

    /// Turns the segment into a `Vec<T>` with the given length and capacity.
    ///
    /// # Safety
    /// 1. `capacity` MUST be equal to the capacity the segment was allocated with
    /// 2. `len` MUST accurately represent the number of undropped, live elements in the segment
    /// 3. `len` MUST be less than or equal to `cap`
    pub unsafe fn into_vec(self, len: usize, capacity: usize) -> Vec<T> {
        Vec::from_raw_parts(self.0.as_ptr(), len, capacity)
    }
}

struct SegList<T: Debug> {
    len: usize,
    capacity: usize,
    segments: Vec<SegPtr<T>>,
    growth_factor: NonZeroUsize,
}

impl<T: Debug> SegList<T> {
    pub const fn new() -> Self {
        Self::with_growth_factor(1)
    }

    pub const fn with_growth_factor(growth_factor_hint: usize) -> Self {
        SegList {
            len: 0,
            capacity: 0,
            segments: Vec::new(),
            growth_factor: unsafe {
                NonZeroUsize::new_unchecked(growth_factor_hint.next_power_of_two())
            },
        }
    }

    pub fn with_capacity(capacity_hint: usize) -> Self {
        Self::with_capacity_and_growth_factor(capacity_hint, 1)
    }

    pub fn with_capacity_and_growth_factor(
        capacity_hint: usize,
        growth_factor_hint: usize,
    ) -> Self {
        let mut v = Self::with_growth_factor(growth_factor_hint);
        v.reserve(capacity_hint);
        v
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn growth_factor(&self) -> usize {
        self.growth_factor.get()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.len() {
            let (seg, offset) = self.segment_and_offset(index);
            // SAFETY:
            // Operations that can invalidate the reference or modify the referenced value requre
            // a mutable reference to self, which is blocked by the immutable reference held in this
            // function call
            Some(unsafe { self.segments.get(seg).unwrap().get_ref(offset) })
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index < self.len() {
            let (seg, offset) = self.segment_and_offset(index);
            // SAFETY:
            // Operations that can invalidate the reference or modify the referenced value requre
            // a mutable reference to self, which is blocked by the mutable reference held in this
            // function call
            Some(unsafe { self.segments.get_mut(seg).unwrap().get_ref_mut(offset) })
        } else {
            None
        }
    }

    pub fn push(&mut self, val: T) {
        self.reserve(1);
        let (seg, offset) = self.segment_and_offset(self.len());
        // SAFETY:
        // seg and offset point to the position in the structure one past the last valid element,
        // and offset is guaranteed to be less than the capacity of seg
        unsafe { self.segments.get_mut(seg).unwrap().write_index(offset, val) };
        self.len += 1;
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len() > 0 {
            self.len -= 1;
            let (seg, offset) = self.segment_and_offset(self.len());
            // SAFETY:
            // The length of the structure has already decreased by one, so the element
            // pointed to by seg and offset is exactly the element to pop.
            Some(unsafe { self.segments.get_mut(seg).unwrap().take_index(offset) })
        } else {
            None
        }
    }

    pub fn reserve(&mut self, additional: usize) {
        let min_cap = match self.len().checked_add(additional) {
            Some(c) => c,
            None => {
                // capacity overflow
                todo!()
            }
        };
        if min_cap <= self.capacity() {
            return;
        }
        let (segment, _) = self.segment_and_offset(min_cap - 1);
        for i in self.segments.len()..=segment {
            let seg_size = self.segment_capacity(i);
            self.segments.push(SegPtr::new(seg_size));
            self.capacity += seg_size;
        }
    }

    pub fn truncate(&mut self, len: usize) {
        if len < self.capacity() {
            let (seg, offset) = self.segment_and_offset(len);
            // SAFETY:
            // self.len() will not change until the end of this function
            unsafe {
                if offset == 0 {
                    self.truncate_segments(seg);
                } else {
                    if len < self.len() {
                        self.truncate_segment(seg, offset);
                    }
                    self.truncate_segments(seg + 1);
                }
            }
            self.capacity = match self.segments.len() {
                0 => 0,
                n => self.segment_capacity(n),
            };
            self.len = len;
        }
    }

    #[allow(unused_unsafe)]
    /// # Safety
    /// 1. `self.len()` must be stable for the duration of this function call.
    unsafe fn truncate_segments(&mut self, from_segment_index: usize) {
        println!("truncating segments {}..", from_segment_index);
        let num_segments = self.segments.len();
        if from_segment_index < num_segments {
            for segment_index in (from_segment_index..num_segments).rev() {
                let segment = self.segments.pop().unwrap();
                let (len, capacity) = self.segment_len_and_capacity(segment_index);
                // SAFETY:
                // 1. The capacity of the segment at segment_index is deterministic
                // 2. The length of the segment at segment_index is deterministic for a given value of `self.len()`
                // 3. The calculated value of len is clamped between 0 and the segment's capacity
                let v = unsafe { segment.into_vec(len, capacity) };
                dbg!(v);
            }
        }
    }

    /// # Safety
    /// 1. `self.len()` must be stable for the duration of this function call
    #[allow(unused_unsafe)]
    unsafe fn truncate_segment(&mut self, segment_index: usize, new_len: usize) {
        println!("truncating segment {} from {}..", segment_index, new_len);
        let num_segments = self.segments.len();
        if segment_index < num_segments {
            let (len, _) = self.segment_len_and_capacity(segment_index);
            if new_len < len {
                // SAFETY:
                // 1. `new_len` is strictly less than `len`
                // 2. `len` will be deterministic for any value of `self.len()`
                // 3. `len` is guarantee to be less than or equal to the capacity of the segment
                unsafe { self.segments[segment_index].deallocate_tail(new_len, len) };
            }
        }
    }

    fn segment_len_and_capacity(&self, segment_index: usize) -> (usize, usize) {
        // The capacity of the segment at segment_index is necessarily equal to the
        // combined capacity of all previous segments due to the property that each
        // allocated segment doubles the total capacity of the data structure.
        let segment_capacity = self.segment_capacity(segment_index);
        // The total number of elements currently stored in the segment at segment_index
        // can be calculated by taking the difference between the total length of the
        // data structure and segment_capacity, clamped to the range (0,segment_capacity).
        // This is correct, because:
        // 1. If the data structure has a len smaller than segment_capacity, meaning that
        //    the segment at segment_index contains no elements (as the filled slots are all
        //    in earlier segments)
        // 2. Otherwise, the data structure has a len larger then segment_capacity, and either the segment
        //    at segment_index is partially full, or it is completely full. In either case
        //    it will not contain more elements than it has capacity.
        (
            self.len()
                .saturating_sub(segment_capacity)
                .clamp(0, segment_capacity),
            segment_capacity,
        )
    }

    fn segment_capacity(&self, segment_index: usize) -> usize {
        match segment_index {
            0 => self.growth_factor.get(),
            n => {
                let pow = u32::try_from(n - 1).expect("fewer than 64 segments");
                match 2usize
                    .checked_pow(pow)
                    .and_then(|n| n.checked_mul(self.growth_factor.get()))
                {
                    Some(size) => size,
                    None => unimplemented!("todo: capacity overflow"),
                }
            }
        }
    }

    fn segment_and_offset(&self, linear_index: usize) -> (usize, usize) {
        let normal = linear_index / self.growth_factor.get();
        let (segment, pow) = match checked_log2_ceil(normal) {
            None => (0usize, 0u32),
            Some(s) => (s as usize + 1, s),
        };
        match 2usize.pow(pow).checked_mul(self.growth_factor.get()) {
            Some(mod_base) => {
                let offset = linear_index % mod_base;
                (segment, offset)
            }
            None => {
                // TODO: index_oob
                unimplemented!("index oob")
            }
        }
    }

    fn swap(&mut self, a: usize, b: usize) {
        if a != b {
            let av = unsafe { &mut *(self.get_mut(a).unwrap() as *mut T) };
            let bv = unsafe { &mut *(self.get_mut(b).unwrap() as *mut T) };
            std::mem::swap(av, bv);
        }
    }
}

impl<T: Debug> Drop for SegList<T> {
    fn drop(&mut self) {
        // drop & deallocate all segments and elements
        self.truncate(0);
    }
}

fn checked_log2_ceil(v: usize) -> Option<u32> {
    if v > 0 {
        Some((usize::BITS - 1) - v.leading_zeros())
    } else {
        None
    }
}

#[cfg(test)]
mod test_utils {
    use std::cell::Cell;

    #[derive(Debug)]
    pub struct DropCounter(Cell<usize>);

    impl DropCounter {
        pub fn new() -> Self {
            DropCounter(Cell::new(0))
        }

        pub fn droppable<T>(&self, value: T) -> Droppable<T> {
            Droppable {
                counter: &self.0,
                value,
            }
        }

        pub fn get(&self) -> usize {
            self.0.get()
        }
    }

    #[derive(Debug)]
    pub struct Droppable<'a, T> {
        counter: &'a Cell<usize>,
        pub value: T,
    }

    impl<'a, T> Drop for Droppable<'a, T> {
        fn drop(&mut self) {
            self.counter.set(self.counter.get() + 1);
        }
    }
}

#[cfg(test)]
mod raw_tests {
    use super::test_utils::*;
    use super::*;

    #[test]
    fn test_seg_list() {
        let counter = DropCounter::new();
        assert_eq!(counter.get(), 0);

        let mut v = SegList::new();
        assert_eq!(v.len(), 0);
        assert_eq!(v.growth_factor.get(), 1);

        v.push(counter.droppable(1));
        v.push(counter.droppable(2));
        v.push(counter.droppable(3));
        v.push(counter.droppable(4));
        v.push(counter.droppable(5));
        assert_eq!(v.len(), 5);
        assert_eq!(v.capacity(), 8);
        assert_eq!(v.segments.len(), 4);

        assert_eq!(v.get(0).unwrap().value, 1);
        assert_eq!(v.get(1).unwrap().value, 2);
        assert_eq!(v.get(2).unwrap().value, 3);
        assert_eq!(v.get(3).unwrap().value, 4);
        assert_eq!(v.get(4).unwrap().value, 5);
        assert!(v.get(5).is_none());

        assert_eq!(v.pop().unwrap().value, 5);
        assert_eq!(v.pop().unwrap().value, 4);
        assert_eq!(v.len(), 3);
        assert_eq!(v.capacity(), 8);
        assert_eq!(counter.get(), 2);

        v.truncate(1);
        assert_eq!(v.len(), 1);
        assert_eq!(v.capacity(), 1);
        assert_eq!(counter.get(), 4);

        // drop(v);
        v.truncate(0);
        assert_eq!(counter.get(), 5);
    }

    // #[test]
    // fn test_segment_logic() {
    //     {
    //         type S = Raw<(), 1>;
    //         assert_eq!(S::segment_size(0), 1);
    //         assert_eq!(S::segment_size(1), 1);
    //         assert_eq!(S::segment_size(2), 2);
    //         assert_eq!(S::segment_size(3), 4);
    //         assert_eq!(S::segment_size(4), 8);
    //         assert_eq!(S::segment_and_offset(2), (2, 0));
    //     }
    //     {
    //         type S = Raw<(), 2>;
    //         assert_eq!(S::segment_size(0), 2);
    //         assert_eq!(S::segment_size(1), 2);
    //         assert_eq!(S::segment_size(2), 4);
    //         assert_eq!(S::segment_size(3), 8);
    //         assert_eq!(S::segment_size(4), 16);
    //         assert_eq!(S::segment_and_offset(2), (1, 0));
    //         assert_eq!(S::segment_and_offset(15), (3, 7));
    //         assert_eq!(S::segment_and_offset(32), (5, 0));
    //         assert_eq!(S::segment_and_offset(51), (5, 19));
    //     }
    //     {
    //         type S = Raw<(), 4>;
    //         assert_eq!(S::segment_size(0), 4);
    //         assert_eq!(S::segment_size(1), 4);
    //         assert_eq!(S::segment_size(2), 8);
    //         assert_eq!(S::segment_size(3), 16);
    //         assert_eq!(S::segment_size(4), 32);
    //         assert_eq!(S::segment_and_offset(2), (0, 2));
    //         assert_eq!(S::segment_and_offset(15), (2, 7));
    //         assert_eq!(S::segment_and_offset(32), (4, 0));
    //         assert_eq!(S::segment_and_offset(51), (4, 19));
    //     }
    //     {
    //         type S = Raw<(), 8>;
    //         assert_eq!(S::segment_size(0), 8);
    //         assert_eq!(S::segment_size(1), 8);
    //         assert_eq!(S::segment_size(2), 16);
    //         assert_eq!(S::segment_size(3), 32);
    //         assert_eq!(S::segment_size(4), 64);
    //         assert_eq!(S::segment_and_offset(0), (0, 0));
    //         assert_eq!(S::segment_and_offset(1), (0, 1));
    //         assert_eq!(S::segment_and_offset(3), (0, 3));
    //         assert_eq!(S::segment_and_offset(5), (0, 5));
    //         assert_eq!(S::segment_and_offset(7), (0, 7));
    //         assert_eq!(S::segment_and_offset(8), (1, 0));
    //         assert_eq!(S::segment_and_offset(9), (1, 1));
    //         assert_eq!(S::segment_and_offset(11), (1, 3));
    //         assert_eq!(S::segment_and_offset(13), (1, 5));
    //         assert_eq!(S::segment_and_offset(15), (1, 7));
    //         assert_eq!(S::segment_and_offset(16), (2, 0));
    //         assert_eq!(S::segment_and_offset(17), (2, 1));
    //         assert_eq!(S::segment_and_offset(19), (2, 3));
    //         assert_eq!(S::segment_and_offset(23), (2, 7));
    //         assert_eq!(S::segment_and_offset(29), (2, 13));
    //         assert_eq!(S::segment_and_offset(31), (2, 15));
    //         assert_eq!(S::segment_and_offset(32), (3, 0));
    //         assert_eq!(S::segment_and_offset(33), (3, 1));
    //         assert_eq!(S::segment_and_offset(45), (3, 13));
    //         assert_eq!(S::segment_and_offset(48), (3, 16));
    //         assert_eq!(S::segment_and_offset(49), (3, 17));
    //         assert_eq!(S::segment_and_offset(50), (3, 18));
    //         assert_eq!(S::segment_and_offset(60), (3, 28));
    //         assert_eq!(S::segment_and_offset(63), (3, 31));
    //         assert_eq!(S::segment_and_offset(64), (4, 0));
    //         assert_eq!(S::segment_and_offset(65), (4, 1));
    //         assert_eq!(S::segment_and_offset(127), (4, 63));
    //         assert_eq!(S::segment_and_offset(128), (5, 0));
    //     }
    // }

    // #[test]
    // fn test_segment_behavior() {
    //     let mut s = Raw::<(), 1>::new();
    //     assert_eq!(s.capacity(), 0);
    //     s.reserve(1);
    //     assert_eq!(s.capacity(), 1);
    //     s.reserve(2);
    //     assert_eq!(s.capacity(), 2);
    //     s.reserve(3);
    //     assert_eq!(s.capacity(), 4);
    //     s.reserve(4);
    //     assert_eq!(s.capacity(), 4);
    //     s.reserve(9);
    //     assert_eq!(s.capacity(), 16);

    //     let mut s = Raw::<(), 8>::new();
    //     assert_eq!(s.capacity(), 0);
    //     s.reserve(1);
    //     assert_eq!(s.capacity(), 8);
    //     s.reserve(9);
    //     assert_eq!(s.capacity(), 16);
    //     s.reserve(17);
    //     assert_eq!(s.capacity(), 32);
    //     s.reserve(33);
    //     assert_eq!(s.capacity(), 64);

    //     let mut s = Raw::<(), 512>::new();
    //     assert_eq!(s.capacity(), 0);
    //     s.reserve(1);
    //     assert_eq!(s.capacity(), 512);
    //     s.reserve(512);
    //     assert_eq!(s.capacity(), 512);
    //     s.reserve(513);
    //     assert_eq!(s.capacity(), 1024);
    // }
}
