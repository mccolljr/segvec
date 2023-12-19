#![allow(clippy::needless_range_loop)]
use num_integer::Roots;

// Note: we do *not* need checked math here, because in all practical applications we will see
//       out-of-memory long before the math overflows

// The math here works with any non-zero value for FACTOR, but using powers of two should
// optimize to more efficient code.

/// Configures the sizes of segments and how to index entries. `Linear`, `Proportional` and
/// `Exponential` implement this trait.
pub trait MemConfig {
    fn new() -> Self;

    /// Called by ctors to assert that the configuration is valid.
    ///
    /// Some `MemConfig` implementations may put constraints on (const generic)
    /// parameters. Currently it is impossible to assert these at compile time in stable
    /// rust. In debug builds we check these constraints when a `SegVec` uses some
    /// config. The three shipped `MemConfig` implementations check here that the 'FACTOR' is
    /// not zero.
    fn debug_assert_config();

    /// Takes the number of allocated segments, returns the total capacity.
    fn capacity(&self, segments: usize) -> usize;

    /// Updates the capacity to the given number of segments.
    #[inline]
    fn update_capacity(&mut self, _segments: usize) {}

    /// Returns the size of the nth segment (starting at 0).
    fn segment_size(&self, segment: usize) -> usize;

    /// Translates a flat index into (segment, offset)
    fn segment_and_offset(&self, index: usize) -> (usize, usize);
}

/// Linear growth, all segments have the same (FACTOR) length.
pub struct Linear<const FACTOR: usize = 1024>;
impl<const FACTOR: usize> MemConfig for Linear<FACTOR> {
    fn new() -> Self {
        Self {}
    }

    #[track_caller]
    fn debug_assert_config() {
        debug_assert_ne!(FACTOR, 0, "FACTOR must be greater than 0")
    }

    #[inline]
    fn capacity(&self, segments: usize) -> usize {
        segments * FACTOR
    }

    #[inline]
    fn segment_size(&self, _segment: usize) -> usize {
        FACTOR
    }

    #[inline]
    fn segment_and_offset(&self, index: usize) -> (usize, usize) {
        (index / FACTOR, index % FACTOR)
    }
}

/// Proportional growth, each segment is segment_number*FACTOR sized.
pub struct Proportional<const FACTOR: usize>;
impl<const FACTOR: usize> MemConfig for Proportional<FACTOR> {
    fn new() -> Self {
        Self {}
    }

    #[track_caller]
    fn debug_assert_config() {
        debug_assert_ne!(FACTOR, 0, "FACTOR must be greater than 0")
    }

    #[inline]
    fn capacity(&self, segments: usize) -> usize {
        segments * (segments + 1) / 2 * FACTOR
    }

    #[inline]
    fn segment_size(&self, segment: usize) -> usize {
        (segment + 1) * FACTOR
    }

    #[inline]
    fn segment_and_offset(&self, index: usize) -> (usize, usize) {
        let linear_segment = index / FACTOR;
        let segment = ((8 * linear_segment + 1).sqrt() - 1) / 2;

        (segment, index - self.capacity(segment))
    }
}

/// Exponential growth, each subsequent segment is as big as the sum of all segments before.
pub struct Exponential<const FACTOR: usize = 16> {
    capacity: usize,
}

impl<const FACTOR: usize> MemConfig for Exponential<FACTOR> {
    fn new() -> Self {
        Self { capacity: 0 }
    }

    #[track_caller]
    fn debug_assert_config() {
        debug_assert_ne!(FACTOR, 0, "FACTOR must be greater than 0")
    }

    #[inline]
    fn capacity(&self, _segments: usize) -> usize {
        self.capacity
    }

    #[inline]
    fn update_capacity(&mut self, segments: usize) {
        self.capacity = if segments == 0 {
            0
        } else {
            2_usize.pow(segments as u32 - 1) * FACTOR
        };
    }

    #[inline]
    fn segment_size(&self, segment: usize) -> usize {
        if segment == 0 {
            FACTOR
        } else {
            2usize.pow(segment as u32 - 1) * FACTOR
        }
    }

    #[inline]
    fn segment_and_offset(&self, index: usize) -> (usize, usize) {
        let linear_segment = index / FACTOR;

        if linear_segment == 0 {
            return (0, index);
        }

        let segment = linear_segment.ilog2() as usize;
        (segment + 1, index - (2_usize.pow(segment as u32) * FACTOR))
    }
}

#[test]
pub fn linear_capacity() {
    let capacities: &[usize] = &[0, 32, 64, 96, 128, 160, 192, 224];

    for i in 0..capacities.len() {
        assert_eq!(Linear::<32>::new().capacity(i), capacities[i])
    }
}

#[test]
pub fn linear_segment_size() {
    let segment_sizes: &[usize] = &[1, 1, 1, 1, 1, 1, 1];

    for i in 0..segment_sizes.len() {
        assert_eq!(Linear::<1>::new().segment_size(i), segment_sizes[i])
    }

    let segment_sizes: &[usize] = &[3, 3, 3, 3, 3, 3, 3];

    for i in 0..segment_sizes.len() {
        assert_eq!(Linear::<3>::new().segment_size(i), segment_sizes[i])
    }
}

#[test]
pub fn linear_segment_and_offset() {
    let segment_indices: &[(usize, usize)] = &[
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
    ];

    for i in 1..segment_indices.len() {
        assert_eq!(Linear::<2>::new().segment_and_offset(i), segment_indices[i])
    }
}

#[test]
pub fn proportional_capacity() {
    // For FACTOR=1
    let capacities: &[usize] = &[0, 1, 3, 6, 10, 15, 21];

    for i in 0..capacities.len() {
        assert_eq!(Proportional::<1>::new().capacity(i), capacities[i])
    }

    // For FACTOR=3
    let capacities: &[usize] = &[0, 3, 9, 18, 30, 45];

    for i in 0..capacities.len() {
        assert_eq!(Proportional::<3>::new().capacity(i), capacities[i])
    }
}

#[test]
pub fn proportional_segment_size() {
    let segment_sizes: &[usize] = &[1, 2, 3, 4, 5, 6, 7, 8];

    for i in 0..segment_sizes.len() {
        assert_eq!(Proportional::<1>::new().segment_size(i), segment_sizes[i])
    }

    let segment_sizes: &[usize] = &[2, 4, 6, 8, 10, 12, 14, 16];

    for i in 0..segment_sizes.len() {
        assert_eq!(Proportional::<2>::new().segment_size(i), segment_sizes[i])
    }

    let segment_sizes: &[usize] = &[3, 6, 9, 12, 15, 18, 21, 24];

    for i in 0..segment_sizes.len() {
        assert_eq!(Proportional::<3>::new().segment_size(i), segment_sizes[i])
    }
}

#[test]
pub fn proportional_segment_and_offset() {
    let segment_indices: &[(usize, usize)] = &[
        (0, 0),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (4, 0),
    ];

    for i in 1..segment_indices.len() {
        assert_eq!(
            Proportional::<1>::new().segment_and_offset(i),
            segment_indices[i]
        )
    }

    let segment_indices: &[(usize, usize)] = &[
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
        (3, 0),
        (3, 1),
    ];

    for i in 1..segment_indices.len() {
        assert_eq!(
            Proportional::<2>::new().segment_and_offset(i),
            segment_indices[i]
        )
    }
}

#[test]
pub fn exponential_capacity() {
    let capacities: &[usize] = &[0, 1, 2, 4, 8, 16, 32, 64];

    for i in 0..capacities.len() {
        let mut e = Exponential::<1>::new();
        e.update_capacity(i);
        assert_eq!(e.capacity(i), capacities[i])
    }
}

#[test]
pub fn exponential_segment_size() {
    let segment_sizes: &[usize] = &[1, 1, 2, 4, 8, 16, 32, 64];

    for i in 0..segment_sizes.len() {
        assert_eq!(Exponential::<1>::new().segment_size(i), segment_sizes[i])
    }

    let segment_sizes: &[usize] = &[4, 4, 8, 16, 32, 64, 128, 256];

    for i in 0..segment_sizes.len() {
        assert_eq!(Exponential::<4>::new().segment_size(i), segment_sizes[i])
    }
}

#[test]
pub fn exponential_segment_and_offset() {
    // FACTOR = 1
    let segment_indices: &[(usize, usize)] =
        &[(0, 0), (1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)];

    for i in 1..segment_indices.len() {
        assert_eq!(
            Exponential::<1>::new().segment_and_offset(i),
            segment_indices[i]
        )
    }

    // FACTOR = 2
    let segment_indices: &[(usize, usize)] = &[
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
        (3, 5),
        (3, 6),
        (3, 7),
        (4, 0),
    ];

    for i in 1..segment_indices.len() {
        assert_eq!(
            Exponential::<2>::new().segment_and_offset(i),
            segment_indices[i]
        )
    }
}

// Expensive tests (hopefully) catching off by one errors, need to be explicitly enabled

#[test]
#[cfg_attr(miri, ignore)]
pub fn linear() {
    type DuT = Linear<64>;

    // segments
    let mut sum = 0_usize;
    for i in 0..10000000 {
        sum += DuT::new().segment_size(i);
        assert_eq!(DuT::new().capacity(i + 1), sum)
    }

    // indices
    for i in 1..10000000 {
        let (segment, index) = DuT::new().segment_and_offset(i);
        assert!(index < DuT::new().segment_size(segment));
        if index == 0 {
            let (psegment, pindex) = DuT::new().segment_and_offset(i - 1);
            assert_eq!(psegment, segment - 1);
            assert!(pindex < DuT::new().segment_size(psegment));
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
pub fn proportional() {
    type DuT = Proportional<4>;

    // segments
    let mut sum = 0_usize;
    for i in 0..10000000 {
        sum += DuT::new().segment_size(i);
        assert_eq!(DuT::new().capacity(i + 1), sum)
    }

    // indices
    for i in 1..10000000 {
        let (segment, index) = DuT::new().segment_and_offset(i);
        assert!(index < DuT::new().segment_size(segment));
        if index == 0 {
            let (psegment, pindex) = DuT::new().segment_and_offset(i - 1);
            assert_eq!(psegment, segment - 1);
            assert!(pindex < DuT::new().segment_size(psegment));
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
pub fn exponential() {
    type DuT = Exponential<4>;

    // segments
    let mut sum = 0_usize;
    for i in 0..60 {
        sum += DuT::new().segment_size(i);
        let mut e = DuT::new();
        e.update_capacity(i + 1);
        assert_eq!(e.capacity(i + 1), sum)
    }

    // indices
    for i in 1..10000000 {
        let (segment, index) = DuT::new().segment_and_offset(i);
        assert!(index < DuT::new().segment_size(segment));
        if index == 0 {
            let (psegment, pindex) = DuT::new().segment_and_offset(i - 1);
            assert_eq!(psegment, segment - 1);
            assert!(pindex < DuT::new().segment_size(psegment));
        }
    }
}
