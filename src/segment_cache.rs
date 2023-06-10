use crate::*;

/// Calculating `segment_and_offset()` can be pretty expensive. While many accesses are close
/// to some previously accessed location within the same or neighboring segments which can be
/// calculated in a simpler way.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct SegmentCache<C: MemConfig> {
    start: usize,
    end: usize,
    segment: usize,
    _config: PhantomData<C>,
}

impl<C: MemConfig> SegmentCache<C> {
    /// creates a new `SegmentCache`, caching the first segment
    pub fn new() -> Self {
        Self {
            start: 0,
            end: C::factor(),
            segment: 0,
            _config: PhantomData,
        }
    }

    /// Segment size without updating the cache
    pub fn segment_size(&self, segment: usize) -> usize {
        if segment == self.segment {
            // cache hit
            self.end - self.start
        } else {
            // cache miss
            C::segment_size(segment)
        }
    }

    /// Index to Segment, updating the cache
    #[inline] // we have only few call sites, inlining won't bloat
    pub fn segment(&mut self, index: usize) -> usize {
        if (self.start..self.end).contains(&index) {
            // cache hit
            self.segment
        } else if index >= self.end && index < self.end + C::factor() {
            // in next segment
            self.start = self.end;
            self.segment += 1;
            self.end = self.start + C::segment_size(self.segment);
            self.segment
        } else {
            self.segment_cold(index)
        }
    }

    #[cold]
    fn segment_cold(&mut self, index: usize) -> usize {
        if index >= self.end && index < self.end + C::factor() {
            // in next segment
            self.start = self.end;
            self.segment += 1;
            self.end = self.start + C::segment_size(self.segment);
            self.segment
        } else if index < self.start && index + C::factor() >= self.start {
            // in previous segment
            self.end = self.start;
            self.segment -= 1;
            self.start = self.end - C::segment_size(self.segment);
            self.segment
        } else {
            // cache miss
            let s = C::segment(index);
            if s > 0 {
                let start = C::capacity(s);
                self.start = start;
                self.end = start + C::segment_size(s);
            } else {
                self.start = 0;
                self.end = C::factor();
            }
            self.segment = s;
            s
        }
    }

    /// Serves segment_and_offset() from a cache, updates the cache when needed.
    // PLANNED: return check_capacity: bool hint when segment becomes bigger, remove capacity again then
    #[inline]
    pub fn segment_and_offset(&mut self, index: usize) -> (usize, usize) {
        if (self.start..self.end).contains(&index) {
            // cache hit
            (self.segment, index - self.start)
        } else {
            // near misses/ misses
            self.segment_and_offset_cold(index)
        }
    }

    #[cold]
    fn segment_and_offset_cold(&mut self, index: usize) -> (usize, usize) {
        if index >= self.end && index < self.end + C::factor() {
            // in next segment
            self.start = self.end;
            self.segment += 1;
            self.end = self.start + C::segment_size(self.segment);
            (self.segment, index - self.start)
        } else if index < self.start && index + C::factor() >= self.start {
            // in previous segment
            self.end = self.start;
            self.segment -= 1;
            self.start = self.end - C::segment_size(self.segment);
            (self.segment, index - self.start)
        } else {
            // cache miss
            let (s, i) = C::segment_and_offset(index);
            if s > 0 {
                let start = C::capacity(s);
                self.start = start;
                self.end = start + C::segment_size(s);
            } else {
                self.start = 0;
                self.end = C::factor();
            }
            self.segment = s;
            (s, i)
        }
    }
}

#[test]
fn smoke() {
    let mut cache = SegmentCache::<Linear<1>>::new();
    assert_eq!(cache.segment_and_offset(0), (0, 0));
}

#[test]
fn hit() {
    let mut cache = SegmentCache::<Linear<2>>::new();
    assert_eq!(cache.segment_and_offset(0), (0, 0));
    assert_eq!(cache.segment_and_offset(1), (0, 1));
}

#[test]
fn next_prev_segment() {
    let mut cache = SegmentCache::<Exponential<2>>::new();
    assert_eq!(cache.segment_and_offset(2), (1, 0));
    assert_eq!(
        cache,
        SegmentCache {
            start: 2,
            end: 4,
            segment: 1,
            _config: PhantomData,
        }
    );
    assert_eq!(cache.segment_and_offset(1), (0, 1));
    assert_eq!(
        cache,
        SegmentCache {
            start: 0,
            end: 2,
            segment: 0,
            _config: PhantomData,
        }
    );
}

#[test]
fn miss() {
    let mut cache = SegmentCache::<Exponential<2>>::new();
    assert_eq!(cache.segment_and_offset(5), (2, 1));
    assert_eq!(
        cache,
        SegmentCache {
            start: 4,
            end: 8,
            segment: 2,
            _config: PhantomData,
        }
    );
}

#[test]
fn segment_size() {
    let cache = SegmentCache::<Exponential<2>>::new();
    assert_eq!(cache.segment_size(0), 2);
    assert_eq!(cache.segment_size(1), 2);
    assert_eq!(cache.segment_size(2), 4);
}
