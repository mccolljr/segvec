use super::*;
use rand::{self, Rng};
use std::{cell::Cell, collections::hash_map::DefaultHasher, hash::Hasher};

struct DropCount<'a, T: Copy>(&'a Cell<usize>, T);

impl<'a, T: Copy> Drop for DropCount<'a, T> {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1)
    }
}

#[test]
fn test_segvec_new() {
    let v = SegVec::<()>::new();
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 0);

    let v = SegVec::<()>::with_capacity(0);
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 0);
}

#[test]
fn test_segvec_with_capacity() {
    const TEST_MAX: usize = if cfg!(miri) {
        2usize.pow(5)
    } else {
        2usize.pow(16)
    };

    for hint in 1..TEST_MAX {
        let v = SegVec::<()>::with_capacity(hint);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), hint.next_power_of_two());
    }
}

#[test]
fn test_segvec_push_pop() {
    const TEST_MAX: usize = if cfg!(miri) {
        2usize.pow(5)
    } else {
        2usize.pow(16)
    };
    let mut v = SegVec::with_capacity(0);
    for i in 0..TEST_MAX {
        v.push(i);
        assert_eq!(v.segments.len(), (v.cap as f64).log2() as usize + 1);
    }
    assert_eq!(v.len(), TEST_MAX);
    assert_eq!(v.capacity(), TEST_MAX);
    for i in 0..TEST_MAX {
        assert_eq!(v.pop(), Some(TEST_MAX - i - 1));
    }
}

#[test]
fn test_segvec_truncate() {
    let dc = Cell::new(0usize);
    const TEST_MAX: usize = 2usize.pow(7);
    let mut v = SegVec::with_capacity(TEST_MAX);
    for i in 0..TEST_MAX {
        v.push(DropCount(&dc, i));
    }
    assert_eq!(v.len(), TEST_MAX);
    assert_eq!(v.capacity(), TEST_MAX);
    assert_eq!(v.segments.len(), 8);
    assert_eq!(dc.get(), 0);

    v.truncate(TEST_MAX + 1);
    assert_eq!(v.len(), TEST_MAX);
    assert_eq!(v.capacity(), TEST_MAX);
    assert_eq!(v.segments.len(), 8);
    assert_eq!(dc.get(), 0);

    v.truncate(TEST_MAX - 1);
    assert_eq!(v.len(), TEST_MAX - 1);
    assert_eq!(v.capacity(), TEST_MAX);
    assert_eq!(v.segments.len(), 8);
    assert_eq!(dc.get(), 1);

    v.truncate(TEST_MAX / 2);
    assert_eq!(v.len(), TEST_MAX / 2);
    assert_eq!(v.capacity(), TEST_MAX / 2);
    assert_eq!(v.segments.len(), 7);
    assert_eq!(dc.get(), TEST_MAX / 2);

    v.truncate(0);
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 0);
    assert_eq!(v.segments.len(), 0);
    assert_eq!(dc.get(), TEST_MAX);
}

#[test]
fn test_segvec_iter() {
    let mut v = SegVec::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);
    assert_eq!(v.len(), 7);
    assert_eq!(v.capacity(), 8);

    assert_eq!(v.iter().size_hint(), (7, Some(7)));
    assert_eq!(
        v.iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        v.iter().rev().copied().collect::<Vec<_>>(),
        vec![7, 6, 5, 4, 3, 2, 1]
    );

    let mut iter = v.iter();
    assert_eq!(iter.next().unwrap(), &1);
    assert_eq!(iter.next_back().unwrap(), &7);
    assert_eq!(iter.size_hint(), (5, Some(5)));
    assert_eq!(iter.next().unwrap(), &2);
    assert_eq!(iter.next_back().unwrap(), &6);
    assert_eq!(iter.size_hint(), (3, Some(3)));
    assert_eq!(iter.next().unwrap(), &3);
    assert_eq!(iter.next_back().unwrap(), &5);
    assert_eq!(iter.size_hint(), (1, Some(1)));
    assert_eq!(iter.next_back().unwrap(), &4);
    assert_eq!(iter.size_hint(), (0, Some(0)));

    assert_eq!(v.len(), 7);
    assert_eq!(v.capacity(), 8);
}

#[test]
fn test_segvec_into_iter() {
    let dc = Cell::new(0usize);
    let mut v = SegVec::new();
    v.push(DropCount(&dc, 1));
    v.push(DropCount(&dc, 2));
    v.push(DropCount(&dc, 3));
    v.push(DropCount(&dc, 4));
    v.push(DropCount(&dc, 5));
    v.push(DropCount(&dc, 6));
    v.push(DropCount(&dc, 7));
    assert_eq!(v.len(), 7);
    assert_eq!(v.capacity(), 8);

    let mut iter = v.into_iter();
    assert_eq!(dc.get(), 0);
    assert_eq!(iter.size_hint(), (7, Some(7)));

    assert_eq!(iter.next().unwrap().1, 1);
    assert_eq!(iter.next_back().unwrap().1, 7);
    assert_eq!(iter.size_hint(), (5, Some(5)));
    assert_eq!(dc.get(), 2);

    assert_eq!(iter.next().unwrap().1, 2);
    assert_eq!(iter.next_back().unwrap().1, 6);
    assert_eq!(iter.size_hint(), (3, Some(3)));
    assert_eq!(dc.get(), 4);

    drop(iter);
    assert_eq!(dc.get(), 7);
}

#[test]
fn test_segvec_from_iter() {
    let v = SegVec::from_iter([1, 2, 3, 4, 5, 6]);
    assert_eq!(v.len(), 6);
    assert_eq!(v.capacity(), 8);
    assert_eq!(v[0], 1);
    assert_eq!(v[1], 2);
    assert_eq!(v[2], 3);
    assert_eq!(v[3], 4);
    assert_eq!(v[4], 5);
    assert_eq!(v[5], 6);
    assert_eq!(v.get(6), None);
}

#[test]
fn test_segvec_insert_remove() {
    let mut v = SegVec::with_capacity(0);
    v.insert(0, 1);
    v.insert(0, 2);
    v.insert(0, 3);
    v.insert(0, 4);
    v.insert(0, 5);
    v.insert(0, 6);
    v.insert(0, 7);
    v.insert(4, 8);
    assert_eq!(v.len(), 8);
    assert_eq!(v.capacity(), 8);
    assert_eq!(
        v.iter().copied().collect::<Vec<_>>(),
        vec![7, 6, 5, 4, 8, 3, 2, 1]
    );

    assert_eq!(v.remove(7), 1);
    assert_eq!(v.remove(4), 8);
    assert_eq!(v.remove(4), 3);
    assert_eq!(v.remove(4), 2);
    assert_eq!(v.len(), 4);
    assert_eq!(v.capacity(), 8);
    assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![7, 6, 5, 4]);
}

#[test]
fn test_segvec_drain() {
    fn make_segvec_8() -> (Box<Cell<usize>>, SegVec<DropCount<'static, i32>>) {
        let dc = Box::into_raw(Box::new(Cell::new(0)));
        let mut v = SegVec::with_capacity(0);
        v.push(DropCount(unsafe { &*dc }, 1));
        v.push(DropCount(unsafe { &*dc }, 2));
        v.push(DropCount(unsafe { &*dc }, 3));
        v.push(DropCount(unsafe { &*dc }, 4));
        v.push(DropCount(unsafe { &*dc }, 5));
        v.push(DropCount(unsafe { &*dc }, 6));
        v.push(DropCount(unsafe { &*dc }, 7));
        v.push(DropCount(unsafe { &*dc }, 8));
        // Safety: tests will drop all items before dropping the boxed cell
        (unsafe { Box::from_raw(dc) }, v)
    }

    for i in 0..=8 {
        // range ..i
        let (dc, mut sv) = make_segvec_8();
        let d = sv.drain((Bound::Unbounded, Bound::Excluded(i)));
        let drained = d.map(|i| i.1).collect::<Vec<_>>();
        assert_eq!(dc.get(), i, "i={}", i);
        assert_eq!(sv.len(), 8 - i, "i={}", i);
        assert_eq!(drained, (1..i as i32 + 1).collect::<Vec<_>>(), "i={}", i);

        // range 0..i
        let (dc, mut sv) = make_segvec_8();
        let d = sv.drain((Bound::Included(0), Bound::Excluded(i)));
        let drained = d.map(|i| i.1).collect::<Vec<_>>();
        assert_eq!(dc.get(), i, "i={}", i);
        assert_eq!(sv.len(), 8 - i, "i={}", i);
        assert_eq!(drained, (1..i as i32 + 1).collect::<Vec<_>>(), "i={}", i);

        if i > 0 {
            // range 1..i
            let (dc, mut sv) = make_segvec_8();
            let d = sv.drain((Bound::Excluded(0), Bound::Excluded(i)));
            let drained = d.map(|i| i.1).collect::<Vec<_>>();
            assert_eq!(dc.get(), i - 1, "i={}", i);
            assert_eq!(sv.len(), 8 - (i - 1), "i={}", i);
            assert_eq!(drained, (2..i as i32 + 1).collect::<Vec<_>>(), "i={}", i);

            // range 1..i
            let (dc, mut sv) = make_segvec_8();
            let d = sv.drain((Bound::Included(1), Bound::Excluded(i)));
            let drained = d.map(|i| i.1).collect::<Vec<_>>();
            assert_eq!(dc.get(), i - 1, "i={}", i);
            assert_eq!(sv.len(), 8 - (i - 1), "i={}", i);
            assert_eq!(drained, (2..i as i32 + 1).collect::<Vec<_>>(), "i={}", i);
        }

        if i > 0 && i < 8 {
            // range 1..=i
            let (dc, mut sv) = make_segvec_8();
            let d = sv.drain((Bound::Excluded(0), Bound::Included(i)));
            let drained = d.map(|i| i.1).collect::<Vec<_>>();
            assert_eq!(dc.get(), i, "i={}", i);
            assert_eq!(sv.len(), 8 - i, "i={}", i);
            assert_eq!(drained, (2..=(i as i32 + 1)).collect::<Vec<_>>(), "i={}", i);

            // range 1..=i
            let (dc, mut sv) = make_segvec_8();
            let d = sv.drain((Bound::Included(1), Bound::Included(i)));
            let drained = d.map(|i| i.1).collect::<Vec<_>>();
            assert_eq!(dc.get(), i, "i={}", i);
            assert_eq!(sv.len(), 8 - i, "i={}", i);
            assert_eq!(drained, (2..=(i as i32 + 1)).collect::<Vec<_>>(), "i={}", i);
        }
    }
}

#[test]
fn test_segvec_slice() {
    let mut v = SegVec::with_capacity(8);
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);
    v.push(8);
    let s1 = v.slice(0..4);
    let s2 = v.slice(4..8);
    let s3 = v.slice(2..6);
    let s4 = v.slice(..);
    let s5 = v.slice(..0);
    // invalid:
    // v.truncate(0); // <- Slices immutably borrow the underlying SegVec
    assert_eq!(s1.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    assert_eq!(s2.iter().copied().collect::<Vec<_>>(), vec![5, 6, 7, 8]);
    assert_eq!(s3.iter().copied().collect::<Vec<_>>(), vec![3, 4, 5, 6]);
    assert_eq!(
        s4.iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(s5.iter().copied().collect::<Vec<_>>(), vec![]);
}

#[test]
fn test_segvec_slice_mut() {
    let mut v = SegVec::with_capacity(8);
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);
    v.push(8);
    let mut s = v.slice_mut(..);
    // invalid:
    // v.push(1000); // <- SliceMuts mutably borrow the underlying SegVec
    s[0] = 100;
    s.into_iter().for_each(|v| *v *= 2);
    assert_eq!(
        v.into_iter().collect::<Vec<_>>(),
        vec![200, 4, 6, 8, 10, 12, 14, 16]
    );
}

#[test]
fn test_segvec_sort() {
    let mut rng = rand::thread_rng();
    for i in 0..1000usize {
        let mut v = SegVec::with_capacity(i);
        while v.len() < v.capacity() {
            v.push(rng.gen_range(0i32..100));
        }

        // sorted descending
        v.sort();
        if i > 0 {
            for j in 0..i - 1 {
                assert!(&v[j] <= &v[j + 1], "{:?}", v);
            }
        }
    }
}

#[test]
fn test_segvec_hash() {
    let mut v1 = SegVec::with_capacity(8);
    v1.push(1);
    v1.push(2);
    let mut v2 = SegVec::with_capacity(4);
    v2.push(1);
    v2.push(2);
    let mut h1 = DefaultHasher::new();
    v1.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    v2.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

#[test]
fn test_segvec_extend() {
    let mut v = SegVec::new();
    v.extend([1, 2, 3, 4, 5]);
    assert_eq!(v.len(), 5);
    assert_eq!(v.capacity(), 8);
    assert_eq!(v.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5]);
}
