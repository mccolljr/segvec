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
fn test_checked_log2_floor() {
    assert_eq!(checked_log2_floor(0), None);
    assert_eq!(checked_log2_floor(1), Some(0));
    assert_eq!(checked_log2_floor(2), Some(1));
    assert_eq!(checked_log2_floor(3), Some(1));
    assert_eq!(checked_log2_floor(4), Some(2));
    assert_eq!(checked_log2_floor(5), Some(2));
    assert_eq!(checked_log2_floor(6), Some(2));
    assert_eq!(checked_log2_floor(7), Some(2));
    assert_eq!(checked_log2_floor(8), Some(3));
}

#[test]
fn test_new() {
    let v = SegVec::<()>::new();
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 0);

    let v = SegVec::<(), Exponential<1>>::with_capacity(0);
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 0);
}

#[test]
#[should_panic(expected = "FACTOR must be greater than 0")]
fn test_new_with_bad_factor() {
    SegVec::<(), Exponential<0>>::new();
}

#[test]
fn test_with_capacity() {
    const TEST_MAX: usize = if cfg!(miri) {
        2usize.pow(5)
    } else {
        2usize.pow(16)
    };

    for hint in 1..TEST_MAX {
        let v = SegVec::<(), Exponential<1>>::with_capacity(hint);
        assert_eq!(v.len(), 0);
        assert_eq!(v.capacity(), hint.next_power_of_two());
    }
}

#[test]
fn test_push_pop() {
    const TEST_MAX: usize = if cfg!(miri) {
        2usize.pow(5)
    } else {
        2usize.pow(16)
    };
    let mut v = SegVec::<usize, Exponential<1>>::with_capacity(0);
    for i in 0..TEST_MAX {
        v.push(i);
        assert_eq!(v.segments.len(), (v.capacity() as f64).log2() as usize + 1);
    }
    assert_eq!(v.len(), TEST_MAX);
    assert_eq!(v.capacity(), TEST_MAX);
    for i in 0..TEST_MAX {
        assert_eq!(v.pop(), Some(TEST_MAX - i - 1));
    }
}

#[test]
fn test_truncate() {
    let dc = Cell::new(0usize);
    const TEST_MAX: usize = 2usize.pow(7);
    let mut v = SegVec::<DropCount<'_, usize>, Exponential<1>>::with_capacity(TEST_MAX);
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
fn test_truncate_custom_factor() {
    let mut s = SegVec::<i32, Exponential<4>>::new();
    for i in 0..10 {
        s.push(i);
    }

    assert_eq!(s.len(), 10);
    assert_eq!(s.capacity(), 16);

    s.truncate(8);
    assert_eq!(s.len(), 8);
    assert_eq!(s.capacity(), 8);
}

#[test]
fn test_iter() {
    let mut v = SegVec::<i32, Exponential<1>>::new();
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
fn test_into_iter() {
    let dc = Cell::new(0usize);
    let mut v = SegVec::<DropCount<'_, usize>, Exponential<1>>::new();
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
fn test_from_iter() {
    let v = SegVec::<i32, Exponential<1>>::from_iter([1, 2, 3, 4, 5, 6]);
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
fn test_insert_remove() {
    let mut v = SegVec::<_, Exponential<1>>::with_capacity(0);
    v.insert(0, 1);
    assert_eq!(v.len(), 1);
    assert_eq!(v.capacity(), 1);
    v.insert(0, 2);
    assert_eq!(v.len(), 2);
    assert_eq!(v.capacity(), 2);
    v.insert(0, 3);
    v.insert(0, 4);
    v.insert(4, 5);
    v.insert(0, 6);
    v.insert(0, 7);
    v.insert(7, 8);
    assert_eq!(v.len(), 8);
    assert_eq!(v.capacity(), 8);
    assert_eq!(
        v.iter().copied().collect::<Vec<_>>(),
        vec![7, 6, 4, 3, 2, 1, 5, 8]
    );
    assert_eq!(v.remove(7), 8);
    assert_eq!(v.remove(4), 2);
    assert_eq!(v.remove(4), 1);
    assert_eq!(v.remove(4), 5);
    assert_eq!(v.len(), 4);
    assert_eq!(v.capacity(), 8);
    assert_eq!(v.iter().copied().collect::<Vec<_>>(), vec![7, 6, 4, 3]);
    assert_eq!(v.remove(1), 6);
    assert_eq!(v.remove(1), 4);
    assert_eq!(v.remove(1), 3);
    assert_eq!(v.remove(0), 7);
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 8);

    let mut rng = rand::thread_rng();
    let mut v = SegVec::<i32, Exponential<512>>::with_capacity(1024);
    for i in 0..1024 {
        v.insert(rng.gen_range(0..=i as usize), rng.gen_range(0..100));
    }
    assert_eq!(v.len(), 1024);
    assert_eq!(v.capacity(), 1024);
    for i in (0..1024).rev() {
        assert!(v.remove(rng.gen_range(0..=i as usize)) < 100);
    }
    assert_eq!(v.len(), 0);
    assert_eq!(v.capacity(), 1024);
}

#[test]
fn test_drain() {
    fn make_segvec_8() -> (
        Box<Cell<usize>>,
        SegVec<DropCount<'static, i32>, Exponential<1>>,
    ) {
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

    let (dc, mut sv) = make_segvec_8();
    let d = sv.drain(2..7);
    drop(d);
    assert_eq!(dc.get(), 5);
    assert_eq!(
        sv.into_iter().map(|i| i.1).collect::<Vec<_>>(),
        vec![1, 2, 8]
    );
    assert_eq!(dc.get(), 8);

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
fn test_slice() {
    let mut v = SegVec::<_, Exponential<1>>::with_capacity(8);
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
    let s6 = v.slice(1..1);
    // invalid:
    // v.truncate(0); // <- Slices immutably borrow the underlying SegVec
    assert_eq!(s1.iter().copied().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    assert_eq!(s2.iter().copied().collect::<Vec<_>>(), vec![5, 6, 7, 8]);
    assert_eq!(s3.iter().copied().collect::<Vec<_>>(), vec![3, 4, 5, 6]);
    assert_eq!(
        s4.iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );
    assert_eq!(s5.iter().copied().collect::<Vec<i32>>(), vec![]);
    assert_eq!(s6.iter().copied().collect::<Vec<i32>>(), vec![]);
}

// this must not compile
// #[test]
// fn test_slice_lifetime() {
//     let mut v = SegVec::<i32>::new();
//     v.push(1);
//     let s = v.slice(..);
//     // drop v while using s later
//     drop(v);
//     assert_eq!(s[0], 1);
// }

#[test]
fn test_subslice() {
    let mut v = SegVec::<_, Exponential<1>>::with_capacity(8);
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);
    v.push(8);
    let slice = v.slice(..);
    assert_eq!(
        slice.iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );

    let subslice = slice.slice(2..5);
    assert_eq!(subslice.iter().copied().collect::<Vec<_>>(), vec![3, 4, 5]);

    let subslice = slice.slice(..5);
    assert_eq!(
        subslice.iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5]
    );

    let subslice = slice.slice(2..);
    assert_eq!(
        subslice.iter().copied().collect::<Vec<_>>(),
        vec![3, 4, 5, 6, 7, 8]
    );

    let subslice = slice.slice(2..=5);
    assert_eq!(
        subslice.iter().copied().collect::<Vec<_>>(),
        vec![3, 4, 5, 6]
    );

    let subslice = slice.slice(..);
    assert_eq!(
        subslice.iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6, 7, 8]
    );
}

#[test]
fn from_slice() {
    let mut v = SegVec::<_, Exponential<1>>::with_capacity(8);
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);
    v.push(8);
    let slice = v.slice(2..5);
    assert_eq!(slice.iter().copied().collect::<Vec<_>>(), vec![3, 4, 5]);

    let v2 = SegVec::<i32, Exponential<1>>::from_iter(&slice);
    assert_eq!(v2.iter().copied().collect::<Vec<_>>(), vec![3, 4, 5]);

    let v2 = SegVec::<i32, Linear<4>>::from_iter(slice);
    assert_eq!(v2.iter().copied().collect::<Vec<_>>(), vec![3, 4, 5]);
}

#[test]
fn test_slice_mut() {
    let mut v = SegVec::<_, Exponential<1>>::with_capacity(8);
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);
    v.push(8);
    let mut s = v.slice_mut(..);
    assert_eq!(s[7], 8);
    // invalid:
    // v.push(1000); // <- SliceMuts mutably borrow the underlying SegVec
    s[0] = 100;
    s.iter_mut().for_each(|v| *v *= 2);
    s.into_iter().for_each(|v| *v *= 2);
    assert_eq!(
        v.iter().copied().collect::<Vec<_>>(),
        vec![400, 8, 12, 16, 20, 24, 28, 32]
    );
}

#[test]
fn test_slice_iter_mut() {
    let mut v = SegVec::<_, Exponential<1>>::with_capacity(8);
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);
    v.push(8);
    let mut s = v.slice_mut(..);
    s.iter_mut().for_each(|v| *v *= 2);
    s.iter_mut().for_each(|v| *v *= 2);

    assert_eq!(
        s.iter().copied().collect::<Vec<_>>(),
        vec![4, 8, 12, 16, 20, 24, 28, 32]
    );
}

#[test]
fn test_slice_mut_into_iter() {
    let mut v = SegVec::<_, Exponential<1>>::with_capacity(8);
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);
    v.push(8);
    let mut s = v.slice_mut(..);
    s[0] = 100;
    s.into_iter().for_each(|v| *v *= 2);

    assert_eq!(
        v.into_iter().collect::<Vec<_>>(),
        vec![200, 4, 6, 8, 10, 12, 14, 16]
    );
}

#[test]
fn test_slice_iter() {
    let mut v = SegVec::<i32, Exponential<1>>::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);
    assert_eq!(v.len(), 7);
    assert_eq!(v.capacity(), 8);

    let s = v.slice(..);

    assert_eq!(s.iter().size_hint(), (7, Some(7)));
    assert_eq!(
        s.iter().copied().collect::<Vec<_>>(),
        vec![1, 2, 3, 4, 5, 6, 7]
    );
    assert_eq!(
        s.iter().rev().copied().collect::<Vec<_>>(),
        vec![7, 6, 5, 4, 3, 2, 1]
    );

    let mut iter = s.iter();
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

    assert_eq!(s.len(), 7);
}

#[test]
fn test_slice_into_iter() {
    let mut v = SegVec::<i32, Exponential<1>>::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);

    let i = v.slice(..).into_iter();

    assert_eq!(i.size_hint(), (7, Some(7)));
    assert_eq!(i.copied().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_segmented_iter() {
    let mut v = SegVec::<i32, Exponential<1>>::new();
    v.push(1);
    v.push(2);
    v.push(3);
    v.push(4);
    v.push(5);
    v.push(6);
    v.push(7);
    assert_eq!(v.len(), 7);
    assert_eq!(v.capacity(), 8);

    let mut iter = v.slice(..).segmented_iter();
    assert_eq!(iter.size_hint(), (4, Some(4)));
    assert_eq!(iter.next().unwrap(), &[1]);
    assert_eq!(iter.size_hint(), (3, Some(3)));
    assert_eq!(iter.next().unwrap(), &[2]);
    assert_eq!(iter.size_hint(), (2, Some(2)));
    assert_eq!(iter.next().unwrap(), &[3, 4]);
    assert_eq!(iter.next().unwrap(), &[5, 6, 7]);
    assert_eq!(iter.size_hint(), (0, Some(0)));

    let mut iter = v.slice(..).segmented_iter();
    assert_eq!(iter.next_back().unwrap(), &[5, 6, 7]);
    assert_eq!(iter.next_back().unwrap(), &[3, 4]);
    assert_eq!(iter.next_back().unwrap(), &[2]);
    assert_eq!(iter.next_back().unwrap(), &[1]);
}

#[test]
fn test_sort() {
    let mut rng = rand::thread_rng();
    for i in 0..1000usize {
        let mut v = SegVec::<_, Exponential<1>>::with_capacity(i);
        while v.len() < v.capacity() {
            v.push(rng.gen_range(0i32..100));
        }

        // sorted descending
        v.sort_unstable();
        if i > 0 {
            for j in 0..i - 1 {
                assert!(&v[j] <= &v[j + 1], "{:?}", v);
            }
        }
    }
}

#[test]
fn test_hash() {
    let mut v1 = SegVec::<_, Exponential<1>>::with_capacity(8);
    v1.push(1);
    v1.push(2);
    let mut v2 = SegVec::<_, Exponential<1>>::with_capacity(4);
    v2.push(1);
    v2.push(2);
    let mut h1 = DefaultHasher::new();
    v1.hash(&mut h1);
    let mut h2 = DefaultHasher::new();
    v2.hash(&mut h2);
    assert_eq!(h1.finish(), h2.finish());
}

#[test]
fn test_extend() {
    let mut v = SegVec::<_, Exponential<1>>::new();
    v.extend([1, 2, 3, 4, 5]);
    assert_eq!(v.len(), 5);
    assert_eq!(v.capacity(), 8);
    assert_eq!(v.into_iter().collect::<Vec<_>>(), vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_extend_ref() {
    let mut v = SegVec::<u8, Exponential<1>>::new();
    v.extend("Hello!".as_bytes());
    assert_eq!(v.len(), 6);
    assert_eq!(v.capacity(), 8);
    assert_eq!(
        v.into_iter().collect::<Vec<_>>(),
        vec![0x48u8, 0x65, 0x6c, 0x6c, 0x6f, 0x21]
    );
}

#[test]
fn test_resize() {
    let mut v = SegVec::<_, Exponential<1>>::new();
    v.resize(8, 12);
    assert_eq!(v.len(), 8);
    assert_eq!(v.capacity(), 8);
    assert_eq!(
        v.iter().cloned().collect::<Vec<_>>(),
        vec![12, 12, 12, 12, 12, 12, 12, 12]
    );
    v.resize(4, 13);
    assert_eq!(v.len(), 4);
    assert_eq!(v.capacity(), 4);
    assert_eq!(v.iter().cloned().collect::<Vec<_>>(), vec![12, 12, 12, 12]);
    v.resize(8, 14);
    assert_eq!(v.len(), 8);
    assert_eq!(v.capacity(), 8);
    assert_eq!(
        v.iter().cloned().collect::<Vec<_>>(),
        vec![12, 12, 12, 12, 14, 14, 14, 14]
    );
}

#[test]
fn test_resize_with() {
    let counter = Cell::new(0i32);
    let mut get_value = || {
        counter.set(counter.get() + 1);
        counter.get()
    };
    let mut v = SegVec::<_, Exponential<1>>::new();
    v.resize_with(8, &mut get_value);
    assert_eq!(v.len(), 8);
    assert_eq!(v.capacity(), 8);
    assert_eq!(counter.get(), 8);
    assert_eq!(v[7], 8);
    v.resize_with(4, &mut get_value);
    assert_eq!(v.len(), 4);
    assert_eq!(v.capacity(), 4);
    assert_eq!(counter.get(), 8);
    assert_eq!(v[3], 4);
    v.resize_with(8, &mut get_value);
    assert_eq!(v.len(), 8);
    assert_eq!(v.capacity(), 8);
    assert_eq!(counter.get(), 12);
    assert_eq!(v[7], 12);
}

#[test]
#[should_panic(expected = "capacity overflow")]
fn test_stress_growth_factor_too_large() {
    let mut sv = SegVec::<u16, Exponential<{ usize::MAX }>>::new();
    sv.reserve(1);
    sv.push(1);
    assert_eq!(sv.len(), 1);
    assert_eq!(sv.capacity(), usize::MAX);
}

#[test]
#[cfg(feature = "small-vec")]
fn test_small_vec_feature() {
    let _: detail::Segments<()> = smallvec::SmallVec::new();
}

#[test]
#[cfg(not(feature = "small-vec"))]
fn test_not_small_vec_feature() {
    let _: detail::Segments<()> = Vec::new();
}

#[test]
#[cfg(feature = "thin-segments")]
fn test_thin_segments_feature() {
    let _: detail::Segment<()> = thin_vec::ThinVec::new();
}

#[test]
#[cfg(not(feature = "thin-segments"))]
fn test_not_thin_segments_feature() {
    let _: detail::Segment<()> = Vec::new();
}
