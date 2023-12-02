use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use segvec::*;

#[inline(always)]
fn fast_prng(state: &mut u32) -> usize {
    let rand = *state;
    *state = rand << 1 ^ ((rand >> 30) & 1) ^ ((rand >> 2) & 1);
    rand as usize
}

pub fn criterion_benchmark(c: &mut Criterion) {
    //const N: i32 = 10000;
    const N: usize = 10000;

    let mut group = c.benchmark_group("iterator & slice");
    group.bench_function("full Vec iteration", |b| {
        let mut v: Vec<usize> = Vec::new();
        let mut r = 0xf00ba;
        for _ in 0..10000 {
            v.push(fast_prng(&mut r));
        }
        b.iter(|| {
            let mut iterator = v.iter();
            while black_box(iterator.next().is_some()) {}
        });
    });

    group.bench_function("full segvec iteration", |b| {
        let mut v: SegVec<usize> = SegVec::new();
        let mut r = 0xf00ba;
        for _ in 0..10000 {
            v.push(fast_prng(&mut r));
        }

        b.iter(|| {
            let mut iterator = v.iter();
            while black_box(iterator.next().is_some()) {}
        });
    });

    group.bench_function("full slice iteration", |b| {
        let mut v: SegVec<usize> = SegVec::new();
        let mut r = 0xf00ba;
        for _ in 0..10000 {
            v.push(fast_prng(&mut r));
        }

        b.iter(|| {
            let mut iterator = v.slice(..).iter();
            while black_box(iterator.next().is_some()) {}
        });
    });

    group.bench_function("slice iteration", |b| {
        let mut v: SegVec<usize> = SegVec::new();
        let mut r = 0xf00ba;
        for _ in 0..10000 {
            v.push(fast_prng(&mut r));
        }

        let slice = v.slice(100..9000);
        b.iter(|| {
            let mut iterator = slice.iter();
            while black_box(iterator.next().is_some()) {}
        });
    });

    group.bench_function("slice indexing", |b| {
        let mut v: SegVec<usize> = SegVec::new();
        let mut r = 0xf00ba;
        for _ in 0..10000 {
            v.push(fast_prng(&mut r));
        }
        let mut r = 0xbaf00;

        b.iter(|| {
            for _ in 0..N {
                _ = black_box(v.get(fast_prng(&mut r) % 8900));
            }
        });
    });

    drop(group);
    let mut group = c.benchmark_group("extend");
    group.measurement_time(Duration::from_secs(20));

    const EXTEND_LEN: usize = 8192;
    group.bench_function("segvec extend", |b| {
        let mut elements = [0; EXTEND_LEN];
        let mut state = 0;
        for x in elements.iter_mut() {
            *x = fast_prng(&mut state);
        }

        b.iter_with_setup(
            || SegVec::<usize>::new(),
            |mut v| {
                v.extend(black_box(&elements));
            },
        );
    });
    group.bench_function("segvec extend from slice", |b| {
        let mut elements = [0; EXTEND_LEN];
        let mut state = 0;
        for x in elements.iter_mut() {
            *x = fast_prng(&mut state);
        }

        b.iter_with_setup(
            || SegVec::<usize>::new(),
            |mut v| {
                v.extend_from_slice(black_box(&elements));
            },
        );
    });

    group.bench_function("vec extend", |b| {
        let mut elements = [0; EXTEND_LEN];
        let mut state = 0;
        for x in elements.iter_mut() {
            *x = fast_prng(&mut state);
        }

        b.iter_with_setup(
            || Vec::<usize>::new(),
            |mut v| {
                v.extend(black_box(&elements));
            },
        );
    });
    group.bench_function("vec extend from slice", |b| {
        let mut elements = [0; EXTEND_LEN];
        let mut state = 0;
        for x in elements.iter_mut() {
            *x = fast_prng(&mut state);
        }

        b.iter_with_setup(
            || Vec::<usize>::new(),
            |mut v| {
                v.extend_from_slice(black_box(&elements));
            },
        );
    });
}

criterion_group!(slice_and_iter_bench, criterion_benchmark);
criterion_main!(slice_and_iter_bench);
