use criterion::{black_box, criterion_group, criterion_main, Criterion};
use segvec::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    //const N: usize = 10000;
    const N: i32 = 1000000;

    let mut group = c.benchmark_group("push values");

    group.bench_function("std Vec", |b| {
        b.iter_with_large_drop(|| {
            let mut v: Vec<i32> = Vec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    });
    group.bench_function("Exponential<1>", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    });
    group.bench_function("Linear<1024>", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Linear> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    });
    group.bench_function("Exponential<16>", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Exponential> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    });
    group.bench_function("Exponential<2500>", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Exponential<2500>> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
