use criterion::{black_box, criterion_group, criterion_main, Criterion};
use segvec::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    //const N: i32 = 10000;
    const N: i32 = 1000000;
    c.bench_function("push values on a std Vec", |b| {
        b.iter_with_large_drop(|| {
            let mut v: Vec<i32> = Vec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    })
    .bench_function("push values with default growth factor", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    })
    .bench_function("push values with large growth factor", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Exponential<2500>> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        })
    })
    .bench_function("push values with linear growth, factor 32", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Linear<32>> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    })
    .bench_function("push values with proportional growth, factor 32", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Proportional<32>> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        })
    })
    .bench_function("push values with exponential growth, factor 32", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Exponential<32>> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    })
    .bench_function("push values with linear growth, factor 32, cached", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Linear<32>> = SegVec::with_capacity(0);
            let mut cache = v.new_cache();
            for i in 0..N {
                v.push_cached(black_box(i), &mut cache);
            }
        });
    })
    .bench_function(
        "push values with proportional growth, factor 32, cached",
        |b| {
            b.iter_with_large_drop(|| {
                let mut v: SegVec<i32, Proportional<32>> = SegVec::with_capacity(0);
                let mut cache = v.new_cache();
                for i in 0..N {
                    v.push_cached(black_box(i), &mut cache);
                }
            })
        },
    )
    .bench_function(
        "push values with exponential growth, factor 32, cached",
        |b| {
            b.iter_with_large_drop(|| {
                let mut v: SegVec<i32, Exponential<32>> = SegVec::with_capacity(0);
                let mut cache = v.new_cache();
                for i in 0..N {
                    v.push_cached(black_box(i), &mut cache);
                }
            });
        },
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
