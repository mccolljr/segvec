use criterion::{black_box, criterion_group, criterion_main, Criterion};
use segvec::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    //const N: i32 = 10000;
    const N: i32 = 100000;
    //const N: i32 = 1000000;

    let mut group = c.benchmark_group("push values");

    group.bench_function("std Vec", |b| {
        b.iter_with_large_drop(|| {
            let mut v: Vec<i32> = Vec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    });

    // too slow
    // group.bench_function("Linear/factor 1", |b| {
    //     b.iter_with_large_drop(|| {
    //         let mut v: SegVec<i32, Linear<1>> = SegVec::with_capacity(0);
    //         for i in 0..N {
    //             v.push(black_box(i));
    //         }
    //     });
    // });

    group.bench_function("Proportional/factor 1", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Proportional<1>> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        })
    });

    group.bench_function("Exponential/factor 1", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Exponential<1>> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    });

    group.bench_function("Linear/factor 1024", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Linear<1024>> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    });
    group.bench_function("Proportional/factor 1024", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Proportional<1024>> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        })
    });

    group.bench_function("Exponential/factor 1024", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Exponential<1024>> = SegVec::with_capacity(0);
            for i in 0..N {
                v.push(black_box(i));
            }
        });
    });

    // too slow
    // group.bench_function("Linear/factor 1/cached", |b| {
    //     b.iter_with_large_drop(|| {
    //         let mut v: SegVec<i32, Linear<1>> = SegVec::with_capacity(0);
    //         let mut cache = v.new_cache();
    //         for i in 0..N {
    //             v.push_cached(black_box(i), &mut cache);
    //         }
    //     });
    // });

    group.bench_function(
        "Proportional/factor 1/cached",
        |b| {
            b.iter_with_large_drop(|| {
                let mut v: SegVec<i32, Proportional<1>> = SegVec::with_capacity(0);
                let mut cache = v.new_cache();
                for i in 0..N {
                    v.push_cached(black_box(i), &mut cache);
                }
            })
        },
    );

    group.bench_function(
        "Exponential/factor 1/cached",
        |b| {
            b.iter_with_large_drop(|| {
                let mut v: SegVec<i32, Exponential<1>> = SegVec::with_capacity(0);
                let mut cache = v.new_cache();
                for i in 0..N {
                    v.push_cached(black_box(i), &mut cache);
                }
            });
        },
    );

    group.bench_function("Linear/factor 1024/cached", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32, Linear<1024>> = SegVec::with_capacity(0);
            let mut cache = v.new_cache();
            for i in 0..N {
                v.push_cached(black_box(i), &mut cache);
            }
        });
    });

    group.bench_function(
        "Proportional/factor 1024/cached",
        |b| {
            b.iter_with_large_drop(|| {
                let mut v: SegVec<i32, Proportional<1024>> = SegVec::with_capacity(0);
                let mut cache = v.new_cache();
                for i in 0..N {
                    v.push_cached(black_box(i), &mut cache);
                }
            })
        },
    );

    group.bench_function(
        "Exponential/factor 1024/cached",
        |b| {
            b.iter_with_large_drop(|| {
                let mut v: SegVec<i32, Exponential<1024>> = SegVec::with_capacity(0);
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
