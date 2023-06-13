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
    const N: usize = 100000;
    //const N: i32 = 1000000;

    let mut group = c.benchmark_group("access/ordered");

    group.bench_function("std Vec", |b| {
        let mut v: Vec<usize> = Vec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        b.iter_with_large_drop(|| {
            for i in 0..N {
                _ = black_box(v.get(black_box(i)));
            }
        });
    });

    group.bench_function("SegVec/Linear<64>", |b| {
        let mut v: SegVec<usize, Linear<64>> = SegVec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        b.iter_with_large_drop(|| {
            for i in 0..N {
                _ = black_box(v.get(black_box(i)));
            }
        });
    });

    group.bench_function("SegVec/Proportional<1>", |b| {
        let mut v: SegVec<usize, Proportional<1>> = SegVec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        b.iter_with_large_drop(|| {
            for i in 0..N {
                _ = black_box(v.get(black_box(i)));
            }
        });
    });

    group.bench_function("SegVec", |b| {
        let mut v: SegVec<usize> = SegVec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        b.iter_with_large_drop(|| {
            for i in 0..N {
                _ = black_box(v.get(black_box(i)));
            }
        });
    });

    drop(group);
    let mut group = c.benchmark_group("access/random");

    group.bench_function("std Vec", |b| {
        let mut v: Vec<usize> = Vec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        let mut prng_state = 0xbabeface_u32;

        b.iter_with_large_drop(|| {
            for _ in 0..N {
                _ = black_box(v.get(fast_prng(&mut prng_state) % N));
            }
        });
    });

    group.bench_function("SegVec/Linear<64>", |b| {
        let mut v: SegVec<usize, Linear<64>> = SegVec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        let mut prng_state = 0xbabeface_u32;

        b.iter_with_large_drop(|| {
            for _ in 0..N {
                _ = black_box(v.get(fast_prng(&mut prng_state) % N));
            }
        });
    });

    group.bench_function("SegVec/Proportional<1>", |b| {
        let mut v: SegVec<usize, Proportional<1>> = SegVec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        let mut prng_state = 0xbabeface_u32;

        b.iter_with_large_drop(|| {
            for _ in 0..N {
                _ = black_box(v.get(fast_prng(&mut prng_state) % N));
            }
        });
    });

    group.bench_function("SegVec", |b| {
        let mut v: SegVec<usize> = SegVec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        let mut prng_state = 0xbabeface_u32;

        b.iter_with_large_drop(|| {
            for _ in 0..N {
                _ = black_box(v.get(fast_prng(&mut prng_state) % N));
            }
        });
    });

    drop(group);
    let mut group = c.benchmark_group("access/semirandom");

    group.bench_function("std Vec", |b| {
        let mut v: Vec<usize> = Vec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        let mut prng_state = 0xbabeface_u32;
        let mut index: usize = 0;

        b.iter_with_large_drop(|| {
            for _ in 0..N {
                index = (index + (fast_prng(&mut prng_state) % 1000))
                    .saturating_sub(500)
                    .min(N);
                _ = black_box(v.get(index));
            }
        });
    });

    group.bench_function("SegVec/Linear<64>", |b| {
        let mut v: SegVec<usize, Linear<64>> = SegVec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        let mut prng_state = 0xbabeface_u32;
        let mut index: usize = 0;

        b.iter_with_large_drop(|| {
            for _ in 0..N {
                index = index
                    .wrapping_add(fast_prng(&mut prng_state) % 1001)
                    .wrapping_sub(500)
                    % N;
                _ = black_box(v.get(index));
            }
        });
    });

    group.bench_function("SegVec/Proportional<1>", |b| {
        let mut v: SegVec<usize, Proportional<1>> = SegVec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        let mut prng_state = 0xbabeface_u32;
        let mut index: usize = 0;

        b.iter_with_large_drop(|| {
            for _ in 0..N {
                index = index
                    .wrapping_add(fast_prng(&mut prng_state) % 1001)
                    .wrapping_sub(500)
                    % N;
                _ = black_box(v.get(index));
            }
        });
    });

    group.bench_function("SegVec", |b| {
        let mut v: SegVec<usize> = SegVec::with_capacity(0);
        for i in 0..N {
            v.push(i);
        }

        let mut prng_state = 0xbabeface_u32;
        let mut index: usize = 0;

        b.iter_with_large_drop(|| {
            for _ in 0..N {
                index = index
                    .wrapping_add(fast_prng(&mut prng_state) % 1001)
                    .wrapping_sub(500)
                    % N;
                _ = black_box(v.get(index));
            }
        });
    });
}

criterion_group!(benches2, criterion_benchmark);
criterion_main!(benches2);
