use criterion::{black_box, criterion_group, criterion_main, Criterion};
use segvec::SegVec;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("push 10k values with default growth factor", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32> = SegVec::with_capacity(0);
            for i in 0..10000 {
                v.push(black_box(i));
            }
        });
    })
    .bench_function("push 10k values with large growth factor", |b| {
        b.iter_with_large_drop(|| {
            let mut v: SegVec<i32> = SegVec::with_capacity_and_factor(0, 2500);
            for i in 0..10000 {
                v.push(black_box(i));
            }
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
