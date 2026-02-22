use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use factoring_core::{factor_ensemble, generate_rsa_target, pollard_rho, trial_division};

fn bench_trial_division(c: &mut Criterion) {
    let mut group = c.benchmark_group("trial_division");
    let mut rng = rand::thread_rng();

    for bits in [16, 24, 32] {
        let target = generate_rsa_target(bits, &mut rng);
        group.bench_with_input(BenchmarkId::from_parameter(bits), &target.n, |b, n| {
            b.iter(|| trial_division(n, 100_000));
        });
    }

    group.finish();
}

fn bench_pollard_rho(c: &mut Criterion) {
    let mut group = c.benchmark_group("pollard_rho");
    let mut rng = rand::thread_rng();

    for bits in [32, 48, 64] {
        let target = generate_rsa_target(bits, &mut rng);
        group.bench_with_input(BenchmarkId::from_parameter(bits), &target.n, |b, n| {
            b.iter(|| pollard_rho(n));
        });
    }

    group.finish();
}

fn bench_ecm(c: &mut Criterion) {
    let mut group = c.benchmark_group("ecm");
    group.sample_size(10);
    let mut rng = rand::thread_rng();

    for bits in [32, 48, 64] {
        let target = generate_rsa_target(bits, &mut rng);
        let params = ecm::EcmParams {
            b1: 1_000,
            b2: 50_000,
            num_curves: 16,
        };
        group.bench_with_input(BenchmarkId::from_parameter(bits), &target.n, |b, n| {
            b.iter(|| ecm::ecm_factor(n, &params));
        });
    }

    group.finish();
}

fn bench_ising(c: &mut Criterion) {
    let mut group = c.benchmark_group("ising_factoring");
    group.sample_size(10);

    for n in [15u64, 35, 77, 143, 323] {
        let bits = 64 - n.leading_zeros();
        let half_bits = (bits as usize / 2).max(2);
        let qubo = ising_factoring::encode_factoring_qubo(n, half_bits, half_bits);
        group.bench_with_input(BenchmarkId::from_parameter(n), &qubo, |b, q| {
            b.iter(|| ising_factoring::simulated_annealing(q, 5.0, 0.01, 1000));
        });
    }

    group.finish();
}

fn bench_compression_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("compression_analysis");
    let mut rng = rand::thread_rng();

    for bits in [32, 64, 128] {
        let target = generate_rsa_target(bits, &mut rng);
        group.bench_with_input(BenchmarkId::from_parameter(bits), &target.n, |b, n| {
            b.iter(|| compression_factor::analyze_compressibility(n));
        });
    }

    group.finish();
}

fn bench_multi_base(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_base_analysis");
    let mut rng = rand::thread_rng();

    for bits in [32, 64, 128] {
        let target = generate_rsa_target(bits, &mut rng);
        group.bench_with_input(BenchmarkId::from_parameter(bits), &target.n, |b, n| {
            b.iter(|| multi_base::analyze(n));
        });
    }

    group.finish();
}

fn bench_period_factor(c: &mut Criterion) {
    let mut group = c.benchmark_group("period_factor");
    let mut rng = rand::thread_rng();

    for bits in [16, 24, 32] {
        let target = generate_rsa_target(bits, &mut rng);
        group.bench_with_input(BenchmarkId::from_parameter(bits), &target.n, |b, n| {
            b.iter(|| quantum_inspired::period_factor(n, 100));
        });
    }

    group.finish();
}

fn bench_group_structure(c: &mut Criterion) {
    let mut group = c.benchmark_group("group_structure");

    for n in [15u64, 77, 143, 323, 1001] {
        let n_big = num_bigint::BigUint::from(n);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n_big, |b, n| {
            b.iter(|| group_structure::sample_orders(n, 50, 10_000));
        });
    }

    group.finish();
}

fn bench_mla_features(c: &mut Criterion) {
    let mut group = c.benchmark_group("mla_features");
    let mut rng = rand::thread_rng();

    for bits in [32, 64, 128] {
        let target = generate_rsa_target(bits, &mut rng);
        group.bench_with_input(BenchmarkId::from_parameter(bits), &target.n, |b, n| {
            b.iter(|| mla_number_theory::compute_features(n));
        });
    }

    group.finish();
}

fn bench_siqs(c: &mut Criterion) {
    let mut group = c.benchmark_group("siqs");
    group.sample_size(10);

    for (n_val, bound) in [(67591u64, 100u64), (1000003 * 1009, 200)] {
        let n = num_bigint::BigUint::from(n_val);
        group.bench_with_input(BenchmarkId::from_parameter(n_val), &n, |b, n| {
            b.iter(|| classical_nfs::siqs_factor(n, bound));
        });
    }

    group.finish();
}

fn bench_ensemble(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_factoring");
    group.sample_size(10);
    let mut rng = rand::thread_rng();

    for bits in [32, 48] {
        let target = generate_rsa_target(bits, &mut rng);
        group.bench_with_input(BenchmarkId::from_parameter(bits), &target.n, |b, n| {
            b.iter(|| factor_ensemble(n, 10_000));
        });
    }

    group.finish();
}

fn bench_alpha_evolve(c: &mut Criterion) {
    let mut group = c.benchmark_group("alpha_evolve");
    group.sample_size(10);

    // Benchmark fitness evaluation of seed programs
    let rho = alpha_evolve::seed_pollard_rho();
    group.bench_function("seed_rho_fitness", |b| {
        b.iter(|| alpha_evolve::fitness::evaluate_fitness(&rho));
    });

    group.finish();
}

fn bench_ssd_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("ssd_strategies");

    let n = 8051u64;
    let divisors: Vec<u64> = ssd_factoring::primes_up_to(90);

    group.bench_function("sequential", |b| {
        b.iter(|| ssd_factoring::trial_division_sequential(n, &divisors));
    });

    use ssd_factoring::SsdFormulation;

    group.bench_function("binary_lift", |b| {
        let strategy = ssd_factoring::binary_lift::BinaryLift;
        b.iter(|| strategy.parallel(n, &divisors));
    });

    group.bench_function("ntt_domain", |b| {
        let strategy = ssd_factoring::ntt_domain::NttDomain { modulus: 998244353 };
        b.iter(|| strategy.parallel(n, &divisors));
    });

    group.bench_function("crt_parallel", |b| {
        let strategy = ssd_factoring::crt_parallel::CrtParallel::new_default();
        b.iter(|| strategy.parallel(n, &divisors));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_trial_division,
    bench_pollard_rho,
    bench_ecm,
    bench_ising,
    bench_compression_analysis,
    bench_multi_base,
    bench_period_factor,
    bench_group_structure,
    bench_mla_features,
    bench_siqs,
    bench_ensemble,
    bench_alpha_evolve,
    bench_ssd_strategies
);
criterion_main!(benches);
