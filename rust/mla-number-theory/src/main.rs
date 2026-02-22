use factoring_core::generate_rsa_target;
use mla_number_theory::{compute_feature_matrix, pca_power_iteration};
use num_bigint::BigUint;

fn main() {
    println!("=== MLA-Inspired Number Theory Compression ===\n");

    let mut rng = rand::thread_rng();
    let bits = 64;
    let num_samples = 50;

    // Generate semiprimes with similar factors
    println!("Generating {} {}-bit semiprimes...", num_samples, bits);
    let targets: Vec<_> = (0..num_samples)
        .map(|_| generate_rsa_target(bits, &mut rng))
        .collect();

    let semiprimes: Vec<BigUint> = targets.iter().map(|t| t.n.clone()).collect();
    let small_factors: Vec<BigUint> = targets.iter().map(|t| t.p.clone().min(t.q.clone())).collect();

    // Compute features
    let feature_matrix = compute_feature_matrix(&semiprimes);
    println!("Feature dimension: {}", feature_matrix[0].len());

    // PCA projection to 2D
    let projected = pca_power_iteration(&feature_matrix, 2, 200);

    println!("\nPCA 2D projection (checking if similar-factor semiprimes cluster):");
    println!("{:>12} | {:>8} | {:>8} | {:>12}", "SmallFactor", "PC1", "PC2", "N");
    for (i, proj) in projected.iter().enumerate() {
        println!(
            "{:>12} | {:>8.4} | {:>8.4} | {}",
            small_factors[i], proj[0], proj[1], semiprimes[i]
        );
    }

    // Check clustering: do semiprimes with similar small factors have similar projections?
    println!("\n--- Correlation Analysis ---");
    let factor_vals: Vec<f64> = small_factors
        .iter()
        .map(|f| f.to_u64_digits().first().copied().unwrap_or(0) as f64)
        .collect();
    let pc1_vals: Vec<f64> = projected.iter().map(|p| p[0]).collect();

    let corr = pearson_correlation(&factor_vals, &pc1_vals);
    println!("Correlation(small_factor, PC1) = {:.4}", corr);
    if corr.abs() > 0.3 {
        println!("  >>> SIGNIFICANT: Factor structure visible in latent space!");
    } else {
        println!("  Low correlation — factor structure not captured by linear projection.");
    }
}

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let cov: f64 = x.iter().zip(y).map(|(xi, yi)| (xi - mean_x) * (yi - mean_y)).sum::<f64>() / n;
    let std_x = (x.iter().map(|xi| (xi - mean_x).powi(2)).sum::<f64>() / n).sqrt();
    let std_y = (y.iter().map(|yi| (yi - mean_y).powi(2)).sum::<f64>() / n).sqrt();
    if std_x * std_y > 1e-10 {
        cov / (std_x * std_y)
    } else {
        0.0
    }
}
