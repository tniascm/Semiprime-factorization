use ai_guided::LinearModel;
use factoring_core::generate_rsa_target;
use mla_number_theory::compute_features;

fn main() {
    println!("=== AI-Guided Factoring Search ===\n");

    let mut rng = rand::thread_rng();
    let bits = 32;
    let num_train = 500;
    let num_test = 100;

    println!("Training on {} {}-bit semiprimes...", num_train, bits);

    // Generate training data
    let mut train_examples = Vec::new();
    for _ in 0..num_train {
        let target = generate_rsa_target(bits, &mut rng);
        let features = compute_features(&target.n);
        let small_factor = target.p.clone().min(target.q.clone());
        let ratio = small_factor.to_u64_digits().first().copied().unwrap_or(0) as f64
            / target.n.to_u64_digits().first().copied().unwrap_or(1) as f64;
        train_examples.push((features.features, ratio));
    }

    // Train model
    let dim = train_examples[0].0.len();
    let mut model = LinearModel::new(dim);
    model.train(&train_examples, 0.01, 1000);

    // Feature importance
    let importance = model.feature_importance();
    let feature_names = compute_features(&generate_rsa_target(bits, &mut rng).n).feature_names;

    println!("\nTop 10 most important features:");
    for (i, (idx, weight)) in importance.iter().take(10).enumerate() {
        let name = if *idx < feature_names.len() {
            &feature_names[*idx]
        } else {
            "unknown"
        };
        println!("  {}. {} (weight: {:.6})", i + 1, name, weight);
    }

    // Test
    println!("\nTesting on {} new semiprimes...", num_test);
    let mut total_error = 0.0;
    for _ in 0..num_test {
        let target = generate_rsa_target(bits, &mut rng);
        let features = compute_features(&target.n);
        let predicted_ratio = model.predict(&features.features);
        let small_factor = target.p.clone().min(target.q.clone());
        let actual_ratio = small_factor.to_u64_digits().first().copied().unwrap_or(0) as f64
            / target.n.to_u64_digits().first().copied().unwrap_or(1) as f64;
        total_error += (predicted_ratio - actual_ratio).abs();
    }
    println!(
        "Average absolute error: {:.6}",
        total_error / num_test as f64
    );
}
