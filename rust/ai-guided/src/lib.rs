//! ML-guided factoring search strategies.
//!
//! Trains simple models on (semiprime → factors) pairs at small scale
//! and examines what patterns emerge.

use num_bigint::BigUint;

/// A training example: semiprime and its known factors.
#[derive(Debug, Clone)]
pub struct TrainingExample {
    pub n: BigUint,
    pub p: BigUint,
    pub q: BigUint,
    pub features: Vec<f64>,
    pub target: f64,
}

/// Simple linear model for predicting factor-related properties.
#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl LinearModel {
    pub fn new(dim: usize) -> Self {
        Self {
            weights: vec![0.0; dim],
            bias: 0.0,
        }
    }

    pub fn predict(&self, features: &[f64]) -> f64 {
        self.bias
            + self
                .weights
                .iter()
                .zip(features)
                .map(|(w, x)| w * x)
                .sum::<f64>()
    }

    /// Train using gradient descent.
    pub fn train(&mut self, examples: &[(Vec<f64>, f64)], learning_rate: f64, epochs: usize) {
        for _ in 0..epochs {
            let mut grad_w = vec![0.0; self.weights.len()];
            let mut grad_b = 0.0;

            for (features, target) in examples {
                let pred = self.predict(features);
                let error = pred - target;
                for (gw, x) in grad_w.iter_mut().zip(features) {
                    *gw += error * x;
                }
                grad_b += error;
            }

            let n = examples.len() as f64;
            for (w, gw) in self.weights.iter_mut().zip(&grad_w) {
                *w -= learning_rate * gw / n;
            }
            self.bias -= learning_rate * grad_b / n;
        }
    }

    /// Get feature importance (absolute weight values).
    pub fn feature_importance(&self) -> Vec<(usize, f64)> {
        let mut importance: Vec<(usize, f64)> = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, w)| (i, w.abs()))
            .collect();
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        importance
    }
}

// TODO: Implement attention-based model for analyzing which features
// of a number the model attends to when predicting factors.
// This would reveal what "patterns" an ML model discovers.

// ---------------------------------------------------------------------------
// KNN Factor Predictor
// ---------------------------------------------------------------------------

/// A stored training example for KNN prediction.
#[derive(Debug, Clone)]
struct KnnExample {
    features: Vec<f64>,
    factor_ratio: f64,
}

/// K-Nearest-Neighbor predictor that searches for factors by analogy.
///
/// Stores (features, factor_ratio) pairs from known factorisations and
/// predicts the likely factor ratio for an unseen semiprime by looking
/// at the `k` most similar training examples in feature space.
#[derive(Debug, Clone)]
pub struct KnnFactorPredictor {
    examples: Vec<KnnExample>,
}

impl KnnFactorPredictor {
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
        }
    }

    /// Store training examples.  Each element is (features, factor_ratio).
    pub fn train(&mut self, examples: &[(Vec<f64>, f64)]) {
        self.examples = examples
            .iter()
            .map(|(f, r)| KnnExample {
                features: f.clone(),
                factor_ratio: *r,
            })
            .collect();
    }

    /// Predict factor ratio for `query` using `k` nearest neighbours.
    ///
    /// Returns a distance-weighted average:  sum(ratio_i / d_i) / sum(1/d_i).
    /// If a neighbour has distance 0 its ratio is returned immediately.
    pub fn predict(&self, query: &[f64], k: usize) -> f64 {
        let neighbours = self.k_nearest(query, k);

        let mut weight_sum = 0.0_f64;
        let mut value_sum = 0.0_f64;

        for (dist, idx) in &neighbours {
            if *dist == 0.0 {
                return self.examples[*idx].factor_ratio;
            }
            let w = 1.0 / dist;
            weight_sum += w;
            value_sum += w * self.examples[*idx].factor_ratio;
        }

        if weight_sum == 0.0 {
            return 0.0;
        }
        value_sum / weight_sum
    }

    /// Return (min_ratio, max_ratio) among the `k` nearest neighbours.
    ///
    /// Useful for bounding the trial-division search range.
    pub fn predict_range(&self, query: &[f64], k: usize) -> (f64, f64) {
        let neighbours = self.k_nearest(query, k);

        let mut min_r = f64::MAX;
        let mut max_r = f64::MIN;

        for (_dist, idx) in &neighbours {
            let r = self.examples[*idx].factor_ratio;
            if r < min_r {
                min_r = r;
            }
            if r > max_r {
                max_r = r;
            }
        }

        (min_r, max_r)
    }

    /// Find k nearest neighbours.  Returns Vec<(distance, index)> sorted
    /// by ascending distance.
    fn k_nearest(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        let mut dists: Vec<(f64, usize)> = self
            .examples
            .iter()
            .enumerate()
            .map(|(i, ex)| (euclidean_distance(query, &ex.features), i))
            .collect();

        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists
    }
}

impl Default for KnnFactorPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Standard Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ---------------------------------------------------------------------------
// Weighted Feature Model
// ---------------------------------------------------------------------------

/// Feature-weighting model that learns per-dimension weights via gradient
/// descent so that a weighted-Euclidean KNN yields better predictions.
#[derive(Debug, Clone)]
pub struct WeightedFeatureModel {
    pub weights: Vec<f64>,
    examples: Vec<KnnExample>,
}

impl WeightedFeatureModel {
    /// Create a new model with `dim` feature dimensions, all weights = 1.
    pub fn new(dim: usize) -> Self {
        Self {
            weights: vec![1.0; dim],
            examples: Vec::new(),
        }
    }

    /// Learn weights that minimise KNN prediction error.
    ///
    /// For every training example we use leave-one-out KNN (k = min(3, n-1))
    /// and push the weights so that the prediction gets closer to the true
    /// ratio.  The gradient is a finite-difference approximation per weight.
    pub fn train_weights(
        &mut self,
        examples: &[(Vec<f64>, f64)],
        epochs: usize,
        lr: f64,
    ) {
        self.examples = examples
            .iter()
            .map(|(f, r)| KnnExample {
                features: f.clone(),
                factor_ratio: *r,
            })
            .collect();

        let n = self.examples.len();
        if n < 2 {
            return;
        }
        let k = 3.min(n - 1);
        let eps = 1e-5;

        for _ in 0..epochs {
            for d in 0..self.weights.len() {
                // Two-sided finite difference for better gradient estimate
                self.weights[d] += eps;
                let loss_plus = self.leave_one_out_loss(k);
                self.weights[d] -= 2.0 * eps;
                let loss_minus = self.leave_one_out_loss(k);
                self.weights[d] += eps; // restore

                let grad = (loss_plus - loss_minus) / (2.0 * eps);
                self.weights[d] -= lr * grad;

                // Clamp to positive values
                if self.weights[d] < 1e-8 {
                    self.weights[d] = 1e-8;
                }
            }
        }
    }

    /// Return the top-`n` features by weight (descending).
    pub fn get_top_features(&self, n: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self
            .weights
            .iter()
            .enumerate()
            .map(|(i, &w)| (i, w))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(n);
        indexed
    }

    // -- internal helpers ---------------------------------------------------

    /// Weighted Euclidean distance using current self.weights.
    fn weighted_distance(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .zip(self.weights.iter())
            .map(|((x, y), w)| w * (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Leave-one-out mean-squared-error using weighted KNN.
    fn leave_one_out_loss(&self, k: usize) -> f64 {
        let n = self.examples.len();
        let mut total_se = 0.0;

        for i in 0..n {
            // find k nearest neighbours excluding i
            let mut dists: Vec<(f64, usize)> = self
                .examples
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .map(|(j, ex)| {
                    (
                        self.weighted_distance(&self.examples[i].features, &ex.features),
                        j,
                    )
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            dists.truncate(k);

            let mut w_sum = 0.0_f64;
            let mut v_sum = 0.0_f64;
            for (d, j) in &dists {
                if *d == 0.0 {
                    w_sum = 1.0;
                    v_sum = self.examples[*j].factor_ratio;
                    break;
                }
                let w = 1.0 / d;
                w_sum += w;
                v_sum += w * self.examples[*j].factor_ratio;
            }
            let pred = if w_sum > 0.0 { v_sum / w_sum } else { 0.0 };
            let err = pred - self.examples[i].factor_ratio;
            total_se += err * err;
        }

        total_se / n as f64
    }
}

// ---------------------------------------------------------------------------
// Guided Trial Division
// ---------------------------------------------------------------------------

/// Attempt trial division in a narrow range centred on a predicted factor.
///
/// * `n`               – the composite to factorise
/// * `predicted_ratio` – estimated  smallest_factor / n  (as f64)
/// * `range_width`     – half-width of the search band  (as fraction of n)
///
/// Searches divisors from `n * (predicted_ratio - range_width)` to
/// `n * (predicted_ratio + range_width)`, returning the first factor found.
pub fn guided_trial_division(
    n: &BigUint,
    predicted_ratio: f64,
    range_width: f64,
) -> Option<BigUint> {
    use num_bigint::ToBigUint;
    use num_integer::Integer;
    use num_traits::{One, Zero};

    let n_f64 = biguint_to_f64(n);

    let center = n_f64 * predicted_ratio;
    let lo = (center - n_f64 * range_width).max(2.0) as u64;
    let hi = (center + n_f64 * range_width).max(2.0) as u64;

    if lo > hi {
        return None;
    }

    let big_lo = lo.to_biguint().unwrap();
    let big_hi = hi.to_biguint().unwrap();

    // Check 2 first if it's in range
    let two = BigUint::from(2u32);
    if two >= big_lo && two <= big_hi && n.is_even() {
        return Some(two);
    }

    // Start at the first odd number >= lo (at least 3)
    let three = BigUint::from(3u32);
    let start = if big_lo <= three {
        three
    } else if big_lo.is_even() {
        &big_lo + BigUint::one()
    } else {
        big_lo.clone()
    };

    let mut candidate = start;
    let step = BigUint::from(2u32);

    while candidate <= big_hi {
        if (n % &candidate).is_zero() {
            return Some(candidate);
        }
        candidate += &step;
    }

    None
}

/// Convert BigUint to f64 (lossy, for approximate arithmetic).
fn biguint_to_f64(n: &BigUint) -> f64 {
    use num_traits::ToPrimitive;
    n.to_f64().unwrap_or(f64::MAX)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knn_basic() {
        // Training data: semiprimes and the ratio  smaller_factor / n.
        // 15 = 3*5,  ratio = 3/15 = 0.2
        // 21 = 3*7,  ratio = 3/21 ≈ 0.143
        // 35 = 5*7,  ratio = 5/35 ≈ 0.143
        // 77 = 7*11, ratio = 7/77 ≈ 0.091
        //
        // Features: [n as f64, sqrt(n), n % 6]
        let examples: Vec<(Vec<f64>, f64)> = vec![
            (vec![15.0, 15.0_f64.sqrt(), 3.0], 3.0 / 15.0),
            (vec![21.0, 21.0_f64.sqrt(), 3.0], 3.0 / 21.0),
            (vec![35.0, 35.0_f64.sqrt(), 5.0], 5.0 / 35.0),
            (vec![77.0, 77.0_f64.sqrt(), 5.0], 7.0 / 77.0),
            (vec![33.0, 33.0_f64.sqrt(), 3.0], 3.0 / 33.0),
            (vec![55.0, 55.0_f64.sqrt(), 1.0], 5.0 / 55.0),
        ];

        let mut knn = KnnFactorPredictor::new();
        knn.train(&examples);

        // Query with features very close to 15 = 3*5 → should predict near 0.2
        let pred = knn.predict(&[15.0, 15.0_f64.sqrt(), 3.0], 3);
        assert!(
            (pred - 0.2).abs() < 0.1,
            "expected prediction near 0.2, got {pred}"
        );

        // predict_range should bracket the true ratio
        let (lo, hi) = knn.predict_range(&[15.0, 15.0_f64.sqrt(), 3.0], 3);
        assert!(lo <= 0.2 + 0.01, "lo={lo} should be <= 0.2");
        assert!(hi >= 0.2 - 0.01, "hi={hi} should be >= 0.2");
    }

    #[test]
    fn test_weighted_features() {
        // Construct data where feature 0 is highly informative (linearly
        // correlated with ratio) while feature 1 is constant noise.
        let mut examples: Vec<(Vec<f64>, f64)> = Vec::new();
        for i in 1..=30 {
            let ratio = 0.03 * i as f64; // 0.03, 0.06, … , 0.90
            let informative = ratio * 10.0; // feature 0: perfectly correlated
            let noise = 5.0; // feature 1: constant → carries no information
            examples.push((vec![informative, noise], ratio));
        }

        let mut model = WeightedFeatureModel::new(2);
        model.train_weights(&examples, 500, 0.05);

        let top = model.get_top_features(2);
        // Feature 0 should have a higher weight than feature 1 because it
        // is the informative one.
        let w0 = model.weights[0];
        let w1 = model.weights[1];
        assert!(
            w0 > w1,
            "informative feature weight ({w0}) should exceed noise feature weight ({w1})"
        );
        // The top-1 feature should be index 0
        assert_eq!(top[0].0, 0, "top feature should be index 0, got {}", top[0].0);
    }

    #[test]
    fn test_guided_trial_division() {
        use num_bigint::ToBigUint;

        // 437 = 19 * 23.  ratio = 19/437 ≈ 0.04348
        let n = 437u64.to_biguint().unwrap();
        let true_ratio = 19.0 / 437.0;

        // Give a reasonably accurate predicted ratio and a tight range
        let result = guided_trial_division(&n, true_ratio, 0.02);
        assert!(result.is_some(), "should find a factor of 437");
        let factor = result.unwrap();
        assert!(
            factor == 19u64.to_biguint().unwrap() || factor == 23u64.to_biguint().unwrap(),
            "expected 19 or 23, got {factor}"
        );

        // Also test a larger semiprime: 10007 * 10009 = 100_160_063
        let n2 = 100_160_063u64.to_biguint().unwrap();
        let ratio2 = 10007.0 / 100_160_063.0;
        let result2 = guided_trial_division(&n2, ratio2, 0.0001);
        assert!(result2.is_some(), "should find a factor of 100160063");
        let f2 = result2.unwrap();
        assert!(
            f2 == 10007u64.to_biguint().unwrap() || f2 == 10009u64.to_biguint().unwrap(),
            "expected 10007 or 10009, got {f2}"
        );
    }
}
