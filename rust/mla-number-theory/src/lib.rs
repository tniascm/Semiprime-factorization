//! Low-rank compression of number-theoretic data.
//!
//! Inspired by DeepSeek's MLA: project high-dimensional modular arithmetic
//! data into low-rank latent space and examine whether factor-related
//! structure emerges.

use factoring_core::to_rns;
use num_bigint::BigUint;
use num_traits::ToPrimitive;

use rayon::prelude::*;

/// Feature vector for a number, computed from its number-theoretic properties.
#[derive(Debug, Clone)]
pub struct NumberFeatures {
    pub n: BigUint,
    pub features: Vec<f64>,
    pub feature_names: Vec<String>,
}

/// Default primes for computing modular residues.
const FEATURE_PRIMES: &[u64] = &[
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
    97, 101, 103, 107, 109, 113, 127, 131,
];

/// Compute feature vector for a number.
pub fn compute_features(n: &BigUint) -> NumberFeatures {
    let mut features = Vec::new();
    let mut names = Vec::new();

    // Modular residues (normalized to [0, 1])
    let residues = to_rns(n, FEATURE_PRIMES);
    for (&modulus, &residue) in FEATURE_PRIMES.iter().zip(residues.iter()) {
        features.push(residue as f64 / modulus as f64);
        names.push(format!("mod_{}", modulus));
    }

    // Digit sums in various bases
    for &base in &[2u32, 3, 6, 10, 16] {
        let digits = factoring_core::to_base(n, base);
        let digit_sum: u32 = digits.iter().sum();
        let num_digits = digits.len();
        features.push(digit_sum as f64 / (num_digits as f64 * (base - 1) as f64));
        names.push(format!("digitsum_base{}", base));
    }

    // Bit length (normalized)
    let bit_len = n.bits() as f64;
    features.push(bit_len / 2048.0); // Normalize to RSA-2048 scale
    names.push("bit_length".to_string());

    // Quadratic residue pattern (is n a QR mod small primes?)
    for &p in &[3u64, 5, 7, 11, 13, 17, 19, 23] {
        let r = to_rns(n, &[p])[0];
        let is_qr = is_quadratic_residue(r, p);
        features.push(if is_qr { 1.0 } else { 0.0 });
        names.push(format!("qr_mod_{}", p));
    }

    NumberFeatures {
        n: n.clone(),
        features,
        feature_names: names,
    }
}

/// Check if a is a quadratic residue mod p (Euler's criterion).
fn is_quadratic_residue(a: u64, p: u64) -> bool {
    if a == 0 {
        return true;
    }
    let exp = (p - 1) / 2;
    mod_pow_u64(a, exp, p) == 1
}

fn mod_pow_u64(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    let mut result = 1u64;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = (result as u128 * base as u128 % modulus as u128) as u64;
        }
        exp /= 2;
        base = (base as u128 * base as u128 % modulus as u128) as u64;
    }
    result
}

/// Compute feature matrix for a batch of numbers.
pub fn compute_feature_matrix(numbers: &[BigUint]) -> Vec<Vec<f64>> {
    numbers
        .par_iter()
        .map(|n| compute_features(n).features)
        .collect()
}

/// Simple PCA via power iteration (finds first k principal components).
pub fn pca_power_iteration(matrix: &[Vec<f64>], k: usize, iterations: usize) -> Vec<Vec<f64>> {
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }
    let d = matrix[0].len();

    // Center the data
    let means: Vec<f64> = (0..d)
        .map(|j| matrix.iter().map(|row| row[j]).sum::<f64>() / n as f64)
        .collect();

    let centered: Vec<Vec<f64>> = matrix
        .iter()
        .map(|row| row.iter().zip(&means).map(|(x, m)| x - m).collect())
        .collect();

    let mut components = Vec::new();
    let mut deflated = centered.clone();

    for _ in 0..k.min(d) {
        // Power iteration
        let mut v: Vec<f64> = (0..d).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();

        for _ in 0..iterations {
            // Compute A^T * A * v
            let av: Vec<f64> = deflated
                .iter()
                .map(|row| row.iter().zip(&v).map(|(x, vi)| x * vi).sum::<f64>())
                .collect();

            let atav: Vec<f64> = (0..d)
                .map(|j| {
                    deflated
                        .iter()
                        .zip(&av)
                        .map(|(row, &avi)| row[j] * avi)
                        .sum::<f64>()
                })
                .collect();

            // Normalize
            let norm: f64 = atav.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                v = atav.iter().map(|x| x / norm).collect();
            }
        }

        // Deflate
        let projections: Vec<f64> = deflated
            .iter()
            .map(|row| row.iter().zip(&v).map(|(x, vi)| x * vi).sum::<f64>())
            .collect();

        for (i, row) in deflated.iter_mut().enumerate() {
            for (j, val) in row.iter_mut().enumerate() {
                *val -= projections[i] * v[j];
            }
        }

        components.push(v);
    }

    // Project data onto principal components
    centered
        .iter()
        .map(|row| {
            components
                .iter()
                .map(|comp| row.iter().zip(comp).map(|(x, c)| x * c).sum::<f64>())
                .collect()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Linear Autoencoder
// ---------------------------------------------------------------------------

/// A simple linear autoencoder with an encoder/decoder bottleneck.
///
/// The encoder maps `input_dim` -> `latent_dim` and the decoder maps
/// `latent_dim` -> `input_dim`.  Both transformations are linear (matrix
/// multiply, no activation).  Training minimises reconstruction MSE via
/// gradient descent.
#[derive(Debug, Clone)]
pub struct LinearAutoencoder {
    pub input_dim: usize,
    pub latent_dim: usize,
    encoder_weights: Vec<Vec<f64>>, // latent_dim x input_dim
    decoder_weights: Vec<Vec<f64>>, // input_dim x latent_dim
}

impl LinearAutoencoder {
    /// Create a new autoencoder with small deterministic pseudo-random weights.
    ///
    /// Uses a simple linear congruential generator seeded with a fixed value so
    /// that results are reproducible across runs.
    pub fn new(input_dim: usize, latent_dim: usize) -> Self {
        // Simple LCG for deterministic weight initialisation
        let mut lcg_state: u64 = 42;
        let mut next_f64 = || -> f64 {
            lcg_state = lcg_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            // Map to small range [-0.1, 0.1]
            let raw = ((lcg_state >> 33) as f64) / (u32::MAX as f64); // [0, 1)
            (raw - 0.5) * 0.2
        };

        let encoder_weights = (0..latent_dim)
            .map(|_| (0..input_dim).map(|_| next_f64()).collect())
            .collect();

        let decoder_weights = (0..input_dim)
            .map(|_| (0..latent_dim).map(|_| next_f64()).collect())
            .collect();

        Self {
            input_dim,
            latent_dim,
            encoder_weights,
            decoder_weights,
        }
    }

    /// Encode: z = W_enc * x
    pub fn encode(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_dim, "input dimension mismatch");
        self.encoder_weights
            .iter()
            .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum::<f64>())
            .collect()
    }

    /// Decode: x_hat = W_dec * z
    pub fn decode(&self, latent: &[f64]) -> Vec<f64> {
        assert_eq!(latent.len(), self.latent_dim, "latent dimension mismatch");
        self.decoder_weights
            .iter()
            .map(|row| row.iter().zip(latent).map(|(w, z)| w * z).sum::<f64>())
            .collect()
    }

    /// Reconstruct: decode(encode(x))
    pub fn reconstruct(&self, input: &[f64]) -> Vec<f64> {
        self.decode(&self.encode(input))
    }

    /// Mean-squared error between input and its reconstruction.
    pub fn reconstruction_error(&self, input: &[f64]) -> f64 {
        let reconstructed = self.reconstruct(input);
        let n = input.len() as f64;
        input
            .iter()
            .zip(&reconstructed)
            .map(|(x, r)| (x - r).powi(2))
            .sum::<f64>()
            / n
    }

    /// Average reconstruction MSE across all examples.
    pub fn batch_reconstruction_error(&self, data: &[Vec<f64>]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        let total: f64 = data.iter().map(|x| self.reconstruction_error(x)).sum();
        total / data.len() as f64
    }

    /// Train the autoencoder via gradient descent on reconstruction MSE.
    ///
    /// For a linear autoencoder the loss for a single sample x is:
    ///   L = ||x - W_dec * W_enc * x||^2
    ///
    /// Let z = W_enc * x, x_hat = W_dec * z, e = x_hat - x.
    ///
    /// Gradients:
    ///   dL/dW_dec = (2/d) * e * z^T
    ///   dL/dW_enc = (2/d) * W_dec^T * e * x^T
    pub fn train(&mut self, data: &[Vec<f64>], epochs: usize, lr: f64) {
        if data.is_empty() {
            return;
        }
        let d = self.input_dim as f64;

        for _ in 0..epochs {
            // Accumulate gradients over the whole batch
            let mut grad_enc = vec![vec![0.0; self.input_dim]; self.latent_dim];
            let mut grad_dec = vec![vec![0.0; self.latent_dim]; self.input_dim];

            for x in data {
                let z = self.encode(x);
                let x_hat = self.decode(&z);

                // error = x_hat - x
                let err: Vec<f64> = x_hat.iter().zip(x).map(|(xh, xi)| xh - xi).collect();

                // dL/dW_dec[i][j] += (2/d) * err[i] * z[j]
                for i in 0..self.input_dim {
                    for j in 0..self.latent_dim {
                        grad_dec[i][j] += (2.0 / d) * err[i] * z[j];
                    }
                }

                // W_dec^T * err  (latent_dim vector)
                let wdt_err: Vec<f64> = (0..self.latent_dim)
                    .map(|j| {
                        self.decoder_weights
                            .iter()
                            .zip(&err)
                            .map(|(row, ei)| row[j] * ei)
                            .sum::<f64>()
                    })
                    .collect();

                // dL/dW_enc[j][k] += (2/d) * wdt_err[j] * x[k]
                for j in 0..self.latent_dim {
                    for k in 0..self.input_dim {
                        grad_enc[j][k] += (2.0 / d) * wdt_err[j] * x[k];
                    }
                }
            }

            let batch_size = data.len() as f64;

            // Update weights (average gradient over batch)
            for j in 0..self.latent_dim {
                for k in 0..self.input_dim {
                    self.encoder_weights[j][k] -= lr * grad_enc[j][k] / batch_size;
                }
            }
            for i in 0..self.input_dim {
                for j in 0..self.latent_dim {
                    self.decoder_weights[i][j] -= lr * grad_dec[i][j] / batch_size;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Clustering Analysis
// ---------------------------------------------------------------------------

/// Results from clustering analysis in latent space.
#[derive(Debug, Clone)]
pub struct ClusteringResult {
    pub silhouette_score: f64,
    pub avg_intra_distance: f64,
    pub avg_inter_distance: f64,
}

/// Euclidean distance between two vectors.
fn euclidean_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Train an autoencoder on the features, encode into latent space, and compute
/// clustering quality via the silhouette score.
///
/// `labels` assigns each feature vector to a cluster (e.g. factor-ratio bin).
pub fn latent_clustering_analysis(
    features: &[Vec<f64>],
    labels: &[usize],
    latent_dim: usize,
) -> ClusteringResult {
    assert_eq!(features.len(), labels.len());
    assert!(!features.is_empty());

    let input_dim = features[0].len();

    // Train autoencoder
    let mut ae = LinearAutoencoder::new(input_dim, latent_dim);
    ae.train(features, 200, 0.01);

    // Encode into latent space
    let latent: Vec<Vec<f64>> = features.iter().map(|f| ae.encode(f)).collect();

    // Determine unique labels
    let mut unique_labels: Vec<usize> = labels.to_vec();
    unique_labels.sort_unstable();
    unique_labels.dedup();

    let n = latent.len();

    let mut total_silhouette = 0.0;
    let mut total_intra = 0.0;
    let mut intra_count = 0usize;
    let mut total_inter = 0.0;
    let mut inter_count = 0usize;

    for i in 0..n {
        let my_label = labels[i];

        // a(i) = average distance to same-label points
        let same: Vec<f64> = (0..n)
            .filter(|&j| j != i && labels[j] == my_label)
            .map(|j| euclidean_distance(&latent[i], &latent[j]))
            .collect();

        let a = if same.is_empty() {
            0.0
        } else {
            same.iter().sum::<f64>() / same.len() as f64
        };

        // b(i) = min over other clusters of average distance to that cluster
        let mut b = f64::INFINITY;
        for &other_label in &unique_labels {
            if other_label == my_label {
                continue;
            }
            let dists: Vec<f64> = (0..n)
                .filter(|&j| labels[j] == other_label)
                .map(|j| euclidean_distance(&latent[i], &latent[j]))
                .collect();
            if !dists.is_empty() {
                let avg = dists.iter().sum::<f64>() / dists.len() as f64;
                if avg < b {
                    b = avg;
                }
            }
        }

        if b.is_infinite() {
            b = 0.0;
        }

        let s = if a.max(b) > 1e-10 {
            (b - a) / a.max(b)
        } else {
            0.0
        };

        total_silhouette += s;

        // accumulate intra / inter distances for reporting
        for &d in &same {
            total_intra += d;
            intra_count += 1;
        }
        for j in 0..n {
            if labels[j] != my_label {
                total_inter += euclidean_distance(&latent[i], &latent[j]);
                inter_count += 1;
            }
        }
    }

    let silhouette_score = total_silhouette / n as f64;
    let avg_intra_distance = if intra_count > 0 {
        total_intra / intra_count as f64
    } else {
        0.0
    };
    let avg_inter_distance = if inter_count > 0 {
        total_inter / inter_count as f64
    } else {
        0.0
    };

    ClusteringResult {
        silhouette_score,
        avg_intra_distance,
        avg_inter_distance,
    }
}

// ---------------------------------------------------------------------------
// Factor Ratio Binning
// ---------------------------------------------------------------------------

/// Compute `small_factor / n` for each semiprime and assign to equal-width bins.
///
/// `semiprimes[i]` and `factors[i]` must correspond.  The smaller of
/// `(factor, n/factor)` is used as the small factor.  Returns a label in
/// `[0, num_bins)` for each entry.
pub fn bin_factor_ratios(
    semiprimes: &[BigUint],
    factors: &[BigUint],
    num_bins: usize,
) -> Vec<usize> {
    assert_eq!(semiprimes.len(), factors.len());
    assert!(num_bins > 0);

    let ratios: Vec<f64> = semiprimes
        .iter()
        .zip(factors)
        .map(|(n, f)| {
            let n_f64 = n.to_f64().unwrap_or(1.0);
            if n_f64 < 1e-10 {
                return 0.0;
            }
            let small = f.min(n).to_f64().unwrap_or(0.0);
            small / n_f64
        })
        .collect();

    // Equal-width bins over [0, max_ratio]
    let max_ratio = ratios
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let max_ratio = if max_ratio < 1e-10 { 1.0 } else { max_ratio };
    let bin_width = max_ratio / num_bins as f64;

    ratios
        .iter()
        .map(|&r| {
            let bin = (r / bin_width) as usize;
            bin.min(num_bins - 1) // clamp upper edge
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Latent Distance Matrix
// ---------------------------------------------------------------------------

/// Encode all data points and compute pairwise Euclidean distances in latent
/// space.  Returns an n x n symmetric matrix with zero diagonal.
pub fn latent_distance_matrix(
    autoencoder: &LinearAutoencoder,
    data: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let latent: Vec<Vec<f64>> = data.iter().map(|x| autoencoder.encode(x)).collect();
    let n = latent.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&latent[i], &latent[j]);
            matrix[i][j] = d;
            matrix[j][i] = d;
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_features() {
        let n = BigUint::from(8051u64);
        let features = compute_features(&n);
        assert!(!features.features.is_empty());
        assert_eq!(features.features.len(), features.feature_names.len());
    }

    #[test]
    fn test_quadratic_residue() {
        assert!(is_quadratic_residue(1, 7));
        assert!(is_quadratic_residue(4, 7)); // 2^2 = 4
        assert!(!is_quadratic_residue(3, 7));
    }

    #[test]
    fn test_pca() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let projected = pca_power_iteration(&data, 2, 100);
        assert_eq!(projected.len(), 3);
        assert_eq!(projected[0].len(), 2);
    }

    #[test]
    fn test_autoencoder_trains() {
        // Create synthetic data: 20 samples, 10 features each, with structure
        let data: Vec<Vec<f64>> = (0..20)
            .map(|i| {
                let base = (i as f64) / 20.0;
                (0..10)
                    .map(|j| base + (j as f64) * 0.1 + ((i * 7 + j * 3) as f64 % 5.0) * 0.05)
                    .collect()
            })
            .collect();

        let mut ae = LinearAutoencoder::new(10, 3);
        let error_before = ae.batch_reconstruction_error(&data);

        ae.train(&data, 500, 0.01);
        let error_after = ae.batch_reconstruction_error(&data);

        assert!(
            error_after < error_before,
            "Reconstruction error should decrease after training: before={}, after={}",
            error_before,
            error_after
        );
    }

    #[test]
    fn test_clustering_separable() {
        // Create two clearly separable clusters in 6D
        let mut features = Vec::new();
        let mut labels = Vec::new();

        // Cluster 0: centered around (10, 10, 10, 10, 10, 10)
        for i in 0..15 {
            let offset = (i as f64) * 0.05;
            features.push(vec![
                10.0 + offset,
                10.0 - offset,
                10.0 + offset * 0.5,
                10.0 - offset * 0.5,
                10.0 + offset * 0.3,
                10.0,
            ]);
            labels.push(0);
        }

        // Cluster 1: centered around (0, 0, 0, 0, 0, 0)
        for i in 0..15 {
            let offset = (i as f64) * 0.05;
            features.push(vec![
                0.0 + offset,
                0.0 - offset,
                0.0 + offset * 0.5,
                0.0 - offset * 0.5,
                0.0 + offset * 0.3,
                0.0,
            ]);
            labels.push(1);
        }

        let result = latent_clustering_analysis(&features, &labels, 3);

        assert!(
            result.silhouette_score > 0.0,
            "Silhouette score should be positive for well-separated clusters: {}",
            result.silhouette_score
        );
        assert!(
            result.avg_inter_distance > result.avg_intra_distance,
            "Inter-cluster distance ({}) should exceed intra-cluster distance ({})",
            result.avg_inter_distance,
            result.avg_intra_distance
        );
    }

    #[test]
    fn test_bin_factor_ratios() {
        // 15 = 3 * 5, factor = 3, ratio = 3/15 = 0.2
        // 21 = 3 * 7, factor = 3, ratio = 3/21 ~ 0.143
        // 35 = 5 * 7, factor = 5, ratio = 5/35 ~ 0.143
        // 77 = 7 * 11, factor = 7, ratio = 7/77 ~ 0.091
        let semiprimes: Vec<BigUint> = vec![15u64, 21, 35, 77]
            .into_iter()
            .map(BigUint::from)
            .collect();
        let factors: Vec<BigUint> = vec![3u64, 3, 5, 7]
            .into_iter()
            .map(BigUint::from)
            .collect();

        let bins = bin_factor_ratios(&semiprimes, &factors, 3);
        assert_eq!(bins.len(), 4);
        // All bins should be in [0, 3)
        for &b in &bins {
            assert!(b < 3, "Bin {} should be less than num_bins (3)", b);
        }
    }

    #[test]
    fn test_latent_distance_matrix() {
        let ae = LinearAutoencoder::new(5, 2);
        let data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
            vec![5.0, 4.0, 3.0, 2.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        let dm = latent_distance_matrix(&ae, &data);

        assert_eq!(dm.len(), 3);
        assert_eq!(dm[0].len(), 3);

        // Diagonal should be zero
        for i in 0..3 {
            assert!(
                dm[i][i].abs() < 1e-10,
                "Diagonal element [{}][{}] = {} should be zero",
                i,
                i,
                dm[i][i]
            );
        }

        // Matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (dm[i][j] - dm[j][i]).abs() < 1e-10,
                    "Matrix should be symmetric: [{}][{}]={} vs [{}][{}]={}",
                    i,
                    j,
                    dm[i][j],
                    j,
                    i,
                    dm[j][i]
                );
            }
        }

        // Non-diagonal entries should be non-negative
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    dm[i][j] >= 0.0,
                    "Distances should be non-negative: [{}][{}]={}",
                    i,
                    j,
                    dm[i][j]
                );
            }
        }
    }
}
