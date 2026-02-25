//! Fincke-Pohst enumeration of short lattice vectors.
//!
//! Given an LLL-reduced basis, enumerate all lattice points within a given
//! radius of the origin. This finds additional short vectors beyond what LLL
//! directly provides, increasing the number of candidate smooth relations.

use lattice_reduction::{gram_schmidt, LatticeBasis};

/// A lattice point with its coordinates and norm.
#[derive(Debug, Clone)]
pub struct LatticePoint {
    /// Integer coordinates in the original basis.
    pub coords: Vec<i64>,
    /// The actual lattice vector (real-valued).
    pub vector: Vec<f64>,
    /// Euclidean norm of the vector.
    pub norm: f64,
}

/// Configuration for lattice enumeration.
#[derive(Debug, Clone)]
pub struct EnumerationConfig {
    /// Maximum Euclidean radius to search.
    pub radius: f64,
    /// Maximum number of vectors to return.
    pub max_vectors: usize,
    /// Skip the zero vector.
    pub skip_zero: bool,
}

impl Default for EnumerationConfig {
    fn default() -> Self {
        Self {
            radius: f64::INFINITY,
            max_vectors: 1000,
            skip_zero: true,
        }
    }
}

/// Enumerate short lattice vectors using the Fincke-Pohst algorithm.
///
/// Given an LLL-reduced basis B, finds all lattice vectors v = sum c_i * b_i
/// with ||v|| <= radius. Uses Gram-Schmidt projections to prune the search tree.
///
/// The algorithm works top-down: for each dimension d, d-1, ..., 1, it computes
/// bounds on the integer coefficient c_d such that the projected norm stays
/// within the radius. This is exponentially faster than brute-force enumeration
/// when the basis is well-reduced (which LLL guarantees).
pub fn enumerate_short_vectors(
    basis: &LatticeBasis,
    config: &EnumerationConfig,
) -> Vec<LatticePoint> {
    let n = basis.len();
    if n == 0 {
        return vec![];
    }
    let m = basis[0].len();

    // Gram-Schmidt orthogonalization
    let (ortho, mu) = gram_schmidt(basis);

    // Norms squared of Gram-Schmidt vectors
    let ortho_norms_sq: Vec<f64> = ortho
        .iter()
        .map(|row| row.iter().map(|x| x * x).sum::<f64>())
        .collect();

    let radius_sq = if config.radius.is_finite() {
        config.radius * config.radius
    } else {
        // Use the shortest basis vector norm * 3 as default radius
        let min_norm_sq: f64 = basis
            .iter()
            .map(|row| row.iter().map(|x| x * x).sum::<f64>())
            .fold(f64::MAX, f64::min);
        min_norm_sq * 9.0
    };

    let mut results = Vec::new();

    // Recursive enumeration using depth-first search with pruning
    let mut coords = vec![0i64; n];
    let mut partial_norms_sq = vec![0.0f64; n + 1]; // partial_norms_sq[i] = projected norm from level i
    let mut center = vec![0.0f64; n]; // center of search interval at each level

    enumerate_recursive(
        n - 1, // start from last dimension
        &mut coords,
        &mut partial_norms_sq,
        &mut center,
        &ortho_norms_sq,
        &mu,
        basis,
        m,
        radius_sq,
        config,
        &mut results,
    );

    // Sort by norm
    results.sort_by(|a, b| a.norm.partial_cmp(&b.norm).unwrap_or(std::cmp::Ordering::Equal));

    // Truncate to max_vectors
    results.truncate(config.max_vectors);

    results
}

/// Recursive Fincke-Pohst enumeration.
///
/// At level `level`, computes the range of c[level] such that the partial
/// norm (from dimensions level..n-1) stays within radius. For each valid
/// c[level], recurses to level-1. At level 0, evaluates the full vector.
#[allow(clippy::too_many_arguments)]
fn enumerate_recursive(
    level: usize,
    coords: &mut Vec<i64>,
    partial_norms_sq: &mut Vec<f64>,
    center: &mut Vec<f64>,
    ortho_norms_sq: &[f64],
    mu: &[Vec<f64>],
    basis: &LatticeBasis,
    m: usize,
    radius_sq: f64,
    config: &EnumerationConfig,
    results: &mut Vec<LatticePoint>,
) {
    if results.len() >= config.max_vectors {
        return;
    }

    let n = coords.len();

    // Compute center of search interval at this level
    let mut c = 0.0;
    for j in (level + 1)..n {
        c -= mu[j][level] * (coords[j] as f64);
    }
    center[level] = c;

    // Remaining squared norm budget at this level
    let remaining_sq = radius_sq - partial_norms_sq[level + 1];
    if remaining_sq < 0.0 {
        return;
    }

    let bstar_sq = ortho_norms_sq[level];
    if bstar_sq < 1e-15 {
        // Degenerate dimension, skip
        coords[level] = 0;
        if level == 0 {
            try_add_point(coords, basis, m, config, results);
        } else {
            partial_norms_sq[level] = partial_norms_sq[level + 1];
            enumerate_recursive(
                level - 1,
                coords,
                partial_norms_sq,
                center,
                ortho_norms_sq,
                mu,
                basis,
                m,
                radius_sq,
                config,
                results,
            );
        }
        return;
    }

    // Range of c[level]: |c[level] - center| <= sqrt(remaining_sq / bstar_sq)
    let half_width = (remaining_sq / bstar_sq).sqrt();
    let c_low = (c - half_width).ceil() as i64;
    let c_high = (c + half_width).floor() as i64;

    // Enumerate in zig-zag order from center for better results faster
    let c_center = c.round() as i64;
    let max_delta = (c_high - c_low + 1).max(0);

    // Process center first, then alternate +delta/-delta
    for delta in 0..=max_delta {
        if results.len() >= config.max_vectors {
            return;
        }

        // For delta=0, process once. For delta>0, process +delta then -delta.
        let iterations = if delta == 0 { 1 } else { 2 };
        for sign_idx in 0..iterations {
            let offset = if sign_idx == 0 { delta } else { -delta };
            let ci = c_center + offset;
            if ci < c_low || ci > c_high {
                continue;
            }
            if results.len() >= config.max_vectors {
                return;
            }

            coords[level] = ci;
            let diff = (ci as f64) - c;
            partial_norms_sq[level] = partial_norms_sq[level + 1] + diff * diff * bstar_sq;

            if partial_norms_sq[level] > radius_sq {
                continue;
            }

            if level == 0 {
                try_add_point(coords, basis, m, config, results);
            } else {
                enumerate_recursive(
                    level - 1,
                    coords,
                    partial_norms_sq,
                    center,
                    ortho_norms_sq,
                    mu,
                    basis,
                    m,
                    radius_sq,
                    config,
                    results,
                );
            }
        }
    }
}

/// Evaluate the full lattice vector and add to results if valid.
fn try_add_point(
    coords: &[i64],
    basis: &LatticeBasis,
    m: usize,
    config: &EnumerationConfig,
    results: &mut Vec<LatticePoint>,
) {
    // Skip zero vector
    if config.skip_zero && coords.iter().all(|&c| c == 0) {
        return;
    }

    // Compute the actual lattice vector: v = sum c_i * b_i
    let mut vector = vec![0.0f64; m];
    for (i, &ci) in coords.iter().enumerate() {
        if ci != 0 {
            let ci_f = ci as f64;
            for (j, v_j) in vector.iter_mut().enumerate() {
                *v_j += ci_f * basis[i][j];
            }
        }
    }

    let norm = vector.iter().map(|x| x * x).sum::<f64>().sqrt();

    results.push(LatticePoint {
        coords: coords.to_vec(),
        vector,
        norm,
    });
}

/// Enumerate lattice points near a target vector.
///
/// Finds lattice vectors v such that ||v - target|| <= radius.
/// Useful for searching neighborhoods of candidate smooth relations.
pub fn enumerate_near_target(
    basis: &LatticeBasis,
    target: &[f64],
    radius: f64,
    max_vectors: usize,
) -> Vec<LatticePoint> {
    // Transform: find short vectors of (basis | -target) lattice
    // Equivalent to: enumerate short vectors and offset by rounding target to nearest lattice point
    let n = basis.len();
    if n == 0 || target.len() != basis[0].len() {
        return vec![];
    }

    // Babai nearest plane to get approximate closest vector
    let (ortho, _mu) = gram_schmidt(basis);
    let m = target.len();
    let mut residual = target.to_vec();
    let mut babai_coords = vec![0i64; n];

    for i in (0..n).rev() {
        let ortho_norm_sq: f64 = ortho[i].iter().map(|x| x * x).sum();
        if ortho_norm_sq < 1e-15 {
            continue;
        }
        let dot: f64 = (0..m).map(|j| residual[j] * ortho[i][j]).sum();
        let ci = (dot / ortho_norm_sq).round() as i64;
        babai_coords[i] = ci;
        let ci_f = ci as f64;
        for j in 0..m {
            residual[j] -= ci_f * basis[i][j];
        }
    }

    // Enumerate short vectors around origin, then offset by Babai coords
    let config = EnumerationConfig {
        radius,
        max_vectors,
        skip_zero: false,
    };

    let mut raw = enumerate_short_vectors(basis, &config);

    // Offset coords by Babai approximation
    for point in &mut raw {
        for (i, c) in point.coords.iter_mut().enumerate() {
            *c += babai_coords[i];
        }
        // Recompute vector and norm
        let m = basis[0].len();
        let mut vector = vec![0.0f64; m];
        for (i, &ci) in point.coords.iter().enumerate() {
            let ci_f = ci as f64;
            for (j, v_j) in vector.iter_mut().enumerate() {
                *v_j += ci_f * basis[i][j];
            }
        }
        let diff: Vec<f64> = vector.iter().zip(target).map(|(v, t)| v - t).collect();
        let norm = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        point.vector = vector;
        point.norm = norm;
    }

    raw.sort_by(|a, b| a.norm.partial_cmp(&b.norm).unwrap_or(std::cmp::Ordering::Equal));
    raw.truncate(max_vectors);
    raw
}

#[cfg(test)]
mod tests {
    use super::*;
    use lattice_reduction::{lll_reduce, LllParams};

    #[test]
    fn test_enumerate_identity() {
        let basis: LatticeBasis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let config = EnumerationConfig {
            radius: 1.01, // just above 1.0 to catch unit vectors only
            max_vectors: 100,
            skip_zero: true,
        };

        let points = enumerate_short_vectors(&basis, &config);

        // Should find 6 vectors: +-e1, +-e2, +-e3
        assert_eq!(
            points.len(),
            6,
            "Identity basis with radius 1.01 should have 6 points, got {}",
            points.len()
        );

        for p in &points {
            assert!((p.norm - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_enumerate_with_lll() {
        let mut basis: LatticeBasis = vec![
            vec![1.0, 1.0, 1.0],
            vec![-1.0, 0.0, 2.0],
            vec![3.0, 5.0, 6.0],
        ];

        let params = LllParams::default();
        lll_reduce(&mut basis, &params);

        let config = EnumerationConfig {
            radius: 5.0,
            max_vectors: 50,
            skip_zero: true,
        };

        let points = enumerate_short_vectors(&basis, &config);

        assert!(!points.is_empty(), "Should find short vectors after LLL");

        // Verify sorted by norm
        for i in 1..points.len() {
            assert!(points[i].norm >= points[i - 1].norm - 1e-10);
        }

        // All norms should be within radius
        for p in &points {
            assert!(p.norm <= 5.0 + 1e-10);
        }
    }

    #[test]
    fn test_enumerate_near_target() {
        let basis: LatticeBasis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let target = vec![2.3, 1.7, 0.5];
        let points = enumerate_near_target(&basis, &target, 1.0, 50);

        assert!(!points.is_empty(), "Should find vectors near target");

        // The closest lattice point should be (2, 2, 0) or (2, 2, 1)
        let closest = &points[0];
        assert!(closest.norm < 1.0 + 1e-10, "Closest should be within radius");
    }

    #[test]
    fn test_max_vectors_limit() {
        let basis: LatticeBasis = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let config = EnumerationConfig {
            radius: 100.0,
            max_vectors: 5,
            skip_zero: true,
        };

        let points = enumerate_short_vectors(&basis, &config);
        assert!(points.len() <= 5, "Should respect max_vectors limit");
    }

    #[test]
    fn test_enumerate_default_radius() {
        let mut basis: LatticeBasis = vec![
            vec![2.0, 1.0],
            vec![1.0, 3.0],
        ];

        let params = LllParams::default();
        lll_reduce(&mut basis, &params);

        let config = EnumerationConfig::default();
        let points = enumerate_short_vectors(&basis, &config);

        assert!(!points.is_empty(), "Default config should find vectors");
    }
}
