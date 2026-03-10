/// Compute Dickman's rho function via differential-delay equation.
///
/// rho(u) gives the probability that a random integer near x is x^{1/u}-smooth.
/// For u <= 1: rho(u) = 1
/// For 1 < u <= 2: rho(u) = 1 - ln(u)
/// For u > 2: computed via numerical integration of rho'(t) = -rho(t-1)/t

use std::sync::OnceLock;

/// Step size for the precomputed table.
const H: f64 = 0.001;
/// Maximum u value in the precomputed table.
const MAX_U: f64 = 20.0;

/// Precomputed Dickman rho table: rho_table[i] = rho(i * H) for i = 0..N.
/// Built once on first access, shared across all calls.
static DICKMAN_TABLE: OnceLock<Vec<f64>> = OnceLock::new();

fn build_table() -> Vec<f64> {
    let n_total = (MAX_U / H).ceil() as usize;
    let mut table: Vec<f64> = Vec::with_capacity(n_total + 1);

    // Fill [0, 1]: rho(t) = 1
    let n1 = (1.0 / H).round() as usize;
    for _ in 0..=n1.min(n_total) {
        table.push(1.0);
    }
    if n_total <= n1 {
        return table;
    }

    // Fill (1, 2]: rho(t) = 1 - ln(t)
    let n2 = (2.0 / H).round() as usize;
    for i in (n1 + 1)..=n2.min(n_total) {
        let t = i as f64 * H;
        table.push(1.0 - t.ln());
    }
    if n_total <= n2 {
        return table;
    }

    // Interpolate rho from the table at an arbitrary point s >= 0.
    let interp = |tbl: &[f64], s: f64| -> f64 {
        if s <= 0.0 {
            return 1.0;
        }
        let idx_f = s / H;
        let idx = idx_f.floor() as usize;
        let frac = idx_f - idx as f64;
        if idx + 1 < tbl.len() {
            tbl[idx] * (1.0 - frac) + tbl[idx + 1] * frac
        } else if idx < tbl.len() {
            tbl[idx]
        } else {
            0.0
        }
    };

    // Fill (2, MAX_U] via RK4.
    for i in (n2 + 1)..=n_total {
        let t = i as f64 * H;
        let rho_cur = *table.last().unwrap();
        let t0 = t - H;

        let k1 = H * (-interp(&table, t0 - 1.0) / t0);
        let k2 = H * (-interp(&table, t0 + H / 2.0 - 1.0) / (t0 + H / 2.0));
        let k3 = k2;
        let k4 = H * (-interp(&table, t0 + H - 1.0) / (t0 + H));

        let rho_next = (rho_cur + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0).max(0.0);
        table.push(rho_next);
    }

    table
}

/// Look up rho(u) from the precomputed table with linear interpolation.
pub fn dickman_rho(u: f64) -> f64 {
    if u <= 0.0 || u <= 1.0 {
        return 1.0;
    }
    if u <= 2.0 {
        return 1.0 - u.ln();
    }

    let table = DICKMAN_TABLE.get_or_init(build_table);

    let idx_f = u / H;
    let idx = idx_f.floor() as usize;
    let frac = idx_f - idx as f64;
    if idx + 1 < table.len() {
        table[idx] * (1.0 - frac) + table[idx + 1] * frac
    } else if idx < table.len() {
        table[idx]
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dickman_rho_trivial() {
        assert!((dickman_rho(0.5) - 1.0).abs() < 1e-10);
        assert!((dickman_rho(1.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_dickman_rho_known_values() {
        // rho(2) = 1 - ln(2) ~ 0.30685
        assert!(
            (dickman_rho(2.0) - 0.30685).abs() < 0.001,
            "rho(2) = {}",
            dickman_rho(2.0)
        );
        // rho(3) ~ 0.04861
        assert!(
            (dickman_rho(3.0) - 0.04861).abs() < 0.01,
            "rho(3) = {}",
            dickman_rho(3.0)
        );
        // rho(4) ~ 0.00491
        assert!(
            (dickman_rho(4.0) - 0.00491).abs() < 0.001,
            "rho(4) = {}",
            dickman_rho(4.0)
        );
        // rho(5) ~ 0.000354
        assert!(
            (dickman_rho(5.0) - 0.000354).abs() < 0.0005,
            "rho(5) = {}",
            dickman_rho(5.0)
        );
    }

    #[test]
    fn test_dickman_rho_monotone_decreasing() {
        for i in 1..20 {
            let u = i as f64;
            assert!(
                dickman_rho(u) >= dickman_rho(u + 0.5),
                "rho should be monotone decreasing at u={}: rho({})={} vs rho({})={}",
                u,
                u,
                dickman_rho(u),
                u + 0.5,
                dickman_rho(u + 0.5)
            );
        }
    }

    #[test]
    fn test_dickman_rho_negative_input() {
        assert_eq!(dickman_rho(-1.0), 1.0);
    }

    #[test]
    fn test_dickman_rho_large_u() {
        let r8 = dickman_rho(8.0);
        assert!(
            r8 > 1e-9 && r8 < 1e-6,
            "rho(8) should be very small but positive: {}",
            r8
        );

        let r10 = dickman_rho(10.0);
        assert!(
            r10 >= 0.0 && r10 < 1e-6,
            "rho(10) should be non-negative and tiny: {}",
            r10
        );
    }
}
