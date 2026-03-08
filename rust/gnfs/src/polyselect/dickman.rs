/// Compute Dickman's rho function via differential-delay equation.
///
/// rho(u) gives the probability that a random integer near x is x^{1/u}-smooth.
/// For u <= 1: rho(u) = 1
/// For 1 < u <= 2: rho(u) = 1 - ln(u)
/// For u > 2: computed via numerical integration of rho'(t) = -rho(t-1)/t
pub fn dickman_rho(u: f64) -> f64 {
    if u <= 0.0 {
        return 1.0;
    }
    if u <= 1.0 {
        return 1.0;
    }
    if u <= 2.0 {
        return 1.0 - u.ln();
    }

    // Numerical integration of rho'(t) = -rho(t-1)/t using RK4.
    //
    // The ODE rho'(t) = -rho(t-1)/t is a delay differential equation where
    // the right-hand side depends only on already-computed history, not on
    // the current value of rho(t). We store rho on a uniform grid starting
    // at t=0 with step h, and use linear interpolation for fractional lookups.

    let h: f64 = 0.0001; // fine step for accuracy
    let n_total = ((u) / h).ceil() as usize;

    // rho_table[i] = rho(i * h) for i = 0, 1, ..., n_total
    let mut rho_table: Vec<f64> = Vec::with_capacity(n_total + 1);

    // Fill [0, 1]: rho(t) = 1
    let n1 = (1.0 / h).round() as usize;
    for _ in 0..=n1.min(n_total) {
        rho_table.push(1.0);
    }
    if n_total <= n1 {
        return rho_table.last().copied().unwrap_or(1.0);
    }

    // Fill (1, 2]: rho(t) = 1 - ln(t)
    let n2 = (2.0 / h).round() as usize;
    for i in (n1 + 1)..=n2.min(n_total) {
        let t = i as f64 * h;
        rho_table.push(1.0 - t.ln());
    }
    if n_total <= n2 {
        return rho_table.last().copied().unwrap_or(0.0);
    }

    // Interpolate rho from the table at an arbitrary point s >= 0.
    // Uses linear interpolation between grid points.
    let interp = |table: &[f64], s: f64| -> f64 {
        if s <= 0.0 {
            return 1.0;
        }
        let idx_f = s / h;
        let idx = idx_f.floor() as usize;
        let frac = idx_f - idx as f64;
        if idx + 1 < table.len() {
            table[idx] * (1.0 - frac) + table[idx + 1] * frac
        } else if idx < table.len() {
            table[idx]
        } else {
            0.0
        }
    };

    // Fill (2, u] via RK4.
    // rho'(t) = -rho(t-1)/t
    // Since rho(t-1) is fully determined by history, the RK4 stages are:
    //   k1 = h * (-rho(t - 1) / t)
    //   k2 = h * (-rho(t + h/2 - 1) / (t + h/2))
    //   k3 = k2  (no y-dependence in the RHS)
    //   k4 = h * (-rho(t + h - 1) / (t + h))
    for i in (n2 + 1)..=n_total {
        let t = i as f64 * h;
        let rho_cur = *rho_table.last().unwrap();

        let t0 = t - h; // = (i-1)*h, the "current" t for the step from t-h to t

        let k1 = h * (-interp(&rho_table, t0 - 1.0) / t0);
        let k2 = h * (-interp(&rho_table, t0 + h / 2.0 - 1.0) / (t0 + h / 2.0));
        let k3 = k2;
        let k4 = h * (-interp(&rho_table, t0 + h - 1.0) / (t0 + h));

        let rho_next = (rho_cur + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0).max(0.0);
        rho_table.push(rho_next);
    }

    rho_table.last().copied().unwrap_or(0.0)
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
            (dickman_rho(3.0) - 0.04861).abs() < 0.001,
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
            (dickman_rho(5.0) - 0.000354).abs() < 0.0001,
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
        // rho(8) ~ 3.23e-8: very small but positive
        let r8 = dickman_rho(8.0);
        assert!(
            r8 > 1e-9 && r8 < 1e-6,
            "rho(8) should be very small but positive: {}",
            r8
        );

        // rho(10) ~ 2.77e-11: at the edge of f64 numerical integration
        // accuracy; we only check it is non-negative (clamped)
        let r10 = dickman_rho(10.0);
        assert!(
            r10 >= 0.0 && r10 < 1e-6,
            "rho(10) should be non-negative and tiny: {}",
            r10
        );
    }
}
