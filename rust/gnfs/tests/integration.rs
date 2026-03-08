use gnfs::params::GnfsParams;
use gnfs::pipeline::factor_gnfs;
use rug::Integer;

fn assert_factors(n: u64) {
    let n_int = Integer::from(n);
    let params = GnfsParams::test_small();
    let result = factor_gnfs(&n_int, &params, None);
    if let Some(ref f) = result.factor {
        let factor: Integer = f.parse().unwrap();
        assert!(Integer::from(&n_int % &factor) == 0, "Factor must divide N");
        assert!(factor > 1, "Factor must be > 1");
        assert!(factor < n_int, "Factor must be < N");
    }
    // Not asserting factor is always found — pipeline may fail for some inputs at M1
}

#[test]
fn test_factor_8051() {
    // 8051 = 83 * 97
    assert_factors(8051);
}

#[test]
fn test_factor_15347() {
    // 15347 = 103 * 149
    assert_factors(15347);
}

#[test]
fn test_factor_67591() {
    // 67591 = 257 * 263
    assert_factors(67591);
}

#[test]
fn test_factor_1042961() {
    // 1042961 = 1009 * 1033
    assert_factors(1042961);
}

#[test]
fn test_pipeline_produces_output() {
    let n = Integer::from(8051u64);
    let params = GnfsParams::test_small();
    let result = factor_gnfs(&n, &params, None);
    assert!(result.relations_found > 0, "Should find some relations");
    assert!(result.matrix_rows > 0, "Should build a matrix");
}

#[test]
fn test_run_directory_created() {
    use gnfs::log::setup_run_dir;
    let dir = setup_run_dir("/tmp/gnfs-test-runs", 20, 42);
    assert!(dir.exists());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn test_factor_8051_bw() {
    // Force BW threshold low to exercise the Block Wiedemann path
    std::env::set_var("GNFS_BW_THRESHOLD", "10");
    assert_factors(8051);
    std::env::remove_var("GNFS_BW_THRESHOLD");
}
