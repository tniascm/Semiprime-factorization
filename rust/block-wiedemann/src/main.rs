use block_wiedemann::matrix::CsrMatrix;
use block_wiedemann::cpu::compute_sequence_cpu;

#[cfg(target_os = "macos")]
use block_wiedemann::gpu::GpuContext;

fn main() {
    println!("=== Block Wiedemann GF(2) SpMV Benchmark ===");

    // Generate a random sparse matrix
    let size = 10000;
    let nnz_per_row = 100;

    let mut row_ptrs = vec![0; size + 1];
    let mut col_indices = Vec::with_capacity(size * nnz_per_row);

    use rand::Rng;
    let mut rng = rand::thread_rng();

    for i in 0..size {
        row_ptrs[i] = col_indices.len() as u32;
        for _ in 0..nnz_per_row {
            col_indices.push(rng.gen_range(0..size) as u32);
        }
        // sort for proper CSR
        let start = row_ptrs[i] as usize;
        let end = col_indices.len();
        col_indices[start..end].sort_unstable();
        let len = col_indices[start..end].len(); if len > 0 { let mut last = col_indices[start]; let mut write_idx = start + 1; for i in start+1..end { if col_indices[i] != last { last = col_indices[i]; col_indices[write_idx] = last; write_idx += 1; } } col_indices.truncate(write_idx); }
    }
    row_ptrs[size] = col_indices.len() as u32;

    let matrix_a = CsrMatrix {
        rows: size,
        cols: size,
        row_ptrs,
        col_indices,
    };

    println!("Constructed {}x{} matrix with {} non-zeros", size, size, matrix_a.col_indices.len());
    let matrix_at = matrix_a.transpose();

    let initial_vector: Vec<u64> = (0..size).map(|_| rng.gen()).collect();
    let iterations = 100;

    println!("Running CPU fallback for {} iterations...", iterations);
    let start = std::time::Instant::now();
    let res_cpu = compute_sequence_cpu(&matrix_a, &matrix_at, &initial_vector, iterations);
    println!("CPU Time: {:?}", start.elapsed());

    #[cfg(target_os = "macos")]
    {
        println!("Initializing Metal GPU context...");
        if let Ok(Some(gpu)) = GpuContext::new() {
            println!("Running GPU SpMV for {} iterations...", iterations);
            let start = std::time::Instant::now();
            let res_gpu = gpu.compute_sequence(&matrix_a, &matrix_at, &initial_vector, iterations).unwrap();
            println!("GPU Time: {:?}", start.elapsed());

            // Verify
            assert_eq!(res_cpu.last().unwrap(), res_gpu.last().unwrap(), "CPU and GPU results mismatch!");
            println!("GPU results verified against CPU.");
        } else {
            println!("Metal not available on this system.");
        }
    }
}
