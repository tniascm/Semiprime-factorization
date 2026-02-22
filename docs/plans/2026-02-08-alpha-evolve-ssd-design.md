# Design: alpha-evolve + ssd-factoring Crates

## Overview

Two new experimental crates extending Project Crazy:

1. **alpha-evolve**: Evolutionary search for novel factoring algorithms using a composable DSL of factoring primitives and genetic programming.
2. **ssd-factoring**: Test whether trial division can be linearized via the State Space Duality theorem (Mamba-2 SSD), exploring three linearization strategies.

## Crate 1: alpha-evolve

### DSL Primitives

The vocabulary of composable operations:

- `mod_pow(base, exp, n)` -- modular exponentiation
- `gcd(a, n)` -- GCD extraction
- `random_element(n)` -- sample from [2, n-2]
- `iterate(f, x, steps)` -- apply function f repeatedly (rho-like walk)
- `accumulate_product(values, n)` -- batch GCD optimization
- `sieve(bound)` -- generate primes up to bound
- `smooth_check(x, primes)` -- test if x factors over a prime base
- `crt_combine(residues, moduli)` -- Chinese Remainder Theorem
- `subtract_gcd(a, b, n)` -- compute gcd(a-b, n)

### Program Representation

Tree of `Primitive` nodes with typed inputs/outputs. Each program takes `n: BigUint` as input and returns `Option<BigUint>` (a factor or None).

### Evolution

- Tournament selection, subtree crossover, primitive/constant mutation
- Population of ~100 programs, ~200 generations
- Fitness on a ladder of 50 semiprimes (5 each at 16, 20, 24, 28, 32, 36, 40, 44, 48, 52 bits)
- Score = sum(bits_factored * speed_bonus), 100ms timeout per attempt
- Programs that crash or loop get fitness 0

### File Structure

```
alpha-evolve/src/
  lib.rs          -- DSL types, program representation, evaluation
  primitives.rs   -- ~10 primitive operations
  evolution.rs    -- tournament selection, crossover, mutation
  fitness.rs      -- semiprime ladder, timing, scoring
  main.rs         -- run evolution, print best programs per generation
```

### Key Types

- `Primitive` -- enum of all DSL operations with typed parameters
- `Program` -- tree of Primitive nodes, compiled to closure
- `Individual` -- Program + fitness score
- `Population` -- Vec<Individual> with selection/crossover/mutation
- `FitnessResult` -- success rate, average time, max bits factored

## Crate 2: ssd-factoring

### Formulation

The SSD theorem: sequential recurrence h_t = A * h_{t-1} + B * x_t has an equivalent parallel matrix form y = M * x for structured A.

Trial division is sequential: test d_1, d_2, ... checking N mod d_t == 0. We explore three linearization strategies to cast this into SSD form.

### Strategy 1: Binary Indicator Lifting

Represent N as a high-dimensional binary vector. Divisibility by d becomes a linear projection. Trades nonlinearity for dimensionality.

### Strategy 2: NTT Domain

Number Theoretic Transform maps modular arithmetic into a domain where convolution becomes pointwise multiplication. If divisibility testing is expressible as convolution, NTT linearizes it.

### Strategy 3: CRT Decomposition

For coprime moduli m_1...m_k, represent N as (N mod m_1, ..., N mod m_k). Each component is independent -- already linear. Question: can we choose moduli that make "find a zero component" parallelizable via SSD?

### File Structure

```
ssd-factoring/src/
  lib.rs          -- core SSD formulation, linearization trait
  binary_lift.rs  -- strategy 1
  ntt_domain.rs   -- strategy 2
  crt_parallel.rs -- strategy 3
  main.rs         -- run all 3, compare to sequential
```

### Key Types

- `SsdFormulation` -- trait: sequential() and parallel() methods
- `BinaryLift`, `NttDomain`, `CrtParallel` -- each implements SsdFormulation
- `DualityReport` -- correctness, dimensionality blowup, timing

## Testing

### alpha-evolve (6 tests)

1. Primitive correctness (mod_pow, sieve, smooth_check)
2. Hand-built known algorithms (rho, trial division) factor correctly
3. Crossover produces type-valid children
4. Mutation produces type-valid programs
5. Known-good programs score higher than random
6. Evolution improves fitness over 50 generations

### ssd-factoring (6 tests)

1. Sequential baseline correctness
2. Binary lift matches sequential for small N
3. NTT round-trip correctness
4. CRT decomposition round-trip correctness
5. Duality agreement (sequential == parallel) for each strategy
6. Dimensionality blowup measurement

## Integration

- Add both crates to workspace Cargo.toml
- Add benchmark groups: bench_alpha_evolve, bench_ssd_strategies
- Update README: 14 crates, new algorithms table entries, updated stats
- Update master synthesis: new experimental findings for AV3 and AV8
