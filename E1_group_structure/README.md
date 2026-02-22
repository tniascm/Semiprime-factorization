# E1: Group-Theoretic Structure of (Z/NZ)*

## Hypothesis

Partial knowledge of the multiplicative group structure of (Z/NZ)* — element
orders, Carmichael function lambda(N), smooth-order elements — may reveal
factor information in poly(log N) time.

## What It Tests

1. **Element order sampling**: Sample random elements of (Z/NZ)* and compute
   their orders. The LCM of observed orders approximates lambda(N) = lcm(p-1, q-1).
2. **Pohlig-Hellman decomposition**: Factor the order bound into prime powers and
   compute order components in each subgroup, combining via CRT.
3. **Baby-step giant-step**: Solve discrete logarithms in O(sqrt(order)) time.
4. **Smooth-order factoring**: Find elements whose orders are B-smooth, then
   reconstruct phi(N) = (p-1)(q-1) from LCM of smooth orders.
5. **Chebotarev density**: Analyze the distribution of element orders for
   signatures that distinguish semiprimes from primes.

## Key Finding

Computing lambda(N) or phi(N) exactly is equivalent to factoring. The smooth-order
approach requires O(sqrt(N)) element-order computations to accumulate enough
information, matching the known Pollard rho / Pollard p-1 barrier. No poly(log N)
shortcut exists through group structure alone.

## Implementation

Rust crate: [`../rust/group-structure/`](../rust/group-structure/)

```bash
cd rust && cargo run -p group-structure
cargo test -p group-structure
```

## Complexity

- Element order (naive): O(ord) multiplications
- Pohlig-Hellman order: O(sum sqrt(p_i^e_i)) per element
- Baby-step giant-step: O(sqrt(order)) time and space
- Smooth-order factoring: O(sqrt(N)) expected element samples

All paths are sub-exponential but NOT poly(log N).
