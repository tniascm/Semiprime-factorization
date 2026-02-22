# E2: ML Feature Extraction and Latent Space Analysis

## Hypothesis

Machine learning models trained on number-theoretic features of small semiprimes
might discover latent patterns that generalize to larger semiprimes, enabling
factor prediction via feature-space similarity.

## What It Tests

Two complementary approaches:

### AI-Guided Factoring (`ai-guided`)
- **Linear regression** on (features, factor_ratio) pairs with gradient descent
- **KNN factor predictor**: distance-weighted k-nearest-neighbor prediction of
  the factor ratio p/N from feature vectors
- **Weighted feature model**: learns per-dimension weights via leave-one-out
  cross-validation to improve KNN accuracy
- **Guided trial division**: narrows search to a predicted factor range

### Low-Rank Compression (`mla-number-theory`)
- **Feature computation**: 45-dimensional vectors from modular residues (32 primes),
  digit sums (5 bases), bit length, and quadratic residue indicators (8 primes)
- **PCA via power iteration**: projects features onto principal components
- **Linear autoencoder**: encoder/decoder bottleneck learns compressed representation;
  trains via gradient descent on reconstruction MSE
- **Latent clustering**: silhouette score analysis of autoencoder embeddings
  grouped by factor-ratio bins

## Key Finding

ML models achieve low prediction error only when the training and test sets share
similar factor distributions (i.e., they memorize rather than generalize). The
feature space is dominated by modular residues mod small primes, which are CRT
observables and therefore subject to the spectral flatness barrier. No learned
representation escapes the CRT product structure.

## Implementation

Rust crates: [`../rust/ai-guided/`](../rust/ai-guided/) and
[`../rust/mla-number-theory/`](../rust/mla-number-theory/)

`ai-guided` depends on `mla-number-theory` for feature computation.

```bash
cd rust && cargo run -p ai-guided
cd rust && cargo run -p mla-number-theory
cargo test -p ai-guided -p mla-number-theory
```

## Complexity

Feature computation is poly(log N) per semiprime. However, the prediction step
requires a training set of factored semiprimes (labeled data), which presupposes
a factoring oracle. The approach is therefore circular: it cannot factor without
already having factors.
