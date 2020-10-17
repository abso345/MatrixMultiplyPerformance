# MatrixMultiplyPerformance
Various methods used to get matrix multiply performance to current library performance
Using AVX2 + FMA instructions
Using copy optimization on A to avoid cache misses.  Found use of prefetch was not as helpful
Did not provide source for the benchmark file to use to compare this against.  Comparison was done against dgemm
This method performed at 55% compared to dgemm.
