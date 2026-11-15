#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include "Complex.h"
#include "Matrix.h"

using Cd = Complex<double>;
using CMatrix = Matrix<Cd>;

// ----------------------------------------------------------------------------
//  Helper functions for testing
// ----------------------------------------------------------------------------

// rows x cols matrix with random complex entries (re, im in [-1, 1])
CMatrix randomMatrix(std::size_t rows, std::size_t cols, std::mt19937& gen) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    CMatrix M(rows, cols);
    for (std::size_t i = 0; i < rows; ++i)
        for (std::size_t j = 0; j < cols; ++j) M[i][j] = Cd(dist(gen), dist(gen));
    return M;
}

// max |A_ij - B_ij|
double maxAbsDiff(const CMatrix& A, const CMatrix& B) {
    double mx = 0.0;
    for (std::size_t i = 0; i < A.rows(); ++i)
        for (std::size_t j = 0; j < A.cols(); ++j) mx = std::max(mx, abs(A[i][j] - B[i][j]));
    return mx;
}

// Compare Strassen with ordinary multiplication + measure the time
bool runTest(std::size_t m, std::size_t k, std::size_t n, std::size_t threshold,
             std::mt19937& gen) {
    using clock = std::chrono::steady_clock;
    CMatrix A = randomMatrix(m, k, gen);
    CMatrix B = randomMatrix(k, n, gen);

    auto t0 = clock::now();
    CMatrix Cnaive = A.multiplyNaive(B);
    auto t1 = clock::now();
    CMatrix Cstr = strassenMultiply(A, B, threshold);
    auto t2 = clock::now();

    double err = maxAbsDiff(Cnaive, Cstr);
    bool ok = err < 1e-8;
    std::chrono::duration<double, std::milli> tn = t1 - t0, ts = t2 - t1;

    std::cout << std::setw(4) << m << "x" << std::setw(4) << k << " * " << std::setw(4) << k
              << "x" << std::setw(4) << n << " | threshold=" << std::setw(3) << threshold
              << " | naive " << std::fixed << std::setprecision(2) << std::setw(9) << tn.count()
              << " ms | strassen " << std::setw(9) << ts.count() << " ms | max err "
              << std::scientific << std::setprecision(2) << err << " | "
              << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

int main() {
    std::mt19937 gen(42);

    // --- 1. Small demonstration example --------------------------------------
    std::cout << "=== Demo: A(3x2) * B(2x4), threshold = 1 ===\n";
    CMatrix A(3, 2), B(2, 4);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            A[i][j] = Cd(static_cast<double>(i) + 1.0, static_cast<double>(j) - 1.0);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            B[i][j] = Cd(static_cast<double>(i) - static_cast<double>(j),
                         static_cast<double>(i) + static_cast<double>(j));

    std::cout << "A =\n";
    A.print();
    std::cout << "B =\n";
    B.print();
    std::cout << "A * B (Strassen) =\n";
    strassenMultiply(A, B, 1).print();
    std::cout << "A * B (naive) =\n";
    A.multiplyNaive(B).print();

    // --- 2. Correctness check on various sizes -------------------------------
    std::cout << "\n=== Correctness check (various sizes, small threshold) ===\n";
    bool allOk = true;
    const std::size_t small[][3] = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {5, 7, 3},
                                    {8, 8, 8}, {13, 9, 11}, {17, 17, 17}, {31, 5, 40}};
    for (const auto& s : small) allOk &= runTest(s[0], s[1], s[2], 2, gen);

    // --- 3. Realistic conditions: threshold = 32 and 64 ----------------------
    std::cout << "\n=== Large matrices (threshold = 32 / 64) ===\n";
    const std::size_t big[][3] = {{100, 100, 100}, {130, 70, 90}, {256, 256, 256}, {300, 300, 300}};
    for (const auto& s : big) {
        allOk &= runTest(s[0], s[1], s[2], 32, gen);
        allOk &= runTest(s[0], s[1], s[2], 64, gen);
    }

    // --- 4. Place for the tasks of the other group members -------------------
    // TODO (member B): CMatrix Inv = inverseGaussJordan(A);
    // TODO (member C): CMatrix Inv = inverseLU(A);
    // Check: maxAbsDiff(strassenMultiply(A, Inv), CMatrix::identity(n)) ~ 0

    std::cout << "\n" << (allOk ? "All tests passed." : "There are errors!") << "\n";
    return allOk ? 0 : 1;
}