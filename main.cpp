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

// ----------------------------------------------------------------------------
//  Tests for the inverse matrix (Gauss-Jordan)
// ----------------------------------------------------------------------------

// Random n x n matrix -> inverse -> check A * A^-1 = E with Strassen; prints time
bool runInverseTest(std::size_t n, std::size_t threshold, std::mt19937& gen) {
    using clock = std::chrono::steady_clock;
    CMatrix A = randomMatrix(n, n, gen);

    auto t0 = clock::now();
    CMatrix Inv = inverseGaussJordan(A);
    auto t1 = clock::now();

    CMatrix P = strassenMultiply(A, Inv, threshold);
    CMatrix P2 = strassenMultiply(Inv, A, threshold);
    CMatrix E = CMatrix::identity(n);
    double err = std::max(maxAbsDiff(P, E), maxAbsDiff(P2, E));
    // random matrices can be ill-conditioned, so the tolerance grows with n
    bool ok = err < 1e-8 * static_cast<double>(n > 10 ? n * n : 10);
    std::chrono::duration<double, std::milli> tg = t1 - t0;

    std::cout << "n = " << std::setw(4) << n << " | threshold=" << std::setw(3) << threshold
        << " | Gauss-Jordan " << std::fixed << std::setprecision(2) << std::setw(9)
        << tg.count() << " ms | max |A*A^-1 - E| " << std::scientific
        << std::setprecision(2) << err << " | " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

// Permutation matrix has zeros on the diagonal -> requires row swaps
bool testPermutationInverse() {
    CMatrix P(3, 3);
    P[0][1] = Cd(1);
    P[1][2] = Cd(1);
    P[2][0] = Cd(1);
    CMatrix Inv = inverseGaussJordan(P);
    // The inverse of a permutation matrix is its transpose
    CMatrix T(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) T[i][j] = P[j][i];
    bool ok = maxAbsDiff(Inv, T) < 1e-12;
    std::cout << "permutation matrix (zero diagonal)      : " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

bool testSingularMatrix() {
    CMatrix S(3, 3);  // second row = 2 * first row
    for (std::size_t j = 0; j < 3; ++j) {
        S[0][j] = Cd(static_cast<double>(j) + 1, 1);
        S[1][j] = S[0][j] * Cd(2);
        S[2][j] = Cd(static_cast<double>(j) * j, -1);
    }
    bool thrown = false;
    try {
        inverseGaussJordan(S);
    }
    catch (const std::runtime_error&) {
        thrown = true;
    }
    std::cout << "singular matrix -> exception            : " << (thrown ? "OK" : "FAIL") << "\n";
    return thrown;
}

bool testNonSquareMatrix() {
    bool thrown = false;
    try {
        inverseGaussJordan(CMatrix(2, 3));
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }
    std::cout << "non-square matrix -> exception          : " << (thrown ? "OK" : "FAIL") << "\n";
    return thrown;
}

bool testFloatInverse(std::mt19937& gen) {
    using Cf = Complex<float>;
    const std::size_t n = 20;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Matrix<Cf> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A[i][j] = Cf(dist(gen), dist(gen));
    Matrix<Cf> Inv = inverseGaussJordan(A);
    Matrix<Cf> P = strassenMultiply(A, Inv, 4);
    float err = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            err = std::max(err, abs(P[i][j] - (i == j ? Cf(1) : Cf(0))));
    bool ok = err < 1e-2f;
    std::cout << "Complex<float>, n = 20, max err " << std::scientific << std::setprecision(2)
        << err << "     : " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

// ----------------------------------------------------------------------------
//  Tests for the inverse matrix (LU decomposition)
// ----------------------------------------------------------------------------

bool runLUInverseTest(std::size_t n, std::size_t threshold, std::mt19937& gen) {
    using clock = std::chrono::steady_clock;
    CMatrix A = randomMatrix(n, n, gen);

    auto t0 = clock::now();
    CMatrix Inv = inverseLU(A);
    auto t1 = clock::now();

    CMatrix P = strassenMultiply(A, Inv, threshold);
    CMatrix P2 = strassenMultiply(Inv, A, threshold);
    CMatrix E = CMatrix::identity(n);
    double err = std::max(maxAbsDiff(P, E), maxAbsDiff(P2, E));
    bool ok = err < 1e-8 * static_cast<double>(n > 10 ? n * n : 10);
    std::chrono::duration<double, std::milli> tlu = t1 - t0;

    std::cout << "n = " << std::setw(4) << n << " | threshold=" << std::setw(3) << threshold
        << " | LU inverse   " << std::fixed << std::setprecision(2) << std::setw(9)
        << tlu.count() << " ms | max |A*A^-1 - E| " << std::scientific
        << std::setprecision(2) << err << " | " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

bool testPermutationInverseLU() {
    CMatrix P(3, 3);
    P[0][1] = Cd(1);
    P[1][2] = Cd(1);
    P[2][0] = Cd(1);
    CMatrix Inv = inverseLU(P);
    CMatrix T(3, 3);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) T[i][j] = P[j][i];
    bool ok = maxAbsDiff(Inv, T) < 1e-12;
    std::cout << "permutation matrix (zero diagonal)      : " << (ok ? "OK" : "FAIL") << "\n";
    return ok;
}

bool testSingularMatrixLU() {
    CMatrix S(3, 3);
    for (std::size_t j = 0; j < 3; ++j) {
        S[0][j] = Cd(static_cast<double>(j) + 1, 1);
        S[1][j] = S[0][j] * Cd(2);
        S[2][j] = Cd(static_cast<double>(j) * j, -1);
    }
    bool thrown = false;
    try {
        inverseLU(S);
    }
    catch (const std::runtime_error&) {
        thrown = true;
    }
    std::cout << "singular matrix -> exception            : " << (thrown ? "OK" : "FAIL") << "\n";
    return thrown;
}

bool testNonSquareMatrixLU() {
    bool thrown = false;
    try {
        inverseLU(CMatrix(2, 3));
    }
    catch (const std::invalid_argument&) {
        thrown = true;
    }
    std::cout << "non-square matrix -> exception          : " << (thrown ? "OK" : "FAIL") << "\n";
    return thrown;
}

bool testFloatInverseLU(std::mt19937& gen) {
    using Cf = Complex<float>;
    const std::size_t n = 20;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    Matrix<Cf> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A[i][j] = Cf(dist(gen), dist(gen));
    Matrix<Cf> Inv = inverseLU(A);
    Matrix<Cf> P = strassenMultiply(A, Inv, 4);
    float err = 0.0f;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            err = std::max(err, abs(P[i][j] - (i == j ? Cf(1) : Cf(0))));
    bool ok = err < 1e-2f;
    std::cout << "Complex<float>, n = 20, max err " << std::scientific << std::setprecision(2)
        << err << "     : " << (ok ? "OK" : "FAIL") << "\n";
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
    const std::size_t small[][3] = { {1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {5, 7, 3},
                                    {8, 8, 8}, {13, 9, 11}, {17, 17, 17}, {31, 5, 40} };
    for (const auto& s : small) allOk &= runTest(s[0], s[1], s[2], 2, gen);

    // --- 3. Realistic conditions: threshold = 32 and 64 ----------------------
    std::cout << "\n=== Large matrices (threshold = 32 / 64) ===\n";
    const std::size_t big[][3] = { {100, 100, 100}, {130, 70, 90}, {256, 256, 256}, {300, 300, 300} };
    for (const auto& s : big) {
        allOk &= runTest(s[0], s[1], s[2], 32, gen);
        allOk &= runTest(s[0], s[1], s[2], 64, gen);
    }

    // --- 4. Task 2: inverse matrix (Gauss-Jordan) ----------------------------
    std::cout << "\n=== Gauss-Jordan inverse: demo 3x3 ===\n";
    {
        CMatrix D(3, 3);
        D[0][0] = Cd(2, 1);  D[0][1] = Cd(1, 0);  D[0][2] = Cd(0, -1);
        D[1][0] = Cd(0, 1);  D[1][1] = Cd(3, 0);  D[1][2] = Cd(1, 1);
        D[2][0] = Cd(1, 0);  D[2][1] = Cd(-1, 2); D[2][2] = Cd(4, 0);
        std::cout << "A =\n";
        D.print();
        CMatrix Dinv = inverseGaussJordan(D);
        std::cout << "A^-1 =\n";
        Dinv.print();
        std::cout << "A * A^-1 (Strassen) =\n";
        strassenMultiply(D, Dinv, 1).print();
    }

    std::cout << "\n=== Gauss-Jordan inverse: check ||A * A^-1 - E|| (Strassen) ===\n";
    const std::size_t invSizes[] = { 1, 2, 3, 5, 10, 33, 64, 100, 200, 300 };
    for (std::size_t n : invSizes) allOk &= runInverseTest(n, 32, gen);
    allOk &= runInverseTest(200, 64, gen);

    std::cout << "\n=== Gauss-Jordan inverse: special cases ===\n";
    allOk &= testPermutationInverse();   // pivoting: zero on the diagonal
    allOk &= testSingularMatrix();       // singular -> exception
    allOk &= testNonSquareMatrix();      // non-square -> exception
    allOk &= testFloatInverse(gen);      // Complex<float>

    // --- 5. Task 3: inverse matrix (LU decomposition) ------------------------
    std::cout << "\n=== LU inverse: demo 3x3 ===\n";
    {
        CMatrix D(3, 3);
        D[0][0] = Cd(2, 1);  D[0][1] = Cd(1, 0);  D[0][2] = Cd(0, -1);
        D[1][0] = Cd(0, 1);  D[1][1] = Cd(3, 0);  D[1][2] = Cd(1, 1);
        D[2][0] = Cd(1, 0);  D[2][1] = Cd(-1, 2); D[2][2] = Cd(4, 0);
        std::cout << "A =\n";
        D.print();
        CMatrix Dinv = inverseLU(D);
        std::cout << "A^-1 (LU) =\n";
        Dinv.print();
        std::cout << "A * A^-1 (Strassen) =\n";
        strassenMultiply(D, Dinv, 1).print();
    }

    std::cout << "\n=== LU inverse: check ||A * A^-1 - E|| (Strassen) ===\n";
    for (std::size_t n : invSizes) allOk &= runLUInverseTest(n, 32, gen);
    allOk &= runLUInverseTest(200, 64, gen);

    std::cout << "\n=== LU inverse: special cases ===\n";
    allOk &= testPermutationInverseLU();   // pivoting: zero on the diagonal
    allOk &= testSingularMatrixLU();       // singular -> exception
    allOk &= testNonSquareMatrixLU();      // non-square -> exception
    allOk &= testFloatInverseLU(gen);      // Complex<float>

    std::cout << "\n" << (allOk ? "All tests passed." : "There are errors!") << "\n";
    return allOk ? 0 : 1;
}