#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Matrix.h"

// ============================================================================
//  Inverse matrix via LU decomposition with partial pivoting (PA = LU)
//
//  Idea:
//    1) Factorise P * A = L * U in-place inside a single matrix LU of size n x n:
//       - L is unit lower-triangular (ones on the diagonal, stored below diag)
//       - U is upper-triangular (stored on and above the diagonal)
//       - P is represented by a permutation array perm[0..n-1]
//    2) For each column j = 0 .. n-1 of the identity matrix E, solve A * x = e_j,
//       which is equivalent to L * U * x = P * e_j:
//       - Forward substitution:  L * y = P * e_j
//       - Back substitution:     U * x = y
//       Write the resulting column x into column j of A^-1.
//
//  Complexity: O(n^3). Strassen (task 1) is used to VERIFY the result:
//  A * A^-1 = E (see main.cpp).
//
//  Throws std::invalid_argument for a non-square matrix and
//  std::runtime_error if the matrix is singular (no inverse exists).
// ============================================================================
template <typename T>
Matrix<T> inverseLU(const Matrix<T>& A) {
    if (!A.isSquare())
        throw std::invalid_argument("inverseLU: matrix must be square");

    const std::size_t n = A.rows();
    if (n == 0) return Matrix<T>();

    using Real = decltype(abs(A[0][0]));  // double for Complex<double>, etc.

    Matrix<T> LU(A);
    std::vector<std::size_t> perm(n);
    Real maxAbs = Real(0);

    for (std::size_t i = 0; i < n; ++i) {
        perm[i] = i;
        for (std::size_t j = 0; j < n; ++j) {
            const Real a = abs(A[i][j]);
            if (a > maxAbs) maxAbs = a;
        }
    }

    if (maxAbs == Real(0))
        throw std::runtime_error("inverseLU: matrix is singular (zero matrix)");

    const Real tol = std::numeric_limits<Real>::epsilon() * Real(n) * maxAbs;

    // --- Step 1: PA = LU decomposition --------------------------------------
    for (std::size_t k = 0; k < n; ++k) {
        // Partial pivoting: find largest abs(LU[r][k]) in rows k..n-1
        std::size_t piv = k;
        Real best = abs(LU[k][k]);
        for (std::size_t r = k + 1; r < n; ++r) {
            const Real a = abs(LU[r][k]);
            if (a > best) {
                best = a;
                piv = r;
            }
        }

        if (best <= tol)
            throw std::runtime_error("inverseLU: matrix is singular");

        // Swap rows in LU and record permutation
        if (piv != k) {
            for (std::size_t j = 0; j < n; ++j) std::swap(LU[k][j], LU[piv][j]);
            std::swap(perm[k], perm[piv]);
        }

        // Compute multipliers (column k of L) and update the trailing submatrix
        const T pivot = LU[k][k];
        for (std::size_t i = k + 1; i < n; ++i) {
            LU[i][k] /= pivot;
            const T factor = LU[i][k];
            if (factor == T()) continue;
            for (std::size_t j = k + 1; j < n; ++j) {
                LU[i][j] -= factor * LU[k][j];
            }
        }
    }

    // --- Step 2: Solve L * U * X = P * E column by column -------------------
    Matrix<T> Inv(n, n);
    std::vector<T> y(n);

    for (std::size_t col = 0; col < n; ++col) {
        // Forward substitution: solve L * y = P * e_col
        for (std::size_t i = 0; i < n; ++i) {
            T sum = (perm[i] == col) ? T(1) : T(0);
            for (std::size_t j = 0; j < i; ++j) {
                sum -= LU[i][j] * y[j];
            }
            y[i] = sum;  // L has 1s on the diagonal
        }

        // Back substitution: solve U * x = y (write directly into Inv[i][col])
        for (std::size_t idx = n; idx > 0; --idx) {
            const std::size_t i = idx - 1;
            T sum = y[i];
            for (std::size_t j = i + 1; j < n; ++j) {
                sum -= LU[i][j] * Inv[j][col];
            }
            Inv[i][col] = sum / LU[i][i];
        }
    }

    return Inv;
}

// Explicit instantiation (the template is declared in Matrix.h, defined here)
template Matrix<Complex<double>> inverseLU<Complex<double>>(
    const Matrix<Complex<double>>&);
template Matrix<Complex<float>> inverseLU<Complex<float>>(
    const Matrix<Complex<float>>&);