#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

#include "Matrix.h"

// ============================================================================
//  Inverse matrix via the Gauss-Jordan method
//
//  Idea: build the augmented matrix [A | E] of size n x 2n and, using
//  elementary row operations, transform the left half into the identity
//  matrix. The right half then becomes A^-1:
//
//      [A | E]  ->  [E | A^-1]
//
//  For every column c = 0 .. n-1:
//    1) Partial pivoting: among rows c..n-1 choose the row whose element in
//       column c has the largest modulus abs(z). This keeps the method
//       numerically stable (and avoids dividing by 0).
//    2) Swap the pivot row with row c.
//    3) Divide row c by the pivot, so that M[c][c] = 1.
//    4) Subtract (M[r][c] * row c) from ALL other rows r != c, so that the
//       whole column c becomes zero except for the diagonal one.
//
//  Complexity: O(n^3) - the algorithm uses only elementary row operations
//  (no matrix-by-matrix products), so Strassen is not needed inside it.
//  Strassen (task 1) is used to VERIFY the result: A * A^-1 = E
//  (see main.cpp).
//
//  Throws std::invalid_argument for a non-square matrix and
//  std::runtime_error if the matrix is singular (no inverse exists).
// ============================================================================
template <typename T>
Matrix<T> inverseGaussJordan(const Matrix<T>& A) {
    if (!A.isSquare())
        throw std::invalid_argument("inverseGaussJordan: matrix must be square");

    const std::size_t n = A.rows();
    if (n == 0) return Matrix<T>();

    using Real = decltype(abs(A[0][0]));  // double for Complex<double>, etc.

    // Augmented matrix [A | E]
    Matrix<T> M(n, 2 * n);
    Real maxAbs = Real(0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            M[i][j] = A[i][j];
            const Real a = abs(A[i][j]);
            if (a > maxAbs) maxAbs = a;
        }
        M[i][n + i] = T(1);
    }

    if (maxAbs == Real(0))
        throw std::runtime_error("inverseGaussJordan: matrix is singular (zero matrix)");

    // Threshold below which the pivot is considered to be zero (relative to the
    // magnitude of the entries, so the check does not depend on scaling)
    const Real tol = std::numeric_limits<Real>::epsilon() * Real(n) * maxAbs;

    for (std::size_t c = 0; c < n; ++c) {
        // 1) Choose the pivot: the largest abs(z) in column c, rows c..n-1
        std::size_t piv = c;
        Real best = abs(M[c][c]);
        for (std::size_t r = c + 1; r < n; ++r) {
            const Real a = abs(M[r][c]);
            if (a > best) {
                best = a;
                piv = r;
            }
        }
        if (best <= tol)
            throw std::runtime_error("inverseGaussJordan: matrix is singular");

        // 2) Move the pivot row to position c
        if (piv != c)
            for (std::size_t j = 0; j < 2 * n; ++j) std::swap(M[c][j], M[piv][j]);

        // 3) Normalise the pivot row (elements left of column c are already 0)
        const T pivot = M[c][c];
        for (std::size_t j = c; j < 2 * n; ++j) M[c][j] /= pivot;
        M[c][c] = T(1);  // remove rounding noise on the diagonal

        // 4) Eliminate column c in all other rows
        for (std::size_t r = 0; r < n; ++r) {
            if (r == c) continue;
            const T factor = M[r][c];
            if (factor == T()) continue;
            for (std::size_t j = c; j < 2 * n; ++j) M[r][j] -= factor * M[c][j];
            M[r][c] = T();  // exactly zero
        }
    }

    // Right half of [E | A^-1]
    return M.block(0, n, n, n);
}

// Explicit instantiation (the template is declared in Matrix.h, defined here)
template Matrix<Complex<double>> inverseGaussJordan<Complex<double>>(
    const Matrix<Complex<double>>&);
template Matrix<Complex<float>> inverseGaussJordan<Complex<float>>(
    const Matrix<Complex<float>>&);
