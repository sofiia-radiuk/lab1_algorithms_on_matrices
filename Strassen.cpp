#include <algorithm>
#include <stdexcept>

#include "Matrix.h"

// ============================================================================
//  Strassen algorithm for matrix multiplication
//
//  For C = A * B (all matrices n x n, n even) split into n/2 x n/2 blocks:
//
//      | A11 A12 |       | B11 B12 |       | C11 C12 |
//  A = |         |,  B = |         |,  C = |         |
//      | A21 A22 |       | B21 B22 |       | C21 C22 |
//
//  and compute 7 products (instead of 8):
//
//      M1 = (A11 + A22) * (B11 + B22)
//      M2 = (A21 + A22) * B11
//      M3 = A11 * (B12 - B22)
//      M4 = A22 * (B21 - B11)
//      M5 = (A11 + A12) * B22
//      M6 = (A21 - A11) * (B11 + B12)
//      M7 = (A12 - A22) * (B21 + B22)
//
//      C11 = M1 + M4 - M5 + M7
//      C12 = M3 + M5
//      C21 = M2 + M4
//      C22 = M1 - M2 + M3 + M6
//
//  Complexity: T(n) = 7 T(n/2) + O(n^2)  =>  O(n^log2(7)) ~ O(n^2.807)
// ============================================================================

namespace {

// Recursive part: A and B are square matrices of the same size n.
// Guarantee (see strassenMultiply): if n > threshold, then n is even.
template <typename T>
Matrix<T> strassenRec(const Matrix<T>& A, const Matrix<T>& B, std::size_t threshold) {
    const std::size_t n = A.rows();

    // Base case: small matrix -> ordinary multiplication
    if (n <= threshold || n % 2 != 0) return A.multiplyNaive(B);

    const std::size_t h = n / 2;

    // Split into blocks
    Matrix<T> A11 = A.block(0, 0, h, h), A12 = A.block(0, h, h, h);
    Matrix<T> A21 = A.block(h, 0, h, h), A22 = A.block(h, h, h, h);
    Matrix<T> B11 = B.block(0, 0, h, h), B12 = B.block(0, h, h, h);
    Matrix<T> B21 = B.block(h, 0, h, h), B22 = B.block(h, h, h, h);

    // 7 recursive products
    Matrix<T> M1 = strassenRec<T>(A11 + A22, B11 + B22, threshold);
    Matrix<T> M2 = strassenRec<T>(A21 + A22, B11, threshold);
    Matrix<T> M3 = strassenRec<T>(A11, B12 - B22, threshold);
    Matrix<T> M4 = strassenRec<T>(A22, B21 - B11, threshold);
    Matrix<T> M5 = strassenRec<T>(A11 + A12, B22, threshold);
    Matrix<T> M6 = strassenRec<T>(A21 - A11, B11 + B12, threshold);
    Matrix<T> M7 = strassenRec<T>(A12 - A22, B21 + B22, threshold);

    // Assemble the blocks of the result
    Matrix<T> C(n, n);
    C.setBlock(0, 0, M1 + M4 - M5 + M7);  // C11
    C.setBlock(0, h, M3 + M5);            // C12
    C.setBlock(h, 0, M2 + M4);            // C21
    C.setBlock(h, h, M1 - M2 + M3 + M6);  // C22
    return C;
}

}  // namespace

// ============================================================================
//  Public function: multiplication of matrices of any dimensions.
//
//  Dynamic zero padding:
//    1) s = max(m, k, n);  p = 0
//    2) while s > threshold:  s = ceil(s / 2),  p = p + 1
//    3) N = s * 2^p  (>= max(m, k, n))  - the size we pad up to
//  This way N can be halved exactly p times down to a size s <= threshold,
//  and there are far fewer extra zeros than when padding up to the nearest
//  power of two. After the multiplication the result is cropped to m x n.
// ============================================================================
template <typename T>
Matrix<T> strassenMultiply(const Matrix<T>& A, const Matrix<T>& B, std::size_t threshold) {
    if (A.cols() != B.rows())
        throw std::invalid_argument("strassenMultiply: A.cols() != B.rows()");
    if (threshold < 1) threshold = 1;

    const std::size_t m = A.rows(), k = A.cols(), n = B.cols();
    if (m == 0 || k == 0 || n == 0) return Matrix<T>(m, n);

    std::size_t s = std::max({m, k, n});
    std::size_t p = 0;
    while (s > threshold) {
        s = (s + 1) / 2;  // ceil(s / 2)
        ++p;
    }
    const std::size_t N = s << p;  // s * 2^p

    Matrix<T> Ap = A.padded(N, N);  // zero padding
    Matrix<T> Bp = B.padded(N, N);
    Matrix<T> Cp = strassenRec<T>(Ap, Bp, threshold);
    return Cp.block(0, 0, m, n);  // crop to m x n
}

// Explicit instantiation (the template is declared in Matrix.h, defined here)
template Matrix<Complex<double>> strassenMultiply<Complex<double>>(
    const Matrix<Complex<double>>&, const Matrix<Complex<double>>&, std::size_t);
template Matrix<Complex<float>> strassenMultiply<Complex<float>>(
    const Matrix<Complex<float>>&, const Matrix<Complex<float>>&, std::size_t);