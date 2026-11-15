#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "Complex.h"

// ============================================================================
//  Matrix<T> - a matrix stored in the T** style (array of pointers to rows).
//  For this lab T = Complex<double>.
//
//  All elements are initialised to zero (T()) on creation.
//  Access: A[i][j] or A(i, j).
// ============================================================================
template <typename T>
class Matrix {
private:
    std::size_t rows_;
    std::size_t cols_;
    T** data_;  // data_[i][j] - element in row i, column j

    // Memory allocation: one contiguous block + an array of row pointers
    void allocate() {
        if (rows_ == 0 || cols_ == 0) {
            data_ = nullptr;
            return;
        }
        T* block = new T[rows_ * cols_]();  // () -> value-initialisation (zeros)
        data_ = new T*[rows_];
        for (std::size_t i = 0; i < rows_; ++i) data_[i] = block + i * cols_;
    }

    void release() {
        if (data_) {
            delete[] data_[0];  // the contiguous block
            delete[] data_;     // the array of row pointers
            data_ = nullptr;
        }
    }

public:
    // --- Constructors / destructor / assignment (rule of five) --------------
    Matrix() : rows_(0), cols_(0), data_(nullptr) {}

    Matrix(std::size_t rows, std::size_t cols) : rows_(rows), cols_(cols), data_(nullptr) {
        allocate();
    }

    Matrix(const Matrix& other) : rows_(other.rows_), cols_(other.cols_), data_(nullptr) {
        allocate();
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) data_[i][j] = other.data_[i][j];
    }

    Matrix(Matrix&& other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(other.data_) {
        other.rows_ = other.cols_ = 0;
        other.data_ = nullptr;
    }

    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            Matrix tmp(other);
            swap(tmp);
        }
        return *this;
    }

    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            release();
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = other.data_;
            other.rows_ = other.cols_ = 0;
            other.data_ = nullptr;
        }
        return *this;
    }

    ~Matrix() { release(); }

    void swap(Matrix& other) noexcept {
        std::swap(rows_, other.rows_);
        std::swap(cols_, other.cols_);
        std::swap(data_, other.data_);
    }

    // --- Dimensions ---------------------------------------------------------
    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    bool isSquare() const { return rows_ == cols_; }

    // --- Element access -----------------------------------------------------
    T* operator[](std::size_t i) { return data_[i]; }              // A[i][j]
    const T* operator[](std::size_t i) const { return data_[i]; }

    T& operator()(std::size_t i, std::size_t j) { return data_[i][j]; }  // A(i, j)
    const T& operator()(std::size_t i, std::size_t j) const { return data_[i][j]; }

    // --- Identity matrix ----------------------------------------------------
    static Matrix identity(std::size_t n) {
        Matrix I(n, n);
        for (std::size_t i = 0; i < n; ++i) I[i][i] = T(1);
        return I;
    }

    // --- Addition / subtraction ---------------------------------------------
    Matrix operator+(const Matrix& other) const {
        checkSameSize(other);
        Matrix R(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) R[i][j] = data_[i][j] + other.data_[i][j];
        return R;
    }

    Matrix operator-(const Matrix& other) const {
        checkSameSize(other);
        Matrix R(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t j = 0; j < cols_; ++j) R[i][j] = data_[i][j] - other.data_[i][j];
        return R;
    }

    // --- Ordinary (classical) O(n^3) multiplication - the Strassen base case -
    // Loop order i-k-j gives sequential memory access (cache friendly)
    Matrix multiplyNaive(const Matrix& other) const {
        if (cols_ != other.rows_)
            throw std::invalid_argument("multiplyNaive: matrix dimensions do not match");
        Matrix R(rows_, other.cols_);
        for (std::size_t i = 0; i < rows_; ++i)
            for (std::size_t k = 0; k < cols_; ++k) {
                const T aik = data_[i][k];
                for (std::size_t j = 0; j < other.cols_; ++j) R[i][j] += aik * other.data_[k][j];
            }
        return R;
    }

    Matrix operator*(const Matrix& other) const { return multiplyNaive(other); }

    // --- Block operations (needed by Strassen) ------------------------------
    // Copy of the submatrix [r0, r0+r) x [c0, c0+c). Elements outside the
    // matrix are 0 (this is exactly the "zero padding").
    Matrix block(std::size_t r0, std::size_t c0, std::size_t r, std::size_t c) const {
        Matrix R(r, c);
        for (std::size_t i = 0; i < r && r0 + i < rows_; ++i)
            for (std::size_t j = 0; j < c && c0 + j < cols_; ++j)
                R[i][j] = data_[r0 + i][c0 + j];
        return R;
    }

    // Write matrix M into this matrix, starting at position (r0, c0)
    void setBlock(std::size_t r0, std::size_t c0, const Matrix& M) {
        if (r0 + M.rows_ > rows_ || c0 + M.cols_ > cols_)
            throw std::out_of_range("setBlock: block does not fit into the matrix");
        for (std::size_t i = 0; i < M.rows_; ++i)
            for (std::size_t j = 0; j < M.cols_; ++j) data_[r0 + i][c0 + j] = M.data_[i][j];
    }

    // New matrix of size newRows x newCols, padded with zeros on the right/bottom
    Matrix padded(std::size_t newRows, std::size_t newCols) const {
        return block(0, 0, newRows, newCols);
    }

    // --- Output -------------------------------------------------------------
    void print(std::ostream& os = std::cout) const {
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) os << data_[i][j] << ' ';
            os << '\n';
        }
    }

private:
    void checkSameSize(const Matrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::invalid_argument("Matrix sizes do not match");
    }
};

// ============================================================================
//  TASK 1: Strassen algorithm (implemented in Strassen.cpp)
//
//  Multiplies A (m x k) by B (k x n) for ANY dimensions.
//  threshold - size at which (N <= threshold) the algorithm switches to the
//  ordinary multiplication (32 or 64 according to the assignment).
//
//  Instantiated for Complex<double> and Complex<float>.
//  THIS FUNCTION IS USED BY THE OTHER TASKS (checking A * A^-1 = E, etc.).
// ============================================================================
template <typename T>
Matrix<T> strassenMultiply(const Matrix<T>& A, const Matrix<T>& B, std::size_t threshold = 64);

// ============================================================================
//  TASK 2 (member B): inverse matrix via the Gauss-Jordan method
//  Create GaussJordan.cpp and implement:
//
//      template <typename T> Matrix<T> inverseGaussJordan(const Matrix<T>& A);
//
//  TASK 3 (member C): inverse matrix via LU decomposition
//  Create LUInverse.cpp and implement:
//
//      template <typename T> Matrix<T> inverseLU(const Matrix<T>& A);
//
//  Hint: use abs(z) from Complex.h to choose the pivot element, and use
//  strassenMultiply(A, Ainv) compared with Matrix<T>::identity(n) to verify
//  the result.
// ============================================================================

#endif  // MATRIX_H