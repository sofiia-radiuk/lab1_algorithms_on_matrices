#ifndef COMPLEX_H
#define COMPLEX_H

#include <cmath>
#include <iostream>

// ============================================================================
//  Complex<T> - a complex number z = re + im * i
//  T - type of the real and imaginary parts (double, float, ...)
//
//  Supports: + - * /, += -= *= /=, unary minus, == !=, conj(), norm(), abs(),
//  and stream output. This is everything needed for Strassen, Gauss-Jordan and
//  LU (use abs(z) or norm() when choosing a pivot element).
// ============================================================================
template <typename T>
class Complex {
public:
    T re;  // real part
    T im;  // imaginary part

    // --- Constructors -------------------------------------------------------
    Complex() : re(T()), im(T()) {}                  // 0 + 0i
    Complex(T real) : re(real), im(T()) {}           // real + 0i (implicit conversion)
    Complex(T real, T imag) : re(real), im(imag) {}

    // --- Compound assignment operators --------------------------------------
    Complex& operator+=(const Complex& o) {
        re += o.re;
        im += o.im;
        return *this;
    }

    Complex& operator-=(const Complex& o) {
        re -= o.re;
        im -= o.im;
        return *this;
    }

    // (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    Complex& operator*=(const Complex& o) {
        T newRe = re * o.re - im * o.im;
        T newIm = re * o.im + im * o.re;
        re = newRe;
        im = newIm;
        return *this;
    }

    // (a + bi)/(c + di) = ((ac + bd) + (bc - ad)i) / (c^2 + d^2)
    Complex& operator/=(const Complex& o) {
        T denom = o.re * o.re + o.im * o.im;
        T newRe = (re * o.re + im * o.im) / denom;
        T newIm = (im * o.re - re * o.im) / denom;
        re = newRe;
        im = newIm;
        return *this;
    }

    // --- Unary minus --------------------------------------------------------
    Complex operator-() const { return Complex(-re, -im); }

    // --- Binary operators (friends, so that 2.0 * z also works) -------------
    friend Complex operator+(Complex a, const Complex& b) { a += b; return a; }
    friend Complex operator-(Complex a, const Complex& b) { a -= b; return a; }
    friend Complex operator*(Complex a, const Complex& b) { a *= b; return a; }
    friend Complex operator/(Complex a, const Complex& b) { a /= b; return a; }

    friend bool operator==(const Complex& a, const Complex& b) {
        return a.re == b.re && a.im == b.im;
    }
    friend bool operator!=(const Complex& a, const Complex& b) { return !(a == b); }

    // --- Helper functions ---------------------------------------------------
    Complex conj() const { return Complex(re, -im); }   // complex conjugate
    T norm() const { return re * re + im * im; }        // |z|^2 (no square root - fast)

    // --- Output format: (a+bi) ----------------------------------------------
    friend std::ostream& operator<<(std::ostream& os, const Complex& z) {
        os << "(" << z.re << (z.im < T() ? "-" : "+")
           << (z.im < T() ? -z.im : z.im) << "i)";
        return os;
    }
};

// Modulus of a complex number: |z| = sqrt(re^2 + im^2)
template <typename T>
T abs(const Complex<T>& z) {
    return std::hypot(z.re, z.im);
}

#endif  // COMPLEX_H