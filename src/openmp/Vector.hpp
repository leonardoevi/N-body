// Vector.hpp
#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <iostream>
#include <cmath>

using namespace std;

template <unsigned int dimension> class Vector {

    array<double, dimension> components;

public:
    Vector() { components.fill(0.0); }

    explicit Vector(array<double, dimension>& components): components(components) {}

    Vector(const initializer_list<double>& values) {
      if (values.size() != dimension) {
        throw invalid_argument("Vector values should be the same size");
      }
      copy(values.begin(), values.end(), components.begin());
    }

    //Is equal operator
    bool operator==(const Vector<dimension>& other) const {
      for (unsigned int i = 0; i < dimension; i++) {
        if (components[i] != other.components[i]) {
          return false;
        }
      }
      return true;
    }

    //Assignment operator
    double& operator[](unsigned int index) {
      if (index >= dimension) {
        throw out_of_range("Index out of range");
      }
      return components[index];
    }
    const double& operator[](unsigned int index) const {
      if (index >= dimension) {
        throw out_of_range("Index out of range");
      }
      return components[index];
    }
    //Addition Operator
    Vector operator+(const Vector& rhs) const {
      Vector result;
      for (unsigned int i = 0; i < dimension; i++) {
        result[i] = components[i] + rhs.components[i];
      }
      return result;
    }
    //Subtraction Operator
    Vector operator-(const Vector& rhs) const {
      Vector result;
      for (unsigned int i = 0; i < dimension; i++) {
        result[i] = components[i] - rhs.components[i];
      }
      return result;
    }
    //Multiplication by scalar operator
    Vector operator*(double scalar) const {
      Vector result;
      for (unsigned int i = 0; i < dimension; i++) {
        result[i] = components[i] * scalar;
      }
      return result;
    }
    // Division by scalar operator
    Vector operator/(double scalar) const {
      if (scalar == 0.0) {
        throw out_of_range("Division by zero");
      }
      Vector result;
      for (unsigned int i = 0; i < dimension; i++) {
        result[i] = components[i] / scalar;
      }
      return result;
    }
    // Inversion Operator
    Vector operator-() const {
      Vector result;
      for (unsigned int i = 0; i < dimension; i++) {
        result[i] = -components[i];
      }
      return result;
    }
    //Addition assignment operator
    Vector& operator+=(const Vector& rhs) {
      for (unsigned int i = 0; i < dimension; i++) {
        components[i] += rhs.components[i];
      }
      return *this;
    }
    //Subtraction assignment operator
    Vector& operator-=(const Vector& rhs) {
      for (unsigned int i = 0; i < dimension; i++) {
        components[i] -= rhs.components[i];
      }
      return *this;
    }

    double dot(const Vector& rhs) const {
      double result = 0.0;
      for (unsigned int i = 0; i < dimension; i++) {
        result += components[i] * rhs.components[i];
      }
      return result;
    }

    [[nodiscard]] double norm() const {
      return sqrt(dot(*this));
    }

    Vector normalize() const {
      double value = norm();
      if (value == 0.0) {
        throw out_of_range("Norm is zero");
      }
      return *this / value;
    }

    void fill(double value) {
      for (unsigned int i = 0; i < dimension; i++) {
        components[i] = value;
      }
    }

    // Stream Output operator
    friend ostream& operator<<(ostream& os, const Vector& rhs) {
      os << '(';
      for (unsigned int i = 0; i < dimension; i++) {
        os << rhs[i];
        if (i != dimension - 1) os << " , ";
      }
      os << ')';
      return os;
    }

};
#endif // VECTOR_HPP