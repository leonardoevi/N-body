#include "Vector.h"

int Vector::get_dim() const {
    return dim;
}

//Is equal operator
bool Vector::operator==(const Vector &rhs) const {
    for (int i = 0; i < dim; i++) {
        if (components[i] != rhs.components[i]) {
            return false;
        }
    }
    return true;
}

//Assignment operator
double &Vector::operator[](unsigned int index) {
    if (index >= dim) {
        throw std::out_of_range("Index out of range");
    }
    return components[index];
}

const double &Vector::operator[](unsigned int index) const {
    if (index >= dim) {
        throw std::out_of_range("Index out of range");
    }
    return components[index];
}

Vector &Vector::operator=(const Vector &rhs) {
    if (this != &rhs) {
        dim = rhs.dim;
        components = std::make_unique<double[]>(dim);
        for (int i = 0; i < dim; i++) {
            components[i] = rhs.components[i];
        }
    }
    return *this;
}

//Addition Operator
Vector Vector::operator+(const Vector &rhs) const {
    Vector result(dim);
    for (int i = 0; i < dim; i++) {
        result.components[i] = components[i] + rhs.components[i];
    }
    return result;
}

//Subtraction Operator
Vector Vector::operator-(const Vector &rhs) const {
    Vector result(dim);
    for (int i = 0; i < dim; i++) {
        result.components[i] = components[i] - rhs.components[i];
    }
    return result;
}

//Multiplication by scalar operator
Vector Vector::operator*(double scalar) const {
    Vector result(dim);
    for (int i = 0; i < dim; i++) {
        result.components[i] = components[i] * scalar;
    }
    return result;
}

// Division by scalar operator
Vector Vector::operator/(double scalar) const {
    Vector result(dim);
    for (int i = 0; i < dim; i++) {
        result.components[i] = components[i] / scalar;
    }
    return result;
}

// Inversion Operator
Vector Vector::operator-() const {
    Vector result(dim);
    for (int i = 0; i < dim; i++) {
        result.components[i] = -components[i];
    }
    return result;
}

//Addition assignment operator
Vector& Vector::operator+=(const Vector &rhs) {
    for (int i = 0; i < dim; i++) {
        components[i] += rhs.components[i];
    }
    return *this;
}

//Subtraction assignment operator
Vector& Vector::operator-=(const Vector &rhs) {
    for (int i = 0; i < dim; i++) {
        components[i] -= rhs.components[i];
    }
    return *this;
}

//Dot product function
double Vector::dot(const Vector &rhs) const {
    double result = 0.0;
    for (int i = 0; i < dim; i++) {
        result += components[i] * rhs.components[i];
    }
    return result;
}

//Returns the norm of the vector
double Vector::norm() const {
    return std::sqrt(dot(*this));
}

// Returns normalized vector
Vector Vector::normalize() const {
    double norm = this->norm();
    if (norm == 0.0) {
        throw std::out_of_range("Norm is zero");
    }
    return std::move(*this / norm);
}

// Fills the array with the value
void Vector::fill(double value) {
    for (int i = 0; i < dim; i++) {
        components[i] = value;
    }
}

// Stream Output operator
std::ostream& operator<<(std::ostream &os, const Vector &vec) {
    os << '(';
    for (unsigned int i = 0; i < vec.dim; i++) {
        os << vec[i];
        if (i != vec.dim - 1) os << " , ";
    }
    os << ')';
    return os;
}






