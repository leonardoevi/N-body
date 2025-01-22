#ifndef VECTOR_H
#define VECTOR_H

#include <array>
#include <iostream>
#include <initializer_list>

class Vector {
    int dim;
    std::unique_ptr<double[]> components;

public:
    // Used for testing
    Vector() : dim(0), components(nullptr) {}

    explicit Vector(const int dim) : dim(dim), components(std::make_unique<double[]>(dim)) {
      for (int i = 0; i < dim; i++) {
        components[i] = 0.0;
      }
    }
    // Copy constructor
    Vector(const Vector& other)
        : dim(other.dim), components(std::make_unique<double[]>(other.dim)) {
        for (int i = 0; i < dim; ++i) {
          components[i] = other.components[i];
        }
      }

    // Constructor for testing
    Vector(const std::initializer_list<double>& init) : dim(init.size()), components(std::make_unique<double[]>(init.size())) {
      int i = 0;
      for (const auto& c : init) {
        components[i++] = c;
      }
    }

    [[nodiscard]] int get_dim() const;

    //Used for testing
    bool operator==(const Vector& rhs) const;

    double& operator[](unsigned int index);

    const double& operator[](unsigned int index) const;

    Vector& operator=(const Vector& rhs);

    Vector operator+(const Vector& rhs) const;

    Vector operator-(const Vector& rhs) const;

    Vector operator*(double scalar) const;

    Vector operator/(double scalar) const;

    Vector operator-() const;

    Vector& operator+=(const Vector& rhs);

    Vector& operator-=(const Vector& rhs);

    [[nodiscard]] double dot(const Vector& rhs) const;

    [[nodiscard]] double norm() const;

    [[nodiscard]] Vector normalize() const;

    void fill(double value);

    friend std::ostream& operator<<(std::ostream& os, const Vector& rhs);

};



#endif //VECTOR_H
