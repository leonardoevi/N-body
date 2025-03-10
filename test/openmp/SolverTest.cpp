#include <gtest/gtest.h>
#include "../../src/openmp/Solver/Solver.h"
#include "../../src/openmp/Solver/ForwardEulerSolver.h"
#include "../../src/openmp/Solver/LeapFrogSolver.h"


constexpr unsigned int num_particles = 2;
constexpr unsigned int dimension = 2;

class SolverTest : public ::testing::Test {
protected:

    double total_time = 0.5;
    double delta_time = 1.0; // In this case we will compute only one step of the simulate function

    std::vector<double> mass;
    std::vector<Vector> initial_positions;
    std::vector<Vector> initial_velocities;

    Solver *forwardEulerSolver = nullptr;
    Solver *leapFrogSolver = nullptr;

    void SetUp() override {
        //Initializing masses
        mass = {1.0, 1.0};

        //Initializing positions and velocities
        initial_positions = std::vector<Vector>(num_particles);
        initial_positions[0] = Vector({0.0, 0.0});
        initial_positions[1] = Vector({1.0, 0.0});

        initial_velocities = std::vector<Vector>(num_particles);

        initial_velocities[0] = Vector({0.0, 0.0});
        initial_velocities[1] = Vector({0.0, 0.0});

        forwardEulerSolver = new ForwardEulerSolver(dimension, num_particles, total_time, delta_time, mass , initial_positions, initial_velocities);
        leapFrogSolver = new LeapFrogSolver(dimension, num_particles, total_time, delta_time, mass , initial_positions, initial_velocities);
    }
    void TearDown() override {
        delete forwardEulerSolver;
        delete leapFrogSolver;
    }
};

TEST_F(SolverTest, ComputeMatrixForwardEuler) {
    forwardEulerSolver->compute_matrix();

    const auto &acceleration_matrix = forwardEulerSolver->get_accelerations();
    for (unsigned int i = 0; i < num_particles; i++) {
        for (unsigned int j = i+1; j < dimension; j++) {
            EXPECT_EQ(acceleration_matrix[i][j], - acceleration_matrix[j][i]) << "Acceleration matrix is not symmetric";
        }
    }
}

TEST_F(SolverTest, ComputeMatrixLeapFrog) {
    leapFrogSolver->compute_matrix();

    const auto &acceleration_matrix = leapFrogSolver->get_accelerations();
    for (unsigned int i = 0; i < num_particles; i++) {
        for (unsigned int j = i+1; j < dimension; j++) {
            EXPECT_EQ(acceleration_matrix[i][j], - acceleration_matrix[j][i]) << "Acceleration matrix is not symmetric";
        }
    }
}

TEST_F(SolverTest, SimulateLeapFrog) {
    leapFrogSolver->simulate("test_output_leapfrog.txt");

    auto final_positions = leapFrogSolver->get_positions();
    // With the data given it can be easily computed that the position of particle 0 is now (5.0, 0)
    EXPECT_EQ(final_positions[0], Vector({5.0, 0.0}));
}

TEST_F(SolverTest, SimulateForwardEuler) {
    forwardEulerSolver->simulate("test_output_forward_euler.txt");

    auto final_positions = forwardEulerSolver->get_positions();
    // With the data given it can be easily computed that the position of particle 0 is now (10, 0)
    EXPECT_EQ(final_positions[0], Vector({10.0, 0.0}));
}



