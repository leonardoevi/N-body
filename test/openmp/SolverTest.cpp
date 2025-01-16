#include <gtest/gtest.h>
#include "../../src/openmp/Solver.cpp"

using namespace std;

constexpr unsigned int num_particles = 2;
constexpr unsigned int dimension = 2;

class SolverTest : public ::testing::Test {
protected:

    double total_time = 0.5;
    double delta_time = 1.0; // In this case we will compute only one step of the simulate function

    vector<double> mass;
    vector<Vector<dimension>> initial_positions;
    vector<Vector<dimension>> initial_velocities;

    Solver<num_particles, dimension> *solver = nullptr;

    void SetUp() override {
        //Initializing masses
        mass = {1.0, 1.0};

        //Initializing positions and velocities
        initial_positions = vector<Vector<dimension>>(num_particles);
        initial_positions[0] = Vector<dimension>({0.0, 0.0});
        initial_positions[1] = Vector<dimension>({1.0, 0.0});

        initial_velocities = vector<Vector<dimension>>(num_particles);

        initial_velocities[0] = Vector<dimension>({0.0, 0.0});
        initial_velocities[1] = Vector<dimension>({0.0, 0.0});

        solver = new Solver<num_particles, dimension>(total_time, delta_time, mass , initial_positions, initial_velocities);
    }
    void TearDown() override {
        delete solver;
    }
};

TEST_F(SolverTest, ComputeMatrix) {
    solver->computeMatrix();

    const auto &acceleration_matrix = solver->get_accelerations();
    for (unsigned int i = 0; i < num_particles; i++) {
        for (unsigned int j = i+1; j < dimension; j++) {
            EXPECT_EQ(acceleration_matrix[i][j], - acceleration_matrix[j][i]) << "Acceleration matrix is not symmetric";
        }
    }
}

TEST_F(SolverTest, SimulateLeapFrog) {
    solver->simulateLeapFrog("test_output.txt");

    auto final_positions = solver->get_positions();
    // With the data given it can be easily computed that the position of particle 0 is now (10, 10)
    EXPECT_EQ(final_positions[0], Vector<dimension>({5.0, 0.0}));
}

TEST_F(SolverTest, SimulateForwardEuler) {
    solver->simulateForwardEuler("test_output.txt");

    auto final_positions = solver->get_positions();
    // With the data given it can be easily computed that the position of particle 0 is now (10, 10)
    EXPECT_EQ(final_positions[0], Vector<dimension>({10.0, 0.0}));
}



