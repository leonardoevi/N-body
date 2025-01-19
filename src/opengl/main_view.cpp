#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

#include "Rendering.h"
#include "../openmp/Vector.hpp"
#include "../../include/define.h"

int main() {
    ifstream file("../out/out.txt");

    if (!file.is_open()) {
      cout << "Error opening file" << endl;
      return 1;
    }

    int num_particles;
    file >> num_particles;
    cout << fixed << std::setprecision(numeric_limits<double>::digits10 + 1);

    std::vector<Vector<DIM>> positions(num_particles);
    std::vector<Vector<DIM>> velocities(num_particles);
    std::vector<double> masses(num_particles);

    // Getting width height ratio of primary monitor
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Get the primary monitor
    GLFWmonitor* primaryMonitor = glfwGetPrimaryMonitor();
    if (!primaryMonitor) {
        std::cerr << "Failed to get primary monitor" << std::endl;
        glfwTerminate();
        return -1;
    }

    int x, y, width, height;
    glfwGetMonitorWorkarea(primaryMonitor, &x, &y, &width, &height);

    Rendering rendering(width/1.5, height/1.5);
    rendering.initialize_rendering();

    for (int i = 0; i < num_particles; i++) {
        file >> masses[i];
    }

    long double current_time;
    while (file >> current_time ) {

        for (int i = 0; i < num_particles; i++) {
            for (int j = 0; j < DIM; j++) {
                file >> positions[i][j];
            }
            for (int k = 0; k < DIM; k++) {
                file >> velocities[i][k];
            }
        }
        rendering.show(positions, velocities, masses);

        while (Rendering::isPaused) {
            rendering.show(positions, velocities, masses);
        }
    }

}
