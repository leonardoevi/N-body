#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <iomanip>

#include "Render.h"
#include "../../include/define.h"


int main() {
    std::ifstream file("../out/out.txt");

    if (!file.is_open()) {
      std::cout << "Error opening file" << std::endl;
      return 1;
    }

    int num_particles, dimension;
    file >> num_particles >> dimension;
    std::cout << std::fixed << std::setprecision(std::numeric_limits<double>::digits10 + 1);

    std::vector<Vector> positions(num_particles, Vector(dimension));
    std::vector<Vector> velocities(num_particles, Vector(dimension));
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

    Render render(width/1.5, height/1.5, dimension);
    render.initialize_rendering();

    for (int i = 0; i < num_particles; i++) {
        file >> masses[i];
    }

    long double current_time;
    // For each time step present in the file reads particles' positions and velocities and displays them. If pausing option is active renders them
    // so that the user can still zoom in and scroll around the screen
    while (file >> current_time ) {

        for (int i = 0; i < num_particles; i++) {
            for (int j = 0; j < DIM; j++) {
                file >> positions[i][j];
            }
            for (int k = 0; k < DIM; k++) {
                file >> velocities[i][k];
            }
        }

        render.show(positions, velocities, masses);

        while (Render::isPaused) {
            render.show(positions, velocities, masses);
        }

    }


}
