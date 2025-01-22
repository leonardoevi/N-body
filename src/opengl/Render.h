#ifndef RENDER_H
#define RENDER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "../openmp/Vector.h"

constexpr float radians(float degrees) {
    constexpr float pi = 3.14159265358979323846f;
    return degrees * (pi / 180.0f);
}

class Render {
    GLFWwindow* window;
    int width, height;
    int dimension;

    static float cameraPosX, cameraPosY, cameraPosZ; // Camera position values
    static float yaw, pitch;
    static float radius; // Fixed distance from the center
    static float lastX, lastY; // Mouse tracking
    static bool firstMouse;
    static bool isDragging; // Tracks if the mouse is being dragged

    public:
      static bool isPaused;

    Render(const int width, const int height, const int dimension): width(width), height(height), dimension(dimension) {
        //Sanity check
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return;
        }
        std::string title = std::to_string(dimension) + "D N-body Simulation";
        window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
        if (!window) {
            glfwTerminate();
            std::cerr << "Failed to create GLFW window" << std::endl;
        }
    }

    void initialize_rendering();

    void set_up_3D_view() const ;

    void show(const std::vector<Vector>& positions, const std::vector<Vector>& velocities, const std::vector<double> masses);

    void render_particles(const std::vector<Vector>& positions, const std::vector<Vector>& velocities, const std::vector<double> masses);

    void render_sphere(float sphere_radius) const;

    static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);

    static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

    static void mouse_callback(GLFWwindow* window, double xpos, double ypos);

    static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);

    static void window_close_callback(GLFWwindow* window);

    static void update_camera_direction();
};

#endif //RENDER_H
