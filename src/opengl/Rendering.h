#ifndef RENDERING_H
#define RENDERING_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <GLKit/GLKMatrix4.h>
#include <cmath>

#include "../../include/define.h"
#include "../openmp/Vector.hpp"

constexpr float radians(float degrees) {
  constexpr float pi = 3.14159265358979323846f;
  return degrees * (pi / 180.0f);
}

class Rendering {
      GLFWwindow* window;
      int width, height;

      static float cameraPosX, cameraPosY, cameraPosZ;
      static float yaw, pitch;
      static float radius; // Fixed distance from the center
      static float lastX, lastY; // Mouse tracking
      static bool firstMouse;
      static bool isDragging; // Tracks if the mouse is being dragged


  public:


  Rendering(const int width, const int height): width(width), height(height) {
          //Sanity check
          if (!glfwInit()) {
            cerr << "Failed to initialize GLFW" << endl;
            return;
          }
          string title = string(TOSTRING(DIM)) + "D N-body Simulation";
          window = glfwCreateWindow(width, height, title.c_str(), NULL, NULL);
          if (!window) {
            glfwTerminate();
            cerr << "Failed to create GLFW window" << endl;
          }
        }

  void initialize_rendering() {
          //Makes the context of the window the main context of the current thread
          glfwMakeContextCurrent(window);
          glewInit();
          glfwGetFramebufferSize(window, &width, &height);
          setup3DView();
        }

  void setup3DView() const {
          glMatrixMode(GL_PROJECTION);
          glLoadIdentity();
          gluPerspective(45.0, (double)width / height, 0.1, 100.0);
          glMatrixMode(GL_MODELVIEW);
          glLoadIdentity();

          // Mouse and keyboard input callbacks
          glfwSetCursorPosCallback(window, mouse_callback);
          glfwSetMouseButtonCallback(window, mouse_button_callback);

          glfwSetScrollCallback(window, scroll_callback);

        }

  static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
          // Adjust the radius based on scroll input
          radius -= static_cast<float>(yoffset);
          // Clamp the radius to prevent zooming too far or too close
          if (radius < 5.0f) radius = 5.0f;     // Minimum zoom level
          if (radius > 100.0f) radius = 100.0f; // Maximum zoom level

          updateCameraDirection(); // Update the camera position based on the new radius
        }

  static void updateCameraDirection() {
          // Convert yaw and pitch to radians
          float yawRad = radians(yaw);
          float pitchRad = radians(pitch);

          // Calculate the new camera position using spherical coordinates
          cameraPosX = radius * cos(pitchRad) * cos(yawRad);
          cameraPosY = radius * sin(pitchRad);
          cameraPosZ = radius * cos(pitchRad) * sin(yawRad);

        }

  static void mouse_callback(GLFWwindow* window, double xpos, double ypos) {
          if (!isDragging) return; // Only update if dragging

          if (firstMouse) {
            lastX = xpos;
            lastY = ypos;
            firstMouse = false;
          }

          float xOffset = xpos - lastX;
          float yOffset = lastY - ypos; // Reversed: y-coordinates go from bottom to top

          lastX = xpos;
          lastY = ypos;

          float sensitivity = 0.1f; // Mouse sensitivity
          xOffset *= sensitivity;
          yOffset *= sensitivity;

          yaw += xOffset;
          pitch += yOffset;

          // Constrain pitch to avoid flipping
          if (pitch > 89.0f) pitch = 89.0f;
          if (pitch < -89.0f) pitch = -89.0f;

          updateCameraDirection(); // Recalculate camera position
        }

  static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
          if (button == GLFW_MOUSE_BUTTON_LEFT) {
            if (action == GLFW_PRESS) {
              isDragging = true;
              firstMouse = true; // Reset the firstMouse flag when starting to drag
            } else if (action == GLFW_RELEASE) {
              isDragging = false;
            }
          }
        }


  void show (const vector<Vector<DIM>>& positions, const vector<Vector<DIM>>& velocities, vector<double>& masses) {
          if (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            glMatrixMode(GL_MODELVIEW);
            // Very important
            glLoadIdentity();

            // Set the camera view
            gluLookAt(
                cameraPosX, cameraPosY, cameraPosZ, // Camera position
                0.0f, 0.0f, 0.0f,// Look-at point (center of the scene)
                0.0f, 1.0f, 0.0f// Up vector
            );

            renderParticles(positions, velocities, masses);

            // Swaps buffer color (a large 2D buffer that contains color values for each pixel of the window
            glfwSwapBuffers(window);
            //Checks if any events(keyboard inputs or mouse movement) are triggered and calls their callback functions
            glfwPollEvents();
          }
          else {
            glfwDestroyWindow(window);
            glfwTerminate();
          }
        }

  void renderParticles(const vector<Vector<DIM>>& positions, const vector<Vector<DIM>>& velocities, vector<double>& masses) {
          double m_max = 0.0f;
          double v_max = 0.1f;
          int i = 0;
          for (const auto& p : velocities) {
            v_max = fmax(v_max, p.norm());
            m_max = fmax(m_max, masses[i]);
            i++;
          }

          i = 0;
          for (const auto& p : positions) {

            float vel = velocities[i].norm();
            float color = fmin(1.0f, vel / v_max); // Normalize distance for color

            glColor3f(color, 1.0f - color, 0.0); // Gradient from blue to red
            glPushMatrix();
            if (DIM == 3)
              glTranslatef(p[0], p[1], p[2]);
            else
              glTranslatef(p[0], p[1], 0.0);
            renderSphere(masses[i] / m_max * 0.1); // Render a sphere with radius 0.1
            glPopMatrix();

            i++;
          }
        }

  void renderSphere(float sphereRadius) const {
          const int stacks = 10;
          const int slices = 10;
          for (int i = 0; i < stacks; ++i) {
            float lat0 = radians(-90.0f + 180.0f * i / stacks);
            float lat1 = radians(-90.0f + 180.0f * (i + 1) / stacks);

            glBegin(GL_QUAD_STRIP);
            for (int j = 0; j <= slices; ++j) {
              float lon = radians(360.0f * j / slices);

              float x0 = cos(lat0) * cos(lon);
              float y0 = sin(lat0);
              float z0 = cos(lat0) * sin(lon);

              float x1 = cos(lat1) * cos(lon);
              float y1 = sin(lat1);
              float z1 = cos(lat1) * sin(lon);

              glVertex3f(sphereRadius * x0, sphereRadius * y0, sphereRadius * z0);
              glVertex3f(sphereRadius * x1, sphereRadius * y1, sphereRadius * z1);
            }
            glEnd();
          }
        }
};

// Initialize static variables
float Rendering::cameraPosX = 0.0f;
float Rendering::cameraPosY = 0.0f;
float Rendering::cameraPosZ = -30.0f;
float Rendering::yaw = -90.0f;
float Rendering::pitch = 0.0f;
float Rendering::radius = 30.0f;
float Rendering::lastX = 400.0f;
float Rendering::lastY = 300.0f;
bool Rendering::firstMouse = true;
bool Rendering::isDragging = false;


#endif //RENDERING_H
