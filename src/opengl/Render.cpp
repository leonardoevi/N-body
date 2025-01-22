#include "Render.h"

// Initialize static variables
float Render::cameraPosX = 0.0f;
float Render::cameraPosY = 0.0f;
float Render::cameraPosZ = -30.0f;
float Render::yaw = -90.0f;
float Render::pitch = 0.0f;
float Render::radius = 30.0f;
float Render::lastX = 400.0f;
float Render::lastY = 300.0f;
bool Render::firstMouse = true;
bool Render::isDragging = false;
bool Render::isPaused = true;


void Render::initialize_rendering(){
    //Makes the context of the window the main context of the current thread
    glfwMakeContextCurrent(window);
    glewInit();
    glfwGetFramebufferSize(window, &width, &height);
    set_up_3D_view();
}

void Render::set_up_3D_view() const {
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0, (double)width / height, 0.1, 100.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    // Mouse and keyboard input callbacks
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);

    glfwSetScrollCallback(window, scroll_callback);
    glfwSetKeyCallback(window, key_callback);

    glfwSetWindowCloseCallback(window, window_close_callback);
}

void Render::show(const std::vector<Vector>& positions, const std::vector<Vector>& velocities, const std::vector<double> masses){
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

        render_particles(positions, velocities, masses);

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

void Render::render_particles(const std::vector<Vector>& positions, const std::vector<Vector>& velocities, const std::vector<double> masses){
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
        float color = fmin(1.0f, vel / v_max); // Normalize velocity for color

        if (color <= 0.5f) {
            // Purple to Orange transition
            glColor3f(color * 2.0f, color, 0.5f);
        } else {
            // Orange to Yellow transition
            glColor3f(1.0f, 0.5f + (color - 0.5f) * 2.0f, (color - 0.5f) * 1.5f);
        }
        glPushMatrix();
        if (dimension == 3)
            glTranslatef(p[0], p[1], p[2]);
        else
            glTranslatef(p[0], p[1], 0.0);
        render_sphere(masses[i] / m_max * 0.05); // Render a sphere with radius 0.1
        glPopMatrix();

        i++;
    }
}

void Render::render_sphere(float sphere_radius) const{
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

            glVertex3f(sphere_radius * x0, sphere_radius * y0, sphere_radius * z0);
            glVertex3f(sphere_radius * x1, sphere_radius * y1, sphere_radius * z1);
        }
        glEnd();
    }
}

void Render::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
        isPaused = !isPaused; // Toggle pause
    }
}

void Render::scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
    // Adjust the radius based on scroll input
    radius -= static_cast<float>(yoffset);
    // Clamp the radius to prevent zooming too far or too close
    if (radius < 5.0f) radius = 5.0f;     // Minimum zoom level
    if (radius > 100.0f) radius = 100.0f; // Maximum zoom level

    update_camera_direction(); // Update the camera position
}

void Render::mouse_callback(GLFWwindow* window, double xpos, double ypos){
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

    update_camera_direction(); // Recalculate camera position
}

void Render::mouse_button_callback(GLFWwindow* window, int button, int action, int mods){
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            isDragging = true;
            firstMouse = true; // Reset the firstMouse flag when starting to drag
        } else if (action == GLFW_RELEASE) {
            isDragging = false;
        }
    }
}

void Render::window_close_callback(GLFWwindow* window){
    // terminate the program
    std::exit(0);
}

void Render::update_camera_direction(){
    // Convert yaw and pitch to radians
    float yawRad = radians(yaw);
    float pitchRad = radians(pitch);

    // Calculate the new camera position using spherical coordinates
    cameraPosX = radius * cos(pitchRad) * cos(yawRad);
    cameraPosY = -radius * sin(pitchRad);
    cameraPosZ = radius * cos(pitchRad) * sin(yawRad);
}