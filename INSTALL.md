# Installation
 
**1. CUDA Execution:**

* **Requirements:**
    * **NVIDIA GPU:** An NVIDIA GPU is required for CUDA execution.
    * **CUDA Toolkit:** Install the CUDA Toolkit from the NVIDIA developer website. This provides the necessary libraries and tools for compiling and running CUDA code.

**2. OpenMP Execution:**

* **Requirements:**
    * **C++ Compiler with OpenMP Support:** Ensure your C++ compiler (e.g., g++, clang++) supports OpenMP.

**3. OpenGL Execution:**

* **Requirements:**
    * **OpenGL:** Install OpenGL libraries.
    * **GLEW:** Install the OpenGL Extension Wrangler (GLEW) library.
    * **GLFW:** Install the GLFW library for window and input handling.

**Supported Machines:**

* **macOS:**
    * **Compiler:** Use clang++.
    * **OpenMP:**
        ```bash
        brew install libomp 
        ```
    * **OpenGL, GLEW, GLFW:**
        ```bash
        brew install glfw glew 
        ```
    * **Build and Run:**
        ```bash
        git clone <repository_url>
        mkdir build
        cd build
        cmake ..
        make 
        # Run executables (e.g., cuda_simulation, openmp_simulation, opengl_visualization)
        ```

* **Linux:**
    * **Build and Run:**
        * Open the project in CLion and load it from the CMakeLists.txt file. (Recommended)
        * Alternatively you can compile the CUDA/OpenMP code by moving to each subfolder of the `src` directory (`cuda`, `openmp`), and run:
            ```bash
            make
            ```
        * Run executables from the `build` folder.

**Execution**

* **Simulation (CUDA/OpenMP):**
    ```bash
    ./<executable_name> <timestep> <total_time> {fe | lf} <output_filename> <input_filename> 
    ```
    * **Arguments:**
        * `<timestep>`: Time step for the simulation.
        * `<total_time>`: Total simulation time.
        * `{fe | lf}`: Integration scheme (`{fe}` for Forward Euler and `{lf}` for Leapfrog).
        * `<output_filename>`: Output file name for simulation results.
        * `<input_filename>`: (Optional) Input file path. If not provided, initial positions and velocities will be generated.

* **Visualization (OpenGL):**
    ```bash
    ./n-body-openGL <output_filename>
    ```

**Input File Format:**

* The input file should have the following format:
    ```
    <number_of_particles> <dimension>
    <mass_of_particle_1> <mass_of_particle_2> ... <mass_of_particle_n>
    <position_x1> <position_y1> <position_z1> <velocity_x1> <velocity_y1> <velocity_z1> 
    <position_x2> <position_y2> <position_z2> <velocity_x2> <velocity_y2> <velocity_z2>
    ...
    <position_xn> <position_yn> <position_zn> <velocity_xn> <velocity_yn> <velocity_zn>
    ```
    * `<number_of_particles>`: The number of particles in the simulation.
    * `<dimension>`: The dimensionality of the simulation (e.g., 3 for 3D).
    * `<mass_of_particle_i>`: Mass of the i-th particle.
    * `<position_xi>`: x-coordinate of the position of the i-th particle.
    * `<velocity_xi>`: x-coordinate of the velocity of the i-th particle.
    * Positions and velocities are provided for each particle in each line.


**A simple test case**

Both the OpenMP and CUDA executables can be run without providing an input file. However,
an example of input state can be found in the folder `asset/LeapFrog_vs_ForwardEuler/lf_vs_fe.txt`.
You can compile the codes and run:
```
./n-body-cuda 0.001 0.5 lf out.txt <path-to>/lf_vs_fe.txt
```
or
```
./n-body-openMP 0.001 0.5 lf out.txt <path-to>/lf_vs_fe.txt
```
and
```
./n-body-openGL <path-to>/out.txt
```
to visualize the results.