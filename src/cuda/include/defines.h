#ifndef DEFINES_H
#define DEFINES_H

// space dimensions (2D or 3D)
#define DIM 3

// gravitational constant
#define G 10.0

// minimum distance to be considered between particles (to avoid dividing by
// zero) @note In modern C++ you use inline constexpr instead of #define:

// inline constexpr double D_MIN = 1.0e-5;
#define D_MIN 1.0e-5

// flag that allows to write the energy of a system in an output file at each
// time step, WARNING this is not done efficiently do not use on large systems.
#define WRITE_ENERGY false

#define TIME_TESTING false

// --- REPLACEABLE BY VARIABLES --- //

#define N_PARTICLES_DEFAULT 3000
#define BLOCK_SIZE                                                             \
  32 // blocks will be 2D: BLOCK_SIZE*BLOCK_SIZE (max is 32 on my GPU)
#define BETTER_REDUCTION true

#endif // DEFINES_H
