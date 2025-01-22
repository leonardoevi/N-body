#ifndef DEFINES_H
#define DEFINES_H

#define DIM 3
#define G 10.0
#define D_MIN 1.0e-5

// REPLACEABLE BY VARIABLES

#define N_PARTICLES_DEFAULT 3000
#define BLOCK_SIZE 32           // blocks will be 2D: BLOCK_SIZE*BLOCK_SIZE (max is 32 on my GPU)
#define BETTER_REDUCTION true

#endif //DEFINES_H
