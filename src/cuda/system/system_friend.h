#ifndef SYSTEM_FRIEND_H
#define SYSTEM_FRIEND_H

#include "System.h"
#include <pthread.h>

/**
 * Function executed by a thread that performs the output of the system state while the new one is
 * being computed on the GPU
 * @param system pointer to the System object, containing the data to be output.
 */
void* write_system_state(void* system);

#endif //SYSTEM_FRIEND_H
