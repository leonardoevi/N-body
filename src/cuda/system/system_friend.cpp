#include "system_friend.h"

void* write_system_state(void* system){
    auto* system_obj = static_cast<System *>(system);

    while (true) {
        pthread_mutex_lock(&system_obj->mutex);

        // wait for the need of writing memory to file
        while (system_obj->print_system == false) {
            // check if termination has been requested
            if (system_obj->kys == true) {
                pthread_mutex_unlock(&system_obj->mutex);
                return nullptr;
            }

            // release the lock and wait for the condition to be signaled
            pthread_cond_wait(&system_obj->cond, &system_obj->mutex);
        }

        // now it is time to write the system to file
        system_obj->write_state();

        #if WRITE_ENERGY
        system_obj->outFileEnergy << system_obj->compute_energy() << "\n";
        #endif

        // set the flag
        system_obj->print_system = false;

        // job is done, release the lock
        pthread_mutex_unlock(&system_obj->mutex);
        pthread_cond_signal(&system_obj->cond);
    }
}