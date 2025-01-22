#ifndef SYSTEMFE_H
#define SYSTEMFE_H

#include "System.h"
#include "system_friend.h"

class SystemFE : public System {

public:

    SystemFE(const unsigned int n_particles_, const double t_max_, const double dt_,
                    std::unique_ptr<double[]> pos_,
                    std::unique_ptr<double[]> vel_,
                    std::unique_ptr<double[]> mass_)
        : System(n_particles_, t_max_, dt_, std::move(pos_), std::move(vel_), std::move(mass_)) {}

    /**
     * Simulates the evolution of the system using FORWARD EULER integration method.
     * @param out_file_name name of the output file to write
     */
    void simulate(const std::string &out_file_name) override;

};


#endif
