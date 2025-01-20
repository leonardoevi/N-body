#ifndef SYSTEMLF_H
#define SYSTEMLF_H

#include "System.h"
#include "system_friend.h"

class SystemLF : public System {

public:

    SystemLF(const unsigned int n_particles_, const double t_max_, const double dt_,
                    std::unique_ptr<double[]> pos_,
                    std::unique_ptr<double[]> vel_,
                    std::unique_ptr<double[]> mass_)
        : System(n_particles_, t_max_, dt_, std::move(pos_), std::move(vel_), std::move(mass_)) {}

    void simulate (const std::string &out_file_name) override;
};


#endif //SYSTEMLF_H
