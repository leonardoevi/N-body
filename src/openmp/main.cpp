#include <iostream>
#include <omp.h>

int main() {

    #pragma omp parallel for
    for (int i = 0; i < 10; i++) {
        std::string s = "Hello, World! from thread: " + std::to_string(omp_get_thread_num()) + "\n";
        std::cout << s;
    }

    return EXIT_SUCCESS;
}
