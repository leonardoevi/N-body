A few small mistakes marked with @note in the code, but nothing serious. Its a pity that tahre is no a makefile or similar to compile the openGL renderer standalone. I do not have a NVIDIA GPU so I cannot use the current Cmke file directly.

The codi is clean, it could hav ebeen organised in am more profesional way, with header files in an include directory. Nice the idea of using openGL.  The choice of the number of spece dimension could have been made via templates, with a possible little gain in efficiency (maybe). 

A note, in the report you say the forward Euler is conditionally stable, which is true, but you did not say the same for leap-frog. Also leap frog is conditinally stable, with a stability condition similar to FE. If you want unconditionally stable schemes you need to move to implicit schemes,

Another note, your implementation is fine, but cannot scale too much, you are storing the full matrix, which grows as $N^2$. You have focused your work on an efficient CUDA implementation of the basic scheme, which is fine, but you could have mentioned metodologies to cover very large problems (in a galaxy we have billions of stars...).


