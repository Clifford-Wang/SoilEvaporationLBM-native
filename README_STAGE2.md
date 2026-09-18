# Stage-2 CUDA validation

This stage keeps the validated Stage-1 geometry/topology logic and adds native CUDA execution for:

- initial rho import from the phase field;
- equilibrium f/F initialization;
- Peng-Robinson EOS;
- Shan-Chen pseudopotential psi;
- initial top dry boundary reconstruction;
- macro-variable reconstruction;
- beta-corrected fluid-fluid interaction force;
- geometry-based adsorption force.

It intentionally stops before MRT collision and streaming. The program validates GPU rho/pressure/psi against independent CPU calculations before accepting the stage.
