# SoilEvaporationLBM Native Stage-1

This is the first validation stage of the native C++/CUDA migration.

Stage-1 deliberately does **not** contain MRT, Shan-Chen, EOS, or CUDA kernels. It validates the data/topology layer first:

- geometry TXT is interpreted with the same Fortran-order reshape as the Python reference;
- `pore_value=0` means pore and any other value means solid;
- phase values `>0.5` are liquid;
- the top buffer occupies `k=nz_geo...nz_geo+n_buffer-1` and is fluid;
- `n_fluid = n_real_pore + nx*ny*n_buffer`;
- `fluid_xyz` follows NumPy `argwhere` ordering for shape `(nx,ny,nz)`;
- X/Y are periodic and Z is non-periodic;
- pull-neighbor out-of-domain = -2, solid = -1;
- fluid-fluid neighbor out-of-domain = -1 with adsorption flag 0;
- solid fluid-fluid neighbor = -1 with adsorption flag 1.

The executable prints node counts and deterministic hashes for:
`fluid_xyz`, `grid_to_idx`, `pull_nb_list`, `is_solid_nb`, `ff_nb_list`, and `ads_solid_nb`.

The Windows Release executable is built by GitHub Actions. The final scientific release will later be C++/CUDA and will not distribute source code.
