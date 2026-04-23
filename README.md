# HMMR

This repository contains the code for **Hessian-based multimodal registration**. The method is explained in [this](https://link.springer.com/chapter/10.1007/978-3-031-47425-5_23) paper.

To run the code, these environment variables must be set in advance:
- fixed_file (.mnc)
- moving_file (.mnc)
- tag_file (.tag)
- output_file_name
- sigma
- force_rigid

Affine registration can be performed using this command:
```console
 python registration.py
```
For nonlinear registration, additional environment variables must be set:
- bspline_spacing
- dynamic_sampling
- affine_matrix_file (.csv)

Nonlinear registration can be performed using this command:
```console
 python registration_nonlinear.py
```
