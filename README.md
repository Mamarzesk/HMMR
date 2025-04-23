# HMMR

This repository contains the code for **Hessian-based multimodal registration**. The method is explained in [this](https://link.springer.com/chapter/10.1007/978-3-031-47425-5_23) paper.

To run the code, these environment variables must be set in advance:
- fixed_file (.mnc)
- moving_file (.mnc)
- tag_file (.tag)
- output_file_name
- sigma

Affine registration can be performed using this command:
```console
 python registration.py
```

|                | BITE<br>pre-| BITE<br>post-| RESECT<br>pre-          | RESECT<br>post-         |  
|----------------|-------------|--------------|-------------------------|-------------------------|
|sigma           |1.5          |1.5           |1.0                      |1.0                      |
|Cases to exclude|1            |14            |11                       |5, 13, 26                |
|fixed_file      |ReconUS      |US3DT         |US_test                  |US_post_test             |
|moving_file     |mr           |MR            |FLAIR_test<br>T1_test_reg|FLAIR_test<br>T1_test_reg|
|tag_file        |X_all        |Tags          |CaseX-MRI-beforeUS       |CaseX-MRI-afterUS        |
=======
