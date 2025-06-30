import os
from functools import reduce

import numpy as np
import torch
from scipy.optimize import differential_evolution

from utils.file_parser import PyMincParser, TagFileParser
from utils.finite_differences import (
    fd_3d_neighbourhood_derivatives,
    fd_3d_volume_derivatives,
    neighbourhood_indices,
)
from utils.helpers import (
    compute_determinant,
    create_image_mask,
    normalize_hessian,
    transform_affine_3d,
)
from utils.logger import LogIO
from utils.similarity import HessianSimilarity


def register(
    fixed,
    moving,
    tag,
    output,
    sigma,
    force_rigid=False,
):
    fixed_parser = PyMincParser(fixed)
    moving_parser = PyMincParser(moving)
    tag_file_parser = TagFileParser(tag)
    fixed_tensor = fixed_parser.get_tensor(sigma, True)
    moving_tensor = moving_parser.get_tensor(sigma, False)
    fixed_grad, fixed_hess = fd_3d_volume_derivatives(fixed_tensor)
    mask = create_image_mask(fixed_tensor, 5)
    fixed_determinant = compute_determinant(fixed_grad, fixed_hess)
    determinant_bounds = torch.quantile(
        fixed_determinant[mask], torch.tensor([0.5, 1.0])
    )
    denominator_threshold = 1e-8
    image_mask = reduce(
        torch.logical_and,
        [
            mask,
            fixed_determinant >= determinant_bounds[0],
            fixed_determinant <= determinant_bounds[1],
        ],
    )
    del fixed_determinant
    masked_fixed_hess = fixed_hess[image_mask]
    masked_fixed_grad = fixed_grad[image_mask]
    del fixed_grad, fixed_hess, fixed_tensor
    fixed_mask_indices = torch.stack(torch.where(image_mask), dim=-1)
    mr_landmarks, us_landmarks = tag_file_parser.extract_landmarks()
    us_landmarks_grid = fixed_parser.position_to_grid(
        us_landmarks, torch.tensor([0.0, 1.0])
    )
    samples_count = 10_000
    samples = torch.randint(fixed_mask_indices.shape[0], (samples_count,))

    bounds = 12 * [(-0.05, 0.05)]

    def validate_affine(
        affine_matrix,
    ):
        affine_matrix = torch.tensor(affine_matrix)
        affine_matrix = affine_matrix.reshape(3, 4)
        moved_us_grid = us_landmarks_grid + transform_affine_3d(
            us_landmarks_grid.double(), affine_matrix.double(), force_rigid
        )
        moved_us_landmarks = fixed_parser.grid_to_position(
            moved_us_grid, torch.tensor([0.0, 1.0])
        )
        landmarks_diff = moved_us_landmarks - mr_landmarks
        return torch.mean(torch.linalg.norm((landmarks_diff), axis=1)).item()

    @LogIO
    def evaluate(
        affine_matrix,
    ) -> float:
        affine_matrix = torch.tensor(affine_matrix)
        affine_matrix = affine_matrix.reshape(3, 4)
        sampled_hess = masked_fixed_hess[samples]
        sampled_grads = masked_fixed_grad[samples]
        sampled_indices = fixed_mask_indices[samples]
        sampled_neighbourhood_indices = neighbourhood_indices(sampled_indices)
        neighbourhoods_position = fixed_parser.compute_positions(
            sampled_neighbourhood_indices
        )
        neighbourhoods_position_grid = fixed_parser.position_to_grid(
            neighbourhoods_position, torch.tensor([0.0, 1.0])
        )
        neighbourhoods_position_grid_shifted = (
            neighbourhoods_position_grid
            + transform_affine_3d(
                neighbourhoods_position_grid, affine_matrix, force_rigid
            )
        )
        neighbourhoods_position_shifted = fixed_parser.grid_to_position(
            neighbourhoods_position_grid_shifted, torch.tensor([0.0, 1.0])
        )
        neighbourhoods_shifted = moving_parser.position_to_index(
            neighbourhoods_position_shifted
        )
        _, deformed_moving_hess = fd_3d_neighbourhood_derivatives(
            moving_tensor, neighbourhoods_shifted
        )
        masked_moving_hess_mag = torch.sum(deformed_moving_hess**2, axis=(-1, -2))
        second_mask = masked_moving_hess_mag > denominator_threshold
        sampled_hess = sampled_hess[second_mask]
        sampled_grads = sampled_grads[second_mask]
        sampled_hess = normalize_hessian(sampled_hess)
        deformed_moving_hess = normalize_hessian(deformed_moving_hess[second_mask])
        similarity_calculator = HessianSimilarity(
            sampled_hess, sampled_grads, deformed_moving_hess
        )
        s_map = similarity_calculator.compute_map()
        s_map = s_map[~torch.isnan(s_map)]
        s_map = s_map[~torch.isinf(s_map)]
        f = s_map.mean()
        return -f.detach().numpy()

    res = differential_evolution(
        evaluate,
        bounds=bounds,
        maxiter=100,
        popsize=1,
        polish=False,
        workers=1,
        tol=0.001,
        disp=True,
        atol=0,
        x0=12 * [0.0],
    )
    np.savetxt(
        os.path.join(output, "affine_results.csv"),
        np.array(
            [
                evaluate.evaluated_outputs,
                [validate_affine(inp) for inp in evaluate.evaluated_inputs],
            ]
        ),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(output, "affine_transformations.csv"), res.x, delimiter=","
    )


def register_from_env():
    fixed_file = os.getenv("fixed_file")
    moving_file = os.getenv("moving_file")
    tag_file = os.getenv("tag_file")
    output_file_name = os.getenv("output_file_name")
    _sigma = os.getenv("sigma")
    if _sigma is None:
        raise RuntimeError("sigma environment not defined")
    sigma = float(_sigma)
    force_rigid = os.getenv("force_rigid").lower() == "true"
    register(fixed_file, moving_file, tag_file, output_file_name, sigma, force_rigid)
