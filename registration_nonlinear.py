import os
from functools import reduce

import numpy as np
import torch
from torch import optim
from torch_cubic_spline_grids import CubicBSplineGrid3d
from tqdm import tqdm

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
from utils.regularizer import compute_deformation_jacobian
from utils.similarity import HessianSimilarity


def register_nonlinear(
    fixed,
    moving,
    tag,
    output,
    sigma,
    bspline_spacing, 
    rigid_file = None, 
    dynamic_sampling = False,
    force_rigid = False,
):
    rigid_matrix = np.genfromtxt(rigid_file) if rigid_file else np.zeros((3, 4))
    fixed_parser = PyMincParser(fixed)
    moving_parser = PyMincParser(moving)
    tag_file_parser = TagFileParser(tag)
    rigid_matrix = torch.tensor(rigid_matrix.reshape(3, 4))
    fixed_tensor = fixed_parser.get_tensor(sigma, True)
    moving_tensor = moving_parser.get_tensor(sigma, False)
    fixed_grad, fixed_hess = fd_3d_volume_derivatives(fixed_tensor)
    mask = create_image_mask(fixed_tensor, 5)
    fixed_determinant = compute_determinant(fixed_grad, fixed_hess)
    determinant_bounds = torch.quantile(fixed_determinant[mask], torch.tensor([0.5, 1.0]))
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
    static_samples = torch.randint(fixed_mask_indices.shape[0], (samples_count,))


    def validate_nonlinear(deformation):
        moved_us_grid = (
            us_landmarks_grid
            + deformation(us_landmarks_grid)
            + transform_affine_3d(
                us_landmarks_grid.double(), rigid_matrix.double(), force_rigid
            )
        )
        moved_us_landmarks = fixed_parser.grid_to_position(
            moved_us_grid, torch.tensor([0.0, 1.0])
        )
        landmarks_diff = moved_us_landmarks - mr_landmarks
        return torch.linalg.norm((landmarks_diff), axis=1).mean().detach().item()


    @LogIO
    def evaluate(deformation) -> torch.Tensor:
        if dynamic_sampling:
            samples = torch.randint(fixed_mask_indices.shape[0], (samples_count,))
        else:
            samples = static_samples
        sampled_hess = masked_fixed_hess[samples]
        sampled_grads = masked_fixed_grad[samples]
        sampled_indices = fixed_mask_indices[samples]
        sampled_neighbourhood_indices = neighbourhood_indices(sampled_indices)
        neighbourhoods_position = fixed_parser.compute_positions(
            sampled_neighbourhood_indices
        )
        mask_position = fixed_parser.compute_positions(sampled_indices)
        mask_position.requires_grad_()
        neighbourhoods_position_grid = fixed_parser.position_to_grid(
            neighbourhoods_position, torch.tensor([0.0, 1.0])
        )
        mask_position_grid = fixed_parser.position_to_grid(
            mask_position, torch.tensor([0.0, 1.0])
        )
        neighbourhoods_position_grid_shifted = (
            neighbourhoods_position_grid
            + deformation(neighbourhoods_position_grid)
            + transform_affine_3d(neighbourhoods_position_grid, rigid_matrix, force_rigid)
        )
        nonlinear_deformation = deformation(mask_position_grid)
        neighbourhoods_position_shifted = fixed_parser.grid_to_position(
            neighbourhoods_position_grid_shifted, torch.tensor([0.0, 1.0])
        )
        deformation_jacobian = compute_deformation_jacobian(
            # mask_position_grid, nonlinear_deformation
            mask_position_grid,
            nonlinear_deformation + mask_position_grid,
        )
        # reg = 100 * torch.mean(deformation_jacobian ** 2)
        reg = 100 * torch.mean((torch.det(deformation_jacobian) - 1) ** 2)
        neighbourhoods_shifted = moving_parser.position_to_index(
            neighbourhoods_position_shifted
        )
        _, deformed_moving_hess = fd_3d_neighbourhood_derivatives(
            moving_tensor, neighbourhoods_shifted
        )
        deformed_moving_hess_mag = torch.sum(deformed_moving_hess**2, axis=(-1, -2))
        second_mask = deformed_moving_hess_mag > denominator_threshold
        sampled_hess = sampled_hess[second_mask]
        sampled_grads = sampled_grads[second_mask]
        deformed_moving_hess = normalize_hessian(deformed_moving_hess[second_mask])
        similarity_calculator = HessianSimilarity(
            sampled_hess, sampled_grads, deformed_moving_hess
        )
        s_map = similarity_calculator.compute_map()
        s_map = s_map[~torch.isnan(s_map)]
        s_map = s_map[~torch.isinf(s_map)]
        f = s_map[s_map > 0.0].mean()
        return -f, -reg


    print(f"bspline_spacing: {bspline_spacing}, patient: {output}")

    bspline = CubicBSplineGrid3d(
        resolution=moving_parser.get_bspline_grid(bspline_spacing), n_channels=3
    )
    learning_rate = 0.0025
    optimizer = optim.Adam(bspline.parameters(), lr=learning_rate)
    num_iterations = 50
    for iteration in tqdm(range(num_iterations)):
        optimizer.zero_grad()
        loss, reg = evaluate(bspline)
        (loss - reg).backward()
        optimizer.step()


    bspline_coefficients = []
    bspline_coefficients.append(next(bspline.parameters()).data.numpy().flatten())


    np.savetxt(
        os.path.join(output, 'nonlinear_results.csv'),
        np.array(
            [
                [item[0] for item in evaluate.evaluated_outputs],
                [item[1] for item in evaluate.evaluated_outputs],
                [validate_nonlinear(inp) for inp in evaluate.evaluated_inputs],
            ]
        ),
        delimiter=",",
    )
    np.savetxt(
        os.path.join(output, 'nonlinear_transformation.csv'),
        np.array(bspline_coefficients),
        delimiter=",",
    )
