import os
from functools import reduce

import numpy as np
import torch
from torch import optim
from torch_cubic_spline_grids import CubicBSplineGrid3d

from utils.file_parser import PyMincParser, TagFileParser
from utils.finite_differences import (
    fd_3d_neighbourhood_derivatives,
    fd_3d_volume_derivatives,
    neighbourhood_indices
)
from utils.helpers import (
    compute_determinant,
    create_image_mask,
    normalize_hessian,
    transform_affine_3d
)
from utils.logger import LogIO
from utils.similarity import HessianSimilarity


def compute_deformation_jacobian(
        points: torch.Tensor, transformed_points: torch.Tensor
) -> torch.Tensor:
    '''
    points shape: ..., n
    transformed_points shape: ..., m
    output shape: ..., m, n
    '''
    deriv_list = []
    output_spatial_dims = transformed_points.shape[-1]
    for dim in range(output_spatial_dims):
        deriv = torch.autograd.grad(
            torch.sum(transformed_points[..., dim]),
            points, retain_graph=True, create_graph=True
        )[0][..., None]
        deriv_list.append(deriv)
    return torch.cat(deriv_list, dim=-1)


fixed_file = os.getenv('fixed_file')
moving_file = os.getenv('moving_file')
output_file_name = os.getenv('output_file_name')
tag_file = os.getenv('tag_file')
sigma = float(os.getenv('sigma'))
affine_file = os.getenv('affine_matrix_file')
affine_matrix = np.genfromtxt(affine_file) if affine_file else np.zeros((3, 4))
fixed_parser = PyMincParser(fixed_file)
moving_parser = PyMincParser(moving_file)
tag_file_parser = TagFileParser(tag_file)
affine_matrix = torch.tensor(affine_matrix.reshape(3, 4))
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
    torch.logical_and, [
        mask,
        fixed_determinant >= determinant_bounds[0],
        fixed_determinant <= determinant_bounds[1]
    ]
)
del fixed_determinant
masked_fixed_hess = fixed_hess[image_mask]
masked_fixed_grad = fixed_grad[image_mask]
del fixed_grad, fixed_hess, fixed_tensor
fixed_mask_indices = torch.stack(torch.where(image_mask), dim=-1)
mr_landmarks, us_landmarks = tag_file_parser.extract_landmarks()
us_landmarks_grid = fixed_parser.position_to_grid(
    us_landmarks, torch.tensor([0., 1.])
)
samples_count = 20_000


def validate_nonlinear(deformation):
    moved_us_grid = (
        us_landmarks_grid
        + deformation(us_landmarks_grid)
        + transform_affine_3d(
            us_landmarks_grid.double(), affine_matrix.double()
        )
    )
    moved_us_landmarks = fixed_parser.grid_to_position(
        moved_us_grid, torch.tensor([0., 1.])
    )
    landmarks_diff = moved_us_landmarks - mr_landmarks
    return torch.linalg.norm((landmarks_diff), axis=1).mean().detach().item()


@LogIO
def evaluate(deformation) -> torch.Tensor:
    samples = torch.randint(fixed_mask_indices.shape[0], (samples_count,))
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
        neighbourhoods_position, torch.tensor([0., 1.])
    )
    mask_position_grid = fixed_parser.position_to_grid(
        mask_position, torch.tensor([0., 1.])
    )
    neighbourhoods_position_grid_shifted = (
        neighbourhoods_position_grid
        + deformation(neighbourhoods_position_grid)
        + transform_affine_3d(neighbourhoods_position_grid, affine_matrix)
    )
    mask_position_grid_shifted = (
        mask_position_grid
        + deformation(mask_position_grid)
        + transform_affine_3d(mask_position_grid, affine_matrix)
    )
    neighbourhoods_position_shifted = fixed_parser.grid_to_position(
        neighbourhoods_position_grid_shifted, torch.tensor([0., 1.])
    )
    deformation_jacobian = compute_deformation_jacobian(
        mask_position_grid, mask_position_grid_shifted
    )
    deformation_jacobian -= torch.tile(torch.eye(3), (samples_count, 1, 1))
    reg = 100 * torch.mean(deformation_jacobian ** 2)
    neighbourhoods_shifted = moving_parser.position_to_index(
        neighbourhoods_position_shifted
    )
    _, deformed_moving_hess = fd_3d_neighbourhood_derivatives(
        moving_tensor, neighbourhoods_shifted
    )
    deformed_moving_hess_mag = torch.sum(
        deformed_moving_hess ** 2, axis=(-1, -2)
    )
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


bspline = CubicBSplineGrid3d(resolution=3 * (21,), n_channels=3)
learning_rate = 0.0025
optimizer = optim.Adam(bspline.parameters(), lr=learning_rate,)
num_iterations = 10
for iteration in range(num_iterations):
    optimizer.zero_grad()
    loss, reg = evaluate(bspline)
    (loss-reg).backward()
    optimizer.step()


bspline_coefficients = []
bspline_coefficients.append(next(bspline.parameters()).data.numpy().flatten())


print(evaluate.evaluated_outputs)
print([validate_nonlinear(inp) for inp in evaluate.evaluated_inputs])
np.savetxt(
    f'{output_file_name}_nonlinear_results.csv',
    np.array([
        [item[0] for item in evaluate.evaluated_outputs],
        [item[1] for item in evaluate.evaluated_outputs],
        [validate_nonlinear(inp) for inp in evaluate.evaluated_inputs]
    ]),
    delimiter=','
)
np.savetxt(
    f'{output_file_name}_nonlinear_transformation.csv',
    np.array(bspline_coefficients),
    delimiter=','
)
