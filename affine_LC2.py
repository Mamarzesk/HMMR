import os
from functools import reduce

import numpy as np
import torch
from scipy.optimize import differential_evolution

from utils.file_parser import PyMincParser, TagFileParser
from utils.finite_differences import (
    fd_3d_neighbourhood_derivatives,
    fd_3d_volume_derivatives,
    neighbourhood_indices
)
from utils.helpers import (
    compute_vector_determinant,
    create_image_mask,
    normalize_grad,
    transform_affine_3d
)
from utils.logger import LogIO
from utils.similarity import LC2


fixed_file = os.getenv('fixed_file')
moving_file = os.getenv('moving_file')
output_file_name = os.getenv('output_file_name')
tag_file = os.getenv('tag_file')
sigma = float(os.getenv('sigma'))
noise_level = float(os.getenv('noise_level'))
force_rigid = os.getenv('force_rigid').lower() == 'true'
fixed_parser = PyMincParser(fixed_file)
moving_parser = PyMincParser(moving_file)
tag_file_parser = TagFileParser(tag_file)
fixed_tensor = fixed_parser.get_tensor(sigma, True)
moving_tensor = moving_parser.get_tensor(sigma, False, noise_level)
fixed_grad = fd_3d_volume_derivatives(fixed_tensor)[0]
fixed_grad_mag = torch.sqrt(torch.einsum('...i,...i', *2*(fixed_grad,)))
mask = create_image_mask(fixed_tensor, 5)
fixed_grad_mag_bounds = torch.quantile(
    fixed_grad_mag[mask], torch.tensor([0.7, 1.0])
)
denominator_threshold = 1e-8
image_mask = reduce(
    torch.logical_and, [
        mask,
        fixed_grad_mag >= fixed_grad_mag_bounds[0],
        fixed_grad_mag <= fixed_grad_mag_bounds[1]
    ]
)
del fixed_grad_mag
masked_fixed_grad = fixed_grad[image_mask]
del fixed_grad, fixed_tensor
fixed_mask_indices = torch.stack(torch.where(image_mask), dim=-1)
mr_landmarks, us_landmarks = tag_file_parser.extract_landmarks()
us_landmarks_grid = fixed_parser.position_to_grid(
    us_landmarks, torch.tensor([0., 1.])
)
torch.manual_seed(0)
samples_count = 10_000
samples = torch.randint(fixed_mask_indices.shape[0], (samples_count,))


def validate_affine(affine_matrix):
    affine_matrix = torch.tensor(affine_matrix)
    affine_matrix = affine_matrix.reshape(3, 4)
    moved_us_grid = (
        us_landmarks_grid +
        transform_affine_3d(us_landmarks_grid.double(), affine_matrix.double(), force_rigid)
    )
    moved_us_landmarks = fixed_parser.grid_to_position(
        moved_us_grid, torch.tensor([0., 1.])
    )
    landmarks_diff = moved_us_landmarks - mr_landmarks
    return torch.mean(torch.linalg.norm((landmarks_diff), axis=1)).item()


@LogIO
def evaluate(affine_matrix) -> float:
    affine_matrix = torch.tensor(affine_matrix)
    affine_matrix = affine_matrix.reshape(3, 4)
    sampled_grads = masked_fixed_grad[samples]
    sampled_indices = fixed_mask_indices[samples]
    sampled_neighbourhood_indices = neighbourhood_indices(sampled_indices)
    neighbourhoods_position = fixed_parser.compute_positions(
        sampled_neighbourhood_indices
    )
    neighbourhoods_position_grid = fixed_parser.position_to_grid(
        neighbourhoods_position, torch.tensor([0., 1.])
    )
    neighbourhoods_position_grid_shifted = (
        neighbourhoods_position_grid
        + transform_affine_3d(neighbourhoods_position_grid, affine_matrix, force_rigid)
    )
    neighbourhoods_position_shifted = fixed_parser.grid_to_position(
        neighbourhoods_position_grid_shifted, torch.tensor([0., 1.])
    )
    neighbourhoods_shifted = moving_parser.position_to_index(
        neighbourhoods_position_shifted
    )
    masked_moving_grad, masked_moving_hess = fd_3d_neighbourhood_derivatives(
        moving_tensor, neighbourhoods_shifted
    )
    masked_moving_hg = torch.matmul(
        masked_moving_hess, masked_moving_grad[..., None]
    )[..., 0]
    determinant = compute_vector_determinant(masked_moving_grad, masked_moving_hg)
    second_mask = determinant > denominator_threshold
    sampled_grads = normalize_grad(sampled_grads)
    similarity_calculator = LC2(
        masked_moving_grad[second_mask],
        masked_moving_hg[second_mask],
        sampled_grads[second_mask]
    )
    s_map = similarity_calculator.compute_map()
    s_map = s_map[~torch.isnan(s_map)]
    s_map = s_map[~torch.isinf(s_map)]
    f = s_map.mean()
    return -f.detach().numpy()


bounds = 12*[(-0.05, 0.05)]
res = differential_evolution(
    evaluate, bounds=bounds, maxiter=100, popsize=1, polish=False,
    workers=1, tol=0.001, disp=True, atol=0, x0=12*[0.],
)
np.savetxt(
    f'{output_file_name}_affine_LC2_results.csv',
    np.array([
        evaluate.evaluated_outputs,
        [validate_affine(inp) for inp in evaluate.evaluated_inputs]
    ]),
    delimiter=','
)
np.savetxt(
    f'{output_file_name}_affine_LC2_transformation.csv',
    res.x,
    delimiter=','
)
