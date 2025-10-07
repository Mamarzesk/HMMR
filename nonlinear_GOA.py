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
    neighbourhood_indices
)
from utils.helpers import (
    create_image_mask,
    normalize_grad,
    transform_affine_3d
)
from utils.logger import LogIO
from utils.regularizer import compute_deformation_jacobian
from utils.similarity import GradientSimilarity


fixed_file = os.getenv('fixed_file')
moving_file = os.getenv('moving_file')
output_file_name = os.getenv('output_file_name')
tag_file = os.getenv('tag_file')
sigma = float(os.getenv('sigma'))
bspline_spacign = float(os.getenv('bspline_spacing'))
force_rigid = os.getenv('force_rigid').lower() == 'true'
dynamic_sampling = os.getenv('dynamic_sampling').lower() == 'true'
affine_file = os.getenv('affine_matrix_file')
affine_matrix = np.genfromtxt(affine_file) if affine_file else np.zeros((3, 4))
fixed_parser = PyMincParser(fixed_file)
moving_parser = PyMincParser(moving_file)
tag_file_parser = TagFileParser(tag_file)
affine_matrix = torch.tensor(affine_matrix.reshape(3, 4))
fixed_tensor = fixed_parser.get_tensor(sigma, True)
moving_tensor = moving_parser.get_tensor(sigma, False)
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
static_samples = torch.randint(fixed_mask_indices.shape[0], (samples_count,))


def validate_nonlinear(deformation):
    moved_us_grid = (
        us_landmarks_grid
        + deformation(us_landmarks_grid)
        + transform_affine_3d(
            us_landmarks_grid.double(),
            affine_matrix.double(),
            force_rigid
        )
    )
    moved_us_landmarks = fixed_parser.grid_to_position(
        moved_us_grid, torch.tensor([0., 1.])
    )
    landmarks_diff = moved_us_landmarks - mr_landmarks
    return torch.linalg.norm((landmarks_diff), axis=1).mean().detach().item()


@LogIO
def evaluate(deformation) -> torch.Tensor:
    if dynamic_sampling:
        samples = torch.randint(fixed_mask_indices.shape[0], (samples_count,))
    else:
        samples = static_samples
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
        + transform_affine_3d(
            neighbourhoods_position_grid, affine_matrix, force_rigid
        )
    )
    nonlinear_deformation = deformation(mask_position_grid)
    neighbourhoods_position_shifted = fixed_parser.grid_to_position(
        neighbourhoods_position_grid_shifted, torch.tensor([0., 1.])
    )
    deformation_jacobian = compute_deformation_jacobian(
        # mask_position_grid, nonlinear_deformation
        mask_position_grid, nonlinear_deformation + mask_position_grid
    )
    # reg = 100 * torch.mean(deformation_jacobian ** 2)
    reg = 100 * torch.mean((torch.det(deformation_jacobian) - 1) ** 2)
    neighbourhoods_shifted = moving_parser.position_to_index(
        neighbourhoods_position_shifted
    )
    masked_moving_grad = fd_3d_neighbourhood_derivatives(
        moving_tensor, neighbourhoods_shifted
    )[0]
    determinant = torch.einsum('...i,...i', *2*(masked_moving_grad,))
    second_mask = determinant > denominator_threshold
    sampled_grads = normalize_grad(sampled_grads)
    similarity_calculator = GradientSimilarity(
        masked_moving_grad[second_mask],
        sampled_grads[second_mask]
    )
    s_map = similarity_calculator.compute_map()
    s_map = s_map[~torch.isnan(s_map)]
    s_map = s_map[~torch.isinf(s_map)]
    f = s_map[s_map > 0.0].mean()
    return -f, -reg


bspline = CubicBSplineGrid3d(
    resolution=moving_parser.get_bspline_grid(bspline_spacign), n_channels=3
)
learning_rate = 0.0025
optimizer = optim.Adam(bspline.parameters(), lr=learning_rate)
num_iterations = 50
for iteration in tqdm(range(num_iterations)):
    optimizer.zero_grad()
    loss, reg = evaluate(bspline)
    (loss-reg).backward()
    optimizer.step()


bspline_coefficients = []
bspline_coefficients.append(next(bspline.parameters()).data.numpy().flatten())


np.savetxt(
    f'{output_file_name}_nonlinear_GOA_results.csv',
    np.array([
        [item[0] for item in evaluate.evaluated_outputs],
        [item[1] for item in evaluate.evaluated_outputs],
        [validate_nonlinear(inp) for inp in evaluate.evaluated_inputs]
    ]),
    delimiter=','
)
np.savetxt(
    f'{output_file_name}_nonlinear_GOA_transformation.csv',
    np.array(bspline_coefficients),
    delimiter=','
)
