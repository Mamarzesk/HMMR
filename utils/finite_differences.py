from typing import Tuple

import torch
import torch.nn.functional as F


CENTER = 'C'
FORWARD = 'F'
BACKWARD = 'B'
DIRS = CENTER + FORWARD + BACKWARD
KEYS = [
    i+j+k
    for i in DIRS
    for j in DIRS
    for k in DIRS
    if CENTER in i+j+k
]
NEIGHBOURS_COUNT = len(KEYS)
DIR_VALUES = {
    CENTER: torch.tensor([0]),
    FORWARD: torch.tensor([1]),
    BACKWARD: torch.tensor([-1])
}
STEPS = [torch.eye(3, dtype=torch.int)[i] for i in range(3)]


def neighbourhood_indices(indices: torch.Tensor) -> torch.Tensor:
    '''
    indices shape: s*3 (integers)
    '''
    count = indices.shape[0]
    neighbour_indices = indices.repeat([NEIGHBOURS_COUNT, 1])

    for index, key in enumerate(KEYS):
        movement = torch.tensor([0, 0, 0], dtype=torch.int)
        for axis in range(3):
            movement += STEPS[axis] * DIR_VALUES[key[axis]]
        neighbour_indices[count * index:count * (index + 1), :] += movement
    return neighbour_indices


def fd_3d_neighbourhood_derivatives(
    image: torch.Tensor, neighbours: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Computes gradient and Hessian of an image given sampled neighbourhoods.
    '''
    count = neighbours.shape[0] // NEIGHBOURS_COUNT
    neighbours = 2 * neighbours / (torch.tensor(image.shape) - 1) - 1
    adjusted_image = torch.einsum('ijk->kji', image)[None, None, ...]
    adjusted_neighbours = neighbours[None, None, None, ...]
    values = F.grid_sample(
        adjusted_image, adjusted_neighbours, align_corners=True
    )[0, 0, 0, 0]
    neighbours_dict = {}
    for index, key in enumerate(KEYS):
        region_values_list = values[count * index:count * (index + 1)]
        neighbours_dict[key] = region_values_list.reshape(count)

    gradient = torch.zeros([count, 3])
    hessian = torch.zeros([count, 3, 3])

    gradient[..., 0] = 0.5 * (neighbours_dict['FCC'] - neighbours_dict['BCC'])
    gradient[..., 1] = 0.5 * (neighbours_dict['CFC'] - neighbours_dict['CBC'])
    gradient[..., 2] = 0.5 * (neighbours_dict['CCF'] - neighbours_dict['CCB'])

    hessian[..., 0, 0] = (
        - neighbours_dict['CCC'] * 2
        + neighbours_dict['FCC']
        + neighbours_dict['BCC']
    )
    hessian[..., 1, 1] = (
        - neighbours_dict['CCC'] * 2
        + neighbours_dict['CFC']
        + neighbours_dict['CBC']
    )
    hessian[..., 2, 2] = (
        - neighbours_dict['CCC'] * 2
        + neighbours_dict['CCF']
        + neighbours_dict['CCB']
    )
    hessian[..., 0, 1] = 0.25 * (
        + neighbours_dict['FFC']
        + neighbours_dict['BBC']
        - neighbours_dict['FBC']
        - neighbours_dict['BFC']
    )
    hessian[..., 1, 2] = 0.25 * (
        + neighbours_dict['CFF']
        + neighbours_dict['CBB']
        - neighbours_dict['CFB']
        - neighbours_dict['CBF']
    )
    hessian[..., 2, 0] = 0.25 * (
        + neighbours_dict['FCF']
        + neighbours_dict['BCB']
        - neighbours_dict['FCB']
        - neighbours_dict['BCF']
    )

    hessian[..., 1, 0] = hessian[..., 0, 1]
    hessian[..., 2, 1] = hessian[..., 1, 2]
    hessian[..., 0, 2] = hessian[..., 2, 0]

    return gradient, hessian


def fd_3d_volume_derivatives(
    image: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    '''
    Computes gradient and Hessian of the entire image using finite difference.
    '''
    gradient = torch.zeros(torch.tensor(image.shape).tolist() + [3])
    hessian = torch.zeros(torch.tensor(image.shape).tolist() + [3, 3])
    image = torch.nn.functional.pad(image, 6*(1, ), 'constant', 0.)

    gradient[..., 0] = 0.5 * (image[2:, 1:-1, 1:-1] - image[:-2, 1:-1, 1:-1])
    gradient[..., 1] = 0.5 * (image[1:-1, 2:, 1:-1] - image[1:-1, :-2, 1:-1])
    gradient[..., 2] = 0.5 * (image[1:-1, 1:-1, 2:] - image[1:-1, 1:-1, :-2])

    hessian[..., 0, 0] = (
        - image[1:-1, 1:-1, 1:-1] * 2
        + image[2:, 1:-1, 1:-1]
        + image[:-2, 1:-1, 1:-1]
    )
    hessian[..., 1, 1] = (
        - image[1:-1, 1:-1, 1:-1] * 2
        + image[1:-1, 2:, 1:-1]
        + image[1:-1, :-2, 1:-1]
    )
    hessian[..., 2, 2] = (
        - image[1:-1, 1:-1, 1:-1] * 2
        + image[1:-1, 1:-1, 2:]
        + image[1:-1, 1:-1, :-2]
    )

    hessian[..., 0, 1] = 0.25 * (
        + image[2:, 2:, 1:-1]
        + image[:-2, :-2, 1:-1]
        - image[2:, :-2, 1:-1]
        - image[:-2, 2:, 1:-1]
    )
    hessian[..., 1, 2] = 0.25 * (
        + image[1:-1, 2:, 2:]
        + image[1:-1, :-2, :-2]
        - image[1:-1, 2:, :-2]
        - image[1:-1, :-2, 2:]
    )
    hessian[..., 2, 0] = 0.25 * (
        + image[2:, 1:-1, 2:]
        + image[:-2, 1:-1, :-2]
        - image[2:, 1:-1, :-2]
        - image[:-2, 1:-1, 2:]
    )

    hessian[..., 1, 0] = hessian[..., 0, 1]
    hessian[..., 2, 1] = hessian[..., 1, 2]
    hessian[..., 0, 2] = hessian[..., 2, 0]

    return gradient, hessian
