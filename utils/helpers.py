import torch
from scipy.ndimage import binary_dilation, binary_erosion


def normalize_hessian(hessian: torch.tensor) -> torch.Tensor:
    hessian_magnitude = torch.sum(hessian ** 2, axis=(-1, -2)) ** 0.5
    hessian_magnitude[hessian_magnitude == 0.0] = 1.0
    return hessian / hessian_magnitude.unsqueeze(-1).unsqueeze(-1)


def normalize_grad(grad: torch.tensor) -> torch.tensor:
    grad_magnitude = torch.sum(grad ** 2, axis=-1) ** 0.5
    grad_magnitude[grad_magnitude == 0.0] = 1.0
    return grad / grad_magnitude.unsqueeze(-1)


def create_image_mask(
    image: torch.tensor, dilation_size: int
) -> torch.Tensor:
    bg_tensor = image == 0.0
    bg = bg_tensor.numpy()
    hole_size = 5
    bg = binary_erosion(bg, iterations=hole_size)
    bg = binary_dilation(bg, iterations=hole_size)
    bg = binary_dilation(bg, iterations=dilation_size)
    return torch.tensor(~bg)


def transform_affine_3d(
        points: torch.Tensor,
        affine_matrix: torch.Tensor,
        force_rigid: bool = False
) -> torch.Tensor:
    if not force_rigid:
        transformed_points = torch.einsum(
            'ij,...j->...i', affine_matrix[:, :-1].double(), points.double()
        )
        return (transformed_points + affine_matrix[:, -1]).float()

    u, _, v = torch.linalg.svd(affine_matrix[:, :-1] + torch.eye(3))
    if torch.linalg.det(u @ v) < 0:
        u[:, -1] *= -1
    rotation_matrix = u @ v - torch.eye(3)
    rotation_matrix = rotation_matrix.double()
    transformed_points = torch.einsum(
        'ij,...j->...i', rotation_matrix, points.double()
    )
    return (transformed_points + affine_matrix[:, -1]).float()


def compute_determinant(
    gradient: torch.Tensor, hessian: torch.Tensor
) -> torch.Tensor:
    dyadic = torch.einsum('...i,...j->...ij', *2*(gradient,))
    xx = torch.sum(hessian * hessian, axis=(-1, -2))
    yy = torch.sum(dyadic * dyadic, axis=(-1, -2))
    xy = torch.sum(hessian * dyadic, axis=(-1, -2))
    return xx*yy - xy**2


def compute_vector_determinant(
    vector1: torch.Tensor, vector2:torch.Tensor
) -> torch.Tensor:
    xx = torch.sum(vector1 * vector1, axis=-1)
    yy = torch.sum(vector2 * vector2, axis=-1)
    xy = torch.sum(vector1 * vector2, axis=-1)
    return xx*yy - xy**2
