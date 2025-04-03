import torch
from scipy.ndimage import binary_dilation, binary_erosion


def normalize_hessian(hessian: torch.tensor) -> torch.Tensor:
    hessian_magnitude = torch.sum(hessian ** 2, axis=(-1, -2)) ** 0.5
    hessian_magnitude[hessian_magnitude == 0.0] = 1.0
    return hessian / hessian_magnitude.unsqueeze(-1).unsqueeze(-1)


def normalize_gradient(gradient: torch.tensor) -> torch.Tensor:
    gradient_magnitude = torch.sum(gradient ** 2, axis=-1) ** 0.5
    gradient_magnitude[gradient_magnitude == 0.0] = 1.0
    return gradient / gradient_magnitude.unsqueeze(-1)


def create_image_maske(
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
        points: torch.Tensor, affine_matrix: torch.Tensor
) -> torch.Tensor:
    return (torch.einsum(
        'ij,...j->...i', affine_matrix[:, :-1].double(), points.double()
    ) + affine_matrix[:, -1]).float()


def compute_determinant(
    gradient: torch.Tensor, hessian:torch.Tensor
) -> torch.Tensor:
    dyadic = torch.einsum('...i,...j->...ij',*2*(gradient,))
    xx = torch.sum(hessian * hessian, axis=(-1, -2))
    yy = torch.sum(dyadic * dyadic, axis=(-1, -2))
    xy = torch.sum(hessian * dyadic, axis=(-1, -2))
    return xx*yy - xy**2
