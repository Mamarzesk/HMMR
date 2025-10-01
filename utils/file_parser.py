from typing import Tuple

import numpy as np
from pyminc.volumes.factory import volumeFromFile, mincVolume
from scipy.ndimage import gaussian_filter
import torch


class PyMincParser:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.image = self.parse()
        self.dims = self.get_dims_ordering()
        self.origin = torch.tensor(self.image.getStarts())[self.dims]
        self.size = torch.tensor(self.image.getSizes())[self.dims]
        self.spacing = torch.tensor(self.image.getSeparations())[self.dims]
        self.end = torch.tensor(
            [
                self.origin[i] + (self.size[i]-1) * self.spacing[i]
                for i in range(self.dims.shape[0])
            ]
        )

    def parse(self) -> mincVolume:
        return volumeFromFile(self.file_path)

    def get_dims_ordering(self) -> np.ndarray:
        ordering = np.argsort(self.image.getDimensionNames())
        return ordering

    def compute_positions(self, indices: torch.Tensor) -> torch.Tensor:
        return self.origin + self.spacing * indices

    def position_to_index(self, positions: torch.Tensor) -> torch.Tensor:
        '''
        Maps positions to indices, not restricted to integers.
        '''
        return (positions - self.origin) / self.spacing

    def position_to_grid(
        self, positions: torch.Tensor, interval: torch.Tensor
    ) -> torch.Tensor:
        '''Maps position coordinates to values between a and b'''
        a, b = interval
        return (b - a)*(positions-self.origin)/(self.end-self.origin) + a

    def grid_to_position(
        self, grid: torch.Tensor, interval: torch.Tensor
    ) -> torch.Tensor:
        '''Maps values between a and b to position coordinates'''
        a, b = interval
        return (self.end-self.origin)*(grid - a)/(b - a) + self.origin

    def get_tensor(
        self, scale: float, remove_background: bool
    ) -> torch.Tensor:
        array = np.array(self.image.getdata().tolist())
        if remove_background:
            bg = array == 0.0
        sigma = (scale / torch.abs(self.spacing)).numpy()
        array = gaussian_filter(array, sigma=sigma)
        if remove_background:
            array[bg] = 0.0
        tensor = torch.tensor(array, dtype=torch.float32)
        return torch.permute(tensor, tuple(self.dims))
    
    def get_bspline_grid(self, node_spacing: float) -> Tuple[int, ...]:
        grid_size = tuple(
            int(length * torch.abs(spacing) // node_spacing)
            for length, spacing in zip(self.size, self.spacing)
        )
        return grid_size


class TagFileParser:
    def __init__(self, file_name: str) -> None:
        self.file_name = file_name
        self.src_landmarks, self.trg_landmarks = self.extract_landmarks()

    def extract_landmarks(self) -> Tuple[torch.Tensor, torch.Tensor]:
        src_landmarks_list = []
        trg_landmarks_list = []
        with open(self.file_name) as f:
            data = f.readlines()
        stripped_data = [item.strip() for item in data]
        for index, line in enumerate(stripped_data):
            if line.startswith('Points'):
                break
        landmarks = stripped_data[index + 1:]
        for landmark in landmarks:
            srcx, srcy, srcz, trgx, trgy, trgz, _ = landmark.split()
            src_landmarks_list.append([float(srcx), float(srcy), float(srcz)])
            trg_landmarks_list.append([float(trgx), float(trgy), float(trgz)])
        src_landmarks = torch.tensor(src_landmarks_list)
        trg_landmarks = torch.tensor(trg_landmarks_list)
        return src_landmarks, trg_landmarks
