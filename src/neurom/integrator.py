import torch
import torch.nn as nn


class Integrator:
    def integrate(self, integrand, measure):
        """
        Args:
            integrand: The field to integrate (N_e, N_q)
            measure: The measure to weight the integrand (N_e, N_q)
        """
        return torch.einsum("eq...,eq...->", integrand, measure)
