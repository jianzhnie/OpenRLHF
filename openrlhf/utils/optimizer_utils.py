import torch
from typing import Union, List, Optional
from torch.nn.utils import clip_grad_value_, clip_grad_norm_


class GradientClipper:
    """
    A class to apply gradient clipping techniques to model parameters.
    Supports clipping by norm (L2, L1) or by value (clipping gradients to a maximum value).
    """

    def __init__(
        self,
        max_grad_norm: float = 1.0,
        max_grad_value: Optional[float] = None,
        clip_type: str = "norm",
        error_if_nonfinite: bool = True,
    ):
        """
        Initializes the gradient clipper.

        Args:
            max_grad_norm (float): Maximum gradient norm for clipping (used when `clip_type` is 'norm').
            max_grad_value (Optional[float]): Maximum absolute value for gradient clipping (used when `clip_type` is 'value').
            clip_type (str): Type of clipping, either 'norm' (based on gradient norm) or 'value' (based on gradient value).
            error_if_nonfinite (bool): If True, raises an error if any gradient has a non-finite value.
        """
        self.max_grad_norm = max_grad_norm
        self.max_grad_value = max_grad_value
        self.clip_type = clip_type
        self.error_if_nonfinite = error_if_nonfinite

    @torch.no_grad()
    def clip_norm(
        self, parameters: Union[torch.Tensor, List[torch.Tensor]], max_norm: float, norm_type: float = 2.0
    ) -> float:
        """
        Applies gradient clipping based on the Lp-norm of the gradients.

        Args:
            parameters (Union[torch.Tensor, List[torch.Tensor]]): Model parameters whose gradients need to be clipped.
            max_norm (float): The maximum norm value the gradients are allowed to have.
            norm_type (float): The type of norm to use, e.g., 2.0 for L2 norm, 1.0 for L1 norm.

        Returns:
            float: The total norm of the gradients before clipping.
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]  # Convert a single tensor to a list for uniform processing.

        # Calculate the total norm of all gradients
        total_norm = clip_grad_norm_(parameters, max_norm, norm_type)

        # Check if the norm is finite
        if self.error_if_nonfinite and not torch.isfinite(total_norm):
            raise RuntimeError(
                f"The total norm of order {norm_type} for gradients from "
                "`parameters` is non-finite, so it cannot be clipped. To disable "
                "this error and scale the gradients by the non-finite norm anyway, "
                "set `error_if_nonfinite=False`."
            )

        return total_norm

    @torch.no_grad()
    def clip_value(self, parameters: Union[torch.Tensor, List[torch.Tensor]], clip_value: float) -> None:
        """
        Clips the gradients by element-wise value, i.e., sets gradients to be within [-clip_value, clip_value].

        Args:
            parameters (Union[torch.Tensor, List[torch.Tensor]]): Model parameters whose gradients need to be clipped.
            clip_value (float): The maximum absolute value for each gradient element.
        """
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]  # Convert a single tensor to a list for uniform processing.

        clip_grad_value_(parameters, clip_value)

    def __call__(self, parameters: Union[torch.Tensor, List[torch.Tensor]]) -> Optional[float]:
        """
        Performs gradient clipping based on the specified clipping type.

        Args:
            parameters (Union[torch.Tensor, List[torch.Tensor]]): Model parameters whose gradients need to be clipped.

        Returns:
            Optional[float]: The total gradient norm (only if `clip_type` is 'norm', otherwise returns None).
        """
        if self.clip_type == "norm" and self.max_grad_norm > 0:
            return self.clip_norm(parameters, self.max_grad_norm)
        elif self.clip_type == "value" and self.max_grad_value is not None:
            self.clip_value(parameters, self.max_grad_value)
            return None
        return None
