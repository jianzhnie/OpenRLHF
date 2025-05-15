from numbers import Number
import numpy as np
import torch


class MovAvg:
    r"""Class for moving average.

    It will automatically exclude the infinity and NaN. Usage:
    ::

        >>> stat = MovAvg(size=66)
        >>> stat.add(torch.tensor(5))
        5.0
        >>> stat.add(float('inf'))  # which will not add to stat
        5.0
        >>> stat.add([6, 7, 8])
        6.5
        >>> stat.get()
        6.5
        >>> print(f'{stat.mean():.2f}±{stat.std():.2f}')
        6.50±1.12
    """

    def __init__(self, size: int = 100) -> None:
        super().__init__()
        self.size = size
        self.cache: list[np.number] = []
        self.banned = [np.inf, np.nan, -np.inf]

    def add(
        self,
        data_array: Number | float | np.number | list | np.ndarray | torch.Tensor,
    ) -> float:
        """Add a scalar into :class:`MovAvg`.

        You can add ``torch.Tensor`` with only one element, a python scalar, or
        a list of python scalar.
        """
        if isinstance(data_array, torch.Tensor):
            data_array = data_array.flatten().cpu().numpy()
        if np.isscalar(data_array):
            data_array = [data_array]
        for number in data_array:  # type: ignore
            if number not in self.banned:
                self.cache.append(number)
        if self.size > 0 and len(self.cache) > self.size:
            self.cache = self.cache[-self.size :]
        return self.get()

    def get(self) -> float:
        """Get the average."""
        if len(self.cache) == 0:
            return 0.0
        return float(np.mean(self.cache))  # type: ignore

    def mean(self) -> float:
        """Get the average. Same as :meth:`get`."""
        return self.get()

    def std(self) -> float:
        """Get the standard deviation."""
        if len(self.cache) == 0:
            return 0.0
        return float(np.std(self.cache))  # type: ignore


class RunningMeanStd:
    """Calculates the running mean and std of a data stream.

    Supports both numpy arrays and torch tensors.

    Based on Welford's online algorithm.


    """

    def __init__(
        self,
        mean: float | np.ndarray | torch.Tensor = 0.0,
        std: float | np.ndarray | torch.Tensor = 1.0,
        clip_max: float | None = 10.0,
        epsilon: float = torch.finfo(torch.float32).eps,
        device: torch.device | None = None,
    ) -> None:
        # Convert inputs to torch tensors
        self.device = device or torch.device("cpu")
        self.mean = torch.as_tensor(mean, dtype=torch.float64, device=self.device)
        self.var = torch.as_tensor(std, dtype=torch.float64, device=self.device) ** 2
        self.count = 0
        self.clip_max = clip_max
        self.eps = epsilon

    def norm(self, data: float | np.ndarray | torch.Tensor) -> torch.Tensor:
        """Normalize input data based on running mean and std."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = torch.as_tensor(data, dtype=torch.float64, device=self.device)

        normalized = (data - self.mean) / torch.sqrt(self.var + self.eps)
        if self.clip_max:
            normalized = torch.clamp(normalized, -self.clip_max, self.clip_max)
        return normalized

    def update(self, data: np.ndarray | torch.Tensor) -> None:
        """Update mean and variance using a batch of data."""
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = torch.as_tensor(data, dtype=torch.float64, device=self.device)

        batch_count = data.shape[0]
        if batch_count == 0:
            return  # 忽略空 batch

        batch_mean = torch.mean(data, dim=0)
        batch_var = torch.var(data, dim=0, unbiased=False)  # 使用有偏方差以匹配numpy行为

        total_count = self.count + batch_count
        delta = batch_mean - self.mean

        # 更新均值和方差
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + (delta**2) * self.count * batch_count / total_count
        new_var = m_2 / total_count

        self.mean, self.var = new_mean, new_var
        self.count = total_count

    def to(self, device: torch.device) -> "RunningMeanStd":
        """Move the normalizer to specified device."""
        self.device = device
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self


# 示例使用
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalizer = RunningMeanStd(device=device)

    # 测试 numpy array
    batch1 = np.array([1.0, 2.0, 0.5])
    batch2 = np.array([3.0, 2.5, 1.5])

    normalizer.update(batch1)
    print("After batch1: Mean =", normalizer.mean.cpu().numpy(), "Std =", torch.sqrt(normalizer.var).cpu().numpy())

    normalizer.update(batch2)
    print("After batch2: Mean =", normalizer.mean.cpu().numpy(), "Std =", torch.sqrt(normalizer.var).cpu().numpy())

    # 测试 torch tensor
    tensor_batch = torch.tensor([[1.5, 2.2], [2.8, 3.1]], device=device)
    normalizer.update(tensor_batch)
    print(
        "After tensor batch: Mean =", normalizer.mean.cpu().numpy(), "Std =", torch.sqrt(normalizer.var).cpu().numpy()
    )

    normed_rewards = normalizer.norm(tensor_batch)
    print("Normalized rewards:", normed_rewards.cpu().numpy())
