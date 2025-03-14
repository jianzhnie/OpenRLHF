import numpy as np


class RunningMeanStd:
    """Calculates the running mean and std of a data stream.

    Based on Welford's online algorithm.


    """

    def __init__(
        self,
        mean: float | np.ndarray = 0.0,
        std: float | np.ndarray = 1.0,
        clip_max: float | None = 10.0,
        epsilon: float = np.finfo(np.float32).eps.item(),
    ) -> None:
        self.mean = np.array(mean, dtype=np.float64)
        self.var = np.array(std, dtype=np.float64) ** 2  # 确保存储的是方差
        self.count = 0
        self.clip_max = clip_max
        self.eps = epsilon

    def norm(self, data_array: float | np.ndarray) -> float | np.ndarray:
        """Normalize input data based on running mean and std."""
        data_array = np.array(data_array, dtype=np.float64)
        normalized = (data_array - self.mean) / np.sqrt(self.var + self.eps)
        if self.clip_max:
            normalized = np.clip(normalized, -self.clip_max, self.clip_max)
        return normalized

    def update(self, data_array: np.ndarray) -> None:
        """Update mean and variance using a batch of data."""
        data_array = np.array(data_array, dtype=np.float64)
        batch_count = len(data_array)
        
        if batch_count == 0:
            return  # 忽略空 batch
        
        batch_mean = np.mean(data_array, axis=0)
        batch_var = np.var(data_array, axis=0)

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

# 示例使用
if __name__ == '__main__':
    normalizer = RunningMeanStd()
    
    batch1 = [1.0, 2.0, 0.5]
    batch2 = [3.0, 2.5, 1.5]
    
    normalizer.update(batch1)
    print("After batch1: Mean =", normalizer.mean, "Std =", np.sqrt(normalizer.var))
    
    normalizer.update(batch2)
    print("After batch2: Mean =", normalizer.mean, "Std =", np.sqrt(normalizer.var))
    
    normed_rewards = normalizer.norm([1.5, 2.2, 2.8, 3.1])
    print("Normalized rewards:", normed_rewards)
