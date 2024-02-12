from typing import List, Tuple

import gpytorch
import numpy as np
import torch
from torcheval.metrics import R2Score
from gpytorch.mlls import ExactMarginalLogLikelihood
from sklearn.metrics import r2_score

# from pypolo.scalers import MinMaxScaler, StandardScaler
from pypolo.scalers import MinMaxScaler, StandardScaler

from .. import BaseModel


class GPyTorchModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPRModel(BaseModel):
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        kernel: gpytorch.kernels.Kernel,
        noise_var: float,
        num_sparsification: int = 10,
        batch_size: int = 128,
    ) -> None:
        self.num_sparsification = num_sparsification
        if self.num_sparsification > 0:
            x_train = x_train[:: self.num_sparsification]
            y_train = y_train[:: self.num_sparsification]
        self._init_scalers(x_train, y_train)
        self.train_x = self._preprocess_x(x_train)
        self.train_y = self._preprocess_y(y_train)
        self._init_kernel_and_likelihood(kernel, noise_var)
        self._init_model()
        self._init_model_evidence()
        self._init_optimizer()
        self.batch_size = batch_size

    @torch.no_grad()
    def add_data(self, x_new: np.ndarray, y_new: np.ndarray) -> None:
        if self.num_sparsification > 0:
            x_new = x_new[:: self.num_sparsification]
            y_new = y_new[:: self.num_sparsification]
        self.new_x = self._preprocess_x(x_new)
        self.new_y = self._preprocess_y(y_new)
        self.train_x = torch.cat([self.train_x, self.new_x])
        self.train_y = torch.cat([self.train_y, self.new_y])
        self.model.set_train_data(self.train_x, self.train_y, strict=False)

    def optimize(
        self, num_steps: int = 100, reinit_optimizer: bool = False
    ) -> List[float]:
        if reinit_optimizer:
            self._init_optimizer()
        self.model.train()
        self.likelihood.train()
        losses = []
        for _ in range(num_steps):
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -self.model_evidence(output, self.train_y)
            loss.backward()
            losses.append(loss.item())
            self.optimizer.step()
        self.model.eval()
        self.likelihood.eval()
        return losses

    @torch.no_grad()
    def predict(
        self, x: np.ndarray, without_likelihood: bool = False, return_std: bool = False, return_cov: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        means = []
        stds = []
        self.model.train()
        self.likelihood.train()
        self.model.eval()
        self.likelihood.eval()
        x = self._preprocess_x(x)
        for x_batch in torch.split(x, self.batch_size):
            predictive_dist = self.model(x_batch)
            if not without_likelihood:
                predictive_dist = self.model.likelihood(predictive_dist)
            mean = predictive_dist.mean.numpy().reshape(-1, 1)
            std = predictive_dist.stddev.numpy().reshape(-1, 1)
            means.append(mean)
            stds.append(std)
        mean = np.vstack(means)
        mean = self.y_scaler.postprocess_mean(mean)

        if return_std:  
            std = np.vstack(stds)
            std = self.y_scaler.postprocess_std(std)
            # print('-----std shape-----', std.shape)
            return mean, std
        elif return_cov:
            # print('-----predictive_dist.covariance_matrix shape-----', x.shape, predictive_dist.covariance_matrix.numpy().shape)
            # return mean, predictive_dist.covariance_matrix.numpy()
            print('-----predictive_dist.variance shape-----', x.shape, predictive_dist.variance.numpy().shape)
            return mean, predictive_dist.covariance_matrix.numpy()
        else:
            std = np.vstack(stds)
            std = self.y_scaler.postprocess_std(std)
            return mean, std

    def _init_scalers(self, x_train, y_train):
        self.x_scaler = MinMaxScaler()
        self.x_scaler.fit(x_train)
        self.y_scaler = StandardScaler()
        self.y_scaler.fit(y_train)

    def _preprocess_x(self, x: np.ndarray) -> torch.Tensor:
        x = torch.tensor(self.x_scaler.preprocess(x), dtype=torch.float64)
        return x

    def _preprocess_y(self, y: np.ndarray) -> torch.Tensor:
        y = torch.tensor(self.y_scaler.preprocess(y), dtype=torch.float64).squeeze(-1)
        return y

    def _init_kernel_and_likelihood(
        self, kernel: gpytorch.kernels.Kernel, noise_var: float
    ) -> None:
        self.kernel = kernel.double()
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        self.likelihood.noise_covar.noise = noise_var

    def _init_model(self) -> None:
        self.model = GPyTorchModel(
            self.train_x, self.train_y, self.kernel, self.likelihood
        )

    def _init_model_evidence(self) -> None:
        self.model_evidence = ExactMarginalLogLikelihood(self.likelihood, self.model)

    def _init_optimizer(self, slow_lr: float = 1e-3, fast_lr: float = 1e-2) -> None:
        slow_params, fast_params = [], []
        print("Model parameters:")
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "nn" in name:
                slow_params.append(param)
                print(f"Slow lr: {slow_lr}", name, param.shape)
            else:
                fast_params.append(param)
                print(f"Fast lr: {fast_lr}", name, param.shape)
        self.optimizer = torch.optim.Adam(
            [
                {"params": slow_params, "lr": slow_lr},
                {"params": fast_params, "lr": fast_lr},
            ],
            lr=slow_lr,
        )

    @property
    def x_train(self) -> np.ndarray:
        return self.x_scaler.postprocess(self.train_x.numpy())

    @torch.no_grad()
    def get_ak_lengthscales(self, x):
        list_features = []
        x = self._preprocess_x(x)
        for x_batch in torch.split(x, self.batch_size):
            features = self.kernel.base_kernel.get_features(x_batch)
            list_features.append(features.numpy())
        features = np.vstack(list_features)
        features = features / features.sum(axis=1, keepdims=True)
        primitive_lengthscales = self.kernel.base_kernel.lengthscales.numpy()
        lengthscales = features @ primitive_lengthscales.reshape(-1, 1)
        return lengthscales

    # def score(self, x, y_true):
    #     y_pred, _ = self.predict(x)
    #     print('-----y_true shape-----', y_true.reshape(-1, 1).shape, type(y_true.reshape(-1, 1)))
    #     print('-----y_pred shape-----', y_pred.shape, type(y_pred))
    #     metric = R2Score()
    #     metric.update(torch.tensor(y_pred), torch.tensor(y_true.reshape(-1, 1)))
    #     return metric.compute().item()
    
    def score(self, x, y_true):
        y_pred, _ = self.predict(x)
        return r2_score(y_true, y_pred)
        