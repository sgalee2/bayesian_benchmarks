import time
import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, devices, output_device):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.MultiDeviceKernel(base_covar_module,
                                                               device_ids=devices,
                                                               output_device=output_device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class RegressionModel:
    def __init__(self, lr, iters, is_test=False, seed=0):
        torch.manual_seed(seed)
        self.is_test = is_test
        self.seed = seed
        self.lr = lr
        self.iters = iters
        self.likelihood = None
        self.model = None
        self.devices = [torch.device('cuda', i)
                        for i in range(torch.cuda.device_count())]
        self.output_device = self.devices[0]

    def fit(self, X, Y):
        X, Y = torch.tensor(X).float(), torch.tensor(Y).float()
        X, Y = X.to(self.output_device), Y.to(self.output_device)
        Y = Y.view(-1)

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = likelihood.to(self.output_device)
        model = ExactGPModel(X, Y, likelihood, self.devices, self.output_device)
        self.model = model.to(self.output_device)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([{'params': self.model.parameters()}],
                                     lr=self.lr)

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        print("Start training")
        start_time = time.time()
        with gpytorch.settings.max_preconditioner_size(5):
            training_iter = 5 if self.is_test else self.iters
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = self.model(X)
                # Calc loss and backprop gradients
                loss = -mll(output, Y)
                loss.backward()
                print('Iter %d/%d - Loss: %.3f   log_lengthscale: %.3f   log_noise: %.3f' % (
                    i + 1, training_iter, loss.item(),
                    self.model.covar_module.module.base_kernel.log_lengthscale.item(),
                    self.model.likelihood.log_noise.item()
                ))
                optimizer.step()
        print("Finished training. Elapsed time: {}".format(time.time() - start_time))
        return

    def predict(self, Xs):
        print("Start Predicting")
        Xs = torch.tensor(Xs).float().to(self.output_device)

        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_pred = self.model(Xs)
            observed_pred = self.likelihood(latent_pred)
            mean, var = observed_pred.mean, observed_pred.variance
            mean, var = mean.cpu().numpy(), var.cpu().numpy()
        print("Finished predicting")
        return mean, var

    def sample(self, Xs, S):
        print("Start sampling")
        Xs = torch.tensor(Xs).float().to(self.output_device)

        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            latent_pred = self.model(Xs)
            observed_pred = self.likelihood(latent_pred)
            samples = observed_pred.sample(torch.Size([S]))
            samples = samples.cpu().numpy()
        print("Finished sampling")
        return samples
