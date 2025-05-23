# Vanilla Variational Autoencoder (VAE)

Here's a review of the mathematical formulas for the original VAE.

## Core Idea

$\{x\}$ represents our image data samples, and $X$ represents the image variable. We struggle to understand the exact distribution of $p(X)$ (making it difficult to directly sample images from this distribution). Therefore, we aim to approximate $p(X)$ by constructing an encoder $p(Z|X)$, where $p(Z)$ is intended to be a Gaussian distribution. We can sample from the Gaussian distribution to obtain $Z$, and then use a decoder $p(X|Z)$ to map $Z$ back into the image space.

Our goal in training the VAE is to minimize the difference between the original image and the reconstructed image, while using KL divergence to constrain the distribution of $Z$ to be close to a standard normal distribution.

## Probability Density Function

The probability density function of a normal distribution $\mathcal{N}(\mu, \sigma^2)$ is:
$$
p(z) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(z-\mu)^2}{2\sigma^2}\right) \\
\log p(z) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2}
$$

Our neural network, after the Encoder part, actually outputs `mu` $\mu$ and `log_var` $\log \sigma^2$.

When calculating the KL divergence, we need to calculate the KL divergence between two normal distributions $p\sim \mathcal{N}(\mu, \sigma)$ and $q\sim \mathcal{N}(0,1)$:

$$
\begin{align}
KL(p(z), q(z))
&= \int p(z) \log\left(\frac{p(z)}{q(z)}\right) dz \\ 

\end{align}
$$

Then we have
$$
\log q(z) = -\frac{1}{2}\log(2\pi) - \frac{z^2}{2}
$$

We can use this estimator to calculate the KL divergence:

$$
\begin{aligned}
\log \frac{p(z)}{q(z)} &= \log p(z) - \log q(z) \\
&= -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(z-\mu)^2}{2\sigma^2} + \frac{1}{2}\log(2\pi) + \frac{z^2}{2} \\
&= -\log\sigma - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2} \\
\end{aligned}
$$

Then the KL divergence is:

$$
\begin{aligned}
KL(p(z), q(z)) &= \int p(z) \log\left(\frac{p(z)}{q(z)}\right) dz \\
&= \int p(z) \left(-\log\sigma - \frac{(z-\mu)^2}{2\sigma^2} + \frac{z^2}{2}\right) dz \\
&= -\log\sigma - \frac{1}{2\sigma^2} {\color{orangered}\int p(z) (z-\mu)^2 dz} + \frac{1}{2} {\color{magenta}\int p(z) z^2 dz} \\
&= -\log\sigma - \frac{1}{2\sigma^2} {\color{orangered}\mathrm{Var}_p(z)} + \frac{1}{2} {\color{magenta}\mathrm{E}_p(z^2)} \\
&= -\log\sigma - \frac{1}{2\sigma^2} {\color{orangered}\sigma^2} + \frac{1}{2} ({\color{magenta}\mathrm{Var}_p(z)+(\mathrm{E}_p(z))^2}) \\
&= -\log\sigma - \frac{1}{2} + \frac{1}{2} (\sigma^2 + \mu^2) \\
\end{aligned}
$$

So in the code, it can usually be written in one line:
```python
kl_loss = 0.5 * torch.sum(mu**2 + torch.exp(log_var) - log_var - 1, dim=1).mean()
```