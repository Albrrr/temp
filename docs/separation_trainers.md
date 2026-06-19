Notes on the implementation in separation_trainers.py (as of 06/16/2026)

## Optimization Objective:
For notational simplicity we break down the feature extractor into two parts. Let $f_\theta$ be the ResNet backbone parameterized by $\theta$ and $h_\phi$ be the classification head parameterized by $\phi$, and let the features extracted by the backbone be $F=f_\theta(x)$ given some input image $x$.

We can then compute the composite loss function $\mathcal{L}_\text{total}$ as follows.
$$
\mathcal{L}_\text{total}=\mathcal{L}_\text{seg}(h_\phi(F),y)+\lambda\mathcal{L}_\text{sep}(F,y)
$$
We define $\mathcal{L}_\text{seg}$ as the standard segmentation loss (NLL+Lovasz+Boundary) and $\mathcal{L}_\text{sep}$ as the Hyperdimensional separation loss. The separation loss maps dense features into a high-dimensional space and enforces contrastive separability.

Given a single pixel's feature vector $v\in\mathbb{R}^d$, it is projected into the hyperspace using a fixed, frozen RP matrix $W_\text{rp}\in\mathbb{R}^{D\times d}$ (shown below where $s\in\mathbb{R}^D$ is the "soft" hypervector, pre-quantization, and $q$ is the final quantized hypervector).
$$
s=W_\text{rp}v\quad q=\text{sign}(s)\in\{-1,1\}^D
$$
To measure the similarity against the $C$ class prototypes $P\in\mathbb{R}^{C\times D}$, we use the cosine similarity $\text{sim}_k$. Since $q$ consists of $\pm 1$, we normalize it and compute the dot product, scaled by temperature $\tau$.
$$
\text{sim}_k=\frac{q\cdot P_k}{\tau\|q\|_2}
$$
The loss for the pixel with true label $y$ is then defined with the InfoNCE contrastive loss, which maximizes the similarity to the true prototype while minimizing similarity to all others.
$$
\mathcal{L}_\text{sep}=-\log\left(\frac{\exp(\text{sim}_y)}{\sum^C_{k=1}\exp(\text{sim}_k)}\right)
$$
The primary bottleneck for this loss is that computing the exact $\nabla_\theta\mathcal{L}_\text{sep}$ requires storing massive intermediate activation graphs of dimension $D_\text{HD}$ (typically ~10k), so we introduce two trainers with different mathematical aprpoximations to bypass this.

## Temporal Zero-Order Trainer:
Zero-Order Optimization, specifically Simultaneous Perturbation Stochastic Approximation, approximates the gradient of the loss with respect to the backbone weights $\theta$ using only forward passes (NOTE: could be changed to being only certain layers if memory becomes a problem). 

At training step $t$, we sample a random perturbation vector $z\sim\mathcal{N}(0,I_{|\theta|})$ from a standard normal distribution with the exact same dimensionality as $\theta$. We then evaluate the total loss at two slightly perturbed points along this direction, scaled by a small finite-difference step $\epsilon$ (NOTE: quantizing the hypervectors for this step would result in very consistent zero gradients, since the small perturbations won't cause a different sign of the soft hypervector).
$$
\mathcal{L}_\text{pos}=\mathcal{L}_\text{total}(\theta_t+\epsilon z)\quad \mathcal{L}_\text{neg}=\mathcal{L}_\text{total}(\theta_t-\epsilon z)
$$
The directional derivative $g_{zo}$ and the synthetic gradient $\hat{\nabla}_\theta\mathcal{L}$ for the entire parameter vector (obtained by scaling the random direction by the directional derivative) can then be calculated as follows.
$$
g_{zo}=\frac{\mathcal{L}_\text{pos}-\mathcal{L}_\text{neg}}{2\epsilon}\quad\hat{\nabla}_\theta\mathcal{L}\approx g_{zo}z
$$
Since this single-sample estimator has really high variance, we apply an Exponential Moving Average with decay rate $\beta$ to stabilize the trajectory over time.
$$
G_t=\beta G_{t-1}+(1-\beta)\hat{\nabla}_\theta\mathcal{L}\quad \theta_{t+1}=\theta_t-\eta G_t
$$

## Direct Feedback Alignment Trainer:
Standard backpropagation computes the gradient for a hidden layer $l$ using the chain rule, passing the error backward layer-by-layer.
$$
\delta_l=W^T_{l+1}\delta_{l+1}\odot\sigma^\prime(a_l)
$$
Direct Feedback Alignment says that the network can still learn if we bypass this sequential chain, and instead project the global error directly to each hidden layer using a fixed random matrix. We first define the global HD-space error $e\in\mathbb{R}^D$ for a given pixel (as the differnece between the normalized soft hypervector and the true class prototype).
$$
e=\frac{s}{\|s\|_2}-P_y
$$
For a specific ResNet layer $l$ with $C_l$ output channels, we draw a fixed random feedback matrix $B_l\in\mathbb{R}^{D\times C_l}$ and then compute the synthetic gradient $\delta_{\text{DFA},l}$ for that  layer by projecting $e$ through $B_l$.
$$
\delta_{\text{DFA},l}=eB_l
$$
These errors are then added to the gradient from the segmentation loss to get the total gradient for that layer (where $\alpha$ is a balance factor).
$$
\delta_{\text{total},l}=\nabla_{a_l}\mathcal{L}_\text{seg}+\alpha\delta_{\text{DFA},l}
$$