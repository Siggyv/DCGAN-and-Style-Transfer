# DCGAN-and-Style-Transfer


### GAN and DCGAN

Developed both a novel Generative Adversarial Network and Deepy Convolutional Generative Adversarial Network.

Both GANS utilize a adversarial binary cross entropy loss. Described generally by: $$
\mathcal{L}_{\text{GAN}}(G, D) = \mathbb{E}_{y}[\log D(y)] + \mathbb{E}_{x,z}[\log(1 - D(G(x, z)))]
$$
