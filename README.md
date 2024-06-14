# DCGAN-and-Style-Transfer


### GAN and DCGAN

Developed both a novel Generative Adversarial Network and Deepy Convolutional Generative Adversarial Network.

Both GANS utilize a adversarial binary cross entropy loss. Described generally by: ![GAN Equation](/imgs/image.png)

  Notes:
    * The loss shows that the generator attempts to minimize and the discriminator attempts to maximize the value/loss function V. This is a minimax algorithm.
    * z is random noise, that the generator uses to 'mold' into a image that will 'fool' the discriminator.
    * The loss can be interpreted as the expectation that x (a real image from the dataset) is from the true distribution of data. Since the discriminator attempts to maximize this, it will want to accurately predict which images are from the real distribution and which are not. 
    * The second part of the loss is the expectation that the discriminator can correctly tell that the generated images are fake. Since the generator, wants to minimize the loss it will attempt to make the discrimintor predict that images are from the true distribution and not the random noise distribution. 

We simplified this into two losses which we can minimize over both, just to make model converging easier. Additionally, we applied a sigmoid function to the outputs in order to keep calculations numerically stable (0 < loss < 1). 

The losses are described as:
![Generator Loss](/imgs/Generator loss.png)
