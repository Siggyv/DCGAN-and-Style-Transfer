# DCGAN-and-Style-Transfer


## GAN and DCGAN

Developed both a novel Generative Adversarial Network and Deepy Convolutional Generative Adversarial Network.

Both GANS utilize a adversarial binary cross entropy loss. Described generally by: ![GAN Equation](/imgs/image.png)

  Explanation:
  
    * The loss shows that the generator attempts to minimize and the discriminator attempts to maximize the value/loss
    function V. This is a minimax algorithm.
    * z is random noise, that the generator uses to 'mold' into a image that will 'fool' the discriminator.
    * The loss can be interpreted as the expectation that x (a real image from the dataset) is from the true distribution
    of data.
    * Since the discriminator attempts to maximize this, it will want to accurately predict which images are from the real 
    distribution and which are not. 
    * The second part of the loss is the expectation that the discriminator can correctly tell that the generated images 
    are fake. 
    * Since the generator, wants to minimize the loss it will attempt to make the discrimintor predict that images are 
    from the true distribution and not the random noise distribution. 

I simplified this into two losses which we can minimize over both, just to make model converging easier. Additionally, I applied a sigmoid function to the outputs in order to keep calculations numerically stable (0 < loss < 1). 

The losses are described as:

![Generator Loss](/imgs/GenLoss.png)

and 

![Discriminator Loss](/imgs/DiscLoss.png)

Code wise, I used BCE (binary cross entropy) to compute the loss for the discriminator, which used the real logits (discriminator output on real images) and fake logits (discriminator output on generated images).

For generator loss, I used the BCE between the fake logits (again discriminator output on generated images) and just ones (since we want these to be discriminated as real images). 

To better understand the training process I have a graph pictured below, this graph makes training seem simple. Which is true, and I utilized the same training and loss calculations for both DCGAN and GAN.

![GAN Training Process](/imgs/gan_training.png)

Note: The only difference between GAN and DCGAN is the use of convolutional layers in DCGAN instead of linear layers in GAN. This is a small, but very important difference as highlighted by the results below. The convolutional layers allow for better spatial understanding, and allow the generator to understand the underlying structure of a number. A convolutional layer would understand the general structure of a 3, while a linear layer would not necessarily.

### Results DCGAN and GAN:

**GAN:**
![GAN Results](/imgs/fc_gan_results.jpg)


**DCGAN:**
![DCGAN Results](/imgs/dc_gan_results.jpg)

Results Evaluation:
  * As seen above, the DCGAN images tend to be more clear and lack the spotiness around the edges. They are sharper. This is a result of the convolutional spatial understanding.
  * Furthermore, the DCGAN has some 'hallucinations' that may be fixable with some regularization such as L2 or L1, but would add some blurriness.
  * Adding reg such as above, may not be a good idea, but could be decent if you clip some grey colors to black or white based on a threshold. Just something to try.

## Style Transfer

I developed a 
