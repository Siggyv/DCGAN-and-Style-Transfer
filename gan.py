from __future__ import print_function

import torch
import torch.utils.data
from torch import nn, optim
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NOISE_DIM = 96


def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device="cpu"):
    """
    Generate a PyTorch Tensor of random noise from Gaussian distribution.

    Input:
    - batch_size: Integer giving the batch size of noise to generate.
    - noise_dim: Integer giving the dimension of noise to generate.

    Output:
    - A PyTorch Tensor of shape (batch_size, noise_dim) containing
      noise from a Gaussian distribution.
    """
    noise = None
    ##############################################################################
    # DONE: Implement sample_noise.                                              #
    ##############################################################################
    noise = torch.randn((batch_size, noise_dim), device=device) #it should not be this easy
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################

    return noise


def discriminator():
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # DONE: Implement discriminator.                                           #
    ############################################################################
    RA = 0.05 #relu alpha
    model = nn.Sequential(
      nn.Linear(784, 400),
      nn.LeakyReLU(RA),
      nn.Linear(400, 200),
      nn.LeakyReLU(RA),
      nn.Linear(200, 100),
      nn.LeakyReLU(RA),
      nn.Linear(100, 1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the architecture
    in the notebook.
    """
    model = None
    ############################################################################
    # DONE: Implement generator. Changed to using leaky relu cause directions say to.                                            #
    ############################################################################
    model = nn.Sequential(
      nn.Linear(noise_dim, 128),
      nn.ReLU(),
      nn.Linear(128, 256),
      nn.ReLU(),
      nn.Linear(256, 512),
      nn.ReLU(),
      nn.Linear(512, 784),
      nn.Tanh()
    )    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss described above.

    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    loss = None
    ##############################################################################
    # DONE: Implement discriminator_loss.                                        #
    ##############################################################################
    # Replace "pass" statement with your code
    real_labels = torch.ones_like(logits_real)
    fake_labels = torch.zeros_like(logits_fake)

    loss_real = nn.functional.binary_cross_entropy_with_logits(logits_real, real_labels)
    loss_fake = nn.functional.binary_cross_entropy_with_logits(logits_fake, fake_labels)

    loss = loss_real + loss_fake
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def generator_loss(logits_fake):
    """
    Computes the generator loss described above.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.

    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    loss = None
    ##############################################################################
    # DONE: Implement generator_loss.                                            #
    ##############################################################################
    # Replace "pass" statement with your code
    real_labels = torch.ones_like(logits_fake)
    loss = nn.functional.binary_cross_entropy_with_logits(logits_fake, real_labels)
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return loss


def get_optimizer(model):
    """
    Construct and return an Adam optimizer for the model with learning rate 1e-3,
    beta1=0.5, and beta2=0.999.

    Input:
    - model: A PyTorch model that we want to optimize.

    Returns:
    - An Adam optimizer for the model with the desired hyperparameters.
    """
    optimizer = None
    ##############################################################################
    # TODO: Implement optimizer.                                                 #
    ##############################################################################
    # Replace "pass" statement with your code
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, betas=(0.5,0.999))
    ##############################################################################
    #                              END OF YOUR CODE                              #
    ##############################################################################
    return optimizer


def run_a_gan(D, G, D_solver, G_solver, loader_train, discriminator_loss, generator_loss, device, show_images, plt, show_every=250,
              batch_size=128, noise_size=96, num_epochs=10):
  """
  Train a GAN!

  Inputs:
  - D, G: PyTorch models for the discriminator and generator
  - D_solver, G_solver: torch.optim Optimizers to use for training the
    discriminator and generator.
  - loader_train: the dataset used to train GAN
  - discriminator_loss, generator_loss: Functions to use for computing the generator and
    discriminator loss, respectively.
  - show_every: Show samples after every show_every iterations.
  - batch_size: Batch size to use for training.
  - noise_size: Dimension of the noise to use as input to the generator.
  - num_epochs: Number of epochs over the training dataset to use for training.
  """
  iter_count = 0
  for epoch in range(num_epochs):
    D.train()
    G.train()
    D = D.to(device)
    G = G.to(device)
    for x, _ in loader_train:
      if len(x) != batch_size:
        continue
      ##############################################################################
      # TODO: Implement an iteration of training the discriminator.                #
      # Replace 'pass' with your code.                                             #
      # Save the overall discriminator loss in the variable 'd_total_error',       #
      # which will be printed after every 'show_every' iterations.                 #
      #                                                                            #
      # IMPORTANT: make sure to pre-process your real data (real images),          #
      # so as to make it in the range [-1,1].                                      #
      ##############################################################################
      d_total_error = None

      # scale x
      scaled_x = x.to(device)
      scaled_x = scaled_x.view(x.size(0), -1) # shape is now (batch_size, 28*28)
      scaled_x = (2 * scaled_x) - 1

      noise = sample_noise(batch_size=batch_size, noise_dim=noise_size, device=device)
      fake_images = G(noise)
      real = D(scaled_x)
      fakes = D(fake_images.detach())
      # start to train discriminator
      D_solver.zero_grad()

      # generate images for loss
      d_total_error = discriminator_loss(real, fakes)

      # step
      d_total_error.backward()
      D_solver.step()
      ##############################################################################
      #                              END OF YOUR CODE                              #
      ##############################################################################


      ##############################################################################
      # TODO: In the same iteration, implement training of the generator now   .   #
      # Replace 'pass' with your code.                                             #
      # Save the generator loss in the variable 'g_error', which will be printed.  #
      # after every 'show_every' iterations, and save the fake images generated    #
      # by G in the variable 'fake_images', which will be used to visualize the    #
      # generated images.
      ##############################################################################
      g_error = None
      # generate new fake images
      new_noise = sample_noise(batch_size=batch_size, noise_dim=noise_size, device=device)
      fake_images = G(new_noise)
      G_solver.zero_grad()

      # loss and training
      g_error = generator_loss(D(fake_images))
      g_error.backward()
      G_solver.step()
      ##############################################################################
      #                              END OF YOUR CODE                              #
      ##############################################################################

      if (iter_count % show_every == 0):
        print('Iter: {}, D: {:.4}, G:{:.4}'.format(iter_count,d_total_error.item(),g_error.item()))
        imgs_numpy = fake_images.data.cpu()#.numpy()
        show_images(imgs_numpy[0:16])
        plt.show()
        print()
      iter_count += 1
    if epoch == num_epochs - 1:
      return imgs_numpy   




def build_dc_classifier():
    """
    Build and return a PyTorch nn.Sequential model for the DCGAN discriminator
    implementing the architecture in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_classifier.                                     #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        nn.Unflatten(1, (1,28,28)),

        nn.Conv2d(in_channels= 1, out_channels=32, kernel_size=5, stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1),
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Flatten(),
        nn.Linear(in_features=4*4*64, out_features=4*4*64),
        nn.LeakyReLU(0.01),
        nn.Linear(in_features=4*4*64, out_features=1)
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model


def build_dc_generator(noise_dim=NOISE_DIM):
    """
    Build and return a PyTorch nn.Sequential model implementing the DCGAN
    generator using the architecture described in the notebook.
    """
    model = None
    ############################################################################
    # TODO: Implement build_dc_generator.                                      #
    ############################################################################
    # Replace "pass" statement with your code
    model = nn.Sequential(
        nn.Linear(in_features= noise_dim, out_features= 1024),
        nn.ReLU(),
        nn.BatchNorm1d(num_features=1024),

        nn.Linear(in_features=1024, out_features= 7 * 7 * 128),
        nn.ReLU(),
        nn.BatchNorm1d(num_features= 7*7*128),
        nn.Unflatten(dim=1, unflattened_size=(128,7,7)),

        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(num_features=64),

        nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1),
        nn.Tanh(),
        nn.Flatten()
    )
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return model
