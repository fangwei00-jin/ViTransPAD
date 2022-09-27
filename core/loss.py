import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()
        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()
        elif type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge': ## hinge loss (loss for training Discriminator and Generator are different)
            if is_disc:
                ## for training Discriminator: loss = E(Relu(1-D(real)))+E(Relu(1+D(fake))), here outputs = D(inputs)
                ## i.e.,  mean(D(real))->1, mean(D(fake))-> -1, mean is E()
                ## If the D is already good enough that mean(D(real)) > 1 or mean(D(fake))<-1, we don't need the training anymore.
                ## Thus, we only train D when its output 1>mean(D(input))>-1.

                if is_real: ## inputing real images
                    outputs = -outputs
                ## mean value of ReLU(1+output).
                ## ReLU is an element-wise operator which won't change the shape of output
                return self.criterion(1 + outputs).mean()
            else:
                ## for training Generator: loss = - E(D(fake)) = E(-D(fake)). For a trained Discriminator D, D(fake) should be -1, D(real) should be 1.
                ## Thus, here training the Generator to output a fake image being good enought to confuses the Discriminator, making D(fake) = 0 instead of -1.
                ## N.B, hinge loss doesn't define the loss = E(1-D(fake)) which would push the Generator to generate an extremely good image which makes the D(input)->1,
                ## i.e. the Discriminator would definately consider the input is a real image instead of a fake one.
                ## Since when D(input)=0, it means that the input is good enough to confuse the Discriminator and it's time to retrain the Discriminator.
                ## That is to say, we don't have to wait the Discriminator being totally lost and then retrain the Discriminator.

                return (-outputs).mean()
        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(
                outputs)
            loss = self.criterion(outputs, labels)
            return loss


