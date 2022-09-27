import cv2
import numpy as np
import torch
import ttach as tta
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
import torch.nn as nn
import matplotlib.pyplot as plt

class BaseCAM:
    def __init__(self,
                 model,
                 target_layers,
                 device,
                 use_cuda=False,
                 reshape_transform=None,
                 compute_input_gradient=False,
                 uses_gradients=True):
        self.model = model.eval()
        self.target_layers = target_layers
        self.cuda = use_cuda
        self.device = device
        if self.cuda:
            #self.model = model.cuda().to(self.device)
            self.model = model.cuda()

        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers, reshape_transform)
        self.bce_loss = nn.CrossEntropyLoss()

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(self,
                        input_tensor,
                        target_layers,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_prob(self, output, target_category):
        prob = 0
        labels = []
        t = int(len(output)/len(target_category))
        for label in target_category:
            labels += [label]*t
        for i in range(len(labels)):
            prob = prob + output[i, labels[i]]
            # label = torch.tensor([target_category[i]], dtype=torch.long).cuda()
            # loss = loss + self.bce_loss(output, label)
        return prob

    def get_cam_image(self,
                      input_tensor,
                      mask_tensor,
                      target_layer,
                      target_category,
                      activations,
                      eigen_smooth=False,
                      grads=[]):
        ###### gradcam ######
        # weights = self.get_cam_weights(input_tensor, target_layer,
        #                                target_category, activations, grads)
        # ###### scorecam ######
        # weights = self.get_cam_weights(input_tensor, mask_tensor, target_layer,
        #                                target_category, activations)

        weights = self.get_cam_weights(input_tensor, mask_tensor, target_layer,
                                       target_category, activations=activations, grads=grads)

        weighted_activations = weights[:, :, None, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam, weights

    def forward(self, input_tensor, mask_tensor, target_category=None, eigen_smooth=False):
        if self.cuda:
            # input_tensor = input_tensor.cuda().to(self.device)
            # mask_tensor = mask_tensor.cuda().to(self.device)
            input_tensor = input_tensor.cuda()
            mask_tensor = mask_tensor.cuda()

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor,
                                                   requires_grad=True)

        output = self.activations_and_grads(input_tensor, mask_tensor)
        pred_labels = np.argmax(output.cpu().data.numpy(), axis=-1)

        # ### save activations map ###
        # maps = self.activations_and_grads.activations
        # maps = maps[0][0] ## video 0, frame 0
        # for i, map in enumerate(maps):
        #     plt.imsave("activation_c%d.jpg"%i,map)

        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert(len(target_category) == input_tensor.size(0))

        if self.uses_gradients: ## GradCAM uses gradients, ScoreCAM does not use gradients
            self.model.zero_grad()
            #target_category_frames = [ label for label in target_category for i in range(int(len(output)/len(target_category)))]
            """
            ### VERY IMPORTANT: in the original code using "get_loss" instead of "get_prob", it's very wrong !!!!!
            ### Actually, Grad_CAM uses the predicted probability of the target class to calculate the weight of each feature map in a specific layer
            ### (by backpropagating the probabiity to the specific layer and then averaging all gradients of each pixel/neuron on the
            ### feature map, and the averaging value is used as the weight of this feature map).
            ### The weighted feature maps then are summed up to generate the heat map which can show
            ### the model's attention region in this layer.
            ### Thus, if we  use the loss "log(probability)" to calculate the weight, the results will be
            ### completely opposite, since the larger of probability the smaller of loss. It means if the model can
            ### correctly predict the class of dog in the image with the probability close to 1, then its loss will close to
            ### zero, and if we calculate the weight of feature map by the very small loss, the weight of the feature map to predict dog will be zero,
            ### but the feature map to predict other class for example cat will be 1, so the heat map will be just opposite
            ### the correct heat map!!!! (This point has be proven in our test code!)
            """
            prob = self.get_prob(output, target_category)
            prob.backward(retain_graph=True)

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor,
                                                   mask_tensor,
                                                   target_category,
                                                   eigen_smooth)
        return self.aggregate_multi_layers(cam_per_layer), pred_labels

    def get_target_width_height(self, input_tensor):
        width, height = input_tensor.size(-1), input_tensor.size(-2)
        return width, height

    def compute_cam_per_layer(
            self,
            input_tensor,
            mask_tensor,
            target_category,
            eigen_smooth):
        activations_list = [a.cpu().data.numpy()
                            for a in self.activations_and_grads.activations]
        grads_list = [g.cpu().data.numpy()
                      for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        if self.uses_gradients:
            for target_layer, layer_activations, layer_grads in \
                    zip(self.target_layers, activations_list, grads_list):
                cam, weights = self.get_cam_image(input_tensor,
                                         mask_tensor,
                                         target_layer,
                                         target_category,
                                         layer_activations,
                                         eigen_smooth,
                                         layer_grads)
        else:
            for target_layer, layer_activations in \
                    zip(self.target_layers, activations_list):
                cam, weights = self.get_cam_image(input_tensor,
                                         mask_tensor,
                                         target_layer,
                                         target_category,
                                         layer_activations,
                                         eigen_smooth)
        cam[cam<0]=0 # works like mute the min-max scale in the function of scale_cam_image
        scaled = self.scale_cam_image(cam, target_size)
        cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer):
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)
        result = np.mean(cam_per_target_layer, axis=1)
        return self.scale_cam_image(result)

    def scale_cam_image(self, cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)

        return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 mask_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):

        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(
                input_tensor, mask_tensor, target_category, eigen_smooth)

        return self.forward(input_tensor, mask_tensor,
                            target_category, eigen_smooth)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(
                f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
