import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            device,
            use_cuda=False,
            reshape_transform=None,
            uses_gradients=False):
        super(ScoreCAM, self).__init__(model, target_layers, device, use_cuda,uses_gradients=uses_gradients,
                                       reshape_transform=reshape_transform)

        if len(target_layers) > 0:
            print("Warning: You are using ScoreCAM with target layers, "
                  "however ScoreCAM will ignore them.")

    def get_cam_weights(self,
                        input_tensor,
                        mask_tensor,
                        target_layer,
                        target_category,
                        activations,
                        grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            if self.cuda:
                #activation_tensor = activation_tensor.cuda().to(self.device)
                activation_tensor = activation_tensor.cuda()

            upsampled = upsample(activation_tensor) ## upsample activation map to input image size

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0] ## [1] is the index of the max value
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0] ## [1] is the index of the min value
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins) ## normalization upsampled activation map to [0-1]

            # input_tensors = input_tensor[:, None,
            #                              :, :] * upsampled[:, :, None, :, :]
            input_tensors = input_tensor[:, :, None,:, :,:] * upsampled[:, :, None, :, :] ## input: batch,frames,channel,w,h

            # ### test if input_tensors is correct
            # input_tensors_all_frames = []
            # for i in range(upsampled.size()[0]):
            #     input_tensor_frame = input_tensor[0,i,:,:,:]
            #     upsampled_frame = upsampled[i,:,:,:]
            #     upsampled_input_frame = []
            #     for j in range(upsampled.size()[1]):
            #         upsampled_frame_channel = upsampled_frame[j, :, :]
            #         upsampled_input_i_channel_j = input_tensor_frame*upsampled_frame_channel
            #         upsampled_input_frame.append(upsampled_input_i_channel_j)
            #     input_tensors_all_frames.append(upsampled_input_frame)

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16

            scores = []
            input_tensors = input_tensors.permute(0,2,1,3,4,5) ## activations as the batches
            for batch_index, tensor in enumerate(input_tensors):
                category = target_category[batch_index]
                for i in tqdm.tqdm(range(0, tensor.size(0), BATCH_SIZE)):
                #for i in range(0, tensor.size(0), BATCH_SIZE):
                    batch = tensor[i: i + BATCH_SIZE, :]
                    #outputs = self.model(batch).cpu().numpy()[:, category]
                    outputs = self.model(batch, mask_tensor).cpu().numpy()[:, category]
                    scores.extend([outputs])
            scores = torch.Tensor(scores)
            scores = scores.permute(1,0)
            scores = scores.view(activations.shape[0], activations.shape[1])

            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights
