class Layershook:
    """ Class for extracting feature map from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.get_activation))

            # # Because of https://github.com/pytorch/pytorch/issues/61519,
            # # we don't use backward hook to record gradients.
            # self.handles.append(
            #     target_layer.register_forward_hook(self.get_gradient))

    def get_activation(self, module, input, output):
        activation = output
        #self.activations.append(activation.cpu().detach())
        self.activations.append(activation)

    def get_gradient(self, module, input, output):
        # Gradients are computed in reverse order
        def _store_grad(grad):
            #elf.gradients = [grad.cpu().detach()] + self.gradients
            self.gradients = [grad] + self.gradients
        ## tensor variable hook only resgisters to the backward propagation respecting to the calculation of gradient
        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        self.model(x)
        return self.activations

    def release(self):
        for handle in self.handles:
            handle.remove()
