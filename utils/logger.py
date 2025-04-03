import torch


class LogIO:
    def __init__(self, func):
        self.func = func
        self.evaluated_inputs = []
        self.evaluated_outputs = []

    def __call__(self, arg):
        self.evaluated_inputs.append(
            arg.detach().numpy() if isinstance(arg, torch.Tensor) else arg
        )
        result = self.func(arg)
        self.evaluated_outputs.append(
            result.detach().item() if isinstance(result, torch.Tensor)
            else result
        )
        return result

    def get_logged_inputs(self):
        return self.evaluated_inputs

    def get_logged_outputs(self):
        return self.evaluated_outputs
