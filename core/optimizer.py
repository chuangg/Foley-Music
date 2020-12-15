from torch.optim.optimizer import Optimizer


class CustomSchedule:
    def __init__(self, d_model, warmup_steps=4000, optimizer: Optimizer = None):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps

        self._step = 0
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        if step is None:
            step = self._step
        arg1 = step ** (-0.5)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.d_model ** (-0.5) * min(arg1, arg2)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict: dict):
        self.optimizer.load_state_dict(state_dict)
