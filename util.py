class EarlyStopping(object):
    def __init__(self, args):
        self.args = args
        self.patience = 100
        self.counter = 0
        self.best_loss = float('inf')

    def reset(self):
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
