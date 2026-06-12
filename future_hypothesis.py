class FutureHypothesis:
    def __init__(self, z_hat):
        self.z_hat = z_hat        # shape (H, 3)
        self.score = 1.0
        self.alive = True
