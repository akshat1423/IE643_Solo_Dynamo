# CycleGAN Loss and Utility Functions
class CycleGANLoss(nn.Module):
    def __init__(self, lambda_cycle=10.0):
        super(CycleGANLoss, self).__init__()
        self.lambda_cycle = lambda_cycle
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()

    def forward(self, pred_real, pred_fake, real_image, recon_image):
        loss_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_cycle = self.lambda_cycle * self.criterion_cycle(real_image, recon_image)
        return loss_GAN + loss_cycle
criterion = CycleGANLoss(lambda_cycle=10.0)
