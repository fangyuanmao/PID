import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class TeVNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=6, args=None):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        smp_model = args.smp_model
        smp_encoder = args.smp_encoder
        smp_encoder_weights = args.smp_encoder_weights
        if smp_encoder_weights == 'None':
            smp_encoder_weights = None
        self.tevnet = getattr(smp, smp_model)(encoder_name=smp_encoder,
                                              encoder_weights=smp_encoder_weights,
                                              in_channels=self.in_channels,
                                              classes=self.out_channels,
                                              )
        
    def forward(self, x):
        preds = self.tevnet(x)
        preds[:,0,:,:] = torch.sigmoid(preds[:,0,:,:])
        preds[:,1,:,:] = F.relu(preds[:,1,:,:])
        return preds