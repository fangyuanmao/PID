from torch import nn
import torch.nn.functional as F
import torch
import segmentation_models_pytorch as smp

class HADARNet(nn.Module):
    def __init__(self,
                 smp_model,
                 smp_encoder,
                 in_channels=3, 
                 out_channels=6,
                 ckpt_path=None,
                 ignore_keys=[]
                 ):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.tevnet = getattr(smp, smp_model)(encoder_name=smp_encoder,
                                              encoder_weights=None,
                                              in_channels=self.in_channels,
                                              classes=self.out_channels,
                                              )
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        for n, _ in list(sd.items()):
            sd[n.replace('module.','')] = sd.pop(n)
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
        else:
            print('Successfully load')
        
    def forward(self, x):
        preds = self.tevnet(x)
        preds[:,0,:,:] = torch.sigmoid(preds[:,0,:,:])
        preds[:,1,:,:] = F.relu(preds[:,1,:,:])
        return preds