from torch import nn
import torch

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

# class ResnetBlock(nn.Module):
#     def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
#         super().__init__()
#         self.noise_func = FeatureWiseAffine(
#             noise_level_emb_dim, dim_out, use_affine_level)

#         self.block1 = Block(dim, dim_out, groups=norm_groups)
#         self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
#         self.res_conv = nn.Conv2d(
#             dim, dim_out, 1) if dim != dim_out else nn.Identity()

#     def forward(self, x, time_emb):
#         b, c, h, w = x.shape
#         h = self.block1(x)
#         h = self.noise_func(h, time_emb)
#         h = self.block2(h)
#         return h + self.res_conv(x)

class TevNet_latent(nn.Module):
    def __init__(self,
                in_channel=3,
                out_channel=10,
                inner_channel=64,
                blocks=1,
                norm_groups=16,
                channel_mults=(1, 2, 4, 8),
                dropout=0,
                image_size=64,
                ckpt_path=None,
                ignore_keys=[]
                ):
        super(TevNet, self).__init__()
        
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks):
                downs.append(Block(
                    pre_channel, channel_mult, groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
        
        # self.mid = nn.ModuleList([
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=True),
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=False)
        # ])
        
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks+1):
                # print(pre_channel, feat_channels, inner_channel, channel_mults[ind])
                ups.append(Block(
                    pre_channel+feat_channels.pop(), channel_mult, groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                # print(1)
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        for n, _ in list(sd.items()):
            # print(n)
            sd[n.replace('module.','')] = sd.pop(n)
        keys = list(sd.keys())
        # print(keys)
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

        
    def forward(self, x):

        feats = []
        for layer in self.downs:
            # print(x.size())
            x = layer(x)
            feats.append(x)

        # for layer in self.mid:
        #     x = x
        # print('ups:')
        for layer in self.ups:
            if isinstance(layer, Block):
            # print(x.size(), feats[-1].size())
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        return self.final_conv(x)

class TevNet(nn.Module):
    def __init__(self,
                in_channel=1,
                out_channel=10,
                inner_channel=64,
                blocks=1,
                norm_groups=16,
                channel_mults=(1, 2, 4, 8),
                dropout=0,
                image_size=256,
                ckpt_path=None,
                ignore_keys=[]
                ):
        super(TevNet, self).__init__()
        
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks):
                downs.append(Block(
                    pre_channel, channel_mult, groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
        
        # self.mid = nn.ModuleList([
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=True),
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=False)
        # ])
        
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks+1):
                # print(pre_channel, feat_channels, inner_channel, channel_mults[ind])
                ups.append(Block(
                    pre_channel+feat_channels.pop(), channel_mult, groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                # print(1)
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        for n, _ in list(sd.items()):
            # print(n)
            sd[n.replace('module.','')] = sd.pop(n)
        keys = list(sd.keys())
        # print(keys)
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

        
    def forward(self, x):

        feats = []
        for layer in self.downs:
            # print(x.size())
            x = layer(x)
            feats.append(x)

        # for layer in self.mid:
        #     x = x
        # print('ups:')
        for layer in self.ups:
            if isinstance(layer, Block):
            # print(x.size(), feats[-1].size())
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        return self.final_conv(x)

class TeXNetp3(nn.Module):
    def __init__(self,
                in_channel=1,
                out_channel=11,
                inner_channel=64,
                blocks=1,
                norm_groups=16,
                channel_mults=(1, 2, 4, 8),
                dropout=0,
                image_size=256,
                ckpt_path=None,
                ignore_keys=[]
                ):
        super(TeXNetp3, self).__init__()
        
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks):
                downs.append(Block(
                    pre_channel, channel_mult, groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
        
        # self.mid = nn.ModuleList([
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=True),
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=False)
        # ])
        
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks+1):
                # print(pre_channel, feat_channels, inner_channel, channel_mults[ind])
                ups.append(Block(
                    pre_channel+feat_channels.pop(), channel_mult, groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                # print(1)
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        for n, _ in list(sd.items()):
            # print(n)
            sd[n.replace('module.','')] = sd.pop(n)
        keys = list(sd.keys())
        # print(keys)
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

        
    def forward(self, x):

        feats = []
        for layer in self.downs:
            # print(x.size())
            x = layer(x)
            feats.append(x)

        # for layer in self.mid:
        #     x = x
        # print('ups:')
        for layer in self.ups:
            if isinstance(layer, Block):
            # print(x.size(), feats[-1].size())
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        return self.final_conv(x)

class TeXNetp8(nn.Module):
    def __init__(self,
                in_channel=1,
                out_channel=66,
                inner_channel=64,
                blocks=1,
                norm_groups=16,
                channel_mults=(1, 2, 4, 8),
                dropout=0,
                image_size=256,
                ckpt_path=None,
                ignore_keys=[]
                ):
        super(TeXNetp8, self).__init__()
        
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks):
                downs.append(Block(
                    pre_channel, channel_mult, groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
        
        # self.mid = nn.ModuleList([
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=True),
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=False)
        # ])
        
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks+1):
                # print(pre_channel, feat_channels, inner_channel, channel_mults[ind])
                ups.append(Block(
                    pre_channel+feat_channels.pop(), channel_mult, groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                # print(1)
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        for n, _ in list(sd.items()):
            # print(n)
            sd[n.replace('module.','')] = sd.pop(n)
        keys = list(sd.keys())
        # print(keys)
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

        
    def forward(self, x):

        feats = []
        for layer in self.downs:
            # print(x.size())
            x = layer(x)
            feats.append(x)

        # for layer in self.mid:
        #     x = x
        # print('ups:')
        for layer in self.ups:
            if isinstance(layer, Block):
            # print(x.size(), feats[-1].size())
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        return self.final_conv(x)

class TeXNetp16(nn.Module):
    def __init__(self,
                in_channel=1,
                out_channel=258,
                inner_channel=64,
                blocks=1,
                norm_groups=16,
                channel_mults=(1, 2, 4, 8),
                dropout=0,
                image_size=256,
                ckpt_path=None,
                ignore_keys=[]
                ):
        super(TeXNetp16, self).__init__()
        
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks):
                downs.append(Block(
                    pre_channel, channel_mult, groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
        
        # self.mid = nn.ModuleList([
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=True),
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=False)
        # ])
        
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks+1):
                # print(pre_channel, feat_channels, inner_channel, channel_mults[ind])
                ups.append(Block(
                    pre_channel+feat_channels.pop(), channel_mult, groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                # print(1)
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        for n, _ in list(sd.items()):
            # print(n)
            sd[n.replace('module.','')] = sd.pop(n)
        keys = list(sd.keys())
        # print(keys)
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

        
    def forward(self, x):

        feats = []
        for layer in self.downs:
            # print(x.size())
            x = layer(x)
            feats.append(x)

        # for layer in self.mid:
        #     x = x
        # print('ups:')
        for layer in self.ups:
            if isinstance(layer, Block):
            # print(x.size(), feats[-1].size())
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        return self.final_conv(x)
    
class TeXNetp32(nn.Module):
    def __init__(self,
                in_channel=1,
                out_channel=1026,
                inner_channel=64,
                blocks=1,
                norm_groups=16,
                channel_mults=(1, 2, 4, 8),
                dropout=0,
                image_size=256,
                ckpt_path=None,
                ignore_keys=[]
                ):
        super(TeXNetp32, self).__init__()
        
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks):
                downs.append(Block(
                    pre_channel, channel_mult, groups=norm_groups, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
        
        # self.mid = nn.ModuleList([
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=True),
        #     ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
        #                        dropout=dropout, with_attn=False)
        # ])
        
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, blocks+1):
                # print(pre_channel, feat_channels, inner_channel, channel_mults[ind])
                ups.append(Block(
                    pre_channel+feat_channels.pop(), channel_mult, groups=norm_groups, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                # print(1)
                ups.append(Upsample(pre_channel))
                now_res = now_res*2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)
        
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
    
    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location=lambda storage, loc: storage)["state_dict"]
        for n, _ in list(sd.items()):
            # print(n)
            sd[n.replace('module.','')] = sd.pop(n)
        keys = list(sd.keys())
        # print(keys)
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

        
    def forward(self, x):

        feats = []
        for layer in self.downs:
            # print(x.size())
            x = layer(x)
            feats.append(x)

        # for layer in self.mid:
        #     x = x
        # print('ups:')
        for layer in self.ups:
            if isinstance(layer, Block):
            # print(x.size(), feats[-1].size())
                x = layer(torch.cat((x, feats.pop()), dim=1))
            else:
                x = layer(x)

        return self.final_conv(x)