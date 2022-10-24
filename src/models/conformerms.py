import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from functools import partial
import mindspore.numpy as np
import math
from mindspore import Tensor, context, Parameter,Model
from mindspore.common import initializer as weight_init
from mindspore.ops import operations as ops
from .misc import Identity
from .misc import DropPath1D as DropPath

class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Dense(hidden_features, out_features)
        self.drop = nn.Dropout(1-drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Cell):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(1-attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(1-proj_drop)
        self.softmax = nn.Softmax()

    def construct(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn =np.matmul(q , k.transpose(0,1,3,2)) * self.scale
        
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = np.matmul(attn , v).transpose(0,2,1,3).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Cell):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, epsilon=1e-6)):
        super().__init__()
        self.norm1 = norm_layer([dim])
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer([dim])
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def construct(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvBlock(nn.Cell):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, has_bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, group=groups, has_bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, has_bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer()

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, has_bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    #def zero_init_last_bn(self):
        #nn.init.zeros_(self.bn3.weight)

    def construct(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x

class FCUDown(nn.Cell):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, epsilon=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1,has_bias=True)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = norm_layer([outplanes])
        self.act = act_layer()

    def construct(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        x = self.sample_pooling(x)
        b,c,h,w=x.shape
        x=x.reshape(b,c,-1).transpose(0,2,1)
        x = self.ln(x)
        x = self.act(x)
        concat_op = ops.Concat(axis=1)
        
        cast_op = ops.Cast()
        #strided_slice = ops.StridedSlice()
        #x_ = strided_slice(x_t, (0, 0, 0), (1024,1,384), (1, 1, 1))
        #x_=cast_op(x_,mindspore.float32)
        #x_=cast_op(x_t[:, :1],mindspore.float32)
        x_=x_t[:, 0][:, None, :]
        x =cast_op(x,x_.dtype)
        
        x = concat_op((x_, x))
        
        #x = concat_op([cast_op(x_t[:, 0][:, None, :],mindspore.float32), x])
        return x

class FCUUp(nn.Cell):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6),):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1,has_bias=True)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def construct(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        #strided_slice = ops.StridedSlice()
        #x_ = strided_slice(x, (0, 1, 0), (1024,197,384), (1, 1, 1))
        x_r = x[:, 1:].transpose(0,2,1).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))
        #resize_bilinear = nn.ResizeBilinear()
        resize=ops.ResizeNearestNeighbor((H * self.up_stride, W * self.up_stride))

        return resize(x_r)

class Med_ConvBlock(nn.Cell):
    """ special case for Convblock with down sampling,
    """
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, has_bias=False)
        self.bn1 = norm_layer([med_planes])
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, group=groups, has_bias=False)
        self.bn2 = norm_layer([med_planes])
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, has_bias=False)
        self.bn3 = norm_layer([inplanes])
        self.act3 = act_layer()

        self.drop_block = drop_block
        self.drop_path = drop_path

    #def zero_init_last_bn(self):
        #nn.init.zeros_(self.bn3.weight)

    def construct(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x

class ConvTransBlock(nn.Cell):
    """
    Basic module for ConvTransformer, keep feature maps for CNN block and patch embeddings for transformer encoder block
    """

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.CellList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)

        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)

        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def construct(self, x, x_t):
        x, x2 = self.cnn_block(x)
        

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)
        
        

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t
        
class Conformer(nn.Cell):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        #assert depth % 3 == 0
        self.embed_dim=embed_dim

        self.cls_token = mindspore.Parameter(Tensor(np.zeros((1,1,embed_dim)),mindspore.float32),name='cls_token', requires_grad=True)
        #self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.trans_dpr=drop_path_rate/(depth-1)  # stochastic depth decay rule

        # Classifier head
        self.trans_norm = nn.LayerNorm([embed_dim])
        self.trans_cls_head = nn.Dense(embed_dim, num_classes) if num_classes > 0 else Identity()
        self.pooling = nn.AvgPool2d(7,7)
        self.conv_cls_head = nn.Dense(int(256 * channel_ratio), num_classes)

        # Stem stage: get the feature maps by conv block (copied form resnet.py)
        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, has_bias=False)  # 1 / 2 [112, 112]
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')  # 1 / 4 [56, 56]

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride,pad_mode="valid",has_bias=True)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0,
                             )

        # 2~4 stage
        self.blocks = nn.CellList()
        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.blocks.append(ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr*(i-1),
                        num_med_block=num_med_block)
            )


        stage_2_channel = int(base_channel * channel_ratio * 2)
        
        # 5~8 stage
        init_stage = fin_stage # 5
        fin_stage = fin_stage + depth // 3 # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.blocks.append(ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr*(i-1),
                        num_med_block=num_med_block)
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.blocks.append(ConvTransBlock(
                        in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr*(i-1),
                        num_med_block=num_med_block, last_fusion=last_fusion)
            )
        self.fin_stage = fin_stage
        self.cls_token.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),self.cls_token.shape,self.cls_token.dtype))

        

        self.init_weights()

    
    def init_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeNormal(mode='fan_out', nonlinearity='relu'),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    
    #def no_weight_decay(self):
        #return {'cls_token'}


    def construct(self, x):
        B = x.shape[0]
        #expand_dims = ops.ExpandDims()
        #cls_tokens = ExpandDims(self.cls_token,(B, -1, -1))
        broadcast_to = ops.BroadcastTo((B, -1, -1))
        #output = broadcast_to(input_tensor)
        cls_tokens = broadcast_to(self.cls_token)
        

        # pdb.set_trace()
        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))
        

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)
        

        x_t = self.trans_patch_conv(x_base).reshape(B,self.embed_dim,-1).transpose(0,2,1)
        concat_op = ops.Concat(axis=1)
        cast_op = ops.Cast()
        cls_tokens = cast_op(cls_tokens, x_t.dtype)
        x_t = concat_op((cls_tokens, x_t))
        x_t = self.trans_1(x_t)
        #print(x_t.shape)
        
        # 2 ~ final 
        for blk in self.blocks:

            x, x_t = blk(x, x_t)
            
            

        # conv classification
        x_p = self.pooling(x).reshape(B,-1)
        
        conv_cls = self.conv_cls_head(x_p)

        # trans classification
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])
        

        return [conv_cls,tran_cls]
                        
def Conformer_tiny_patch16(pretrained=False, **kwargs):
    model = Conformer(patch_size=16, channel_ratio=1, embed_dim=384, depth=12,
                      num_heads=6, mlp_ratio=4, qkv_bias=True, **kwargs)
    return model

if __name__ == "__main__":
    from mindspore import context
    from mindspore import dtype as mstype
    #import numpy as np

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    data = Tensor(np.ones([2, 3, 224, 224]), dtype=mstype.float32)
    model = Conformer_tiny_patch16()
    # 验证前向传播
    out = model(data)
    print(out.shape)
    #params = 0.
    
    #for name, param in model.parameters_and_names():
    	# 获取参数, 获取名字
        #params += np.prod(param.shape)
        #print(name)
    #print(params, 26604328)
