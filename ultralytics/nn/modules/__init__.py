# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules. Visualize with:

from ultralytics.nn.modules import *
import torch
import os

x = torch.ones(1, 128, 40, 40)
m = Conv(128, 128)
f = f'{m._get_name()}.onnx'
torch.onnx.export(m, x, f)
os.system(f'onnxsim {f} {f} && open {f}')
"""

from .block import (C1, C2, C3, C3TR, DFL, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x, GhostBottleneck,
                    HGBlock, HGStem, Proto, RepC3,Bottle2neckv2,Bottle2neckv3,SSPPFCSPC,LSKBlock,SPPFCSPC,
                   SimFusion_3in,SimFusion_4in,IFM,InjectionMultiSum_Auto_pool,PyramidPoolAgg,AdvPoolFusion,TopBasicLayer,GSConv,VoVGSCSP,C2f_DBB,BiLevelRoutingAttention,space_to_depth,Auxiliary,CoordAtt,Bottle2neck,C2f_Res2Block,CAM,RCSOSA,C3STR,ContextGuidedBlock,ContextGuidedBlock_Down,C2f_ContextGuided,MSBlock,ACmix,BiFPN_Add2,BiFPN_Add3,MobileNetV3,ASFF2,ASFF3,SimPPFCSPC,SwinV2_CSPB,C2f_Faster,Fusion,C2f_ScConv,FocalModulation,Partial_conv3,PConv,C2f_ODConv,MY_EMSConv,C2f_EMSC)
from .conv import (CBAM, ChannelAttention, Concat, Conv, Conv2, ConvTranspose, DWConv, DWConvTranspose2d, Focus,
                   GhostConv, LightConv, RepConv, SpatialAttention)
from .head import Classify, Detect, Pose, RTDETRDecoder, Segment,Detect_Efficient,Detect_DyHead
from .transformer import (AIFI, MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer, LayerNorm2d,
                          MLPBlock, MSDeformAttn, TransformerBlock, TransformerEncoderLayer, TransformerLayer)

__all__ = ('Conv', 'Conv2', 'LightConv', 'RepConv', 'DWConv', 'DWConvTranspose2d', 'ConvTranspose', 'Focus',
           'GhostConv', 'ChannelAttention', 'SpatialAttention', 'CBAM', 'Concat', 'TransformerLayer',
           'TransformerBlock', 'MLPBlock', 'LayerNorm2d', 'DFL', 'HGBlock', 'HGStem', 'SPP', 'SPPF', 'C1', 'C2', 'C3',
           'C2f', 'C3x', 'C3TR', 'C3Ghost', 'GhostBottleneck', 'Bottleneck', 'BottleneckCSP', 'Proto', 'Detect',
           'Segment', 'Pose', 'Classify', 'TransformerEncoderLayer', 'RepC3', 'RTDETRDecoder', 'AIFI',
           'DeformableTransformerDecoder', 'DeformableTransformerDecoderLayer', 'MSDeformAttn', 'MLP','Bottle2neckv2','Bottle2neckv3',
           'SSPPFCSPC','LSKBlock','Detect_Efficient','SPPFCSPC','SimFusion_3in','SimFusion_4in',
           'IFM','InjectionMultiSum_Auto_pool','PyramidPoolAgg','AdvPoolFusion','TopBasicLayer','GSConv','VoVGSCSP','C2f_DBB','BiLevelRoutingAttention','space_to_depth','Auxiliary','CoordAtt','Bottle2neck','C2f_Res2Block','CAM','Detect_DyHead','RCSOSA','C3STR','ContextGuidedBlock','ContextGuidedBlock_Down','C2f_ContextGuided','MSBlock','ACmix','BiFPN_Add2','BiFPN_Add3','MobileNetV3','ASFF2','ASFF3','SimPPFCSPC','SwinV2_CSPB','C2f_Faster','Fusion','C2f_ScConv','FocalModulation','Partial_conv3','PConv','C2f_ODConv','MY_EMSConv','C2f_EMSC')
