import torch
import torch.nn as nn
from torch.nn import functional as F
from .gcc import GraphResBlock
from .ogcc import OGraphResBlock
from funcs_utils import init_weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=2048+3, nhead=1, dim_feedforward=2048+3, kdim=256+3+3, vdim=256+3+3, dropout=0.0, activation="gelu"):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim_feedforward, nhead, dropout=dropout, kdim=kdim, vdim=vdim)
        self.linear = nn.Linear(d_model, dim_feedforward)
        self.linear1 = nn.Linear(dim_feedforward, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim_feedforward)
        self.norm1 = nn.LayerNorm(dim_feedforward)
        self.norm2 = nn.LayerNorm(dim_feedforward)
        self.activation = F.gelu

    def forward(self, src_q, src_k, src_v):
        src_q, src_k, src_v = src_q.permute(1,0,2), src_k.permute(1,0,2), src_v.permute(1,0,2)
        src = src_q
        src2, _ = self.cross_attn(src_q, src_k, src_v)

        src = src + self.dropout(src2)
        src = self.norm(src)
        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src.permute(1,0,2)

class TGformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.Transformer_layer1,self.Transformer_layer2,self.Transformer_layer3 = [],[],[]
        self.graphResBlock1, self.graphResBlock2, self.graphResBlock3 = [],[],[]
        self.ographResBlock1, self.ographResBlock2, self.ographResBlock3 = [],[],[]
        self.num_layers = 4
        self.dim = [1024,512,256]
        self.trans_linear1 = nn.Linear(2051, 1024)
        self.trans_linear2 = nn.Linear(1024, 512)
        self.trans_linear3 = nn.Linear(512, 256)
        self.trans_linear4 = nn.Linear(256, 3)
        for i in range(self.num_layers):
            transformer_layer = TransformerEncoderLayer(d_model=self.dim[0], nhead=4, dim_feedforward=self.dim[0], kdim=self.dim[0], vdim=self.dim[0])
            self.Transformer_layer1.append(transformer_layer)
            graphResBlock = GraphResBlock(in_channels=self.dim[0], out_channels=self.dim[0], mesh_type='body')
            self.graphResBlock1.append(graphResBlock)

            # graphResBlock = OGraphResBlock(in_channels=self.dim[0], out_channels=self.dim[0], mesh_type='body')
            # self.ographResBlock1.append(graphResBlock)

        for i in range(self.num_layers):
            transformer_layer = TransformerEncoderLayer(d_model=self.dim[1], nhead=4, dim_feedforward=self.dim[1], kdim=self.dim[1], vdim=self.dim[1])
            self.Transformer_layer2.append(transformer_layer)
            graphResBlock = GraphResBlock(in_channels=self.dim[1], out_channels=self.dim[1], mesh_type='body')
            self.graphResBlock2.append(graphResBlock)

            graphResBlock = OGraphResBlock(in_channels=self.dim[1], out_channels=self.dim[1], mesh_type='body')
            self.ographResBlock2.append(graphResBlock)

        for i in range(self.num_layers):
            transformer_layer = TransformerEncoderLayer(d_model=self.dim[2], nhead=4, dim_feedforward=self.dim[2], kdim=self.dim[2], vdim=self.dim[2])
            self.Transformer_layer3.append(transformer_layer)
            graphResBlock = GraphResBlock(in_channels=self.dim[2], out_channels=self.dim[2], mesh_type='body')
            self.graphResBlock3.append(graphResBlock)

            #graphResBlock = OGraphResBlock(in_channels=self.dim[2], out_channels=self.dim[2], mesh_type='body')
            #self.ographResBlock3.append(graphResBlock)

        self.Transformer_layer1 = nn.ModuleList(self.Transformer_layer1)
        self.Transformer_layer2 = nn.ModuleList(self.Transformer_layer2)
        self.Transformer_layer3 = nn.ModuleList(self.Transformer_layer3)
        self.graphResBlock1 = nn.ModuleList(self.graphResBlock1)
        self.graphResBlock2 = nn.ModuleList(self.graphResBlock2)
        self.graphResBlock3 = nn.ModuleList(self.graphResBlock3)

        # self.ographResBlock1 = nn.ModuleList(self.ographResBlock1)
        self.ographResBlock2 = nn.ModuleList(self.ographResBlock2)
        #self.ographResBlock3 = nn.ModuleList(self.ographResBlock3)


    def init_weights(self):
        self.apply(init_weights)

    def forward(self, transformertokens,graph):
        
        transformertokens = self.trans_linear1(transformertokens)
        for i in range(self.num_layers):
            transformertokens = self.Transformer_layer1[i](transformertokens, transformertokens, transformertokens)
            joint_token = transformertokens[:,:73,:]
            human_token = transformertokens[:,73:-64,:]
            objectv_token = transformertokens[:,-64:,:]
            human_token = self.graphResBlock1[i](human_token)
            # objectv_token = self.ographResBlock1[i](objectv_token,graph)
            transformertokens = torch.cat((joint_token,human_token,objectv_token),dim=1)


        transformertokens = self.trans_linear2(transformertokens)
        for i in range(self.num_layers):
            transformertokens = self.Transformer_layer2[i](transformertokens, transformertokens, transformertokens)
            joint_token = transformertokens[:,:73,:]
            human_token = transformertokens[:,73:-64,:]
            objectv_token = transformertokens[:,-64:,:]
            human_token = self.graphResBlock2[i](human_token)
            objectv_token = self.ographResBlock2[i](objectv_token,graph)
            transformertokens = torch.cat((joint_token,human_token,objectv_token),dim=1)

        transformertokens = self.trans_linear3(transformertokens)
        for i in range(self.num_layers):
            transformertokens = self.Transformer_layer3[i](transformertokens, transformertokens, transformertokens)
            joint_token = transformertokens[:,:73,:]
            human_token = transformertokens[:,73:-64,:]
            objectv_token = transformertokens[:,-64:,:]
            human_token = self.graphResBlock3[i](human_token)
            # objectv_token = self.ographResBlock3[i](objectv_token,graph)
            transformertokens = torch.cat((joint_token,human_token,objectv_token),dim=1)
            


        transformertokens = self.trans_linear4(transformertokens)

        joint = transformertokens[:,:73,:]
        human = transformertokens[:,73:-64,:]
        objectv = transformertokens[:,-64:,:]


        return joint, human, objectv
    
