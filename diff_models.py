import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from linear_attention_transformer import LinearAttentionTransformer


def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)

def get_linear_trans(heads=8,layers=1,channels=64,localheads=0,localwindow=0):

  return LinearAttentionTransformer(
        dim = channels,
        depth = layers,
        heads = heads,
        max_seq_len = 256,
        n_local_attn_heads = 0, 
        local_attn_window_size = 0,
    )

def Conv1d_with_init(in_channels, out_channels, kernel_size,** kwargs):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size,** kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table


class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                    is_linear=config["is_linear"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape

        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)

        diffusion_emb = self.diffusion_embedding(diffusion_step)

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x



    # 在diff_models.py中修改ResidualBlock类，增加自适应注意力权重计算
class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads, is_linear=False):
        super().__init__()
        # 原有初始化代码...
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)
        
        # 新增：注意力自适应权重模块（计算局部波动程度）
        self.attention_gate = nn.Sequential(
            Conv1d_with_init(channels, channels//2, 3, padding=1),
            nn.ReLU(),
            Conv1d_with_init(channels//2, 1, 1),
            nn.Sigmoid()
        )

        self.is_linear = is_linear
        # 原有注意力层初始化...
        # ---------------------- 新增：初始化 time_layer 和 feature_layer ----------------------
        # 时间维度注意力层（time_layer）
        if self.is_linear:
            # 线性注意力（原CSDI中用于长序列，用Linear实现）
            self.time_layer = nn.Linear(channels, channels)
        else:
            # 多头自注意力（原CSDI默认，用MultiheadAttention实现）
            self.time_layer = nn.MultiheadAttention(
                embed_dim=channels,  # 输入特征维度
                num_heads=nheads,    # 头数，来自配置参数
                batch_first=False    # 注意：原代码中维度顺序为 (seq_len, batch, dim)
            )
        
        # 特征维度注意力层（feature_layer）
        if self.is_linear:
            self.feature_layer = nn.Linear(channels, channels)
        else:
            self.feature_layer = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=nheads,
                batch_first=False
            )
    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)

        # 新增：计算时间维度注意力权重（基于局部波动）
        time_volatility = torch.std(y, dim=-1, keepdim=True)  # 局部波动率
        time_gate = self.attention_gate(y) * time_volatility  # 波动越大，权重越高
        
        if self.is_linear:
            attn_output = self.time_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            attn_output = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        
        # 应用自适应权重
        y = attn_output * time_gate + y * (1 - time_gate)  # 残差融合
        
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    # 同理修改forward_feature方法，增加特征维度的自适应权重
    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        
        # 新增：计算特征维度注意力权重
        feature_volatility = torch.std(y, dim=-1, keepdim=True)
        feature_gate = self.attention_gate(y) * feature_volatility
        
        if self.is_linear:
            attn_output = self.feature_layer(y.permute(0, 2, 1)).permute(0, 2, 1)
        else:   其他:
            attn_output = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        
        # 应用自适应权重
        y = attn_output * feature_gate + y * (1 - feature_gate)
        
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip
