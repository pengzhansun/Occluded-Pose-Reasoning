# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import matplotlib.pyplot as plt
from .rtm import RTMCCBlock
warnings.simplefilter("ignore")
import numpy as np

class Embedding_linear(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding_linear, self).__init__()
        self.embed = nn.Sequential(
                          nn.Linear(vocab_size, embed_dim),
                          nn.ReLU(),
                        #   nn.Linear(embed_dim, embed_dim)
                        )
    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out
    

# register buffer in Pytorch ->
# If you have parameters in your model, which should be saved and restored in the state_dict,
# but not trained by the optimizer, you should register them as buffers.


class PositionalEmbedding(nn.Module):
    def __init__(self,max_seq_len,embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim
        self.max_seq_len = max_seq_len

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        self.pe = pe.unsqueeze(0).cuda()

        # self.pe = nn.Embedding(max_seq_len, embed_model_dim)
        

        # self.register_buffer('pe', pe)


    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)

        # position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=x.device)
        # position_ids = position_ids.unsqueeze(0).repeat(x.shape[0], 1)

        # pe = self.pe(position_ids)

        # x += pe

        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    #512 dim
        self.n_heads = n_heads   #8
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #512/8 = 64  . each key,query, value will be of 64d
       
        #key,query and value matrixes    #64 x 64   
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 

        nn.init.xavier_uniform_(self.query_matrix.weight)
        nn.init.xavier_uniform_(self.key_matrix.weight)
        nn.init.xavier_uniform_(self.value_matrix.weight)

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        seq_length_query = query.size(1)
        
        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
      
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)

        # attention_scores = scores.clone().detach().cpu()
        # print(attention_scores)

        # print(v.shape)
        # print(scores.shape)
        # exit()
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##torch.Size([32, 2, 17, 17]) * torch.Size([32, 2, 17, 256]) -> torch.Size([32, 2, 17, 256])
        
        
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        # print(concat.shape)

        output = self.out(concat) #(32,10,512) -> (32,10,512)
        # print(output.shape)
        # exit()
       
        # return output, attention_scores
        return output


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                          nn.Linear(embed_dim, expansion_factor*embed_dim),
                          nn.SiLU(),
                          nn.Linear(expansion_factor*embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)

    def forward(self,key,query,value):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
        """
        
        # attention_out, attention_scores = self.attention(key,query,value)  #32x10x512
        attention_out = self.attention(key,query,value)  #32x10x512
        attention_residual_out = attention_out + value  #32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) #32x10x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) #32x10x512

        # return norm2_out, attention_scores
        return norm2_out



class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, cfg, seq_len, vocab_size, embed_dim, output_dim, pe=False, num_layers=2, expansion_factor=1, n_heads=8):
        super(TransformerEncoder, self).__init__()
        
        self.output_dim = output_dim

        self.embedding_layer = Embedding_linear(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.pe = pe

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
        self.fc = nn.Linear(embed_dim, cfg.MODEL.HEAD_INPUT)

        # self.mlp_head_x = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)
        # self.mlp_head_y = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)

        # self.mlp_head_x = nn.Linear(embed_dim, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)
        # self.mlp_head_y = nn.Linear(embed_dim, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)


    def get_GT_visibility_mask(self, visible_state):
        
        visible_state = visible_state.cpu().numpy()
        # print(visible_state)
        # print(visible_state.shape)
        is_visible = np.equal(visible_state, 2).astype(np.float32)
        # print(is_visible.shape)
        visible_mask = torch.from_numpy(is_visible)

        is_occluded = np.not_equal(visible_state, 2).astype(np.float32)
        occluded_mask = torch.from_numpy(is_occluded)

        mask = visible_mask + occluded_mask * 0.01

        return mask.cuda()
    

    def get_visibility_mask(self, visible_state):
        
        visible_state = visible_state.cpu().numpy()
        # print(visible_state)
        # print(visible_state.shape)
        is_visible = np.equal(visible_state, 1).astype(np.float32)
        # print(is_visible.shape)
        visible_mask = torch.from_numpy(is_visible)

        is_occluded = np.equal(visible_state, 0).astype(np.float32)
        occluded_mask = torch.from_numpy(is_occluded)

        mask = visible_mask + occluded_mask * 0.01

        return mask.cuda()

    def forward(self, x, visibility_state, occlusion_mask_strategy):
        attention_score_list = []
        # print('x.shape:', x.shape)
        x = x.view(x.shape[0], x.shape[1], -1)
        
        # print(x.dtype)
        if occlusion_mask_strategy:
            
            # print(visibility_state)
            mask = self.get_visibility_mask(visibility_state)

            # print(mask)
            x_masked = x.mul(mask.unsqueeze(2))
            out = self.embedding_layer(x_masked)

        else:
            out = self.embedding_layer(x)
            # print('you should be here')
        
        if self.pe:
            out = self.positional_encoder(out)
        for layer in self.layers:
            # out, attention_scores = layer(out,out,out)  # torch.Size([32, 17, 512]) 2 attention head (256 * 2 = 512)
            out = layer(out,out,out)
            # print(attention_scores.shape)
            # attention_score_list.append(attention_scores)
        
        # attention_score = torch.cat(attention_score_list, dim=1)
        
        out = self.fc(out) + x
        # return out, attention_score
        return out


class TransformerEncoder_OccMap(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, cfg, seq_len, vocab_size, embed_dim, output_dim, pe=False, num_layers=2, expansion_factor=1, n_heads=8):
        super(TransformerEncoder_OccMap, self).__init__()
        
        self.output_dim = output_dim

        self.embedding_layer = Embedding_linear(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.pe = pe

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
        self.fc = nn.Linear(embed_dim, cfg.MODEL.HEAD_INPUT)
        


    def forward(self, x):
        # print('x.shape:', x.shape)
        x = x.view(x.shape[0], x.shape[1], -1)
        out = self.embedding_layer(x)

        # print(x.dtype)
        if self.pe:
            out = self.positional_encoder(out)
        for layer in self.layers:
            out = layer(out,out,out)

        out = self.fc(out) + x
        return out



class TransformerOPORTEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, cfg, seq_len, vocab_size, embed_dim, output_dim, pe=False, num_layers=2, expansion_factor=1, n_heads=8):
        super(TransformerOPORTEncoder, self).__init__()
        
        self.output_dim = output_dim
        self.conv_learn_tokens = nn.Conv1d(256, cfg.MODEL.NUM_JOINTS, cfg.MODEL.HEATMAP_SIZE[0]*cfg.MODEL.HEATMAP_SIZE[1])

        self.embedding_layer = Embedding_linear(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.pe = pe

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
    
        self.fc = nn.Linear(embed_dim, cfg.MODEL.HEAD_INPUT)

        # self.mlp_head_x = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)
        # self.mlp_head_y = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)

        # self.mlp_head_x = nn.Linear(embed_dim, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)
        # self.mlp_head_y = nn.Linear(embed_dim, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)


    def get_visibility_mask(self, visible_state):
        
        visible_state = visible_state.cpu().numpy()
        # print(visible_state)
        # print(visible_state.shape)
        is_visible = np.equal(visible_state, 2).astype(np.float32)
        # print(is_visible.shape)
        visible_mask = torch.from_numpy(is_visible)

        is_occluded = np.not_equal(visible_state, 2).astype(np.float32)
        occluded_mask = torch.from_numpy(is_occluded)

        mask = visible_mask + occluded_mask * 0.01

        return mask.cuda()

    def forward(self, x, visibility_state, occlusion_mask_strategy):
        # print('x.shape:', x.shape)
        x = x.view(x.shape[0], x.shape[1], -1)
        print(x.shape)
        exit()
        # Here, we use conv_learn_tokens
        x = self.conv_learn_tokens(x)
        
        # concat img_feat with coords
        # x = torch.cat((x, coords), 2)  # dim = heatmap.shape[0] * heatmap.shape[1] + 3 (x, y, score)
        
        
        # print(x.dtype)
        if occlusion_mask_strategy:
            mask = self.get_visibility_mask(visibility_state)
            x_masked = x.mul(mask.unsqueeze(2))
            out = self.embedding_layer(x_masked)

        else:
            out = self.embedding_layer(x)

        # print(x.dtype)
        if self.pe:
            out = self.positional_encoder(out)
        for layer in self.layers:
            out = layer(out,out,out)

        out = self.fc(out)
        # out = self.fc(out) + x
        # out = x

        # x_coord = self.mlp_head_x(out)
        # y_coord = self.mlp_head_y(out)

        # out = x[:, :, :self.output_dim] + out

        # return x_coord, y_coord #32x10x512
        return out



class Output(nn.Module):
    def __init__(self, cfg):
        super(Output, self).__init__()
        self.mlp_head_x = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)
        self.mlp_head_y = nn.Linear(cfg.MODEL.HEAD_INPUT, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)
    def forward(self, x):
        x_coord = self.mlp_head_x(x)
        y_coord = self.mlp_head_y(x)
        return x_coord, y_coord

class TransformerEncoder_0(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, cfg, seq_len, vocab_size, embed_dim, output_dim, pe=False, num_layers=2, expansion_factor=2, n_heads=8):
        super(TransformerEncoder_0, self).__init__()
        
        self.output_dim = output_dim

        self.embedding_layer = Embedding_linear(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.pe = pe

        # self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])

        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, expansion_factor=expansion_factor)
        # self.gau = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.gau = RTMCCBlock(
            embed_dim,
            embed_dim,
            embed_dim,
            s=128,
            expansion_factor=expansion_factor,
            dropout_rate=0.,
            drop_path=0.,
            attn_type='self-attn',
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False)

        self.fc = nn.Linear(embed_dim, vocab_size)

        self.mlp_head_x = nn.Linear(vocab_size, int(cfg.MODEL.IMAGE_SIZE[0]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)
        self.mlp_head_y = nn.Linear(vocab_size, int(cfg.MODEL.IMAGE_SIZE[1]*cfg.MODEL.SIMDR_SPLIT_RATIO), bias=False)

    def forward(self, x):
        # print('x.shape:', x.shape)
        x = x.view(x.shape[0], x.shape[1], -1)
        out = self.embedding_layer(x)
        if self.pe:
            out = self.positional_encoder(out)
        # for layer in self.layers:
        #     out = layer(out,out,out)

        # out = self.transformer_encoder(out)

        out = self.gau(out)

        out = self.fc(out) + x

        x_coord = self.mlp_head_x(out)
        y_coord = self.mlp_head_y(out)

        # out = x[:, :, :self.output_dim] + out

        return x_coord, y_coord #32x10x512


# class JointViabilityNet(nn.Module):
#     def __init__(self):
#         super(JointViabilityNet, self).__init__()
#         self.conv1 = nn.Conv1d(896, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
#         self.maxpool = nn.MaxPool1d(kernel_size=3, stride=3)
#         self.flatten = nn.Flatten()
#         self.fc1 = nn.Linear(2688, 64)
#         self.dropout = nn.Dropout(0.5)
#         self.fc2 = nn.Linear(64, 1)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         x = x.permute(0, 2, 1)  # reshape to [batch_size, num_joint, input_feature_dim]
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = self.maxpool(x)
#         print(x.shape)
#         x = self.flatten(x)
#         print(x.shape)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         x = self.sigmoid(self.fc2(x))
#         x = x.view(-1, 17, 1)   # reshape to [batch_size, num_joint, 1]
#         return x


class JointViabilityNet(nn.Module):
    def __init__(self):
        super(JointViabilityNet, self).__init__()
        self.fc1 = nn.Linear(896, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        x = x.squeeze()
        return x


class HRNetJointVisibilityNet(nn.Module):
    def __init__(self):
        super(HRNetJointVisibilityNet, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv1d(in_channels=3072, out_channels=64, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1, padding=0)
        
    def forward(self, x):
        # apply convolutional layers
        x = x.view(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        # reshape output tensor to match the desired output shape
        x = x.view(-1, 17)
        # apply sigmoid activation function to get probabilities
        x = torch.sigmoid(x)
        return x


# class OccHeatmapNet(nn.Module):
#     def __init__(OccHeatmapNet, self):
#         super(OccHeatmapNet, self).__init__()

        








if __name__ == '__main__':
    input = torch.ones((32, 10, 3), dtype=torch.float)

    transformer = TransformerEncoder(10, 3, 64, 2)

    output = transformer(input)

    print(output.shape)

