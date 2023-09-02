
import os
import torch
import torch.nn as nn
from modules import TransformerEncoder
from modules import TransformerEncoderLayer
from modules import SinCosPositionalEmbedding
from gpt_transformers import GPT


class GPTLongSeqLSTM(nn.Module):
    def __init__(
            self, num_classes, vocab_size=0, seg_seqlen=30,
            embed_dim=384, num_layers=6, num_heads=6,
            embed_dim_seg=192, num_layers_seg=1, dropout=0., bidirectional=True,
            pad_token_id=5, norm_layer=nn.LayerNorm,
            pretrained = None,
        ):
        super().__init__()
        self.transformer = GPT(
            vocab_size = vocab_size,
            max_seqlen = seg_seqlen,
            embed_dim = embed_dim, 
            num_layers = num_layers,
            num_heads = num_heads,
            eos_token_id = pad_token_id,
            pad_token_id = pad_token_id,
        )
        self.norm_seq = norm_layer(embed_dim)
        self.seg_encoder = nn.LSTM(
            embed_dim, embed_dim_seg, num_layers_seg, 
            batch_first=True, dropout=dropout, bidirectional=bidirectional,
        )
        self.fc = nn.Linear(embed_dim_seg*2, num_classes)
        
        if pretrained is not None:
            load_from_pretrained_lm(self, pretrained)

    def forward(self, x, input_mask=None):
        """
        x: (batch_size, num_segments, seg_seqlen)
        input_mask: (batch_size, num_segments, seg_seqlen)
        """
        batch_size, num_segments, seg_seqlen = x.size()
        x = x.view(-1, seg_seqlen)
        if input_mask is not None:
            seg_mask = (input_mask.sum(-1) > 0)
            seg_lengths = (seg_mask.sum(-1) - 1).long()
            input_mask = input_mask.view(-1, seg_seqlen)
            input_lengths = (input_mask.sum(-1) - 1).long()
        else:
            seg_lengths = -1
            input_lengths = -1
        output = self.transformer(input_ids=x, attention_mask=input_mask)[0]
        x = self.norm_seq(output)
        x = x[torch.arange(x.size(0), device=x.device), input_lengths]
        x = x.view(batch_size, num_segments, -1)
        x, _ = self.seg_encoder(x)
        x = x[:, -1, :]
        # x = x[torch.arange(x.size(0), device=x.device), seg_lengths]
        x = self.fc(x)
        return x
    

class GPTLongSeqTransformer(nn.Module):
    def __init__(
            self, num_classes, vocab_size=0, seg_seqlen=30, max_segnum=100,
            embed_dim=384, num_layers=6, num_heads=6, mlp_ratio=4.0,
            embed_dim_seg=192, num_layers_seg=1, num_heads_seg=6,
            pad_token_id=5, norm_layer=nn.LayerNorm,
            pretrained = None,
        ):
        super().__init__()
        self.transformer = GPT(
            vocab_size = vocab_size,
            max_seqlen = seg_seqlen,
            embed_dim = embed_dim,
            num_layers = num_layers, 
            num_heads = num_heads,
            eos_token_id = pad_token_id,
            pad_token_id = pad_token_id,
        )
        self.norm_seq = norm_layer(embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim_seg)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim_seg))
        self.pos_embed = SinCosPositionalEmbedding(embed_dim_seg, max_segnum, True)
        self.seg_encoder = TransformerEncoder(
            encoder_layer = TransformerEncoderLayer(
                d_model = embed_dim_seg,
                n_heads = num_heads_seg,
                d_ff = int(embed_dim*mlp_ratio)
            ),
            num_layers = num_layers_seg,
        )
        self.norm_seg = norm_layer(embed_dim_seg)
        self.fc = nn.Linear(embed_dim_seg, num_classes)

        if pretrained is not None:
            load_from_pretrained_lm(self, pretrained)

    def forward(self, x, input_mask=None):
        """
        x: (batch_size, num_segments, seg_seqlen)
        input_mask: (batch_size, num_segments, seg_seqlen)
        """
        batch_size, num_segments, seg_seqlen = x.size()
        x = x.view(-1, seg_seqlen)
        if input_mask is not None:
            seg_mask = (input_mask.sum(-1) > 0).float()
            seg_lengths = (seg_mask.sum(-1) - 1).long()
            input_mask = input_mask.view(-1, seg_seqlen)
            input_lengths = (input_mask.sum(-1) - 1).long()
        else:
            seg_lengths = -1
            input_lengths = -1
        output = self.transformer(input_ids=x, attention_mask=input_mask)[0]
        x = self.norm_seq(output)
        x = x[torch.arange(x.size(0), device=x.device), input_lengths]
        # project to embed_dim_seg
        x = self.proj(x)
        # reshape to (batch_size, num_segments, embed_dim_seg)
        x = x.view(batch_size, num_segments, -1)
        # append cls token
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (bs, num_patches+1, emb_dim)
        seg_mask = torch.cat((torch.ones((x.size(0), 1), device=x.device).float(), seg_mask), dim=1)
        # add pos embed w/o cls token
        x = self.pos_embed(x)
        # apply Transformer blocks
        outputs, attn_weights = self.seg_encoder(x, seg_mask)
        x = outputs[-1] # only use the last layer
        x = self.norm_seg(x)
        # pooling: cls token or mean pooling
        x = x[:, 0, :] # cls token
        # x = x[torch.arange(x.size(0), device=x.device), seg_lengths] # mean pooling
        x = self.fc(x)
        return x, attn_weights
    

def load_from_pretrained_lm(model, pretrained):
    """
    pretrained: path to pretrained model
    """
    if os.path.isfile(pretrained):
        checkpoint = torch.load(pretrained, map_location='cpu')
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict, strict=False)
        print("=> loaded checkpoint '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))
    

if __name__ == '__main__':

    sm_dir = '/home/yuty2009/data/eegdata/sleep/shhs/'
    sm_pretrained = sm_dir + '/output/gpt/session_20230707233436_30_192_1_6/checkpoint/chkpt_0050.pth.tar'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1243)

    model = GPTLongSeqLSTM(
        num_classes = 2,
        vocab_size = 6,
        seg_seqlen = 30,
        embed_dim = 192,
        num_layers = 1,
        num_heads = 6,
        pretrained = sm_pretrained,
    ).to(device)
    x = torch.vstack((
        torch.cat((torch.zeros(8, 20, 30), 5*torch.ones(8, 30, 30)), dim=1),   # b, num_segments, seg_seqlen
        torch.cat((torch.zeros(8, 30, 30), 5*torch.ones(8, 20, 30)), dim=1),   # b, num_segments, seg_seqlen
        torch.cat((torch.zeros(8, 40, 30), 5*torch.ones(8, 10, 30)), dim=1),   # b, num_segments, seg_seqlen
        torch.zeros(8, 50, 30),   # b, num_segments, seg_seqlen
    )).long()
    x[0][0][0:10] = 1
    mask = torch.ones(32, 50, 30).long()
    output = model(x.to(device), mask.to(device))
    print(output.shape)  # 32, 2
