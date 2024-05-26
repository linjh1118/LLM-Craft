import torch
import torch.nn as nn

"""Learn from B站 UP主 小鹿乙xx"""

class PatchEmbedding(nn.Module):
    """功能：传入图片；【对图片进行分patch；每个patch投影成一个词向量】；加上cls这个句首聚合词向量；加上随机的位置编码"""
    def __init__(self, in_channels=3, embed_dim=768, patch_size=14, img_size=196, dropout=0.1):
        super(PatchEmbedding, self).__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, 
                      kernel_size=patch_size, stride=patch_size),  # (bs, emb_dim, n_row, n_col)
            nn.Flatten(2),  # 将矩阵排列的词向量 变成 列表排列
            )
        
        self.cls_emb = nn.Parameter(data=torch.randn(size=[1, 1, embed_dim]), requires_grad=True)
        
        self.n_patches = (img_size // patch_size) ** 2
        self.n_postions = self.n_patches + 1

        self.pos_emb = nn.Parameter(data=torch.randn(size=[1, self.n_postions, embed_dim]), requires_grad=True)

        self.dropout = nn.Dropout(p=dropout)
        
        print('finish init class `PatchEmbedding`')

    def forward(self, pixel_imgs: torch.FloatTensor):
        patch_emb = self.patcher(pixel_imgs) # (bs, emb_dim, n_row*n_col)
        patch_emb = patch_emb.permute(0, 2, 1)
        
        batch_size = patch_emb.shape[0]
        batch_cls_emb = self.cls_emb.expand(batch_size, 1, -1)
        patch_emb = torch.cat([batch_cls_emb, patch_emb], dim=1)

        cur_n_patches = patch_emb.size[1]
        patch_emb = patch_emb + self.pos_emb[:, :cur_n_patches, :]  # 支持更小的图片，所以对pos_emb做一个截取
        
        patch_emb = self.dropout(patch_emb)

        return patch_emb

class ViT(nn.Module):
    def __init__(self, n_class, num_layers, embed_dim, n_head, dropout, activation) -> None:
        super(ViT, self).__init__()

        self.patch_emb_layer = PatchEmbedding(in_channels=3, embed_dim=768, patch_size=14, img_size=196, dropout=0.1)

        each_encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_head, 
                                                        dropout=dropout, activation=activation,
                                                        norm_first=True, batch_first=True
                                                        )
        self.encoder_layers = nn.TransformerEncoder(encoder_layer=each_encoder_layer, num_layers=num_layers)
        
        self.classify_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=n_class)
            )

    def forward(self, pixel_imgs):
        x = self.patch_emb_layer(pixel_imgs)
        x = self.encoder_layers(x)
        batch_logits = self.classify_head(x[:, 0, :]) # take cls
        return batch_logits

