import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

class LlavaConfig:
    def __init__(self, v_dim, t_dim):
        self.v_dim = v_dim
        self.t_dim = t_dim
        

class LlavaMultiModelProjector(nn.Module):    
    def __init__(self, config: LlavaConfig):
        v_dim, t_dim = config.v_dim, config.t_dim
        self.mlp = nn.Sequential(
            nn.Linear(v_dim, v_dim),
            nn.LeakyReLU(),
            nn.Linear(v_dim, v_dim)
        )
    
    def forward(self, img_features):
        return self.mlp(img_features)

class Llava(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()
        self.vision_encoder = AutoModel.from_pretrained(config.vision_encoder_config)
        self.projector = LlavaMultiModelProjector(config)
        self.lm = AutoModelForCausalLM.from_pretrained(config.lm_config)
        self.vit_layer_id_selected = config.layer_id_in_visual_encoder
        
    def _merge
    
    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
    ):
        # prepare text emb
        input_embs = self.get_input_embeddings()(input_ids)
        # prepare vision emb
        img_outputs = self.vision_encoder(pixel_values, output_hidden_states=True)
        selected_img_outputs = img_outputs.hidden_states[self.vit_layer_id_selected]
        img_embs = self.projector(selected_img_outputs)
        # merge
        input_embs, attention_mask, labels, position_ids = self._merge_two_embs(input_embs, img_embs)
        
            
        
        

        
        
        
        
        