import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast

output_hidden_states = True
output_attentions = True

class LlamaModel(nn.Module):
    DOCSTRING="""简单说说整体的结构，嵌入层后，接上32层block，然后rmsNorm后，通过线性层解码"""
    def __init__(self, config):
        super().__init__()
        self.input_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([LlamaLayer(config) for _ in range(config.num_hidden_layers)])
        self.rmsNorm = LLamaRMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_init()
    
    def forward(self, input_ids, attention_mask, position_ids, past_key_values):
        input_embeds = self.input_embeddings(input_ids)
        hidden_states = input_embeds
        # 往往要记录下词向量情况
        all_hidden_states = (input_embeds, )
        all_attentions = ()
        for layer in self.layers:
            model_output = layer(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)
            hidden_states = model_output['hidden_states']
            # 这里可以记录下每层的hidden_states
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if output_attentions:
                all_attentions += (model_output['attentions'])
        else:
            next_past_key_values = model_output['past_key_values']
        
        hidden_states = self.rmsNorm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_states = hidden_states,
            hidden_states = all_hidden_states,
            past_key_values = next_past_key_values,
            attentions=all_attentions
        )

a="""我基本都懂了，之后有时间把源码放上来"""
        
class LlamaLayer(nn.Module):
    DOCSTRING="""经典的Llama解码层Block, 之前的LN(x + FFN/SelfAttn(x))，现在变成 x + FFN/SelfAttn(LN(x))。 也就是合并后norm，变成了合并前，单单对主路进行norm
    ROPE的实现很简单：懂原理就行，就是在q和k向量相乘之前，对q和k分别乘一个复数，做一下旋转，进而引入位置信息。更具体举一个例子，如果q属于第一个token，那么q就旋转10度，k属于第三个token，那么q就旋转30度.
    然后加上了rope的k和v后，相乘后，复数抵销就会得到一个20度的信息。说说实现的细节，m位置的qm向量(q1q2q3q4),两两配对，q1和q2，
    通过emθ旋转，变成[q_{m1}cos(mΘ) - q_{m2}sin(mΘ), q_{m2}cos(mΘ) + q_{m1}sin(mΘ)]，然后他再实现时，做了简化，将原始的乘上cos，然后加上反转的sin即可.
    
    GQA(1234头共同一组kv，5678共同一组kv)的实现也特别简单。就是在产出的k_v的时候，本来是产出n_head * dim, 变成n_group * dim, 然后计算的时候，repeat_k_v再长出12个头的kv，去计算。
    所以GQA主要是节省了缓存的kv，当然也由于kv更瘦了，所以算kv的计算小了，但是计算qkv交互计算的时候，并节省计算量。
    """
    def __init__(self, ):
        super().__init__()
    
    def forward(hidden_states, attention_mask, position_ids, past_key_values):
        attentions = None
        return BaseModelOutputWithPast(
            last_hidden_state=None,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attentions=attentions
        )
        
        
                
        
        