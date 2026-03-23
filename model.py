import torch

from peft import get_peft_model
from torch import nn
from transformers import T5ForConditionalGeneration, AutoConfig

class CustomByT5Model(nn.Module):
    
    def __init__(self, model_name='google/byt5-small', input_dim=255, load_weights=True, lora_config=None):
        
        super().__init__()
        
        if load_weights:
            self.byt5 = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.byt5 = T5ForConditionalGeneration(config)

        if lora_config:
            self.byt5 = get_peft_model(self.byt5, lora_config)
        
        self.projection = nn.Linear(input_dim, self.byt5.config.d_model)

        self.generation_config = self.byt5.generation_config
        # Ez dakit jarri beharko nukeen:
        self.config = self.byt5.config

   
    def forward(self, input_ids, input_vectors, text_attention_mask, vectors_attention_mask, labels=None, **kwargs): # labels esplizituki jarri gabe ebaluazioan arazoak ematen ditu
        
        if 'num_items_in_batch' in kwargs:
            kwargs.pop('num_items_in_batch') # huggingface bertsioa eguneratzean arazoak ematen hasi zelako. Ez dakit zertarako behar den
        
        projected_inputs = self.projection(input_vectors) # (batch_size, vectors_seq_len, d_model)
        text_embeds = self.byt5.encoder.embed_tokens(input_ids) # (batch_size, text_seq_len, d_model) Baliteke ez funtzionatzea batch_size dimentsio bakarra baino gehiago badaude
        input_embeds = torch.cat((text_embeds, projected_inputs), dim=1) # (batch_size, text_seq_len + vectors_seq_len, d_model)
        attention_mask = torch.cat((text_attention_mask, vectors_attention_mask), dim=1)
        outputs = self.byt5(inputs_embeds=input_embeds,
                            attention_mask=attention_mask,
                            labels=labels,
                            **kwargs)

        return outputs

    def generate(self, input_ids, input_vectors, text_attention_mask, vectors_attention_mask, **kwargs):
        projected_inputs = self.projection(input_vectors)
        text_embeds = self.byt5.encoder.embed_tokens(input_ids)
        input_embeds = torch.cat((text_embeds, projected_inputs), dim=1)
        attention_mask = torch.cat((text_attention_mask, vectors_attention_mask), dim=1)
        # igual aldatu beharko nuke beti tamaina berekoak sortu ditzan, gero -100ak ez agertzeko?
        outputs = self.byt5.generate(inputs_embeds=input_embeds,
                                     attention_mask=attention_mask,
                                     **kwargs)
        return outputs



class VectorsOnlyCustomByT5Model(nn.Module):
    def __init__(self, model_name='google/byt5-small', input_dim=255, load_weights=True, lora_config=None):
        super().__init__()
        
        if load_weights:
            self.byt5 = T5ForConditionalGeneration.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.byt5 = T5ForConditionalGeneration(config)

        if lora_config:
            self.byt5 = get_peft_model(self.byt5, lora_config)
        
        self.projection = nn.Linear(input_dim, self.byt5.config.d_model)

        self.generation_config = self.byt5.generation_config
        # Ez dakit jarri beharko nukeen:
        self.config = self.byt5.config

    #def forward(self, input_vectors, attention_mask=None, labels=None): # decoder_input_ids=None, ?
    def forward(self, input_vectors, labels=None, **kwargs): # labels esplizituki jarri gabe ebaluazioan arazoak ematen ditu
        if 'num_items_in_batch' in kwargs:
            kwargs.pop('num_items_in_batch') # huggingface bertsioa eguneratzean arazoak ematen hasi zelako. Ez dakit zertarako behar den
        
        # Project input vectors to the required dimension
        projected_inputs = self.projection(input_vectors)

        # Pass through ByT5 model
        outputs = self.byt5(inputs_embeds=projected_inputs,
                            labels=labels,
                            **kwargs)

        return outputs
    
    def generate(self, input_vectors, **kwargs):
        projected_inputs = self.projection(input_vectors)
        # igual aldatu beharko nuke beti tamaina berekoak sortu ditzan, gero -100ak ez agertzeko?
        outputs = self.byt5.generate(inputs_embeds=projected_inputs, **kwargs)
        return outputs
    
