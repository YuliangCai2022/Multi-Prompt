import torch
import torch.nn as nn

class Prompt(nn.Module):
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', prompt_pool=False, 
                 prompt_key=False, pool_size=None, top_k=None, batchwise_prompt=False, prompt_key_init='uniform',):
        super().__init__()

        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt

        self.text_prompts = {}
        self.text_prompts['key'] = []
        self.text_prompts['prompt'] = []
        self.image_prompts = {}
        self.image_prompts['key'] = []
        self.image_prompts['prompt'] = []
        self.combine_prompts = {}
        self.combine_prompts['key'] = []
        self.combine_prompts['prompt'] = []

        if self.prompt_pool:
            prompt_pool_shape = (pool_size, length, embed_dim)
            prompt_pool_combine_shape = (pool_size, 1, embed_dim)
            if prompt_init == 'zero':
                text_prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                image_prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                combine_prompt = nn.Parameter(torch.zeros(prompt_pool_combine_shape))
            elif prompt_init == 'uniform':
                text_prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                image_prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
                combine_prompt = nn.Parameter(torch.zeros(prompt_pool_combine_shape))
                nn.init.uniform_(text_prompt, -1, 1)
                nn.init.uniform_(image_prompt, -1, 1)
                nn.init.uniform_(combine_prompt, -1, 1)

                self.image_prompts['prompt'].append(image_prompt)
                self.text_prompts['prompt'].append(text_prompt)
                self.combine_prompts['prompt'].append(combine_prompt)
        
        # if using learnable prompt keys
        if prompt_key:
            key_shape = (pool_size, embed_dim)
            if prompt_key_init == 'zero':
                text_prompt_key = nn.Parameter(torch.zeros(key_shape))
                image_prompt_key = nn.Parameter(torch.zeros(key_shape))
                combine_prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                text_prompt_key = nn.Parameter(torch.zeros(key_shape))
                image_prompt_key = nn.Parameter(torch.zeros(key_shape))
                combine_prompt_key = nn.Parameter(torch.zeros(key_shape))
                nn.init.uniform_(text_prompt_key, -1, 1)
                nn.init.uniform_(image_prompt_key, -1, 1)
                nn.init.uniform_(combine_prompt_key, -1, 1)
                self.image_prompts['key'].append(image_prompt_key)
                self.text_prompts['key'].append(text_prompt_key)
                self.combine_prompts['key'].append(combine_prompt_key)

        else:
            # else use mean of prompt as key
            # only compatible with prompt, not prefix
            prompt_mean = torch.mean(self.text_prompt, dim=1)
            self.text_prompt_key = prompt_mean

            prompt_mean = torch.mean(self.image_prompt, dim=1)
            self.image_prompt_key = prompt_mean

            prompt_mean = torch.mean(self.combine_prompt, dim=1)
            self.combine_prompt_key = prompt_mean
    
    def update_pool(self):
        prompt_pool_shape = (self.pool_size, self.length, self.embed_dim)
        prompt_pool_combine_shape = (self.pool_size, 1, self.embed_dim)
        text_prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        image_prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        combine_prompt = nn.Parameter(torch.zeros(prompt_pool_combine_shape))
        nn.init.uniform_(text_prompt, -1, 1)
        nn.init.uniform_(image_prompt, -1, 1)
        nn.init.uniform_(combine_prompt, -1, 1)

        self.image_prompts['prompt'].append(image_prompt)
        self.text_prompts['prompt'].append(text_prompt)
        self.combine_prompts['prompt'].append(combine_prompt)

        key_shape = (self.pool_size, self.embed_dim)

        text_prompt_key = nn.Parameter(torch.zeros(key_shape))
        image_prompt_key = nn.Parameter(torch.zeros(key_shape))
        combine_prompt_key = nn.Parameter(torch.zeros(key_shape))
        nn.init.uniform_(text_prompt_key, -1, 1)
        nn.init.uniform_(image_prompt_key, -1, 1)
        nn.init.uniform_(combine_prompt_key, -1, 1)

        self.image_prompts['key'].append(image_prompt_key)
        self.text_prompts['key'].append(text_prompt_key)
        self.combine_prompts['key'].append(combine_prompt_key)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, t_or_i=0, task_id=-1):
        # t_or_i: 0 is text, 1 is image, 2 is combine
        prompt_key = None
        prompt = None
        if t_or_i == 0:
            #prompt = self.text_prompt
            #prompt_key = self.text_prompt_key
            prompt = self.text_prompts['prompt'][task_id]
            prompt_key = self.text_prompts['key'][task_id]
        elif t_or_i == 1:
            #prompt = self.image_prompt
            #prompt_key = self.image_prompt_key
            prompt = self.image_prompts['prompt'][task_id]
            prompt_key = self.image_prompts['key'][task_id]
        elif t_or_i == 2:
            #prompt = self.combine_prompt
            #prompt_key = self.combine_prompt_key
            prompt = self.combine_prompts['prompt'][task_id]
            prompt_key = self.combine_prompts['key'][task_id]

        out = dict()
        if self.prompt_pool:
            if self.embedding_key == 'mean':
                x_embed_mean = torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'max':
                x_embed_mean = torch.max(x_embed, dim=1)[0]
            elif self.embedding_key == 'mean_max':
                x_embed_mean = torch.max(x_embed, dim=1)[0] + 2 * torch.mean(x_embed, dim=1)
            elif self.embedding_key == 'cls':
                if cls_features is None:
                    x_embed_mean = torch.max(x_embed, dim=1)[0] # B, C
                else:
                    x_embed_mean = cls_features
            else:
                raise NotImplementedError("Not supported way of calculating embedding keys!")

            prompt_norm = self.l2_normalize(prompt_key, dim=1).to("cuda:0") # Pool_size, C
            x_embed_norm = self.l2_normalize(x_embed_mean, dim=1) # B, C

            similarity = torch.matmul(x_embed_norm, prompt_norm.t()) # B, Pool_size
            
            if prompt_mask is None:
                _, idx = torch.topk(similarity, k=self.top_k, dim=1) # B, top_k
                if self.batchwise_prompt:
                    prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                    # In jnp.unique, when the 'size' is specified and there are fewer than the indicated number of elements,
                    # the remaining elements will be filled with 'fill_value', the default is the minimum value along the specified dimension.
                    # Unless dimension is specified, this will be flattend if it is not already 1D.
                    if prompt_id.shape[0] < self.pool_size:
                        prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                        id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                    _, major_idx = torch.topk(id_counts, k=self.top_k) # top_k
                    major_prompt_id = prompt_id[major_idx] # top_k
                    # expand to batch
                    idx = major_prompt_id.expand(x_embed.shape[0], -1) # B, top_k
            else:
                idx = prompt_mask # B, top_k
            prompt = prompt.to("cuda:0")
            batched_prompt_raw = prompt[idx] # B, top_k, length, C
            batch_size, top_k, length, c = batched_prompt_raw.shape
            batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c) # B, top_k * length, C

            out['prompt_idx'] = idx

            # Debugging, return sim as well
            out['prompt_norm'] = prompt_norm
            out['x_embed_norm'] = x_embed_norm
            out['similarity'] = similarity

            # Put pull_constraint loss calculation inside
            batched_key_norm = prompt_norm[idx] # B, top_k, C
            out['selected_key'] = batched_key_norm
            x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
            sim = batched_key_norm * x_embed_norm # B, top_k, C
            reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar

            out['reduce_sim'] = reduce_sim
        else:
            if self.prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(self.length, self.embed_dim))
            elif self.prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(self.length, self.embed_dim))
                nn.init.uniform_(self.prompt)
            batched_prompt = self.prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        # The input with the prompt concatenated to the front. [B, prompt+token, C]
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)

        # for the combined prompt, only return the prompts
        if t_or_i == 2:
            return batched_prompt

        return out
