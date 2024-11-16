import torch
import torch.nn as nn
import math
import tiktoken


class SelfAttention(nn.Module):
    
    def __init__(self, embed_dim):
        super().__init__()
        
        d = embed_dim
        
        self.q_weight = nn.Linear(d, d)
        self.k_weight = nn.Linear(d, d)
        self.v_weight = nn.Linear(d, d)

    
    def forward(self, x):

        # x and matrices shape (B, N, d)

        # scaling before matrix multiplication for numerical stability
        q_mat = self.q_weight(x)
        q_mat = q_mat/math.sqrt(q_mat.shape[-1])

        k_mat = self.k_weight(x)
        v_mat = self.v_weight(x)

        # attention shape (B, N, N)
        
        # both matmul and bmm works, bmm more efficient
        attention = torch.bmm(q_mat, k_mat.transpose(-1, -2))


        N = attention.shape[-1]

        # torch.full generates a tensor filled with -1e9
        # torch.triu sets the lower triangle of a matrix to zero, the diagonal should not be masked.
        mask = torch.triu(torch.full((N, N), -1e9, device=x.device), diagonal=1)
        
        causal_attention = attention + mask

        # torch softmax handles numerical stability itself
        attention_scores = torch.softmax(causal_attention, dim=-1)
        
        out = torch.bmm(attention_scores, v_mat)

        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, embed_dim):
        
        super().__init__()

        self.n_heads = n_heads
        self.embed_dim = d = embed_dim
        self.head_dim = self.embed_dim // self.n_heads

        self.q_weights = nn.Linear(d, d)
        self.k_weights = nn.Linear(d, d)
        self.v_weights = nn.Linear(d, d)

        self.o_weights =  nn.Linear(d, d)
    
    def attention(self, q_mat, k_mat, v_mat):
        
        # scaling before matrix multiplication for numerical stability
        q_mat = q_mat/math.sqrt(q_mat.shape[-1])

        # attention shape (B, N, N)
        
        # both matmul and bmm works, bmm more efficient
        attention = torch.bmm(q_mat, k_mat.transpose(-1, -2))

        N = attention.shape[-1]

        # torch.full generates a tensor filled with -1e9
        # torch.triu sets the lower triangle of a matrix to zero, the diagonal should not be masked.
        mask = torch.triu(torch.full((N, N), -1e9, device=x.device), diagonal=1)
        
        causal_attention = attention + mask

        # torch softmax handles numerical stability itself
        attention_scores = torch.softmax(causal_attention, dim=-1)
        
        out = torch.bmm(attention_scores, v_mat)

        return out

    def forward(self, x):

        B, N, D = x.shape

        q_mat = self.q_weights(x)
        k_mat = self.k_weights(x)
        v_mat = self.v_weights(x)


        q_mat_reshaped = q_mat.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k_mat_reshaped = k_mat.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v_mat_reshaped = v_mat.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        
        outputs = []

        for i in range(self.n_heads):
            q_head = q_mat_reshaped[:, i, ...]
            k_head = k_mat_reshaped[:, i, ...]
            v_head = v_mat_reshaped[:, i, ...]

            out_head = self.attention(q_head, k_head, v_head)

            outputs.append(out_head)
        
        outputs = torch.cat(outputs, dim=-1)

        out = self.o_weights(outputs)

        return out


class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(embed_dim))
        self.shift = nn.Parameter(torch.zeros(embed_dim))

    # normalizing each token
    def forward(self, x, eps=1e-5):
        # x (B, N, D)
        mean = torch.mean(x, dim=-1, keepdim=True)
        variance = torch.var(x, dim=-1, keepdim=True)
        x_normalized = (x-mean) / torch.sqrt(variance+eps)
        out = self.scale * x_normalized + self.shift
        return out

class FeedForward(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class DecoderTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        embed_dim = cfg['embed_dim'] 
        self.ln1 = LayerNorm(embed_dim=embed_dim)
        self.ln2 = LayerNorm(embed_dim=embed_dim)
        self.multi_attention = MultiHeadAttention(n_heads=cfg['n_heads'], embed_dim=embed_dim)
        self.feed_forward = FeedForward(embed_dim=embed_dim)
        self.dropout = nn.Dropout(cfg['drop_rate'])

    def forward(self, x):
        init_x = x
        
        x = self.ln1(x)
        x = self.multi_attention(x)
        x = self.dropout(x)
        
        x = x + init_x
        init_x = x
        
        x = self.ln2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        
        out = x + init_x
        return out


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # we need vocab size for token embedding, and last linear layer
        vocab_size = cfg['vocab_size']
        # we need context length for position embedding
        context_length = cfg['context_length']
        embed_dim = cfg['embed_dim']
        n_layers = cfg['n_layers']
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(context_length, embed_dim)
        self.dropout = nn.Dropout(cfg['drop_rate'])
        self.transformer_blocks = nn.Sequential(
            *[DecoderTransformerBlock(cfg) for _ in range(n_layers)]
        )
        self.last_ln = LayerNorm(embed_dim=embed_dim)
        self.out_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x_token_embeddings = self.token_embedding(x)
        _, context_length = x.shape
        pos_embeddings = torch.arange(context_length, device=x.device)
        x_pos_embeddings = self.position_embedding(pos_embeddings)

        x = x_token_embeddings + x_pos_embeddings
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.last_ln(x)
        out = self.out_head(x)

        return out


def generate(model, token_ids, max_new_tokens=1):
    
    for _ in range(max_new_tokens):
        
        logits = model(token_ids)
        selected_tokens = torch.argmax(logits, dim=-1)[:, -1].unsqueeze(-1)
        print(selected_tokens.shape)
        token_ids = torch.cat([token_ids, selected_tokens], dim=-1)

    return token_ids


def encode(text, tokenizer):
    token_ids = torch.tensor(tokenizer.encode(text)) # batch dimension
    return token_ids

def decode(token_ids, tokenizer):
    if token_ids.dim() > 1:
        token_ids = token_ids.squeeze(0)
    token_ids = token_ids.tolist()
    text = tokenizer.decode(token_ids)
    return text

def batch_decode(token_ids, tokenizer):
    pass


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


from load_data import create_data_loader



def train(model, optimizer, tokenizer, loss_fn_type="CrossEntropy", context_length=512, stride=512, train_cfg=None):

    if loss_fn_type == "CrossEntropy":
        loss_fn = nn.CrossEntropyLoss()

    batch_size = train_cfg['batch_size']
    n_epochs = train_cfg['n_epochs']
    train_loader, val_loader = create_data_loader(tokenizer, context_length, stride, batch_size)
    model.train()
    loss_steps = []
    for i,(input_ids, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.shape[-1]), labels.shape(-1, 1))
        loss_steps.append(loss)
        loss.backward()
        optimizer.step()
        print(f'loss step{i}: {loss_steps[-1]}')

    return model


if __name__ == "__main__":
    
    model_cfg = {
        "vocab_size": 50257,    # Vocabulary size
        "context_length": 1024, # Context length
        "embed_dim": 768,         # Embedding dimension
        "n_heads": 12,          # Number of attention heads
        "n_layers": 12,         # Number of layers
        "drop_rate": 0.1,       # Dropout rate
    }

    train_cfg = {
        'batch_size': 2,
        'n_epochs': 1
    }

    # model = MultiHeadAttention(cfg)
    # model = LayerNorm(cfg)

    # model = DecoderTransformerBlock(cfg)

    model = GPT(model_cfg)

    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total number of parameters: {total_params:,}")

    tokenizer = tiktoken.get_encoding("gpt2")



    optimizer = torch.optim.SGD(model.parameters())

    train(model, optimizer, tokenizer, train_cfg=train_cfg)


    # x = torch.tensor(
    #     [[1, 2, 3, 4], [5, 1, 3, 2], [4, 0, 2, 9], [3, 2, 5, 3]],
    #     dtype=torch.long
    # )
    
    # text1 = "Every effort moves you"
    # text2 = "Hi, you are"

    # x = encode(text1, tokenizer)
    # y = encode(text2, tokenizer)

    # token_ids =  torch.stack([x, y], dim=0)
    # out = generate(model=model, token_ids=token_ids, max_new_tokens=10)
    
    # print(out)
    
    # response = decode(out[0], tokenizer)

    # print(response)



