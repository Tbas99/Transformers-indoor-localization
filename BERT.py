import math
import torch
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        """
        Args:
            seq_len: length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len,self.embed_dim)
        for pos in range(max_seq_len):
            for i in range(0,self.embed_dim,2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
      
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim)
        #add constant to embedding
        seq_len = x.size(1)
        x = x + torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False)

        return x

class BERT(nn.Module):
    def __init__(self, embed_dim, hidden_size, src_vocab_size, seq_length, num_layers=6, n_heads=6, dropout_prob=0.1):
        super(BERT, self).__init__()
        self.token_embeddings = nn.Embedding(src_vocab_size, embed_dim)
        self.position_embeddings = PositionalEmbedding(seq_length, embed_dim)
        self.embedding_layer_norm = nn.LayerNorm(embed_dim)
        self.embedding_dropout = nn.Dropout(p=dropout_prob)

        self.encoders = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=n_heads * 16,
                nhead=n_heads,
                dim_feedforward=2048,
                dropout=dropout_prob,
                activation="gelu",
            ),
            num_layers=num_layers,
        )

        #self.pooler_layer = nn.Linear(hidden_size, hidden_size)
        #self.pooled_output_activate = nn.Tanh()

        self.fc1 = nn.Linear(59520, 4096)
        self.fc2 = nn.Linear(4096, 3)


    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """

        number_embeddings = self.token_embeddings(x)
        positional_embeddings = self.position_embeddings(number_embeddings)

        embeddings = self.embedding_layer_norm(positional_embeddings)
        embeddings = self.embedding_dropout(embeddings)

        encoder_outputs = self.encoders(embeddings.permute(1, 0, 2))
        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        out = torch.flatten(encoder_outputs, 1)
        out = self.fc1(out)
        out = self.fc2(out)

        return out