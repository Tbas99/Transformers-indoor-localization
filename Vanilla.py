import math
import torch
import torch.nn as nn

class Transformer(nn.Module):
    """Vanilla transformer model (encoder only) by Vaswani Et Al."""

    def __init__(self, seqLen, embDim, nHeads, hidDim, nLayers, dropout=0.5):
        """
        Args:
            seqLen: The length of the input sequence
            embDim: The number of expected features in the encoder inputs
            nHeads: The number of heads in ``nn.MultiheadAttention``
            hidDim: The dimension of the feedforward network model in ``nn.TransformerEncoder``
            nLayers: The number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
            dropout: Dropout probability (in the range [0,1])
        """
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(embDim, dropout)
        encoder_layers = TransformerEncoderLayer(embeddingDimension, numberOfHeads, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(sequenceLength, embeddingDimension)
        self.embeddingDimension = embeddingDimension
        self.linear = nn.Linear(embeddingDimension, sequenceLength)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, srcMask = None):
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            srcMask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        if srcMask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            srcMask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(device)
        output = self.transformer_encoder(src, srcMask)
        output = self.linear(output)
        return output