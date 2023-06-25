import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelConfig:
    def __init__(
        self,
        max_sequence_length: int = 50,
        max_size_value: int = 5100,
        X_padding: int = 0,
        y_padding: int = -1,
        encoder_hidden_size: int = 128,
        dropout_rate: float = 0.1,
        num_attention_heads: int = 4,
        num_types: int = 2,
        num_encoder_layers: int = 2,
        num_labels: int = 2,
        learning_rate: float = 0.0001,
    ) -> None:
        """Config class for model."""
        self.max_sequence_length = max_sequence_length
        self.max_size_value = max_size_value
        self.X_padding = X_padding
        self.y_padding = y_padding
        self.encoder_hidden_size = encoder_hidden_size
        self.dropout_rate = dropout_rate
        self.num_attention_heads = num_attention_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.num_types = num_types


class SequenceLabelingModel(nn.Module):
    def __init__(self, hidden_size, num_labels, classifier_dropout=0.1):
        super(SequenceLabelingModel, self).__init__()
        self.encoder_model = TransformerEncoder()
        self.dense = nn.Linear(hidden_size, num_labels)
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, inputs_left_space, inputs_size, inputs_type):
        output = self.encoder_model(inputs_left_space, inputs_size, inputs_type)
        output = self.dropout(output)
        output = self.dense(output)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.config = ModelConfig()
        self.left_space_embedding = nn.Embedding(
            self.config.max_size_value + 1, self.config.encoder_hidden_size
        )
        self.size_embedding = nn.Embedding(
            self.config.max_size_value + 1, self.config.encoder_hidden_size
        )
        self.type_embedding = nn.Embedding(
            self.config.num_types, self.config.encoder_hidden_size
        )
        self.positional_encoding = PositionalEncoding(self.config.encoder_hidden_size)
        self.transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self.config.encoder_hidden_size,
                    self.config.num_attention_heads,
                    self.config.dropout_rate,
                )
                for _ in range(self.config.num_encoder_layers)
            ]
        )

    def forward(self, inputs_left_space, inputs_size, inputs_type):
        left_space_embedded_inputs = self.left_space_embedding(inputs_left_space)
        size_embedded_inputs = self.size_embedding(inputs_size)
        type_embedded_inputs = self.type_embedding(inputs_type)
        embedded_inputs = (
            left_space_embedded_inputs + size_embedded_inputs + type_embedded_inputs
        )
        positional_inputs = self.positional_encoding(embedded_inputs)

        output = positional_inputs
        for transformer_layer in self.transformer_layers:
            output = transformer_layer(output)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout_rate):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_rate
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        attended_inputs, _ = self.self_attention(inputs, inputs, inputs)
        attended_inputs = self.dropout(attended_inputs)
        inputs = self.norm1(inputs + attended_inputs)

        feed_forward_output = self.feed_forward(inputs)
        feed_forward_output = self.dropout(feed_forward_output)
        inputs = self.norm2(inputs + feed_forward_output)

        return inputs


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_length=5000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2)
            * -(torch.log(torch.tensor(10000.0)) / hidden_size)
        )

        positional_encoding = torch.zeros(max_length, hidden_size)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer("positional_encoding", positional_encoding)

    def forward(self, inputs):
        return inputs + self.positional_encoding[:, : inputs.size(1), :]
