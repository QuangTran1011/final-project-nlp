import tensorflow as tf
import numpy as np
from positional_embedding import PositionalEmbedding


class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def get_config(self):
        config = super().get_config()
        return config           


class SelfAttention(BaseAttention):
    def call(self, x):
        att_x = self.mha(x, x, x)
        x = self.add([x, att_x])
        x = self.layernorm(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.seq = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(dff, activation="relu"),
                tf.keras.layers.Dense(d_model),
                tf.keras.layers.Dropout(dropout_rate),
            ]
        )
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "dff": self.dff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.self_attention = SelfAttention(
            num_heads=num_heads, key_dim=d_model, dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, dff)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x


class minBert(tf.keras.Model):
    def __init__(
        self, *,name, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1, **kwargs
    ):
        super(minBert, self).__init__(name=name, **kwargs)

        self.num_layers = num_layers
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.num_heads = num_heads
        self.dff = dff

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

        self.enc_layers = [
            EncoderLayer(
                d_model=d_model, num_heads=num_heads, dff=dff, dropout_rate=dropout_rate
            )
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def get_config(self):
        config = super(minBert, self).get_config()
        config.update(
            {
                "name": self.name,
                "num_layers": self.num_layers,
                "d_model": self.d_model,
                "num_heads": self.num_heads,
                "dff": self.dff,
                "vocab_size": self.vocab_size,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config

    def call(self, x):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

        # Add dropout.
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)

        return x  # Shape `(batch_size, seq_len, d_model)`.
