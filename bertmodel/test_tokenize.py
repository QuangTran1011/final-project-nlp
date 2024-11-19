import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_text as tf_text
from positional_embedding import PositionalEmbedding
from bert import minBert
import sentencepiece as spm

with open("./minbert_bpe.model", "rb") as f:
    pre_trained_tokenizer = f.read()
    
tokenizer = tf_text.SentencepieceTokenizer(pre_trained_tokenizer, out_type=tf.int32)

# eos = tokenizer.string_to_id("</s>").numpy()
# print("EOS: " + str(eos))
input_str = 'Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers.'

a = tokenizer.tokenize(input_str)
print(tokenizer.detokenize(a).numpy().decode('utf-8'))