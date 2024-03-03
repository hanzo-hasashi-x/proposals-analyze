import tensorflow as tf
from tensorflow import keras

import numpy as np

def load_embed_weights(weights_path,vocab_size,embedding_dim):
    with open(weights_path, 'r', encoding='utf-8') as vectors_file:
        vectors = vectors_file.read().split('\n')

    weights = np.zeros([vocab_size, embedding_dim])

    for index, vector in enumerate(vectors, 1):
        if vector == '': continue
        weights[index] = [float(weight) for weight in vector.split('\t')]
        
    return weights

def positional_encoding(length,depth):
    depth /= 2
    
    positions = np.arange(length)[:, np.newaxis] # (seq,1)
    depths = np.arange(depth)[np.newaxis, :]/depth # (1, depth)
    
    angles_rates = 1 / (1000**depths) # (1,depth)
    angles_rads = positions*angles_rates # (pos,depth)
    
    pos_encoding = np.concatenate(
        [np.sin(angles_rads), np.cos(angles_rads)],
        axis=-1
    )
    
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, embed_weights_path=None):
        super().__init__()
        self.d_model = d_model
        
        if not embed_weights_path:
            self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        else:
            weights = load_embed_weights(embed_weights_path,vocab_size, d_model)
            self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True,
                                                       weights=[weights])
            
            self.embedding.trainable = False
            
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)
        
    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args,**kwargs)
    
    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x+self.pos_encoding[tf.newaxis, :length, :]
        return x
    
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output,attn_scores = self.mha(
            query = x,
            key=context,
            value=context,
            return_attention_scores=True
        )
        
        self.last_attn_scores = attn_scores
        
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        
        return x
    
    
class GlobalSelfAttention(BaseAttention):
    def call(self,x):
        attn_output = self.mha(
            query=x, value=x, key=x
        )
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
    
class CausalSelfAttention(BaseAttention):
    def call(self,x):
        attn_output = self.mha(
            query=x,value=x,key=x,
            use_causal_mask=True
        )
        x = self.add([x,attn_output])
        x = self.layernorm(x)
        return x
    
class FeedForward(keras.layers.Layer):
    def __init__(self,d_model,dff,dropout_rate=.1):
        super().__init__()
        self.seq = keras.Sequential([
            tf.keras.layers.Dense(dff,activation='relu'),
            keras.layers.Dense(d_model),
            keras.layers.Dropout(dropout_rate)
        ])
        self.add = keras.layers.Add()
        self.layer_norm = keras.layers.LayerNormalization()
        
    def call(self,x):
        x = self.add([x,self.seq(x)])
        x = self.layer_norm(x)
        return x
    
class EncodeLayer(keras.layers.Layer):
    def __init__(self,*,d_model,num_heads,dff,dropout_rate=.1):
        super().__init__()
        
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        
        self.fnn = FeedForward(d_model,dff)
        
    def call(self,x):
        x = self.self_attention(x)
        x = self.fnn(x)
        return x

class Encoder(keras.layers.Layer):
    def __init__(self, *, num_layers,d_model,num_heads,
                 dff,vocab_size,embed_weights_path=None,dropout_rate=.1):
        super().__init__()
        
        self.d_model, self.num_layers = d_model, num_layers
        
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
            d_model=d_model, embed_weights_path=embed_weights_path)
        
        self.enc_layers = [
            EncodeLayer(d_model=d_model,
                        num_heads=num_heads, dff=dff,
                        dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        
        self.dropout = keras.layers.Dropout(dropout_rate)
        
    def call(self, x):
        x = self.pos_embedding(x)
        
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
            
        return x
    
class DecodeLayer(keras.layers.Layer):
    def __init__(self,*,d_model,num_heads,dff,dropout_rate=.1):
        super(DecodeLayer,self).__init__()
        
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.fnn = FeedForward(d_model,dff)
        
    def call(self,x,context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)
        
        self.last_attn_scores = self.cross_attention.last_attn_scores
        
        x = self.fnn(x)
        return x
    
class Decoder(keras.layers.Layer):
    def __init__(self, *, num_layers,d_model,num_heads,
                 dff,vocab_size,embed_weights_path=None,dropout_rate=.1):
        super().__init__()
        
        self.d_model, self.num_layers = d_model, num_layers
        
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                        d_model=d_model, embed_weights_path=embed_weights_path)
        
        self.dec_layers = [
            DecodeLayer(d_model=d_model,
                        num_heads=num_heads, dff=dff,
                        dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.last_attn_scores = None
        
    def call(self, x, context):
        x = self.pos_embedding(x)
        
        x = self.dropout(x)
        
        for i in range(self.num_layers):
            x = self.dec_layers[i](x,context)
            
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
            
        return x

class Transformer(keras.Model):
    def __init__(self,*,num_layers,d_model,num_heads,dff,
                 vocab_size, target_embed_dim,embed_weights_path=None,dropout_rate=.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads,dff=dff,
                               vocab_size=vocab_size,
                               embed_weights_path=embed_weights_path,
                               dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                               num_heads=num_heads,dff=dff,
                               vocab_size=vocab_size,
                               embed_weights_path=embed_weights_path,
                               dropout_rate=dropout_rate)
        
        self.final_layer = tf.keras.layers.Dense(target_embed_dim)
        # self.final_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self,inputs):
        context,x = inputs
        context = self.encoder(context)
        
        x = self.decoder(x,context)
        
        logits = self.final_layer(x)
        
        try:
            del logits._keras_mask # so it won't scale losses or metrics
        except AttributeError: pass
    
        return tf.reduce_mean(logits, axis=1)
    
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model,warmup_steps=4000):
        super().__init__()
        
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        
        self.warmup_steps = warmup_steps
        
    def __call__(self,step):
        step = tf.cast(step,dtype=tf.float32)
        arg1=tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1,arg2)
    
    
def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss

def masked_loss(label, pred):
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_object(label, pred)

    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
    return loss


def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0

    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(match)/tf.reduce_sum(mask)

