import joblib
import tensorflow as tf
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D, Input, Embedding,Concatenate,BatchNormalization, Dropout, Add, GRU, AveragePooling2D
import numpy as np
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image


chexnet_weights = "densenet.h5"

def create_chexnet(chexnet_weights = chexnet_weights,input_size=(224,224)):
  model = tf.keras.applications.DenseNet121(include_top=False,input_shape = input_size+(3,)) 

  x = model.output 
  x = GlobalAveragePooling2D()(x)
  x = Dense(14, activation="sigmoid", name="chexnet_output")(x) 

  chexnet = tf.keras.Model(inputs = model.input,outputs = x)
  chexnet.load_weights(chexnet_weights)
  chexnet = tf.keras.Model(inputs = model.input,outputs = chexnet.layers[-3].output)  
  return chexnet


class encoder_block(tf.keras.layers.Layer):
  def __init__(self,
               name = "image_encoder"
               ):
    super().__init__()
    self.chexnet = create_chexnet(input_size = (224,224))
    self.chexnet.trainable = False
    self.avgpool = AveragePooling2D((1,1))

  def call(self,data):
    op = self.chexnet(data) 
    op = self.avgpool(op) 
    op = tf.reshape(op,shape = (-1,op.shape[1]*op.shape[2],op.shape[3])) 
    return op


def encoder(image,dense_dim,dropout_rate):

  image_encoder = encoder_block(name="image_encoder")
  encoder_features = image_encoder(image) 
  dense = Dense(dense_dim,name = "encoder_dense",activation = "relu") 
  dense_features = dense(encoder_features)



  batch_norm = BatchNormalization(name = "encoder_batch_norm")(dense_features)
  dropout = Dropout(dropout_rate,name = "encoder_dropout")(batch_norm)
  return dropout


class global_attention(tf.keras.layers.Layer):
  """
  calculate global attention
  """
  def __init__(self,dense_dim):
    super().__init__()

    self.w1 = Dense(units = dense_dim) 
    self.w2 = Dense(units = dense_dim) 
    self.v = Dense(units = 1) 



  def call(self,encoder_output,decoder_h):
    decoder_h = tf.expand_dims(decoder_h,axis=1) 
    tanh_input = self.w1(encoder_output) + self.w2(decoder_h) 
    tanh_output =  tf.nn.tanh(tanh_input)
    attention_weights = tf.nn.softmax(self.v(tanh_output),axis=1) 
    op = attention_weights*encoder_output
    context_vector = tf.reduce_sum(op,axis=1)


    return context_vector,attention_weights


class decoder_block(tf.keras.layers.Layer):
  """
  decodes a single token
  """
  def __init__(self,vocab_size, embedding_dim, max_pad, dense_dim ,name = "token_decoder"):
    super().__init__()
    self.dense_dim = dense_dim
    self.embedding = Embedding(input_dim = vocab_size+1,
                                output_dim = embedding_dim,
                                input_length=max_pad,
                                mask_zero=True, 
                                name = 'decoder_embedding'
                              )
    self.LSTM = GRU(units=self.dense_dim,
                    return_state=True,
                    name = 'decoder_LSTM'
                    )
    self.attention = global_attention(dense_dim = dense_dim)
    self.concat = Concatenate(axis=-1)
    self.dense = Dense(dense_dim,name = 'decoder_dense',activation = 'relu')
    self.final = Dense(vocab_size+1,activation='softmax')
    self.concat = Concatenate(axis=-1)
    self.add =Add()
  @tf.function
  def call(self,input_to_decoder, encoder_output, decoder_h):

    embedding_op = self.embedding(input_to_decoder) 
    

    context_vector,attention_weights = self.attention(encoder_output,decoder_h) 
    context_vector_time_axis = tf.expand_dims(context_vector,axis=1)
    concat_input = self.concat([context_vector_time_axis,embedding_op])
    
    output,decoder_h = self.LSTM(concat_input,initial_state = decoder_h)

    

    output = self.final(output)
    return output,decoder_h,attention_weights


class decoder(tf.keras.Model):

  def __init__(self,max_pad, embedding_dim,dense_dim,batch_size ,vocab_size):
    super().__init__()
    self.token_decoder = decoder_block(vocab_size = vocab_size, embedding_dim = embedding_dim, max_pad = max_pad, dense_dim = dense_dim)
    self.output_array = tf.TensorArray(tf.float32,size=max_pad)
    self.max_pad = max_pad
    self.batch_size = batch_size
    self.dense_dim =dense_dim
    
  @tf.function
  def call(self,encoder_output,caption):
    decoder_h = tf.zeros_like(encoder_output[:,0])
    output_array = tf.TensorArray(tf.float32,size=self.max_pad)
    for timestep in range(self.max_pad): 
      output,decoder_h,_ = self.token_decoder(caption[:,timestep:timestep+1], encoder_output, decoder_h)
      output_array = output_array.write(timestep,output) 

    self.output_array = tf.transpose(output_array.stack(),[1,0,2])
    return self.output_array


def create_model():
  input_size = (224,224)
  tokenizer = joblib.load('tokenizer.pkl')
  max_pad = 29
  batch_size = 100
  vocab_size = len(tokenizer.word_index)
  embedding_dim = 300
  dense_dim = 512
  dropout_rate = 0.2


  tf.keras.backend.clear_session()
  image = Input(shape = (input_size + (3,))) 
  caption = Input(shape = (max_pad,))

  encoder_output = encoder(image,dense_dim,dropout_rate) 

  output = decoder(max_pad, embedding_dim,dense_dim,batch_size ,vocab_size)(encoder_output,caption)
  model = tf.keras.Model(inputs = [image,caption], outputs = output)
  model_filename = 'Encoder_Decoder_global_attention.h5'
  model_save = model_filename
  model.load_weights(model_save)

  return model,tokenizer


def greedy_search_predict(image,model,tokenizer,input_size = (224,224)):

  image = tf.expand_dims(cv2.resize(image,input_size,interpolation = cv2.INTER_NEAREST),axis=0) 
  image = model.get_layer('encoder_block')(image)
  image = model.get_layer('encoder_dense')(image)

  enc_op = model.get_layer('encoder_batch_norm')(image)  
  enc_op = model.get_layer('encoder_dropout')(enc_op)


  decoder_h,_ = tf.zeros_like(enc_op[:,0]),tf.zeros_like(enc_op[:,0])
  a = []
  max_pad = 29
  for i in range(max_pad):
    if i==0: 
      caption = np.array(tokenizer.texts_to_sequences(['<cls>'])) 
    output,decoder_h,_ = model.get_layer('decoder').token_decoder(caption,enc_op,decoder_h)

    #prediction
    max_prob = tf.argmax(output,axis=-1)  
    caption = np.array([max_prob]) 
    if max_prob==np.squeeze(tokenizer.texts_to_sequences(['<end>'])): 
      break
    else:
      a.append(tf.squeeze(max_prob).numpy())
  return tokenizer.sequences_to_texts([a])[0] 


def predict_captions(image,model_tokenizer = None):
  if model_tokenizer == None:
    model,tokenizer = create_model()
  else:
    model,tokenizer = model_tokenizer[0],model_tokenizer[1]
  predicted_caption = []
  for img in image:
    caption = greedy_search_predict(img,model,tokenizer)
    predicted_caption.append(caption)

  return predicted_caption

if __name__ == "__main__":
    model_tokenizer = create_model()

    IMAGES_DIR = "NLMCXR_png" 
    captions = []

    for image in os.listdir(IMAGES_DIR):
        print(image)
        image = Image.open(f"{IMAGES_DIR}/{image}").convert("RGB")
        image = np.array(image)/255
        caption = predict_captions([image], model_tokenizer)
        print(caption)
        captions.append(tuple([image, caption[0]]))
    captions = pd.DataFrame(captions)
    captions.to_csv("captions.csv")
