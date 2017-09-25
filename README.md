
# Language Translation
In this project, you’re going to take a peek into the realm of neural network machine translation.  You’ll be training a sequence to sequence model on a dataset of English and French sentences that can translate new sentences from English to French.
## Get the Data
Since translating the whole language of English to French will take lots of time to train, we have provided you with a small portion of the English corpus.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import problem_unittests as tests

source_path = 'data/small_vocab_en'
target_path = 'data/small_vocab_fr'
source_text = helper.load_data(source_path)
target_text = helper.load_data(target_path)
```

## Explore the Data
Play around with view_sentence_range to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in source_text.split()})))

sentences = source_text.split('\n')
word_counts = [len(sentence.split()) for sentence in sentences]
print('Number of sentences: {}'.format(len(sentences)))
print('Average number of words in a sentence: {}'.format(np.average(word_counts)))

print()
print('English sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(source_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
print()
print('French sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(target_text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 227
    Number of sentences: 137861
    Average number of words in a sentence: 13.225277634719028
    
    English sentences 0 to 10:
    new jersey is sometimes quiet during autumn , and it is snowy in april .
    the united states is usually chilly during july , and it is usually freezing in november .
    california is usually quiet during march , and it is usually hot in june .
    the united states is sometimes mild during june , and it is cold in september .
    your least liked fruit is the grape , but my least liked is the apple .
    his favorite fruit is the orange , but my favorite is the grape .
    paris is relaxing during december , but it is usually chilly in july .
    new jersey is busy during spring , and it is never hot in march .
    our least liked fruit is the lemon , but my least liked is the grape .
    the united states is sometimes busy during january , and it is sometimes warm in november .
    
    French sentences 0 to 10:
    new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
    les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
    california est généralement calme en mars , et il est généralement chaud en juin .
    les états-unis est parfois légère en juin , et il fait froid en septembre .
    votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .
    son fruit préféré est l'orange , mais mon préféré est le raisin .
    paris est relaxant en décembre , mais il est généralement froid en juillet .
    new jersey est occupé au printemps , et il est jamais chaude en mars .
    notre fruit est moins aimé le citron , mais mon moins aimé est le raisin .
    les états-unis est parfois occupé en janvier , et il est parfois chaud en novembre .


## Implement Preprocessing Function
### Text to Word Ids
As you did with other RNNs, you must turn the text into a number so the computer can understand it. In the function `text_to_ids()`, you'll turn `source_text` and `target_text` from words to ids.  However, you need to add the `<EOS>` word id at the end of `target_text`.  This will help the neural network predict when the sentence should end.

You can get the `<EOS>` word id by doing:
```python
target_vocab_to_int['<EOS>']
```
You can get other word ids using `source_vocab_to_int` and `target_vocab_to_int`.


```python
def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    """
    Convert source and target text to proper word ids
    :param source_text: String that contains all the source text.
    :param target_text: String that contains all the target text.
    :param source_vocab_to_int: Dictionary to go from the source words to an id
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: A tuple of lists (source_id_text, target_id_text)
    """
    vocab = set([word for line in source_text.split('\n') for word in line.split()])
    
    source_id_text = [[source_vocab_to_int.get(word, source_vocab_to_int['<UNK>']) 
                       for word in line.split()] for line in source_text.split('\n')]
    target_id_text = [[target_vocab_to_int.get(word, source_vocab_to_int['<UNK>']) 
                       for word in line.split()] for line in target_text.split('\n')]
        
    for line in target_id_text:
        line.append(target_vocab_to_int['<EOS>'])
    
    return source_id_text, target_id_text

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_text_to_ids(text_to_ids)
```

    Tests Passed


### Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
helper.preprocess_and_save_data(source_path, target_path, text_to_ids)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np
import helper
import problem_unittests as tests

(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
```

### Check the Version of TensorFlow and Access to GPU
This will check to make sure you have the correct version of TensorFlow and access to a GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf
from tensorflow.python.layers.core import Dense

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.1'), 'Please use TensorFlow version 1.1 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.3.0


    /home/aladim/anaconda3/envs/dlnd-tf-lab/lib/python3.5/site-packages/ipykernel/__main__.py:15: UserWarning: No GPU found. Please use a GPU to train your neural network.


## Build the Neural Network
You'll build the components necessary to build a Sequence-to-Sequence model by implementing the following functions below:
- `model_inputs`
- `process_decoder_input`
- `encoding_layer`
- `decoding_layer_train`
- `decoding_layer_infer`
- `decoding_layer`
- `seq2seq_model`

### Input
Implement the `model_inputs()` function to create TF Placeholders for the Neural Network. It should create the following placeholders:

- Input text placeholder named "input" using the TF Placeholder name parameter with rank 2.
- Targets placeholder with rank 2.
- Learning rate placeholder with rank 0.
- Keep probability placeholder named "keep_prob" using the TF Placeholder name parameter with rank 0.
- Target sequence length placeholder named "target_sequence_length" with rank 1
- Max target sequence length tensor named "max_target_len" getting its value from applying tf.reduce_max on the target_sequence_length placeholder. Rank 0.
- Source sequence length placeholder named "source_sequence_length" with rank 1

Return the placeholders in the following the tuple (input, targets, learning rate, keep probability, target sequence length, max target sequence length, source sequence length)


```python
def model_inputs():
    """
    Create TF Placeholders for input, targets, learning rate, and lengths of source and target sequences.
    :return: Tuple (input, targets, learning rate, keep probability, target sequence length,
    max target sequence length, source sequence length)
    """
    #Based in the example of notebook sequence_to_sequence_implementation.ipynb
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    """
    I passed the shape to learning_rate and keep_probability because of 
    tensorflow warning in version 1.3.
    Link with the solution by the (meme) user calvinalvin: https://github.com/openai/pixel-cnn/issues/17
    """
    learning_rate = tf.placeholder(tf.float32, name='learning_rate', shape=())
    keep_probability = tf.placeholder(tf.float32, name='keep_prob', shape=())
    target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, [None], name='source_sequence_length')
    return inputs, targets, learning_rate, keep_probability, target_sequence_length, max_target_sequence_length, source_sequence_length


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_model_inputs(model_inputs)
```

    Tests Passed


### Process Decoder Input
Implement `process_decoder_input` by removing the last word id from each batch in `target_data` and concat the GO ID to the begining of each batch.


```python
def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    """
    Preprocess target data for encoding
    :param target_data: Target Placehoder
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param batch_size: Batch Size
    :return: Preprocessed target data
    """
    #sequence_to_sequence_implementation.ipynb
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    return tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_process_encoding_input(process_decoder_input)
```

    Tests Passed


### Encoding
Implement `encoding_layer()` to create a Encoder RNN layer:
 * Embed the encoder input using [`tf.contrib.layers.embed_sequence`](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/embed_sequence)
 * Construct a [stacked](https://github.com/tensorflow/tensorflow/blob/6947f65a374ebf29e74bb71e36fd82760056d82c/tensorflow/docs_src/tutorials/recurrent.md#stacking-multiple-lstms) [`tf.contrib.rnn.LSTMCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMCell) wrapped in a [`tf.contrib.rnn.DropoutWrapper`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/DropoutWrapper)
 * Pass cell and embedded input to [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)


```python
def build_multi_rnn_cell(rnn_size, num_layers, keep_prob=None):
    def build_cell(rnn_size, keep_prob=None):
        lstm = tf.contrib.rnn.LSTMCell(rnn_size)
        if keep_prob is not None:
            return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        else:
            return lstm
    
    return tf.contrib.rnn.MultiRNNCell([build_cell(rnn_size, keep_prob) for _ in range(num_layers)])
```


```python
from imp import reload
reload(tests)

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):
    """
    Create encoding layer
    :param rnn_inputs: Inputs for the RNN
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param keep_prob: Dropout keep probability
    :param source_sequence_length: a list of the lengths of each sequence in the batch
    :param source_vocab_size: vocabulary size of source data
    :param encoding_embedding_size: embedding size of source data
    :return: tuple (RNN output, RNN state)
    """
    #based on sequence_to_sequence_implementation.ipynbs
    enc_inputs = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)
    
    cell = build_multi_rnn_cell(rnn_size, num_layers, keep_prob)
    output, initial_state = tf.nn.dynamic_rnn(cell, enc_inputs, sequence_length=source_sequence_length, dtype=tf.float32)
    
    return output, initial_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_encoding_layer(encoding_layer)
```

    Tests Passed


### Decoding - Training
Create a training decoding layer:
* Create a [`tf.contrib.seq2seq.TrainingHelper`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/TrainingHelper) 
* Create a [`tf.contrib.seq2seq.BasicDecoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder)
* Obtain the decoder outputs from [`tf.contrib.seq2seq.dynamic_decode`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode)


```python

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, 
                         target_sequence_length, max_summary_length, 
                         output_layer, keep_prob):
    """
    Create a decoding layer for training
    :param encoder_state: Encoder State
    :param dec_cell: Decoder RNN Cell
    :param dec_embed_input: Decoder embedded input
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_summary_length: The length of the longest sequence in the batch
    :param output_layer: Function to apply the output layer
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing training logits and sample_id
    """
    #based on sequence_to_sequence_implementation.ipynb
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,
                                     sequence_length=target_sequence_length,
                                     time_major=False)
    
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, 
                                                training_helper,
                                                encoder_state,
                                                output_layer)
    
    training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder, 
                                                                impute_finished=True,
                                                                maximum_iterations=max_summary_length)[0]
    
    return training_decoder_output


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_train(decoding_layer_train)
```

    Tests Passed


### Decoding - Inference
Create inference decoder:
* Create a [`tf.contrib.seq2seq.GreedyEmbeddingHelper`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/GreedyEmbeddingHelper)
* Create a [`tf.contrib.seq2seq.BasicDecoder`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/BasicDecoder)
* Obtain the decoder outputs from [`tf.contrib.seq2seq.dynamic_decode`](https://www.tensorflow.org/api_docs/python/tf/contrib/seq2seq/dynamic_decode)


```python
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,
                         end_of_sequence_id, max_target_sequence_length,
                         vocab_size, output_layer, batch_size, keep_prob):
    """
    Create a decoding layer for inference
    :param encoder_state: Encoder state
    :param dec_cell: Decoder RNN Cell
    :param dec_embeddings: Decoder embeddings
    :param start_of_sequence_id: GO ID
    :param end_of_sequence_id: EOS Id
    :param max_target_sequence_length: Maximum length of target sequences
    :param vocab_size: Size of decoder/target vocabulary
    :param decoding_scope: TenorFlow Variable Scope for decoding
    :param output_layer: Function to apply the output layer
    :param batch_size: Batch size
    :param keep_prob: Dropout keep probability
    :return: BasicDecoderOutput containing inference logits and sample_id
    """
    #based on sequence_to_sequence_implementation.ipynbs
    start_tokens = tf.tile(tf.constant([start_of_sequence_id], dtype=tf.int32), 
                           [batch_size])
    
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings, 
                                             start_tokens,
                                             end_of_sequence_id)
    
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,
                                                        inference_helper,
                                                        encoder_state,
                                                        output_layer)
    
    inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,
                                                                 impute_finished=True,
                                                                 maximum_iterations=max_target_sequence_length)[0]
    
    return inference_decoder_output



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer_infer(decoding_layer_infer)
```

    Tests Passed


### Build the Decoding Layer
Implement `decoding_layer()` to create a Decoder RNN layer.

* Embed the target sequences
* Construct the decoder LSTM cell (just like you constructed the encoder cell above)
* Create an output layer to map the outputs of the decoder to the elements of our vocabulary
* Use the your `decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_target_sequence_length, output_layer, keep_prob)` function to get the training logits.
* Use your `decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id, max_target_sequence_length, vocab_size, output_layer, batch_size, keep_prob)` function to get the inference logits.

Note: You'll need to use [tf.variable_scope](https://www.tensorflow.org/api_docs/python/tf/variable_scope) to share variables between training and inference.


```python
from tensorflow.python.layers.core import Dense

def decoding_layer(dec_input, encoder_state,
                   target_sequence_length, max_target_sequence_length,
                   rnn_size, num_layers, 
                   target_vocab_to_int, target_vocab_size,
                   batch_size, keep_prob, 
                   decoding_embedding_size):
    """
    Create decoding layer
    :param dec_input: Decoder input
    :param encoder_state: Encoder state
    :param target_sequence_length: The lengths of each sequence in the target batch
    :param max_target_sequence_length: Maximum length of target sequences
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :param target_vocab_size: Size of target vocabulary
    :param batch_size: The size of the batch
    :param keep_prob: Dropout keep probability
    :param decoding_embedding_size: Decoding embedding size
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    with tf.variable_scope("decode") as scope:
        
        dec_embeddings = tf.Variable(tf.truncated_normal([target_vocab_size, decoding_embedding_size]))
        dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
        
        dec_cell = build_multi_rnn_cell(rnn_size, num_layers, keep_prob)
        
        output_layer = Dense(target_vocab_size)
        
        decoding_train = decoding_layer_train(encoder_state, dec_cell, 
                             dec_embed_input, target_sequence_length, 
                             max_target_sequence_length, output_layer, 
                             keep_prob)
        
        start_of_sequence_id = target_vocab_to_int['<GO>']
        end_of_sequence_id = target_vocab_to_int['<EOS>']
        
    with tf.variable_scope("decode", reuse=True) as scope:
        
        decoding_infer = decoding_layer_infer(encoder_state, dec_cell, 
                             dec_embeddings, start_of_sequence_id, 
                             end_of_sequence_id, max_target_sequence_length, 
                             target_vocab_size, output_layer, 
                             batch_size, keep_prob)
        
    return decoding_train, decoding_infer



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_decoding_layer(decoding_layer)
```

    Tests Passed


### Build the Neural Network
Apply the functions you implemented above to:

- Encode the input using your `encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,  source_sequence_length, source_vocab_size, encoding_embedding_size)`.
- Process target data using your `process_decoder_input(target_data, target_vocab_to_int, batch_size)` function.
- Decode the encoded input using your `decoding_layer(dec_input, enc_state, target_sequence_length, max_target_sentence_length, rnn_size, num_layers, target_vocab_to_int, target_vocab_size, batch_size, keep_prob, dec_embedding_size)` function.


```python
def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  source_sequence_length, target_sequence_length,
                  max_target_sentence_length,
                  source_vocab_size, target_vocab_size,
                  enc_embedding_size, dec_embedding_size,
                  rnn_size, num_layers, target_vocab_to_int):
    """
    Build the Sequence-to-Sequence part of the neural network
    :param input_data: Input placeholder
    :param target_data: Target placeholder
    :param keep_prob: Dropout keep probability placeholder
    :param batch_size: Batch Size
    :param source_sequence_length: Sequence Lengths of source sequences in the batch
    :param target_sequence_length: Sequence Lengths of target sequences in the batch
    :param source_vocab_size: Source vocabulary size
    :param target_vocab_size: Target vocabulary size
    :param enc_embedding_size: Decoder embedding size
    :param dec_embedding_size: Encoder embedding size
    :param rnn_size: RNN Size
    :param num_layers: Number of layers
    :param target_vocab_to_int: Dictionary to go from the target words to an id
    :return: Tuple of (Training BasicDecoderOutput, Inference BasicDecoderOutput)
    """
    output, enc_state = encoding_layer(input_data, rnn_size, 
                                       num_layers, keep_prob,  
                                       source_sequence_length, source_vocab_size, 
                                       enc_embedding_size)
    
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    
    decoding_train, decoding_infer = decoding_layer(dec_input, enc_state, 
                                                    target_sequence_length, max_target_sentence_length, 
                                                    rnn_size, num_layers, target_vocab_to_int, 
                                                    target_vocab_size, batch_size, 
                                                    keep_prob, dec_embedding_size)
    
    return decoding_train, decoding_infer


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_seq2seq_model(seq2seq_model)
```

    Tests Passed


## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `num_layers` to the number of layers.
- Set `encoding_embedding_size` to the size of the embedding for the encoder.
- Set `decoding_embedding_size` to the size of the embedding for the decoder.
- Set `learning_rate` to the learning rate.
- Set `keep_probability` to the Dropout keep probability
- Set `display_step` to state how many steps between each debug output statement


```python
# Number of Epochs
epochs = 20
# Batch Size
batch_size = 512
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 128
decoding_embedding_size = 128
# Learning Rate
learning_rate = 0.01
# Dropout Keep Probability
keep_probability = 0.8
display_step = 100
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
save_path = 'checkpoints/dev'
(source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = helper.load_preprocess()
max_target_sentence_length = max([len(sentence) for sentence in source_int_text])

train_graph = tf.Graph()
with train_graph.as_default():
    input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()

    #sequence_length = tf.placeholder_with_default(max_target_sentence_length, None, name='sequence_length')
    input_shape = tf.shape(input_data)

    train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                   targets,
                                                   keep_prob,
                                                   batch_size,
                                                   source_sequence_length,
                                                   target_sequence_length,
                                                   max_target_sequence_length,
                                                   len(source_vocab_to_int),
                                                   len(target_vocab_to_int),
                                                   encoding_embedding_size,
                                                   decoding_embedding_size,
                                                   rnn_size,
                                                   num_layers,
                                                   target_vocab_to_int)


    training_logits = tf.identity(train_logits.rnn_output, name='logits')
    inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):
        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

```

Batch and pad the source and target sequences


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
def pad_sentence_batch(sentence_batch, pad_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    """Batch targets, sources, and the lengths of their sentences together"""
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths

```

### Train
Train the neural network on the preprocessed data. If you have a hard time getting a good loss, check the forms to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
def get_accuracy(target, logits):
    """
    Calculate accuracy
    """
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(
            logits,
            [(0,0),(0,max_seq - logits.shape[1])],
            'constant')

    return np.mean(np.equal(target, logits))

# Split data to training and validation sets
train_source = source_int_text[batch_size:]
train_target = target_int_text[batch_size:]
valid_source = source_int_text[:batch_size]
valid_target = target_int_text[:batch_size]
(valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,
                                                                                                             valid_target,
                                                                                                             batch_size,
                                                                                                             source_vocab_to_int['<PAD>'],
                                                                                                             target_vocab_to_int['<PAD>']))                                                                                                  
with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(epochs):
        for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                get_batches(train_source, train_target, batch_size,
                            source_vocab_to_int['<PAD>'],
                            target_vocab_to_int['<PAD>'])):

            _, loss = sess.run(
                [train_op, cost],
                {input_data: source_batch,
                 targets: target_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths,
                 keep_prob: keep_probability})


            if batch_i % display_step == 0 and batch_i > 0:


                batch_train_logits = sess.run(
                    inference_logits,
                    {input_data: source_batch,
                     source_sequence_length: sources_lengths,
                     target_sequence_length: targets_lengths,
                     keep_prob: 1.0})


                batch_valid_logits = sess.run(
                    inference_logits,
                    {input_data: valid_sources_batch,
                     source_sequence_length: valid_sources_lengths,
                     target_sequence_length: valid_targets_lengths,
                     keep_prob: 1.0})

                train_acc = get_accuracy(target_batch, batch_train_logits)

                valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)

                print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                      .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_path)
    print('Model Trained and Saved')
```

    Epoch   0 Batch  100/269 - Train Accuracy: 0.5432, Validation Accuracy: 0.5389, Loss: 0.6807
    Epoch   0 Batch  200/269 - Train Accuracy: 0.6126, Validation Accuracy: 0.6318, Loss: 0.5348
    Epoch   1 Batch  100/269 - Train Accuracy: 0.7849, Validation Accuracy: 0.7829, Loss: 0.3053
    Epoch   1 Batch  200/269 - Train Accuracy: 0.8432, Validation Accuracy: 0.8486, Loss: 0.1893
    Epoch   2 Batch  100/269 - Train Accuracy: 0.9243, Validation Accuracy: 0.9312, Loss: 0.0707
    Epoch   2 Batch  200/269 - Train Accuracy: 0.9452, Validation Accuracy: 0.9523, Loss: 0.0488
    Epoch   3 Batch  100/269 - Train Accuracy: 0.9507, Validation Accuracy: 0.9500, Loss: 0.0403
    Epoch   3 Batch  200/269 - Train Accuracy: 0.9659, Validation Accuracy: 0.9607, Loss: 0.0313
    Epoch   4 Batch  100/269 - Train Accuracy: 0.9576, Validation Accuracy: 0.9716, Loss: 0.0303
    Epoch   4 Batch  200/269 - Train Accuracy: 0.9711, Validation Accuracy: 0.9688, Loss: 0.0252
    Epoch   5 Batch  100/269 - Train Accuracy: 0.9618, Validation Accuracy: 0.9644, Loss: 0.0251
    Epoch   5 Batch  200/269 - Train Accuracy: 0.9771, Validation Accuracy: 0.9638, Loss: 0.0180
    Epoch   6 Batch  100/269 - Train Accuracy: 0.9683, Validation Accuracy: 0.9696, Loss: 0.0219
    Epoch   6 Batch  200/269 - Train Accuracy: 0.9769, Validation Accuracy: 0.9646, Loss: 0.0178
    Epoch   7 Batch  100/269 - Train Accuracy: 0.9746, Validation Accuracy: 0.9698, Loss: 0.0200
    Epoch   7 Batch  200/269 - Train Accuracy: 0.9799, Validation Accuracy: 0.9657, Loss: 0.0166
    Epoch   8 Batch  100/269 - Train Accuracy: 0.9756, Validation Accuracy: 0.9744, Loss: 0.0185
    Epoch   8 Batch  200/269 - Train Accuracy: 0.9796, Validation Accuracy: 0.9684, Loss: 0.0143
    Epoch   9 Batch  100/269 - Train Accuracy: 0.9744, Validation Accuracy: 0.9673, Loss: 0.0164
    Epoch   9 Batch  200/269 - Train Accuracy: 0.9774, Validation Accuracy: 0.9733, Loss: 0.0168
    Epoch  10 Batch  100/269 - Train Accuracy: 0.9794, Validation Accuracy: 0.9620, Loss: 0.0201
    Epoch  10 Batch  200/269 - Train Accuracy: 0.9832, Validation Accuracy: 0.9645, Loss: 0.0175
    Epoch  11 Batch  100/269 - Train Accuracy: 0.9744, Validation Accuracy: 0.9656, Loss: 0.0221
    Epoch  11 Batch  200/269 - Train Accuracy: 0.9847, Validation Accuracy: 0.9672, Loss: 0.0157
    Epoch  12 Batch  100/269 - Train Accuracy: 0.9769, Validation Accuracy: 0.9700, Loss: 0.0201
    Epoch  12 Batch  200/269 - Train Accuracy: 0.9823, Validation Accuracy: 0.9714, Loss: 0.0165
    Epoch  13 Batch  100/269 - Train Accuracy: 0.9818, Validation Accuracy: 0.9642, Loss: 0.0214
    Epoch  13 Batch  200/269 - Train Accuracy: 0.9807, Validation Accuracy: 0.9658, Loss: 0.0176
    Epoch  14 Batch  100/269 - Train Accuracy: 0.9736, Validation Accuracy: 0.9715, Loss: 0.0229
    Epoch  14 Batch  200/269 - Train Accuracy: 0.9846, Validation Accuracy: 0.9734, Loss: 0.0174
    Epoch  15 Batch  100/269 - Train Accuracy: 0.9733, Validation Accuracy: 0.9743, Loss: 0.0203
    Epoch  15 Batch  200/269 - Train Accuracy: 0.9763, Validation Accuracy: 0.9702, Loss: 0.0194
    Epoch  16 Batch  100/269 - Train Accuracy: 0.9713, Validation Accuracy: 0.9688, Loss: 0.0202
    Epoch  16 Batch  200/269 - Train Accuracy: 0.9782, Validation Accuracy: 0.9727, Loss: 0.0172
    Epoch  17 Batch  100/269 - Train Accuracy: 0.9775, Validation Accuracy: 0.9702, Loss: 0.0229
    Epoch  17 Batch  200/269 - Train Accuracy: 0.9843, Validation Accuracy: 0.9636, Loss: 0.0194
    Epoch  18 Batch  100/269 - Train Accuracy: 0.9798, Validation Accuracy: 0.9671, Loss: 0.0235
    Epoch  18 Batch  200/269 - Train Accuracy: 0.9792, Validation Accuracy: 0.9701, Loss: 0.0212
    Epoch  19 Batch  100/269 - Train Accuracy: 0.9760, Validation Accuracy: 0.9680, Loss: 0.0227
    Epoch  19 Batch  200/269 - Train Accuracy: 0.9784, Validation Accuracy: 0.9681, Loss: 0.0201
    Model Trained and Saved


### Save Parameters
Save the `batch_size` and `save_path` parameters for inference.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params(save_path)
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = helper.load_preprocess()
load_path = helper.load_params()
```

## Sentence to Sequence
To feed a sentence into the model for translation, you first need to preprocess it.  Implement the function `sentence_to_seq()` to preprocess new sentences.

- Convert the sentence to lowercase
- Convert words into ids using `vocab_to_int`
 - Convert words not in the vocabulary, to the `<UNK>` word id.


```python
def sentence_to_seq(sentence, vocab_to_int):
    """
    Convert a sentence to a sequence of ids
    :param sentence: String
    :param vocab_to_int: Dictionary to go from the words to an id
    :return: List of word ids
    """
    sentence = sentence.lower()
    unk_id = vocab_to_int['<UNK>']
    int_sentence = [vocab_to_int.get(word, unk_id) for word in sentence.split()]
    return int_sentence


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_sentence_to_seq(sentence_to_seq)
```

    Tests Passed


## Translate
This will translate `translate_sentence` from English to French.


```python
translate_sentence = 'he saw a old yellow truck .'

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_path + '.meta')
    loader.restore(sess, load_path)

    input_data = loaded_graph.get_tensor_by_name('input:0')
    logits = loaded_graph.get_tensor_by_name('predictions:0')
    target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
    source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                         target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                         source_sequence_length: [len(translate_sentence)]*batch_size,
                                         keep_prob: 1.0})[0]

print('Input')
print('  Word Ids:      {}'.format([i for i in translate_sentence]))
print('  English Words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))

print('\nPrediction')
print('  Word Ids:      {}'.format([i for i in translate_logits]))
print('  French Words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))

```

    INFO:tensorflow:Restoring parameters from checkpoints/dev
    Input
      Word Ids:      [222, 27, 198, 195, 48, 106, 20]
      English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']
    
    Prediction
      Word Ids:      [121, 84, 24, 356, 67, 61, 142, 14, 1]
      French Words: il a conduit le nouveau camion bleu . <EOS>


## Imperfect Translation
You might notice that some sentences translate better than others.  Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words.  For this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.

You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided.  Just make sure you play with the WMT10 corpus after you've submitted this project.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_language_translation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
