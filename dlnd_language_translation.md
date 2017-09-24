
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

    TensorFlow Version: 1.2.1
    

    C:\Users\Kurosaki-X\Anaconda3\envs\dlnd-tf-lab\lib\site-packages\ipykernel\__main__.py:15: UserWarning: No GPU found. Please use a GPU to train your neural network.
    

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
        
        dec_cell = build_multi_rnn_cell(rnn_size, num_layers)
        
        output_layer = Dense(target_vocab_size)
        
        decoding_train = decoding_layer_train(encoder_state, dec_cell, 
                             dec_embed_input, target_sequence_length, 
                             max_target_sequence_length, output_layer, 
                             keep_prob)
        
        start_of_sequence_id = target_vocab_to_int['<GO>']
        end_of_sequence_id = target_vocab_to_int['<EOS>']
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
epochs = 40
# Batch Size
batch_size = 512
# RNN Size
rnn_size = 30
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.01
# Dropout Keep Probability
keep_probability = 0.8
display_step = 5
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

    Epoch   0 Batch    5/269 - Train Accuracy: 0.2324, Validation Accuracy: 0.3096, Loss: 4.6996
    Epoch   0 Batch   10/269 - Train Accuracy: 0.2327, Validation Accuracy: 0.3096, Loss: 3.7212
    Epoch   0 Batch   15/269 - Train Accuracy: 0.2591, Validation Accuracy: 0.3096, Loss: 3.4783
    Epoch   0 Batch   20/269 - Train Accuracy: 0.2377, Validation Accuracy: 0.3096, Loss: 3.4326
    Epoch   0 Batch   25/269 - Train Accuracy: 0.2781, Validation Accuracy: 0.3427, Loss: 3.3232
    Epoch   0 Batch   30/269 - Train Accuracy: 0.3211, Validation Accuracy: 0.3608, Loss: 3.1381
    Epoch   0 Batch   35/269 - Train Accuracy: 0.3368, Validation Accuracy: 0.3676, Loss: 3.0250
    Epoch   0 Batch   40/269 - Train Accuracy: 0.3085, Validation Accuracy: 0.3723, Loss: 3.0916
    Epoch   0 Batch   45/269 - Train Accuracy: 0.3096, Validation Accuracy: 0.3721, Loss: 3.0333
    Epoch   0 Batch   50/269 - Train Accuracy: 0.3281, Validation Accuracy: 0.3874, Loss: 2.9709
    Epoch   0 Batch   55/269 - Train Accuracy: 0.3524, Validation Accuracy: 0.3835, Loss: 2.7959
    Epoch   0 Batch   60/269 - Train Accuracy: 0.3650, Validation Accuracy: 0.3795, Loss: 2.6694
    Epoch   0 Batch   65/269 - Train Accuracy: 0.3712, Validation Accuracy: 0.3974, Loss: 2.6781
    Epoch   0 Batch   70/269 - Train Accuracy: 0.3858, Validation Accuracy: 0.4048, Loss: 2.6217
    Epoch   0 Batch   75/269 - Train Accuracy: 0.3756, Validation Accuracy: 0.3967, Loss: 2.5748
    Epoch   0 Batch   80/269 - Train Accuracy: 0.3942, Validation Accuracy: 0.4094, Loss: 2.4697
    Epoch   0 Batch   85/269 - Train Accuracy: 0.3789, Validation Accuracy: 0.4165, Loss: 2.4693
    Epoch   0 Batch   90/269 - Train Accuracy: 0.3704, Validation Accuracy: 0.4335, Loss: 2.5134
    Epoch   0 Batch   95/269 - Train Accuracy: 0.4170, Validation Accuracy: 0.4438, Loss: 2.2970
    Epoch   0 Batch  100/269 - Train Accuracy: 0.4386, Validation Accuracy: 0.4460, Loss: 2.1786
    Epoch   0 Batch  105/269 - Train Accuracy: 0.4296, Validation Accuracy: 0.4535, Loss: 2.1555
    Epoch   0 Batch  110/269 - Train Accuracy: 0.4094, Validation Accuracy: 0.4339, Loss: 2.0645
    Epoch   0 Batch  115/269 - Train Accuracy: 0.3656, Validation Accuracy: 0.4347, Loss: 2.0692
    Epoch   0 Batch  120/269 - Train Accuracy: 0.3801, Validation Accuracy: 0.4345, Loss: 1.9844
    Epoch   0 Batch  125/269 - Train Accuracy: 0.4192, Validation Accuracy: 0.4490, Loss: 1.7858
    Epoch   0 Batch  130/269 - Train Accuracy: 0.3855, Validation Accuracy: 0.4540, Loss: 1.8460
    Epoch   0 Batch  135/269 - Train Accuracy: 0.3990, Validation Accuracy: 0.4667, Loss: 1.7658
    Epoch   0 Batch  140/269 - Train Accuracy: 0.4414, Validation Accuracy: 0.4577, Loss: 1.5809
    Epoch   0 Batch  145/269 - Train Accuracy: 0.2585, Validation Accuracy: 0.2791, Loss: 1.5235
    Epoch   0 Batch  150/269 - Train Accuracy: 0.2483, Validation Accuracy: 0.2453, Loss: 1.4642
    Epoch   0 Batch  155/269 - Train Accuracy: 0.2721, Validation Accuracy: 0.2617, Loss: 1.3289
    Epoch   0 Batch  160/269 - Train Accuracy: 0.2991, Validation Accuracy: 0.2936, Loss: 1.3602
    Epoch   0 Batch  165/269 - Train Accuracy: 0.2796, Validation Accuracy: 0.2907, Loss: 1.3474
    Epoch   0 Batch  170/269 - Train Accuracy: 0.4081, Validation Accuracy: 0.4227, Loss: 1.2607
    Epoch   0 Batch  175/269 - Train Accuracy: 0.4605, Validation Accuracy: 0.4810, Loss: 1.2414
    Epoch   0 Batch  180/269 - Train Accuracy: 0.4852, Validation Accuracy: 0.4893, Loss: 1.1806
    Epoch   0 Batch  185/269 - Train Accuracy: 0.4958, Validation Accuracy: 0.5015, Loss: 1.1559
    Epoch   0 Batch  190/269 - Train Accuracy: 0.4807, Validation Accuracy: 0.4927, Loss: 1.1084
    Epoch   0 Batch  195/269 - Train Accuracy: 0.4799, Validation Accuracy: 0.5045, Loss: 1.1126
    Epoch   0 Batch  200/269 - Train Accuracy: 0.4676, Validation Accuracy: 0.4988, Loss: 1.1143
    Epoch   0 Batch  205/269 - Train Accuracy: 0.4629, Validation Accuracy: 0.4910, Loss: 1.0403
    Epoch   0 Batch  210/269 - Train Accuracy: 0.4909, Validation Accuracy: 0.4970, Loss: 1.0146
    Epoch   0 Batch  215/269 - Train Accuracy: 0.5130, Validation Accuracy: 0.5021, Loss: 0.9431
    Epoch   0 Batch  220/269 - Train Accuracy: 0.5159, Validation Accuracy: 0.5062, Loss: 0.9374
    Epoch   0 Batch  225/269 - Train Accuracy: 0.4869, Validation Accuracy: 0.5082, Loss: 0.9869
    Epoch   0 Batch  230/269 - Train Accuracy: 0.4915, Validation Accuracy: 0.5102, Loss: 0.9450
    Epoch   0 Batch  235/269 - Train Accuracy: 0.5084, Validation Accuracy: 0.5166, Loss: 0.9153
    Epoch   0 Batch  240/269 - Train Accuracy: 0.5497, Validation Accuracy: 0.5131, Loss: 0.8381
    Epoch   0 Batch  245/269 - Train Accuracy: 0.4941, Validation Accuracy: 0.5205, Loss: 0.9405
    Epoch   0 Batch  250/269 - Train Accuracy: 0.4941, Validation Accuracy: 0.5202, Loss: 0.9037
    Epoch   0 Batch  255/269 - Train Accuracy: 0.5418, Validation Accuracy: 0.5302, Loss: 0.8300
    Epoch   0 Batch  260/269 - Train Accuracy: 0.4910, Validation Accuracy: 0.5135, Loss: 0.8882
    Epoch   0 Batch  265/269 - Train Accuracy: 0.5082, Validation Accuracy: 0.5281, Loss: 0.8656
    Epoch   1 Batch    5/269 - Train Accuracy: 0.4855, Validation Accuracy: 0.5314, Loss: 0.8606
    Epoch   1 Batch   10/269 - Train Accuracy: 0.5108, Validation Accuracy: 0.5385, Loss: 0.8356
    Epoch   1 Batch   15/269 - Train Accuracy: 0.5161, Validation Accuracy: 0.5309, Loss: 0.7927
    Epoch   1 Batch   20/269 - Train Accuracy: 0.5170, Validation Accuracy: 0.5315, Loss: 0.8174
    Epoch   1 Batch   25/269 - Train Accuracy: 0.5337, Validation Accuracy: 0.5479, Loss: 0.8263
    Epoch   1 Batch   30/269 - Train Accuracy: 0.5442, Validation Accuracy: 0.5500, Loss: 0.7676
    Epoch   1 Batch   35/269 - Train Accuracy: 0.5436, Validation Accuracy: 0.5421, Loss: 0.7760
    Epoch   1 Batch   40/269 - Train Accuracy: 0.5299, Validation Accuracy: 0.5691, Loss: 0.7857
    Epoch   1 Batch   45/269 - Train Accuracy: 0.5301, Validation Accuracy: 0.5524, Loss: 0.7785
    Epoch   1 Batch   50/269 - Train Accuracy: 0.5240, Validation Accuracy: 0.5393, Loss: 0.7675
    Epoch   1 Batch   55/269 - Train Accuracy: 0.5657, Validation Accuracy: 0.5692, Loss: 0.7188
    Epoch   1 Batch   60/269 - Train Accuracy: 0.5564, Validation Accuracy: 0.5408, Loss: 0.6881
    Epoch   1 Batch   65/269 - Train Accuracy: 0.5391, Validation Accuracy: 0.5517, Loss: 0.7022
    Epoch   1 Batch   70/269 - Train Accuracy: 0.5605, Validation Accuracy: 0.5630, Loss: 0.7030
    Epoch   1 Batch   75/269 - Train Accuracy: 0.5609, Validation Accuracy: 0.5668, Loss: 0.6868
    Epoch   1 Batch   80/269 - Train Accuracy: 0.5510, Validation Accuracy: 0.5455, Loss: 0.6782
    Epoch   1 Batch   85/269 - Train Accuracy: 0.5541, Validation Accuracy: 0.5676, Loss: 0.6811
    Epoch   1 Batch   90/269 - Train Accuracy: 0.5308, Validation Accuracy: 0.5737, Loss: 0.7144
    Epoch   1 Batch   95/269 - Train Accuracy: 0.5665, Validation Accuracy: 0.5743, Loss: 0.6656
    Epoch   1 Batch  100/269 - Train Accuracy: 0.5831, Validation Accuracy: 0.5683, Loss: 0.6459
    Epoch   1 Batch  105/269 - Train Accuracy: 0.5560, Validation Accuracy: 0.5556, Loss: 0.6629
    Epoch   1 Batch  110/269 - Train Accuracy: 0.5576, Validation Accuracy: 0.5484, Loss: 0.6355
    Epoch   1 Batch  115/269 - Train Accuracy: 0.5478, Validation Accuracy: 0.5665, Loss: 0.6713
    Epoch   1 Batch  120/269 - Train Accuracy: 0.5701, Validation Accuracy: 0.5748, Loss: 0.6590
    Epoch   1 Batch  125/269 - Train Accuracy: 0.5800, Validation Accuracy: 0.5657, Loss: 0.6230
    Epoch   1 Batch  130/269 - Train Accuracy: 0.5408, Validation Accuracy: 0.5587, Loss: 0.6519
    Epoch   1 Batch  135/269 - Train Accuracy: 0.5438, Validation Accuracy: 0.5534, Loss: 0.6736
    Epoch   1 Batch  140/269 - Train Accuracy: 0.5975, Validation Accuracy: 0.5725, Loss: 0.6292
    Epoch   1 Batch  145/269 - Train Accuracy: 0.5826, Validation Accuracy: 0.5796, Loss: 0.6138
    Epoch   1 Batch  150/269 - Train Accuracy: 0.5771, Validation Accuracy: 0.5798, Loss: 0.6181
    Epoch   1 Batch  155/269 - Train Accuracy: 0.5905, Validation Accuracy: 0.5827, Loss: 0.5809
    Epoch   1 Batch  160/269 - Train Accuracy: 0.6036, Validation Accuracy: 0.5901, Loss: 0.6004
    Epoch   1 Batch  165/269 - Train Accuracy: 0.5734, Validation Accuracy: 0.6007, Loss: 0.6183
    Epoch   1 Batch  170/269 - Train Accuracy: 0.5882, Validation Accuracy: 0.5881, Loss: 0.5837
    Epoch   1 Batch  175/269 - Train Accuracy: 0.6133, Validation Accuracy: 0.6072, Loss: 0.6026
    Epoch   1 Batch  180/269 - Train Accuracy: 0.5803, Validation Accuracy: 0.5787, Loss: 0.5855
    Epoch   1 Batch  185/269 - Train Accuracy: 0.6129, Validation Accuracy: 0.6088, Loss: 0.5775
    Epoch   1 Batch  190/269 - Train Accuracy: 0.5935, Validation Accuracy: 0.5930, Loss: 0.5679
    Epoch   1 Batch  195/269 - Train Accuracy: 0.5947, Validation Accuracy: 0.5961, Loss: 0.5778
    Epoch   1 Batch  200/269 - Train Accuracy: 0.6094, Validation Accuracy: 0.6142, Loss: 0.5942
    Epoch   1 Batch  205/269 - Train Accuracy: 0.5831, Validation Accuracy: 0.5879, Loss: 0.5615
    Epoch   1 Batch  210/269 - Train Accuracy: 0.6158, Validation Accuracy: 0.5976, Loss: 0.5569
    Epoch   1 Batch  215/269 - Train Accuracy: 0.6295, Validation Accuracy: 0.6163, Loss: 0.5296
    Epoch   1 Batch  220/269 - Train Accuracy: 0.5879, Validation Accuracy: 0.5958, Loss: 0.5378
    Epoch   1 Batch  225/269 - Train Accuracy: 0.6038, Validation Accuracy: 0.6214, Loss: 0.5630
    Epoch   1 Batch  230/269 - Train Accuracy: 0.6008, Validation Accuracy: 0.6160, Loss: 0.5544
    Epoch   1 Batch  235/269 - Train Accuracy: 0.6311, Validation Accuracy: 0.6166, Loss: 0.5412
    Epoch   1 Batch  240/269 - Train Accuracy: 0.6625, Validation Accuracy: 0.6358, Loss: 0.4988
    Epoch   1 Batch  245/269 - Train Accuracy: 0.5930, Validation Accuracy: 0.6262, Loss: 0.5744
    Epoch   1 Batch  250/269 - Train Accuracy: 0.6304, Validation Accuracy: 0.6458, Loss: 0.5545
    Epoch   1 Batch  255/269 - Train Accuracy: 0.6408, Validation Accuracy: 0.6409, Loss: 0.5178
    Epoch   1 Batch  260/269 - Train Accuracy: 0.6050, Validation Accuracy: 0.6331, Loss: 0.5604
    Epoch   1 Batch  265/269 - Train Accuracy: 0.6456, Validation Accuracy: 0.6541, Loss: 0.5434
    Epoch   2 Batch    5/269 - Train Accuracy: 0.6116, Validation Accuracy: 0.6483, Loss: 0.5503
    Epoch   2 Batch   10/269 - Train Accuracy: 0.6226, Validation Accuracy: 0.6496, Loss: 0.5451
    Epoch   2 Batch   15/269 - Train Accuracy: 0.6456, Validation Accuracy: 0.6702, Loss: 0.5085
    Epoch   2 Batch   20/269 - Train Accuracy: 0.6340, Validation Accuracy: 0.6448, Loss: 0.5349
    Epoch   2 Batch   25/269 - Train Accuracy: 0.6241, Validation Accuracy: 0.6631, Loss: 0.5480
    Epoch   2 Batch   30/269 - Train Accuracy: 0.6594, Validation Accuracy: 0.6630, Loss: 0.5091
    Epoch   2 Batch   35/269 - Train Accuracy: 0.6488, Validation Accuracy: 0.6552, Loss: 0.5166
    Epoch   2 Batch   40/269 - Train Accuracy: 0.6473, Validation Accuracy: 0.6630, Loss: 0.5254
    Epoch   2 Batch   45/269 - Train Accuracy: 0.6174, Validation Accuracy: 0.6562, Loss: 0.5227
    Epoch   2 Batch   50/269 - Train Accuracy: 0.6353, Validation Accuracy: 0.6574, Loss: 0.5239
    Epoch   2 Batch   55/269 - Train Accuracy: 0.6741, Validation Accuracy: 0.6656, Loss: 0.4833
    Epoch   2 Batch   60/269 - Train Accuracy: 0.6690, Validation Accuracy: 0.6588, Loss: 0.4712
    Epoch   2 Batch   65/269 - Train Accuracy: 0.6446, Validation Accuracy: 0.6671, Loss: 0.4874
    Epoch   2 Batch   70/269 - Train Accuracy: 0.6726, Validation Accuracy: 0.6634, Loss: 0.4878
    Epoch   2 Batch   75/269 - Train Accuracy: 0.6882, Validation Accuracy: 0.6798, Loss: 0.4737
    Epoch   2 Batch   80/269 - Train Accuracy: 0.6636, Validation Accuracy: 0.6737, Loss: 0.4717
    Epoch   2 Batch   85/269 - Train Accuracy: 0.6530, Validation Accuracy: 0.6779, Loss: 0.4746
    Epoch   2 Batch   90/269 - Train Accuracy: 0.6296, Validation Accuracy: 0.6642, Loss: 0.4995
    Epoch   2 Batch   95/269 - Train Accuracy: 0.6660, Validation Accuracy: 0.6767, Loss: 0.4639
    Epoch   2 Batch  100/269 - Train Accuracy: 0.6538, Validation Accuracy: 0.6652, Loss: 0.4603
    Epoch   2 Batch  105/269 - Train Accuracy: 0.6474, Validation Accuracy: 0.6586, Loss: 0.4754
    Epoch   2 Batch  110/269 - Train Accuracy: 0.6838, Validation Accuracy: 0.6867, Loss: 0.4567
    Epoch   2 Batch  115/269 - Train Accuracy: 0.6702, Validation Accuracy: 0.6792, Loss: 0.4865
    Epoch   2 Batch  120/269 - Train Accuracy: 0.6805, Validation Accuracy: 0.6919, Loss: 0.4727
    Epoch   2 Batch  125/269 - Train Accuracy: 0.6936, Validation Accuracy: 0.6842, Loss: 0.4491
    Epoch   2 Batch  130/269 - Train Accuracy: 0.6686, Validation Accuracy: 0.6963, Loss: 0.4666
    Epoch   2 Batch  135/269 - Train Accuracy: 0.6555, Validation Accuracy: 0.6892, Loss: 0.4880
    Epoch   2 Batch  140/269 - Train Accuracy: 0.6803, Validation Accuracy: 0.6885, Loss: 0.4574
    Epoch   2 Batch  145/269 - Train Accuracy: 0.6999, Validation Accuracy: 0.6945, Loss: 0.4474
    Epoch   2 Batch  150/269 - Train Accuracy: 0.6998, Validation Accuracy: 0.6864, Loss: 0.4343
    Epoch   2 Batch  155/269 - Train Accuracy: 0.7115, Validation Accuracy: 0.7000, Loss: 0.4194
    Epoch   2 Batch  160/269 - Train Accuracy: 0.7011, Validation Accuracy: 0.7177, Loss: 0.4282
    Epoch   2 Batch  165/269 - Train Accuracy: 0.7104, Validation Accuracy: 0.7164, Loss: 0.4440
    Epoch   2 Batch  170/269 - Train Accuracy: 0.7066, Validation Accuracy: 0.7201, Loss: 0.4258
    Epoch   2 Batch  175/269 - Train Accuracy: 0.6872, Validation Accuracy: 0.7097, Loss: 0.4396
    Epoch   2 Batch  180/269 - Train Accuracy: 0.7099, Validation Accuracy: 0.7081, Loss: 0.4269
    Epoch   2 Batch  185/269 - Train Accuracy: 0.7177, Validation Accuracy: 0.7154, Loss: 0.4155
    Epoch   2 Batch  190/269 - Train Accuracy: 0.7094, Validation Accuracy: 0.7149, Loss: 0.4168
    Epoch   2 Batch  195/269 - Train Accuracy: 0.7059, Validation Accuracy: 0.7150, Loss: 0.4200
    Epoch   2 Batch  200/269 - Train Accuracy: 0.7072, Validation Accuracy: 0.7057, Loss: 0.4357
    Epoch   2 Batch  205/269 - Train Accuracy: 0.7248, Validation Accuracy: 0.7114, Loss: 0.4002
    Epoch   2 Batch  210/269 - Train Accuracy: 0.7131, Validation Accuracy: 0.7195, Loss: 0.4028
    Epoch   2 Batch  215/269 - Train Accuracy: 0.7461, Validation Accuracy: 0.7280, Loss: 0.3865
    Epoch   2 Batch  220/269 - Train Accuracy: 0.7304, Validation Accuracy: 0.7239, Loss: 0.3892
    Epoch   2 Batch  225/269 - Train Accuracy: 0.7197, Validation Accuracy: 0.7267, Loss: 0.4046
    Epoch   2 Batch  230/269 - Train Accuracy: 0.7201, Validation Accuracy: 0.7327, Loss: 0.4021
    Epoch   2 Batch  235/269 - Train Accuracy: 0.7338, Validation Accuracy: 0.7290, Loss: 0.3889
    Epoch   2 Batch  240/269 - Train Accuracy: 0.7551, Validation Accuracy: 0.7261, Loss: 0.3620
    Epoch   2 Batch  245/269 - Train Accuracy: 0.6931, Validation Accuracy: 0.7097, Loss: 0.4116
    Epoch   2 Batch  250/269 - Train Accuracy: 0.7334, Validation Accuracy: 0.7244, Loss: 0.4069
    Epoch   2 Batch  255/269 - Train Accuracy: 0.7451, Validation Accuracy: 0.7358, Loss: 0.3753
    Epoch   2 Batch  260/269 - Train Accuracy: 0.7115, Validation Accuracy: 0.7373, Loss: 0.4083
    Epoch   2 Batch  265/269 - Train Accuracy: 0.7342, Validation Accuracy: 0.7351, Loss: 0.3974
    Epoch   3 Batch    5/269 - Train Accuracy: 0.7077, Validation Accuracy: 0.7227, Loss: 0.3979
    Epoch   3 Batch   10/269 - Train Accuracy: 0.7225, Validation Accuracy: 0.7285, Loss: 0.3968
    Epoch   3 Batch   15/269 - Train Accuracy: 0.7399, Validation Accuracy: 0.7372, Loss: 0.3671
    Epoch   3 Batch   20/269 - Train Accuracy: 0.7366, Validation Accuracy: 0.7382, Loss: 0.3878
    Epoch   3 Batch   25/269 - Train Accuracy: 0.7173, Validation Accuracy: 0.7356, Loss: 0.4012
    Epoch   3 Batch   30/269 - Train Accuracy: 0.7217, Validation Accuracy: 0.7221, Loss: 0.3746
    Epoch   3 Batch   35/269 - Train Accuracy: 0.7396, Validation Accuracy: 0.7379, Loss: 0.3964
    Epoch   3 Batch   40/269 - Train Accuracy: 0.7215, Validation Accuracy: 0.7379, Loss: 0.3905
    Epoch   3 Batch   45/269 - Train Accuracy: 0.7292, Validation Accuracy: 0.7507, Loss: 0.3860
    Epoch   3 Batch   50/269 - Train Accuracy: 0.7214, Validation Accuracy: 0.7405, Loss: 0.3881
    Epoch   3 Batch   55/269 - Train Accuracy: 0.7467, Validation Accuracy: 0.7350, Loss: 0.3500
    Epoch   3 Batch   60/269 - Train Accuracy: 0.7628, Validation Accuracy: 0.7330, Loss: 0.3388
    Epoch   3 Batch   65/269 - Train Accuracy: 0.7373, Validation Accuracy: 0.7417, Loss: 0.3566
    Epoch   3 Batch   70/269 - Train Accuracy: 0.7592, Validation Accuracy: 0.7326, Loss: 0.3573
    Epoch   3 Batch   75/269 - Train Accuracy: 0.7563, Validation Accuracy: 0.7532, Loss: 0.3435
    Epoch   3 Batch   80/269 - Train Accuracy: 0.7450, Validation Accuracy: 0.7446, Loss: 0.3448
    Epoch   3 Batch   85/269 - Train Accuracy: 0.7553, Validation Accuracy: 0.7538, Loss: 0.3436
    Epoch   3 Batch   90/269 - Train Accuracy: 0.7380, Validation Accuracy: 0.7504, Loss: 0.3631
    Epoch   3 Batch   95/269 - Train Accuracy: 0.7480, Validation Accuracy: 0.7553, Loss: 0.3391
    Epoch   3 Batch  100/269 - Train Accuracy: 0.7630, Validation Accuracy: 0.7542, Loss: 0.3381
    Epoch   3 Batch  105/269 - Train Accuracy: 0.7502, Validation Accuracy: 0.7470, Loss: 0.3349
    Epoch   3 Batch  110/269 - Train Accuracy: 0.7559, Validation Accuracy: 0.7562, Loss: 0.3273
    Epoch   3 Batch  115/269 - Train Accuracy: 0.7499, Validation Accuracy: 0.7476, Loss: 0.3519
    Epoch   3 Batch  120/269 - Train Accuracy: 0.7593, Validation Accuracy: 0.7567, Loss: 0.3415
    Epoch   3 Batch  125/269 - Train Accuracy: 0.7620, Validation Accuracy: 0.7559, Loss: 0.3283
    Epoch   3 Batch  130/269 - Train Accuracy: 0.7433, Validation Accuracy: 0.7599, Loss: 0.3366
    Epoch   3 Batch  135/269 - Train Accuracy: 0.7431, Validation Accuracy: 0.7558, Loss: 0.3549
    Epoch   3 Batch  140/269 - Train Accuracy: 0.7530, Validation Accuracy: 0.7594, Loss: 0.3366
    Epoch   3 Batch  145/269 - Train Accuracy: 0.7588, Validation Accuracy: 0.7525, Loss: 0.3272
    Epoch   3 Batch  150/269 - Train Accuracy: 0.7731, Validation Accuracy: 0.7654, Loss: 0.3161
    Epoch   3 Batch  155/269 - Train Accuracy: 0.7709, Validation Accuracy: 0.7577, Loss: 0.3096
    Epoch   3 Batch  160/269 - Train Accuracy: 0.7582, Validation Accuracy: 0.7547, Loss: 0.3144
    Epoch   3 Batch  165/269 - Train Accuracy: 0.7674, Validation Accuracy: 0.7719, Loss: 0.3228
    Epoch   3 Batch  170/269 - Train Accuracy: 0.7628, Validation Accuracy: 0.7679, Loss: 0.3023
    Epoch   3 Batch  175/269 - Train Accuracy: 0.7560, Validation Accuracy: 0.7682, Loss: 0.3199
    Epoch   3 Batch  180/269 - Train Accuracy: 0.7799, Validation Accuracy: 0.7702, Loss: 0.3035
    Epoch   3 Batch  185/269 - Train Accuracy: 0.7910, Validation Accuracy: 0.7747, Loss: 0.3003
    Epoch   3 Batch  190/269 - Train Accuracy: 0.7784, Validation Accuracy: 0.7759, Loss: 0.2957
    Epoch   3 Batch  195/269 - Train Accuracy: 0.7651, Validation Accuracy: 0.7719, Loss: 0.2982
    Epoch   3 Batch  200/269 - Train Accuracy: 0.7747, Validation Accuracy: 0.7800, Loss: 0.3155
    Epoch   3 Batch  205/269 - Train Accuracy: 0.7928, Validation Accuracy: 0.7740, Loss: 0.2896
    Epoch   3 Batch  210/269 - Train Accuracy: 0.7857, Validation Accuracy: 0.7793, Loss: 0.2885
    Epoch   3 Batch  215/269 - Train Accuracy: 0.8036, Validation Accuracy: 0.7880, Loss: 0.2818
    Epoch   3 Batch  220/269 - Train Accuracy: 0.7895, Validation Accuracy: 0.7757, Loss: 0.2753
    Epoch   3 Batch  225/269 - Train Accuracy: 0.7701, Validation Accuracy: 0.7860, Loss: 0.2878
    Epoch   3 Batch  230/269 - Train Accuracy: 0.7794, Validation Accuracy: 0.7814, Loss: 0.2843
    Epoch   3 Batch  235/269 - Train Accuracy: 0.7894, Validation Accuracy: 0.7931, Loss: 0.2771
    Epoch   3 Batch  240/269 - Train Accuracy: 0.7895, Validation Accuracy: 0.7773, Loss: 0.2558
    Epoch   3 Batch  245/269 - Train Accuracy: 0.7586, Validation Accuracy: 0.7781, Loss: 0.2945
    Epoch   3 Batch  250/269 - Train Accuracy: 0.7763, Validation Accuracy: 0.7895, Loss: 0.2878
    Epoch   3 Batch  255/269 - Train Accuracy: 0.7858, Validation Accuracy: 0.7860, Loss: 0.2755
    Epoch   3 Batch  260/269 - Train Accuracy: 0.7680, Validation Accuracy: 0.7788, Loss: 0.2937
    Epoch   3 Batch  265/269 - Train Accuracy: 0.8030, Validation Accuracy: 0.7844, Loss: 0.2734
    Epoch   4 Batch    5/269 - Train Accuracy: 0.7500, Validation Accuracy: 0.7798, Loss: 0.2802
    Epoch   4 Batch   10/269 - Train Accuracy: 0.7798, Validation Accuracy: 0.7726, Loss: 0.2797
    Epoch   4 Batch   15/269 - Train Accuracy: 0.7744, Validation Accuracy: 0.7913, Loss: 0.2609
    Epoch   4 Batch   20/269 - Train Accuracy: 0.7631, Validation Accuracy: 0.7795, Loss: 0.2716
    Epoch   4 Batch   25/269 - Train Accuracy: 0.7665, Validation Accuracy: 0.7849, Loss: 0.2888
    Epoch   4 Batch   30/269 - Train Accuracy: 0.7714, Validation Accuracy: 0.7932, Loss: 0.2709
    Epoch   4 Batch   35/269 - Train Accuracy: 0.7946, Validation Accuracy: 0.7854, Loss: 0.2752
    Epoch   4 Batch   40/269 - Train Accuracy: 0.7760, Validation Accuracy: 0.7974, Loss: 0.2671
    Epoch   4 Batch   45/269 - Train Accuracy: 0.7846, Validation Accuracy: 0.7871, Loss: 0.2717
    Epoch   4 Batch   50/269 - Train Accuracy: 0.7685, Validation Accuracy: 0.7944, Loss: 0.2744
    Epoch   4 Batch   55/269 - Train Accuracy: 0.7897, Validation Accuracy: 0.7936, Loss: 0.2421
    Epoch   4 Batch   60/269 - Train Accuracy: 0.7986, Validation Accuracy: 0.7965, Loss: 0.2363
    Epoch   4 Batch   65/269 - Train Accuracy: 0.7872, Validation Accuracy: 0.8011, Loss: 0.2497
    Epoch   4 Batch   70/269 - Train Accuracy: 0.8296, Validation Accuracy: 0.8005, Loss: 0.2417
    Epoch   4 Batch   75/269 - Train Accuracy: 0.8141, Validation Accuracy: 0.8007, Loss: 0.2408
    Epoch   4 Batch   80/269 - Train Accuracy: 0.8262, Validation Accuracy: 0.7985, Loss: 0.2421
    Epoch   4 Batch   85/269 - Train Accuracy: 0.8097, Validation Accuracy: 0.8022, Loss: 0.2336
    Epoch   4 Batch   90/269 - Train Accuracy: 0.7840, Validation Accuracy: 0.8034, Loss: 0.2488
    Epoch   4 Batch   95/269 - Train Accuracy: 0.7886, Validation Accuracy: 0.8040, Loss: 0.2325
    Epoch   4 Batch  100/269 - Train Accuracy: 0.8051, Validation Accuracy: 0.7988, Loss: 0.2312
    Epoch   4 Batch  105/269 - Train Accuracy: 0.7967, Validation Accuracy: 0.8023, Loss: 0.2341
    Epoch   4 Batch  110/269 - Train Accuracy: 0.8148, Validation Accuracy: 0.8036, Loss: 0.2277
    Epoch   4 Batch  115/269 - Train Accuracy: 0.7953, Validation Accuracy: 0.8049, Loss: 0.2526
    Epoch   4 Batch  120/269 - Train Accuracy: 0.8011, Validation Accuracy: 0.7958, Loss: 0.2340
    Epoch   4 Batch  125/269 - Train Accuracy: 0.8112, Validation Accuracy: 0.8064, Loss: 0.2273
    Epoch   4 Batch  130/269 - Train Accuracy: 0.7940, Validation Accuracy: 0.7995, Loss: 0.2306
    Epoch   4 Batch  135/269 - Train Accuracy: 0.7879, Validation Accuracy: 0.8047, Loss: 0.2498
    Epoch   4 Batch  140/269 - Train Accuracy: 0.8074, Validation Accuracy: 0.8137, Loss: 0.2365
    Epoch   4 Batch  145/269 - Train Accuracy: 0.8181, Validation Accuracy: 0.8098, Loss: 0.2232
    Epoch   4 Batch  150/269 - Train Accuracy: 0.8153, Validation Accuracy: 0.8081, Loss: 0.2221
    Epoch   4 Batch  155/269 - Train Accuracy: 0.8111, Validation Accuracy: 0.8089, Loss: 0.2180
    Epoch   4 Batch  160/269 - Train Accuracy: 0.8158, Validation Accuracy: 0.8144, Loss: 0.2230
    Epoch   4 Batch  165/269 - Train Accuracy: 0.8085, Validation Accuracy: 0.8095, Loss: 0.2226
    Epoch   4 Batch  170/269 - Train Accuracy: 0.8158, Validation Accuracy: 0.8290, Loss: 0.2118
    Epoch   4 Batch  175/269 - Train Accuracy: 0.8124, Validation Accuracy: 0.8169, Loss: 0.2281
    Epoch   4 Batch  180/269 - Train Accuracy: 0.8225, Validation Accuracy: 0.8178, Loss: 0.2133
    Epoch   4 Batch  185/269 - Train Accuracy: 0.8320, Validation Accuracy: 0.8145, Loss: 0.2227
    Epoch   4 Batch  190/269 - Train Accuracy: 0.7985, Validation Accuracy: 0.8066, Loss: 0.2167
    Epoch   4 Batch  195/269 - Train Accuracy: 0.8069, Validation Accuracy: 0.8258, Loss: 0.2189
    Epoch   4 Batch  200/269 - Train Accuracy: 0.7988, Validation Accuracy: 0.8105, Loss: 0.2227
    Epoch   4 Batch  205/269 - Train Accuracy: 0.8306, Validation Accuracy: 0.8131, Loss: 0.2191
    Epoch   4 Batch  210/269 - Train Accuracy: 0.8132, Validation Accuracy: 0.8109, Loss: 0.2073
    Epoch   4 Batch  215/269 - Train Accuracy: 0.8541, Validation Accuracy: 0.8219, Loss: 0.1994
    Epoch   4 Batch  220/269 - Train Accuracy: 0.8340, Validation Accuracy: 0.8210, Loss: 0.1971
    Epoch   4 Batch  225/269 - Train Accuracy: 0.8139, Validation Accuracy: 0.8200, Loss: 0.2003
    Epoch   4 Batch  230/269 - Train Accuracy: 0.8152, Validation Accuracy: 0.8216, Loss: 0.2051
    Epoch   4 Batch  235/269 - Train Accuracy: 0.8305, Validation Accuracy: 0.8221, Loss: 0.1939
    Epoch   4 Batch  240/269 - Train Accuracy: 0.8375, Validation Accuracy: 0.8212, Loss: 0.1830
    Epoch   4 Batch  245/269 - Train Accuracy: 0.8040, Validation Accuracy: 0.8160, Loss: 0.2109
    Epoch   4 Batch  250/269 - Train Accuracy: 0.8271, Validation Accuracy: 0.8271, Loss: 0.2034
    Epoch   4 Batch  255/269 - Train Accuracy: 0.8279, Validation Accuracy: 0.8279, Loss: 0.1987
    Epoch   4 Batch  260/269 - Train Accuracy: 0.7987, Validation Accuracy: 0.8204, Loss: 0.2136
    Epoch   4 Batch  265/269 - Train Accuracy: 0.8298, Validation Accuracy: 0.8288, Loss: 0.2002
    Epoch   5 Batch    5/269 - Train Accuracy: 0.7929, Validation Accuracy: 0.8273, Loss: 0.1984
    Epoch   5 Batch   10/269 - Train Accuracy: 0.8295, Validation Accuracy: 0.8237, Loss: 0.1965
    Epoch   5 Batch   15/269 - Train Accuracy: 0.8254, Validation Accuracy: 0.8195, Loss: 0.1836
    Epoch   5 Batch   20/269 - Train Accuracy: 0.8354, Validation Accuracy: 0.8373, Loss: 0.1948
    Epoch   5 Batch   25/269 - Train Accuracy: 0.8263, Validation Accuracy: 0.8286, Loss: 0.2065
    Epoch   5 Batch   30/269 - Train Accuracy: 0.8139, Validation Accuracy: 0.8367, Loss: 0.1916
    Epoch   5 Batch   35/269 - Train Accuracy: 0.8352, Validation Accuracy: 0.8340, Loss: 0.2028
    Epoch   5 Batch   40/269 - Train Accuracy: 0.8162, Validation Accuracy: 0.8389, Loss: 0.1887
    Epoch   5 Batch   45/269 - Train Accuracy: 0.8352, Validation Accuracy: 0.8343, Loss: 0.1970
    Epoch   5 Batch   50/269 - Train Accuracy: 0.8228, Validation Accuracy: 0.8316, Loss: 0.2025
    Epoch   5 Batch   55/269 - Train Accuracy: 0.8404, Validation Accuracy: 0.8321, Loss: 0.1789
    Epoch   5 Batch   60/269 - Train Accuracy: 0.8305, Validation Accuracy: 0.8371, Loss: 0.1753
    Epoch   5 Batch   65/269 - Train Accuracy: 0.8299, Validation Accuracy: 0.8443, Loss: 0.1861
    Epoch   5 Batch   70/269 - Train Accuracy: 0.8466, Validation Accuracy: 0.8394, Loss: 0.1861
    Epoch   5 Batch   75/269 - Train Accuracy: 0.8507, Validation Accuracy: 0.8421, Loss: 0.1817
    Epoch   5 Batch   80/269 - Train Accuracy: 0.8490, Validation Accuracy: 0.8395, Loss: 0.1759
    Epoch   5 Batch   85/269 - Train Accuracy: 0.8411, Validation Accuracy: 0.8443, Loss: 0.1711
    Epoch   5 Batch   90/269 - Train Accuracy: 0.8386, Validation Accuracy: 0.8437, Loss: 0.1850
    Epoch   5 Batch   95/269 - Train Accuracy: 0.8410, Validation Accuracy: 0.8417, Loss: 0.1760
    Epoch   5 Batch  100/269 - Train Accuracy: 0.8557, Validation Accuracy: 0.8382, Loss: 0.1711
    Epoch   5 Batch  105/269 - Train Accuracy: 0.8424, Validation Accuracy: 0.8426, Loss: 0.1745
    Epoch   5 Batch  110/269 - Train Accuracy: 0.8423, Validation Accuracy: 0.8409, Loss: 0.1697
    Epoch   5 Batch  115/269 - Train Accuracy: 0.8301, Validation Accuracy: 0.8512, Loss: 0.1817
    Epoch   5 Batch  120/269 - Train Accuracy: 0.8465, Validation Accuracy: 0.8501, Loss: 0.1738
    Epoch   5 Batch  125/269 - Train Accuracy: 0.8570, Validation Accuracy: 0.8319, Loss: 0.1672
    Epoch   5 Batch  130/269 - Train Accuracy: 0.8319, Validation Accuracy: 0.8348, Loss: 0.1703
    Epoch   5 Batch  135/269 - Train Accuracy: 0.8315, Validation Accuracy: 0.8474, Loss: 0.1867
    Epoch   5 Batch  140/269 - Train Accuracy: 0.8373, Validation Accuracy: 0.8587, Loss: 0.1766
    Epoch   5 Batch  145/269 - Train Accuracy: 0.8560, Validation Accuracy: 0.8458, Loss: 0.1667
    Epoch   5 Batch  150/269 - Train Accuracy: 0.8477, Validation Accuracy: 0.8526, Loss: 0.1679
    Epoch   5 Batch  155/269 - Train Accuracy: 0.8488, Validation Accuracy: 0.8496, Loss: 0.1622
    Epoch   5 Batch  160/269 - Train Accuracy: 0.8487, Validation Accuracy: 0.8475, Loss: 0.1656
    Epoch   5 Batch  165/269 - Train Accuracy: 0.8537, Validation Accuracy: 0.8551, Loss: 0.1645
    Epoch   5 Batch  170/269 - Train Accuracy: 0.8634, Validation Accuracy: 0.8589, Loss: 0.1569
    Epoch   5 Batch  175/269 - Train Accuracy: 0.8468, Validation Accuracy: 0.8421, Loss: 0.1729
    Epoch   5 Batch  180/269 - Train Accuracy: 0.8610, Validation Accuracy: 0.8502, Loss: 0.1581
    Epoch   5 Batch  185/269 - Train Accuracy: 0.8718, Validation Accuracy: 0.8476, Loss: 0.1602
    Epoch   5 Batch  190/269 - Train Accuracy: 0.8419, Validation Accuracy: 0.8447, Loss: 0.1666
    Epoch   5 Batch  195/269 - Train Accuracy: 0.8359, Validation Accuracy: 0.8497, Loss: 0.1780
    Epoch   5 Batch  200/269 - Train Accuracy: 0.8391, Validation Accuracy: 0.8403, Loss: 0.1878
    Epoch   5 Batch  205/269 - Train Accuracy: 0.8515, Validation Accuracy: 0.8559, Loss: 0.1725
    Epoch   5 Batch  210/269 - Train Accuracy: 0.8528, Validation Accuracy: 0.8442, Loss: 0.1593
    Epoch   5 Batch  215/269 - Train Accuracy: 0.8769, Validation Accuracy: 0.8531, Loss: 0.1534
    Epoch   5 Batch  220/269 - Train Accuracy: 0.8604, Validation Accuracy: 0.8517, Loss: 0.1586
    Epoch   5 Batch  225/269 - Train Accuracy: 0.8390, Validation Accuracy: 0.8642, Loss: 0.1594
    Epoch   5 Batch  230/269 - Train Accuracy: 0.8504, Validation Accuracy: 0.8519, Loss: 0.1587
    Epoch   5 Batch  235/269 - Train Accuracy: 0.8725, Validation Accuracy: 0.8564, Loss: 0.1490
    Epoch   5 Batch  240/269 - Train Accuracy: 0.8576, Validation Accuracy: 0.8503, Loss: 0.1420
    Epoch   5 Batch  245/269 - Train Accuracy: 0.8500, Validation Accuracy: 0.8506, Loss: 0.1656
    Epoch   5 Batch  250/269 - Train Accuracy: 0.8651, Validation Accuracy: 0.8662, Loss: 0.1542
    Epoch   5 Batch  255/269 - Train Accuracy: 0.8703, Validation Accuracy: 0.8566, Loss: 0.1565
    Epoch   5 Batch  260/269 - Train Accuracy: 0.8340, Validation Accuracy: 0.8635, Loss: 0.1663
    Epoch   5 Batch  265/269 - Train Accuracy: 0.8588, Validation Accuracy: 0.8603, Loss: 0.1512
    Epoch   6 Batch    5/269 - Train Accuracy: 0.8476, Validation Accuracy: 0.8520, Loss: 0.1587
    Epoch   6 Batch   10/269 - Train Accuracy: 0.8735, Validation Accuracy: 0.8572, Loss: 0.1483
    Epoch   6 Batch   15/269 - Train Accuracy: 0.8601, Validation Accuracy: 0.8662, Loss: 0.1389
    Epoch   6 Batch   20/269 - Train Accuracy: 0.8779, Validation Accuracy: 0.8703, Loss: 0.1466
    Epoch   6 Batch   25/269 - Train Accuracy: 0.8463, Validation Accuracy: 0.8576, Loss: 0.1681
    Epoch   6 Batch   30/269 - Train Accuracy: 0.8546, Validation Accuracy: 0.8598, Loss: 0.1563
    Epoch   6 Batch   35/269 - Train Accuracy: 0.8649, Validation Accuracy: 0.8604, Loss: 0.1594
    Epoch   6 Batch   40/269 - Train Accuracy: 0.8500, Validation Accuracy: 0.8674, Loss: 0.1538
    Epoch   6 Batch   45/269 - Train Accuracy: 0.8694, Validation Accuracy: 0.8719, Loss: 0.1598
    Epoch   6 Batch   50/269 - Train Accuracy: 0.8441, Validation Accuracy: 0.8639, Loss: 0.1632
    Epoch   6 Batch   55/269 - Train Accuracy: 0.8690, Validation Accuracy: 0.8607, Loss: 0.1398
    Epoch   6 Batch   60/269 - Train Accuracy: 0.8747, Validation Accuracy: 0.8705, Loss: 0.1388
    Epoch   6 Batch   65/269 - Train Accuracy: 0.8583, Validation Accuracy: 0.8690, Loss: 0.1458
    Epoch   6 Batch   70/269 - Train Accuracy: 0.8666, Validation Accuracy: 0.8572, Loss: 0.1442
    Epoch   6 Batch   75/269 - Train Accuracy: 0.8842, Validation Accuracy: 0.8696, Loss: 0.1424
    Epoch   6 Batch   80/269 - Train Accuracy: 0.8632, Validation Accuracy: 0.8676, Loss: 0.1401
    Epoch   6 Batch   85/269 - Train Accuracy: 0.8706, Validation Accuracy: 0.8692, Loss: 0.1365
    Epoch   6 Batch   90/269 - Train Accuracy: 0.8637, Validation Accuracy: 0.8691, Loss: 0.1431
    Epoch   6 Batch   95/269 - Train Accuracy: 0.8756, Validation Accuracy: 0.8624, Loss: 0.1319
    Epoch   6 Batch  100/269 - Train Accuracy: 0.8843, Validation Accuracy: 0.8738, Loss: 0.1387
    Epoch   6 Batch  105/269 - Train Accuracy: 0.8671, Validation Accuracy: 0.8733, Loss: 0.1398
    Epoch   6 Batch  110/269 - Train Accuracy: 0.8661, Validation Accuracy: 0.8654, Loss: 0.1354
    Epoch   6 Batch  115/269 - Train Accuracy: 0.8624, Validation Accuracy: 0.8694, Loss: 0.1473
    Epoch   6 Batch  120/269 - Train Accuracy: 0.8768, Validation Accuracy: 0.8783, Loss: 0.1423
    Epoch   6 Batch  125/269 - Train Accuracy: 0.8921, Validation Accuracy: 0.8661, Loss: 0.1288
    Epoch   6 Batch  130/269 - Train Accuracy: 0.8716, Validation Accuracy: 0.8711, Loss: 0.1359
    Epoch   6 Batch  135/269 - Train Accuracy: 0.8616, Validation Accuracy: 0.8672, Loss: 0.1456
    Epoch   6 Batch  140/269 - Train Accuracy: 0.8628, Validation Accuracy: 0.8765, Loss: 0.1434
    Epoch   6 Batch  145/269 - Train Accuracy: 0.8742, Validation Accuracy: 0.8659, Loss: 0.1292
    Epoch   6 Batch  150/269 - Train Accuracy: 0.8801, Validation Accuracy: 0.8742, Loss: 0.1354
    Epoch   6 Batch  155/269 - Train Accuracy: 0.8739, Validation Accuracy: 0.8757, Loss: 0.1363
    Epoch   6 Batch  160/269 - Train Accuracy: 0.8762, Validation Accuracy: 0.8645, Loss: 0.1349
    Epoch   6 Batch  165/269 - Train Accuracy: 0.8761, Validation Accuracy: 0.8706, Loss: 0.1362
    Epoch   6 Batch  170/269 - Train Accuracy: 0.8588, Validation Accuracy: 0.8706, Loss: 0.1291
    Epoch   6 Batch  175/269 - Train Accuracy: 0.8622, Validation Accuracy: 0.8679, Loss: 0.1459
    Epoch   6 Batch  180/269 - Train Accuracy: 0.8797, Validation Accuracy: 0.8554, Loss: 0.1318
    Epoch   6 Batch  185/269 - Train Accuracy: 0.8863, Validation Accuracy: 0.8721, Loss: 0.1301
    Epoch   6 Batch  190/269 - Train Accuracy: 0.8565, Validation Accuracy: 0.8733, Loss: 0.1399
    Epoch   6 Batch  195/269 - Train Accuracy: 0.8540, Validation Accuracy: 0.8698, Loss: 0.1347
    Epoch   6 Batch  200/269 - Train Accuracy: 0.8632, Validation Accuracy: 0.8620, Loss: 0.1348
    Epoch   6 Batch  205/269 - Train Accuracy: 0.8724, Validation Accuracy: 0.8667, Loss: 0.1297
    Epoch   6 Batch  210/269 - Train Accuracy: 0.8696, Validation Accuracy: 0.8763, Loss: 0.1338
    Epoch   6 Batch  215/269 - Train Accuracy: 0.8825, Validation Accuracy: 0.8734, Loss: 0.1213
    Epoch   6 Batch  220/269 - Train Accuracy: 0.8841, Validation Accuracy: 0.8694, Loss: 0.1240
    Epoch   6 Batch  225/269 - Train Accuracy: 0.8573, Validation Accuracy: 0.8746, Loss: 0.1240
    Epoch   6 Batch  230/269 - Train Accuracy: 0.8761, Validation Accuracy: 0.8791, Loss: 0.1279
    Epoch   6 Batch  235/269 - Train Accuracy: 0.9006, Validation Accuracy: 0.8790, Loss: 0.1193
    Epoch   6 Batch  240/269 - Train Accuracy: 0.8794, Validation Accuracy: 0.8766, Loss: 0.1159
    Epoch   6 Batch  245/269 - Train Accuracy: 0.8604, Validation Accuracy: 0.8676, Loss: 0.1295
    Epoch   6 Batch  250/269 - Train Accuracy: 0.8870, Validation Accuracy: 0.8745, Loss: 0.1231
    Epoch   6 Batch  255/269 - Train Accuracy: 0.8830, Validation Accuracy: 0.8714, Loss: 0.1235
    Epoch   6 Batch  260/269 - Train Accuracy: 0.8589, Validation Accuracy: 0.8848, Loss: 0.1327
    Epoch   6 Batch  265/269 - Train Accuracy: 0.8778, Validation Accuracy: 0.8726, Loss: 0.1288
    Epoch   7 Batch    5/269 - Train Accuracy: 0.8718, Validation Accuracy: 0.8737, Loss: 0.1350
    Epoch   7 Batch   10/269 - Train Accuracy: 0.8973, Validation Accuracy: 0.8729, Loss: 0.1206
    Epoch   7 Batch   15/269 - Train Accuracy: 0.8791, Validation Accuracy: 0.8766, Loss: 0.1133
    Epoch   7 Batch   20/269 - Train Accuracy: 0.8895, Validation Accuracy: 0.8767, Loss: 0.1247
    Epoch   7 Batch   25/269 - Train Accuracy: 0.8664, Validation Accuracy: 0.8770, Loss: 0.1380
    Epoch   7 Batch   30/269 - Train Accuracy: 0.8730, Validation Accuracy: 0.8748, Loss: 0.1214
    Epoch   7 Batch   35/269 - Train Accuracy: 0.8890, Validation Accuracy: 0.8728, Loss: 0.1347
    Epoch   7 Batch   40/269 - Train Accuracy: 0.8699, Validation Accuracy: 0.8774, Loss: 0.1224
    Epoch   7 Batch   45/269 - Train Accuracy: 0.8794, Validation Accuracy: 0.8739, Loss: 0.1287
    Epoch   7 Batch   50/269 - Train Accuracy: 0.8572, Validation Accuracy: 0.8709, Loss: 0.1313
    Epoch   7 Batch   55/269 - Train Accuracy: 0.8996, Validation Accuracy: 0.8814, Loss: 0.1132
    Epoch   7 Batch   60/269 - Train Accuracy: 0.8920, Validation Accuracy: 0.8796, Loss: 0.1113
    Epoch   7 Batch   65/269 - Train Accuracy: 0.8849, Validation Accuracy: 0.8851, Loss: 0.1212
    Epoch   7 Batch   70/269 - Train Accuracy: 0.8950, Validation Accuracy: 0.8758, Loss: 0.1219
    Epoch   7 Batch   75/269 - Train Accuracy: 0.8968, Validation Accuracy: 0.8909, Loss: 0.1230
    Epoch   7 Batch   80/269 - Train Accuracy: 0.8881, Validation Accuracy: 0.8759, Loss: 0.1167
    Epoch   7 Batch   85/269 - Train Accuracy: 0.8863, Validation Accuracy: 0.8770, Loss: 0.1141
    Epoch   7 Batch   90/269 - Train Accuracy: 0.8795, Validation Accuracy: 0.8771, Loss: 0.1204
    Epoch   7 Batch   95/269 - Train Accuracy: 0.9005, Validation Accuracy: 0.8830, Loss: 0.1098
    Epoch   7 Batch  100/269 - Train Accuracy: 0.9055, Validation Accuracy: 0.8766, Loss: 0.1125
    Epoch   7 Batch  105/269 - Train Accuracy: 0.8853, Validation Accuracy: 0.8890, Loss: 0.1135
    Epoch   7 Batch  110/269 - Train Accuracy: 0.8810, Validation Accuracy: 0.8839, Loss: 0.1125
    Epoch   7 Batch  115/269 - Train Accuracy: 0.8681, Validation Accuracy: 0.8837, Loss: 0.1233
    Epoch   7 Batch  120/269 - Train Accuracy: 0.8854, Validation Accuracy: 0.8833, Loss: 0.1236
    Epoch   7 Batch  125/269 - Train Accuracy: 0.8979, Validation Accuracy: 0.8763, Loss: 0.1085
    Epoch   7 Batch  130/269 - Train Accuracy: 0.8764, Validation Accuracy: 0.8855, Loss: 0.1144
    Epoch   7 Batch  135/269 - Train Accuracy: 0.8705, Validation Accuracy: 0.8779, Loss: 0.1177
    Epoch   7 Batch  140/269 - Train Accuracy: 0.8785, Validation Accuracy: 0.8787, Loss: 0.1226
    Epoch   7 Batch  145/269 - Train Accuracy: 0.8859, Validation Accuracy: 0.8783, Loss: 0.1105
    Epoch   7 Batch  150/269 - Train Accuracy: 0.8859, Validation Accuracy: 0.8808, Loss: 0.1183
    Epoch   7 Batch  155/269 - Train Accuracy: 0.8902, Validation Accuracy: 0.8833, Loss: 0.1077
    Epoch   7 Batch  160/269 - Train Accuracy: 0.8909, Validation Accuracy: 0.8846, Loss: 0.1146
    Epoch   7 Batch  165/269 - Train Accuracy: 0.8879, Validation Accuracy: 0.8830, Loss: 0.1132
    Epoch   7 Batch  170/269 - Train Accuracy: 0.8785, Validation Accuracy: 0.8841, Loss: 0.1091
    Epoch   7 Batch  175/269 - Train Accuracy: 0.8821, Validation Accuracy: 0.8849, Loss: 0.1184
    Epoch   7 Batch  180/269 - Train Accuracy: 0.9015, Validation Accuracy: 0.8803, Loss: 0.1064
    Epoch   7 Batch  185/269 - Train Accuracy: 0.9038, Validation Accuracy: 0.8847, Loss: 0.1042
    Epoch   7 Batch  190/269 - Train Accuracy: 0.8679, Validation Accuracy: 0.8844, Loss: 0.1156
    Epoch   7 Batch  195/269 - Train Accuracy: 0.8794, Validation Accuracy: 0.8919, Loss: 0.1089
    Epoch   7 Batch  200/269 - Train Accuracy: 0.8896, Validation Accuracy: 0.8837, Loss: 0.1093
    Epoch   7 Batch  205/269 - Train Accuracy: 0.8912, Validation Accuracy: 0.8844, Loss: 0.1053
    Epoch   7 Batch  210/269 - Train Accuracy: 0.8823, Validation Accuracy: 0.8795, Loss: 0.1030
    Epoch   7 Batch  215/269 - Train Accuracy: 0.8995, Validation Accuracy: 0.8850, Loss: 0.0998
    Epoch   7 Batch  220/269 - Train Accuracy: 0.8939, Validation Accuracy: 0.8862, Loss: 0.1054
    Epoch   7 Batch  225/269 - Train Accuracy: 0.8748, Validation Accuracy: 0.8897, Loss: 0.1000
    Epoch   7 Batch  230/269 - Train Accuracy: 0.8889, Validation Accuracy: 0.8911, Loss: 0.1062
    Epoch   7 Batch  235/269 - Train Accuracy: 0.9092, Validation Accuracy: 0.8888, Loss: 0.0963
    Epoch   7 Batch  240/269 - Train Accuracy: 0.8851, Validation Accuracy: 0.8916, Loss: 0.0977
    Epoch   7 Batch  245/269 - Train Accuracy: 0.8966, Validation Accuracy: 0.8938, Loss: 0.1035
    Epoch   7 Batch  250/269 - Train Accuracy: 0.8969, Validation Accuracy: 0.8907, Loss: 0.1030
    Epoch   7 Batch  255/269 - Train Accuracy: 0.9020, Validation Accuracy: 0.8910, Loss: 0.1018
    Epoch   7 Batch  260/269 - Train Accuracy: 0.8768, Validation Accuracy: 0.8871, Loss: 0.1083
    Epoch   7 Batch  265/269 - Train Accuracy: 0.8926, Validation Accuracy: 0.8896, Loss: 0.1015
    Epoch   8 Batch    5/269 - Train Accuracy: 0.8950, Validation Accuracy: 0.8866, Loss: 0.1083
    Epoch   8 Batch   10/269 - Train Accuracy: 0.9068, Validation Accuracy: 0.8865, Loss: 0.0958
    Epoch   8 Batch   15/269 - Train Accuracy: 0.8971, Validation Accuracy: 0.8928, Loss: 0.0914
    Epoch   8 Batch   20/269 - Train Accuracy: 0.8961, Validation Accuracy: 0.8884, Loss: 0.0983
    Epoch   8 Batch   25/269 - Train Accuracy: 0.8793, Validation Accuracy: 0.8807, Loss: 0.1105
    Epoch   8 Batch   30/269 - Train Accuracy: 0.8895, Validation Accuracy: 0.8884, Loss: 0.1026
    Epoch   8 Batch   35/269 - Train Accuracy: 0.8912, Validation Accuracy: 0.8840, Loss: 0.1177
    Epoch   8 Batch   40/269 - Train Accuracy: 0.8807, Validation Accuracy: 0.8830, Loss: 0.1026
    Epoch   8 Batch   45/269 - Train Accuracy: 0.8977, Validation Accuracy: 0.8904, Loss: 0.1077
    Epoch   8 Batch   50/269 - Train Accuracy: 0.8716, Validation Accuracy: 0.8805, Loss: 0.1177
    Epoch   8 Batch   55/269 - Train Accuracy: 0.9134, Validation Accuracy: 0.8821, Loss: 0.0943
    Epoch   8 Batch   60/269 - Train Accuracy: 0.9012, Validation Accuracy: 0.8923, Loss: 0.0930
    Epoch   8 Batch   65/269 - Train Accuracy: 0.9028, Validation Accuracy: 0.8955, Loss: 0.1002
    Epoch   8 Batch   70/269 - Train Accuracy: 0.9023, Validation Accuracy: 0.8833, Loss: 0.1012
    Epoch   8 Batch   75/269 - Train Accuracy: 0.9057, Validation Accuracy: 0.8905, Loss: 0.1008
    Epoch   8 Batch   80/269 - Train Accuracy: 0.8918, Validation Accuracy: 0.8859, Loss: 0.0981
    Epoch   8 Batch   85/269 - Train Accuracy: 0.8892, Validation Accuracy: 0.8856, Loss: 0.0975
    Epoch   8 Batch   90/269 - Train Accuracy: 0.8864, Validation Accuracy: 0.8872, Loss: 0.0974
    Epoch   8 Batch   95/269 - Train Accuracy: 0.9004, Validation Accuracy: 0.8944, Loss: 0.0916
    Epoch   8 Batch  100/269 - Train Accuracy: 0.9076, Validation Accuracy: 0.8888, Loss: 0.0943
    Epoch   8 Batch  105/269 - Train Accuracy: 0.8889, Validation Accuracy: 0.8962, Loss: 0.0962
    Epoch   8 Batch  110/269 - Train Accuracy: 0.8905, Validation Accuracy: 0.8878, Loss: 0.0959
    Epoch   8 Batch  115/269 - Train Accuracy: 0.8780, Validation Accuracy: 0.8850, Loss: 0.1034
    Epoch   8 Batch  120/269 - Train Accuracy: 0.9015, Validation Accuracy: 0.8879, Loss: 0.1005
    Epoch   8 Batch  125/269 - Train Accuracy: 0.9110, Validation Accuracy: 0.8807, Loss: 0.0908
    Epoch   8 Batch  130/269 - Train Accuracy: 0.8928, Validation Accuracy: 0.8904, Loss: 0.0984
    Epoch   8 Batch  135/269 - Train Accuracy: 0.8877, Validation Accuracy: 0.8936, Loss: 0.0982
    Epoch   8 Batch  140/269 - Train Accuracy: 0.8878, Validation Accuracy: 0.8877, Loss: 0.1055
    Epoch   8 Batch  145/269 - Train Accuracy: 0.8915, Validation Accuracy: 0.8840, Loss: 0.0908
    Epoch   8 Batch  150/269 - Train Accuracy: 0.8895, Validation Accuracy: 0.8838, Loss: 0.1001
    Epoch   8 Batch  155/269 - Train Accuracy: 0.8996, Validation Accuracy: 0.8879, Loss: 0.0942
    Epoch   8 Batch  160/269 - Train Accuracy: 0.8959, Validation Accuracy: 0.8856, Loss: 0.0995
    Epoch   8 Batch  165/269 - Train Accuracy: 0.9049, Validation Accuracy: 0.8926, Loss: 0.0967
    Epoch   8 Batch  170/269 - Train Accuracy: 0.8923, Validation Accuracy: 0.8916, Loss: 0.0934
    Epoch   8 Batch  175/269 - Train Accuracy: 0.8962, Validation Accuracy: 0.8871, Loss: 0.0999
    Epoch   8 Batch  180/269 - Train Accuracy: 0.9078, Validation Accuracy: 0.8953, Loss: 0.0898
    Epoch   8 Batch  185/269 - Train Accuracy: 0.9127, Validation Accuracy: 0.8905, Loss: 0.0899
    Epoch   8 Batch  190/269 - Train Accuracy: 0.8807, Validation Accuracy: 0.8854, Loss: 0.0947
    Epoch   8 Batch  195/269 - Train Accuracy: 0.8935, Validation Accuracy: 0.9024, Loss: 0.0937
    Epoch   8 Batch  200/269 - Train Accuracy: 0.9016, Validation Accuracy: 0.8945, Loss: 0.0907
    Epoch   8 Batch  205/269 - Train Accuracy: 0.8956, Validation Accuracy: 0.8852, Loss: 0.0890
    Epoch   8 Batch  210/269 - Train Accuracy: 0.8885, Validation Accuracy: 0.8983, Loss: 0.0864
    Epoch   8 Batch  215/269 - Train Accuracy: 0.9066, Validation Accuracy: 0.8895, Loss: 0.0851
    Epoch   8 Batch  220/269 - Train Accuracy: 0.8954, Validation Accuracy: 0.8834, Loss: 0.0901
    Epoch   8 Batch  225/269 - Train Accuracy: 0.8950, Validation Accuracy: 0.8947, Loss: 0.0861
    Epoch   8 Batch  230/269 - Train Accuracy: 0.8994, Validation Accuracy: 0.8930, Loss: 0.0894
    Epoch   8 Batch  235/269 - Train Accuracy: 0.9214, Validation Accuracy: 0.9000, Loss: 0.0838
    Epoch   8 Batch  240/269 - Train Accuracy: 0.9013, Validation Accuracy: 0.8988, Loss: 0.0804
    Epoch   8 Batch  245/269 - Train Accuracy: 0.8938, Validation Accuracy: 0.9018, Loss: 0.0896
    Epoch   8 Batch  250/269 - Train Accuracy: 0.9080, Validation Accuracy: 0.8979, Loss: 0.0924
    Epoch   8 Batch  255/269 - Train Accuracy: 0.9038, Validation Accuracy: 0.9038, Loss: 0.0909
    Epoch   8 Batch  260/269 - Train Accuracy: 0.8743, Validation Accuracy: 0.8950, Loss: 0.0963
    Epoch   8 Batch  265/269 - Train Accuracy: 0.9009, Validation Accuracy: 0.9015, Loss: 0.0911
    Epoch   9 Batch    5/269 - Train Accuracy: 0.8925, Validation Accuracy: 0.8851, Loss: 0.0962
    Epoch   9 Batch   10/269 - Train Accuracy: 0.8906, Validation Accuracy: 0.8881, Loss: 0.2120
    Epoch   9 Batch   15/269 - Train Accuracy: 0.8553, Validation Accuracy: 0.8445, Loss: 0.1205
    Epoch   9 Batch   20/269 - Train Accuracy: 0.8612, Validation Accuracy: 0.8553, Loss: 0.1452
    Epoch   9 Batch   25/269 - Train Accuracy: 0.8391, Validation Accuracy: 0.8716, Loss: 0.1316
    Epoch   9 Batch   30/269 - Train Accuracy: 0.8622, Validation Accuracy: 0.8543, Loss: 0.1570
    Epoch   9 Batch   35/269 - Train Accuracy: 0.8509, Validation Accuracy: 0.8438, Loss: 0.1481
    Epoch   9 Batch   40/269 - Train Accuracy: 0.8355, Validation Accuracy: 0.8509, Loss: 0.1293
    Epoch   9 Batch   45/269 - Train Accuracy: 0.8683, Validation Accuracy: 0.8584, Loss: 0.1201
    Epoch   9 Batch   50/269 - Train Accuracy: 0.8523, Validation Accuracy: 0.8817, Loss: 0.1182
    Epoch   9 Batch   55/269 - Train Accuracy: 0.9005, Validation Accuracy: 0.8761, Loss: 0.0976
    Epoch   9 Batch   60/269 - Train Accuracy: 0.8931, Validation Accuracy: 0.8826, Loss: 0.0931
    Epoch   9 Batch   65/269 - Train Accuracy: 0.9029, Validation Accuracy: 0.8933, Loss: 0.0909
    Epoch   9 Batch   70/269 - Train Accuracy: 0.9003, Validation Accuracy: 0.8858, Loss: 0.0920
    Epoch   9 Batch   75/269 - Train Accuracy: 0.9074, Validation Accuracy: 0.8967, Loss: 0.0927
    Epoch   9 Batch   80/269 - Train Accuracy: 0.8928, Validation Accuracy: 0.8867, Loss: 0.0892
    Epoch   9 Batch   85/269 - Train Accuracy: 0.8966, Validation Accuracy: 0.8853, Loss: 0.0845
    Epoch   9 Batch   90/269 - Train Accuracy: 0.8949, Validation Accuracy: 0.8948, Loss: 0.0878
    Epoch   9 Batch   95/269 - Train Accuracy: 0.9059, Validation Accuracy: 0.8980, Loss: 0.0802
    Epoch   9 Batch  100/269 - Train Accuracy: 0.9058, Validation Accuracy: 0.9054, Loss: 0.0840
    Epoch   9 Batch  105/269 - Train Accuracy: 0.8992, Validation Accuracy: 0.8941, Loss: 0.0835
    Epoch   9 Batch  110/269 - Train Accuracy: 0.9057, Validation Accuracy: 0.9046, Loss: 0.0821
    Epoch   9 Batch  115/269 - Train Accuracy: 0.8943, Validation Accuracy: 0.8980, Loss: 0.0895
    Epoch   9 Batch  120/269 - Train Accuracy: 0.9100, Validation Accuracy: 0.8969, Loss: 0.0875
    Epoch   9 Batch  125/269 - Train Accuracy: 0.9089, Validation Accuracy: 0.8884, Loss: 0.0807
    Epoch   9 Batch  130/269 - Train Accuracy: 0.9016, Validation Accuracy: 0.9043, Loss: 0.0830
    Epoch   9 Batch  135/269 - Train Accuracy: 0.8989, Validation Accuracy: 0.9014, Loss: 0.0842
    Epoch   9 Batch  140/269 - Train Accuracy: 0.8968, Validation Accuracy: 0.8970, Loss: 0.0891
    Epoch   9 Batch  145/269 - Train Accuracy: 0.9043, Validation Accuracy: 0.8996, Loss: 0.0780
    Epoch   9 Batch  150/269 - Train Accuracy: 0.9036, Validation Accuracy: 0.9023, Loss: 0.0828
    Epoch   9 Batch  155/269 - Train Accuracy: 0.9100, Validation Accuracy: 0.9061, Loss: 0.0765
    Epoch   9 Batch  160/269 - Train Accuracy: 0.9096, Validation Accuracy: 0.9042, Loss: 0.0821
    Epoch   9 Batch  165/269 - Train Accuracy: 0.9112, Validation Accuracy: 0.9021, Loss: 0.0830
    Epoch   9 Batch  170/269 - Train Accuracy: 0.8959, Validation Accuracy: 0.9033, Loss: 0.0785
    Epoch   9 Batch  175/269 - Train Accuracy: 0.8941, Validation Accuracy: 0.9034, Loss: 0.0920
    Epoch   9 Batch  180/269 - Train Accuracy: 0.9206, Validation Accuracy: 0.9107, Loss: 0.0774
    Epoch   9 Batch  185/269 - Train Accuracy: 0.9182, Validation Accuracy: 0.9086, Loss: 0.0770
    Epoch   9 Batch  190/269 - Train Accuracy: 0.8959, Validation Accuracy: 0.9176, Loss: 0.0819
    Epoch   9 Batch  195/269 - Train Accuracy: 0.8919, Validation Accuracy: 0.9113, Loss: 0.0812
    Epoch   9 Batch  200/269 - Train Accuracy: 0.9010, Validation Accuracy: 0.9041, Loss: 0.0763
    Epoch   9 Batch  205/269 - Train Accuracy: 0.9100, Validation Accuracy: 0.9071, Loss: 0.0803
    Epoch   9 Batch  210/269 - Train Accuracy: 0.9030, Validation Accuracy: 0.9101, Loss: 0.0744
    Epoch   9 Batch  215/269 - Train Accuracy: 0.9050, Validation Accuracy: 0.9021, Loss: 0.0755
    Epoch   9 Batch  220/269 - Train Accuracy: 0.9024, Validation Accuracy: 0.9058, Loss: 0.0803
    Epoch   9 Batch  225/269 - Train Accuracy: 0.8979, Validation Accuracy: 0.9086, Loss: 0.0784
    Epoch   9 Batch  230/269 - Train Accuracy: 0.9096, Validation Accuracy: 0.9055, Loss: 0.0786
    Epoch   9 Batch  235/269 - Train Accuracy: 0.9302, Validation Accuracy: 0.9111, Loss: 0.0707
    Epoch   9 Batch  240/269 - Train Accuracy: 0.9172, Validation Accuracy: 0.9128, Loss: 0.0709
    Epoch   9 Batch  245/269 - Train Accuracy: 0.9146, Validation Accuracy: 0.9133, Loss: 0.0777
    Epoch   9 Batch  250/269 - Train Accuracy: 0.9103, Validation Accuracy: 0.9098, Loss: 0.0786
    Epoch   9 Batch  255/269 - Train Accuracy: 0.9190, Validation Accuracy: 0.9140, Loss: 0.0776
    Epoch   9 Batch  260/269 - Train Accuracy: 0.8986, Validation Accuracy: 0.9098, Loss: 0.0846
    Epoch   9 Batch  265/269 - Train Accuracy: 0.9187, Validation Accuracy: 0.9155, Loss: 0.0789
    Epoch  10 Batch    5/269 - Train Accuracy: 0.9157, Validation Accuracy: 0.8970, Loss: 0.0780
    Epoch  10 Batch   10/269 - Train Accuracy: 0.9136, Validation Accuracy: 0.9116, Loss: 0.0719
    Epoch  10 Batch   15/269 - Train Accuracy: 0.9157, Validation Accuracy: 0.9028, Loss: 0.0669
    Epoch  10 Batch   20/269 - Train Accuracy: 0.9142, Validation Accuracy: 0.9110, Loss: 0.0760
    Epoch  10 Batch   25/269 - Train Accuracy: 0.9011, Validation Accuracy: 0.9078, Loss: 0.0860
    Epoch  10 Batch   30/269 - Train Accuracy: 0.9211, Validation Accuracy: 0.9139, Loss: 0.0718
    Epoch  10 Batch   35/269 - Train Accuracy: 0.9182, Validation Accuracy: 0.9060, Loss: 0.0901
    Epoch  10 Batch   40/269 - Train Accuracy: 0.9014, Validation Accuracy: 0.9048, Loss: 0.0770
    Epoch  10 Batch   45/269 - Train Accuracy: 0.9218, Validation Accuracy: 0.9094, Loss: 0.0817
    Epoch  10 Batch   50/269 - Train Accuracy: 0.8910, Validation Accuracy: 0.9054, Loss: 0.0880
    Epoch  10 Batch   55/269 - Train Accuracy: 0.9215, Validation Accuracy: 0.9025, Loss: 0.0728
    Epoch  10 Batch   60/269 - Train Accuracy: 0.9060, Validation Accuracy: 0.9118, Loss: 0.0778
    Epoch  10 Batch   65/269 - Train Accuracy: 0.9231, Validation Accuracy: 0.9142, Loss: 0.0921
    Epoch  10 Batch   70/269 - Train Accuracy: 0.8993, Validation Accuracy: 0.8934, Loss: 0.0920
    Epoch  10 Batch   75/269 - Train Accuracy: 0.9223, Validation Accuracy: 0.9054, Loss: 0.1017
    Epoch  10 Batch   80/269 - Train Accuracy: 0.8896, Validation Accuracy: 0.8808, Loss: 0.0919
    Epoch  10 Batch   85/269 - Train Accuracy: 0.8875, Validation Accuracy: 0.8923, Loss: 0.0851
    Epoch  10 Batch   90/269 - Train Accuracy: 0.8904, Validation Accuracy: 0.8922, Loss: 0.0871
    Epoch  10 Batch   95/269 - Train Accuracy: 0.9076, Validation Accuracy: 0.9044, Loss: 0.0787
    Epoch  10 Batch  100/269 - Train Accuracy: 0.9139, Validation Accuracy: 0.8973, Loss: 0.0874
    Epoch  10 Batch  105/269 - Train Accuracy: 0.9053, Validation Accuracy: 0.9001, Loss: 0.0818
    Epoch  10 Batch  110/269 - Train Accuracy: 0.9038, Validation Accuracy: 0.9126, Loss: 0.0768
    Epoch  10 Batch  115/269 - Train Accuracy: 0.9006, Validation Accuracy: 0.9024, Loss: 0.0811
    Epoch  10 Batch  120/269 - Train Accuracy: 0.9088, Validation Accuracy: 0.9121, Loss: 0.0808
    Epoch  10 Batch  125/269 - Train Accuracy: 0.9182, Validation Accuracy: 0.9037, Loss: 0.0728
    Epoch  10 Batch  130/269 - Train Accuracy: 0.9056, Validation Accuracy: 0.9086, Loss: 0.0750
    Epoch  10 Batch  135/269 - Train Accuracy: 0.9104, Validation Accuracy: 0.9102, Loss: 0.0750
    Epoch  10 Batch  140/269 - Train Accuracy: 0.9024, Validation Accuracy: 0.9097, Loss: 0.0814
    Epoch  10 Batch  145/269 - Train Accuracy: 0.9097, Validation Accuracy: 0.9079, Loss: 0.0703
    Epoch  10 Batch  150/269 - Train Accuracy: 0.9149, Validation Accuracy: 0.9084, Loss: 0.0745
    Epoch  10 Batch  155/269 - Train Accuracy: 0.9189, Validation Accuracy: 0.9178, Loss: 0.0668
    Epoch  10 Batch  160/269 - Train Accuracy: 0.9174, Validation Accuracy: 0.9158, Loss: 0.0748
    Epoch  10 Batch  165/269 - Train Accuracy: 0.9146, Validation Accuracy: 0.8999, Loss: 0.0739
    Epoch  10 Batch  170/269 - Train Accuracy: 0.9145, Validation Accuracy: 0.9199, Loss: 0.0729
    Epoch  10 Batch  175/269 - Train Accuracy: 0.9089, Validation Accuracy: 0.9134, Loss: 0.0817
    Epoch  10 Batch  180/269 - Train Accuracy: 0.9267, Validation Accuracy: 0.9146, Loss: 0.0658
    Epoch  10 Batch  185/269 - Train Accuracy: 0.9337, Validation Accuracy: 0.9187, Loss: 0.0691
    Epoch  10 Batch  190/269 - Train Accuracy: 0.9032, Validation Accuracy: 0.9188, Loss: 0.0736
    Epoch  10 Batch  195/269 - Train Accuracy: 0.9010, Validation Accuracy: 0.9156, Loss: 0.0713
    Epoch  10 Batch  200/269 - Train Accuracy: 0.9065, Validation Accuracy: 0.9150, Loss: 0.0686
    Epoch  10 Batch  205/269 - Train Accuracy: 0.9178, Validation Accuracy: 0.9179, Loss: 0.0714
    Epoch  10 Batch  210/269 - Train Accuracy: 0.9049, Validation Accuracy: 0.9114, Loss: 0.0669
    Epoch  10 Batch  215/269 - Train Accuracy: 0.9120, Validation Accuracy: 0.9184, Loss: 0.0698
    Epoch  10 Batch  220/269 - Train Accuracy: 0.9237, Validation Accuracy: 0.9167, Loss: 0.0746
    Epoch  10 Batch  225/269 - Train Accuracy: 0.9047, Validation Accuracy: 0.9123, Loss: 0.0697
    Epoch  10 Batch  230/269 - Train Accuracy: 0.9174, Validation Accuracy: 0.9108, Loss: 0.0710
    Epoch  10 Batch  235/269 - Train Accuracy: 0.9417, Validation Accuracy: 0.9110, Loss: 0.0644
    Epoch  10 Batch  240/269 - Train Accuracy: 0.9169, Validation Accuracy: 0.9197, Loss: 0.0666
    Epoch  10 Batch  245/269 - Train Accuracy: 0.9070, Validation Accuracy: 0.9155, Loss: 0.0719
    Epoch  10 Batch  250/269 - Train Accuracy: 0.9042, Validation Accuracy: 0.8992, Loss: 0.0706
    Epoch  10 Batch  255/269 - Train Accuracy: 0.9239, Validation Accuracy: 0.9156, Loss: 0.0702
    Epoch  10 Batch  260/269 - Train Accuracy: 0.8966, Validation Accuracy: 0.9089, Loss: 0.0754
    Epoch  10 Batch  265/269 - Train Accuracy: 0.9244, Validation Accuracy: 0.9138, Loss: 0.0695
    Epoch  11 Batch    5/269 - Train Accuracy: 0.9136, Validation Accuracy: 0.9070, Loss: 0.0709
    Epoch  11 Batch   10/269 - Train Accuracy: 0.9250, Validation Accuracy: 0.9160, Loss: 0.0625
    Epoch  11 Batch   15/269 - Train Accuracy: 0.9249, Validation Accuracy: 0.9114, Loss: 0.0589
    Epoch  11 Batch   20/269 - Train Accuracy: 0.9150, Validation Accuracy: 0.9070, Loss: 0.0701
    Epoch  11 Batch   25/269 - Train Accuracy: 0.9108, Validation Accuracy: 0.9108, Loss: 0.0778
    Epoch  11 Batch   30/269 - Train Accuracy: 0.9253, Validation Accuracy: 0.9206, Loss: 0.0659
    Epoch  11 Batch   35/269 - Train Accuracy: 0.9193, Validation Accuracy: 0.9078, Loss: 0.0828
    Epoch  11 Batch   40/269 - Train Accuracy: 0.9082, Validation Accuracy: 0.9115, Loss: 0.0710
    Epoch  11 Batch   45/269 - Train Accuracy: 0.9203, Validation Accuracy: 0.9150, Loss: 0.0737
    Epoch  11 Batch   50/269 - Train Accuracy: 0.8986, Validation Accuracy: 0.9056, Loss: 0.0777
    Epoch  11 Batch   55/269 - Train Accuracy: 0.9250, Validation Accuracy: 0.9118, Loss: 0.0646
    Epoch  11 Batch   60/269 - Train Accuracy: 0.9173, Validation Accuracy: 0.9202, Loss: 0.0683
    Epoch  11 Batch   65/269 - Train Accuracy: 0.9374, Validation Accuracy: 0.9128, Loss: 0.0673
    Epoch  11 Batch   70/269 - Train Accuracy: 0.9164, Validation Accuracy: 0.9067, Loss: 0.0720
    Epoch  11 Batch   75/269 - Train Accuracy: 0.9213, Validation Accuracy: 0.9169, Loss: 0.0713
    Epoch  11 Batch   80/269 - Train Accuracy: 0.9107, Validation Accuracy: 0.9109, Loss: 0.0699
    Epoch  11 Batch   85/269 - Train Accuracy: 0.9140, Validation Accuracy: 0.9173, Loss: 0.0682
    Epoch  11 Batch   90/269 - Train Accuracy: 0.9225, Validation Accuracy: 0.9142, Loss: 0.0697
    Epoch  11 Batch   95/269 - Train Accuracy: 0.9276, Validation Accuracy: 0.9134, Loss: 0.0634
    Epoch  11 Batch  100/269 - Train Accuracy: 0.9175, Validation Accuracy: 0.9137, Loss: 0.0675
    Epoch  11 Batch  105/269 - Train Accuracy: 0.9080, Validation Accuracy: 0.9074, Loss: 0.0670
    Epoch  11 Batch  110/269 - Train Accuracy: 0.9042, Validation Accuracy: 0.9155, Loss: 0.0702
    Epoch  11 Batch  115/269 - Train Accuracy: 0.9129, Validation Accuracy: 0.9094, Loss: 0.0725
    Epoch  11 Batch  120/269 - Train Accuracy: 0.9104, Validation Accuracy: 0.9200, Loss: 0.0718
    Epoch  11 Batch  125/269 - Train Accuracy: 0.9129, Validation Accuracy: 0.9190, Loss: 0.0688
    Epoch  11 Batch  130/269 - Train Accuracy: 0.9107, Validation Accuracy: 0.9166, Loss: 0.0680
    Epoch  11 Batch  135/269 - Train Accuracy: 0.9121, Validation Accuracy: 0.9199, Loss: 0.0690
    Epoch  11 Batch  140/269 - Train Accuracy: 0.9088, Validation Accuracy: 0.9174, Loss: 0.0745
    Epoch  11 Batch  145/269 - Train Accuracy: 0.9072, Validation Accuracy: 0.9160, Loss: 0.0642
    Epoch  11 Batch  150/269 - Train Accuracy: 0.9160, Validation Accuracy: 0.9148, Loss: 0.0680
    Epoch  11 Batch  155/269 - Train Accuracy: 0.9268, Validation Accuracy: 0.9270, Loss: 0.0630
    Epoch  11 Batch  160/269 - Train Accuracy: 0.9182, Validation Accuracy: 0.9126, Loss: 0.0658
    Epoch  11 Batch  165/269 - Train Accuracy: 0.9152, Validation Accuracy: 0.9104, Loss: 0.0676
    Epoch  11 Batch  170/269 - Train Accuracy: 0.9128, Validation Accuracy: 0.9183, Loss: 0.0639
    Epoch  11 Batch  175/269 - Train Accuracy: 0.9142, Validation Accuracy: 0.9198, Loss: 0.0783
    Epoch  11 Batch  180/269 - Train Accuracy: 0.9237, Validation Accuracy: 0.9181, Loss: 0.0620
    Epoch  11 Batch  185/269 - Train Accuracy: 0.9354, Validation Accuracy: 0.9191, Loss: 0.0623
    Epoch  11 Batch  190/269 - Train Accuracy: 0.9060, Validation Accuracy: 0.9206, Loss: 0.0656
    Epoch  11 Batch  195/269 - Train Accuracy: 0.9057, Validation Accuracy: 0.9110, Loss: 0.0654
    Epoch  11 Batch  200/269 - Train Accuracy: 0.9209, Validation Accuracy: 0.9189, Loss: 0.0611
    Epoch  11 Batch  205/269 - Train Accuracy: 0.9180, Validation Accuracy: 0.9208, Loss: 0.0652
    Epoch  11 Batch  210/269 - Train Accuracy: 0.9141, Validation Accuracy: 0.9165, Loss: 0.0602
    Epoch  11 Batch  215/269 - Train Accuracy: 0.9190, Validation Accuracy: 0.9164, Loss: 0.0627
    Epoch  11 Batch  220/269 - Train Accuracy: 0.9266, Validation Accuracy: 0.9197, Loss: 0.0678
    Epoch  11 Batch  225/269 - Train Accuracy: 0.9104, Validation Accuracy: 0.9205, Loss: 0.0635
    Epoch  11 Batch  230/269 - Train Accuracy: 0.9178, Validation Accuracy: 0.9182, Loss: 0.0644
    Epoch  11 Batch  235/269 - Train Accuracy: 0.9436, Validation Accuracy: 0.9213, Loss: 0.0583
    Epoch  11 Batch  240/269 - Train Accuracy: 0.9220, Validation Accuracy: 0.9157, Loss: 0.0588
    Epoch  11 Batch  245/269 - Train Accuracy: 0.9201, Validation Accuracy: 0.9183, Loss: 0.0639
    Epoch  11 Batch  250/269 - Train Accuracy: 0.9131, Validation Accuracy: 0.9118, Loss: 0.0642
    Epoch  11 Batch  255/269 - Train Accuracy: 0.9182, Validation Accuracy: 0.9190, Loss: 0.0630
    Epoch  11 Batch  260/269 - Train Accuracy: 0.9036, Validation Accuracy: 0.9174, Loss: 0.0674
    Epoch  11 Batch  265/269 - Train Accuracy: 0.9238, Validation Accuracy: 0.9181, Loss: 0.0642
    Epoch  12 Batch    5/269 - Train Accuracy: 0.9179, Validation Accuracy: 0.8975, Loss: 0.0662
    Epoch  12 Batch   10/269 - Train Accuracy: 0.9331, Validation Accuracy: 0.9184, Loss: 0.0609
    Epoch  12 Batch   15/269 - Train Accuracy: 0.8864, Validation Accuracy: 0.8803, Loss: 0.1170
    Epoch  12 Batch   20/269 - Train Accuracy: 0.9117, Validation Accuracy: 0.8936, Loss: 0.1258
    Epoch  12 Batch   25/269 - Train Accuracy: 0.8804, Validation Accuracy: 0.8846, Loss: 0.1030
    Epoch  12 Batch   30/269 - Train Accuracy: 0.8948, Validation Accuracy: 0.8896, Loss: 0.0964
    Epoch  12 Batch   35/269 - Train Accuracy: 0.8978, Validation Accuracy: 0.8892, Loss: 0.0941
    Epoch  12 Batch   40/269 - Train Accuracy: 0.8945, Validation Accuracy: 0.9056, Loss: 0.0931
    Epoch  12 Batch   45/269 - Train Accuracy: 0.9045, Validation Accuracy: 0.8932, Loss: 0.0828
    Epoch  12 Batch   50/269 - Train Accuracy: 0.9043, Validation Accuracy: 0.9128, Loss: 0.0865
    Epoch  12 Batch   55/269 - Train Accuracy: 0.9302, Validation Accuracy: 0.9119, Loss: 0.0685
    Epoch  12 Batch   60/269 - Train Accuracy: 0.9114, Validation Accuracy: 0.9191, Loss: 0.0682
    Epoch  12 Batch   65/269 - Train Accuracy: 0.9257, Validation Accuracy: 0.9125, Loss: 0.0666
    Epoch  12 Batch   70/269 - Train Accuracy: 0.9175, Validation Accuracy: 0.9029, Loss: 0.0723
    Epoch  12 Batch   75/269 - Train Accuracy: 0.9254, Validation Accuracy: 0.9142, Loss: 0.0685
    Epoch  12 Batch   80/269 - Train Accuracy: 0.9068, Validation Accuracy: 0.9069, Loss: 0.0654
    Epoch  12 Batch   85/269 - Train Accuracy: 0.9124, Validation Accuracy: 0.9202, Loss: 0.0631
    Epoch  12 Batch   90/269 - Train Accuracy: 0.9225, Validation Accuracy: 0.9126, Loss: 0.0629
    Epoch  12 Batch   95/269 - Train Accuracy: 0.9319, Validation Accuracy: 0.9162, Loss: 0.0592
    Epoch  12 Batch  100/269 - Train Accuracy: 0.9205, Validation Accuracy: 0.9162, Loss: 0.0645
    Epoch  12 Batch  105/269 - Train Accuracy: 0.9166, Validation Accuracy: 0.9127, Loss: 0.0597
    Epoch  12 Batch  110/269 - Train Accuracy: 0.9123, Validation Accuracy: 0.9207, Loss: 0.0603
    Epoch  12 Batch  115/269 - Train Accuracy: 0.9196, Validation Accuracy: 0.9209, Loss: 0.0670
    Epoch  12 Batch  120/269 - Train Accuracy: 0.9298, Validation Accuracy: 0.9205, Loss: 0.0651
    Epoch  12 Batch  125/269 - Train Accuracy: 0.9304, Validation Accuracy: 0.9192, Loss: 0.0609
    Epoch  12 Batch  130/269 - Train Accuracy: 0.9154, Validation Accuracy: 0.9175, Loss: 0.0618
    Epoch  12 Batch  135/269 - Train Accuracy: 0.9117, Validation Accuracy: 0.9165, Loss: 0.0620
    Epoch  12 Batch  140/269 - Train Accuracy: 0.9142, Validation Accuracy: 0.9149, Loss: 0.0700
    Epoch  12 Batch  145/269 - Train Accuracy: 0.9204, Validation Accuracy: 0.9143, Loss: 0.0578
    Epoch  12 Batch  150/269 - Train Accuracy: 0.9229, Validation Accuracy: 0.9252, Loss: 0.0628
    Epoch  12 Batch  155/269 - Train Accuracy: 0.9209, Validation Accuracy: 0.9193, Loss: 0.0575
    Epoch  12 Batch  160/269 - Train Accuracy: 0.9209, Validation Accuracy: 0.9114, Loss: 0.0628
    Epoch  12 Batch  165/269 - Train Accuracy: 0.9194, Validation Accuracy: 0.9254, Loss: 0.0611
    Epoch  12 Batch  170/269 - Train Accuracy: 0.9214, Validation Accuracy: 0.9255, Loss: 0.0565
    Epoch  12 Batch  175/269 - Train Accuracy: 0.9152, Validation Accuracy: 0.9209, Loss: 0.0697
    Epoch  12 Batch  180/269 - Train Accuracy: 0.9383, Validation Accuracy: 0.9259, Loss: 0.0566
    Epoch  12 Batch  185/269 - Train Accuracy: 0.9332, Validation Accuracy: 0.9279, Loss: 0.0574
    Epoch  12 Batch  190/269 - Train Accuracy: 0.9161, Validation Accuracy: 0.9234, Loss: 0.0610
    Epoch  12 Batch  195/269 - Train Accuracy: 0.9228, Validation Accuracy: 0.9237, Loss: 0.0592
    Epoch  12 Batch  200/269 - Train Accuracy: 0.9299, Validation Accuracy: 0.9231, Loss: 0.0586
    Epoch  12 Batch  205/269 - Train Accuracy: 0.9257, Validation Accuracy: 0.9156, Loss: 0.0629
    Epoch  12 Batch  210/269 - Train Accuracy: 0.9215, Validation Accuracy: 0.9165, Loss: 0.0586
    Epoch  12 Batch  215/269 - Train Accuracy: 0.9134, Validation Accuracy: 0.9268, Loss: 0.0603
    Epoch  12 Batch  220/269 - Train Accuracy: 0.9307, Validation Accuracy: 0.9262, Loss: 0.0613
    Epoch  12 Batch  225/269 - Train Accuracy: 0.9062, Validation Accuracy: 0.9248, Loss: 0.0615
    Epoch  12 Batch  230/269 - Train Accuracy: 0.9253, Validation Accuracy: 0.9210, Loss: 0.0595
    Epoch  12 Batch  235/269 - Train Accuracy: 0.9438, Validation Accuracy: 0.9197, Loss: 0.0560
    Epoch  12 Batch  240/269 - Train Accuracy: 0.9259, Validation Accuracy: 0.9165, Loss: 0.0568
    Epoch  12 Batch  245/269 - Train Accuracy: 0.9229, Validation Accuracy: 0.9211, Loss: 0.0594
    Epoch  12 Batch  250/269 - Train Accuracy: 0.9113, Validation Accuracy: 0.9099, Loss: 0.0615
    Epoch  12 Batch  255/269 - Train Accuracy: 0.9260, Validation Accuracy: 0.9168, Loss: 0.0615
    Epoch  12 Batch  260/269 - Train Accuracy: 0.9128, Validation Accuracy: 0.9195, Loss: 0.0644
    Epoch  12 Batch  265/269 - Train Accuracy: 0.9311, Validation Accuracy: 0.9267, Loss: 0.0598
    Epoch  13 Batch    5/269 - Train Accuracy: 0.9247, Validation Accuracy: 0.9094, Loss: 0.0601
    Epoch  13 Batch   10/269 - Train Accuracy: 0.9355, Validation Accuracy: 0.9140, Loss: 0.0543
    Epoch  13 Batch   15/269 - Train Accuracy: 0.9331, Validation Accuracy: 0.9158, Loss: 0.0531
    Epoch  13 Batch   20/269 - Train Accuracy: 0.9274, Validation Accuracy: 0.9063, Loss: 0.0649
    Epoch  13 Batch   25/269 - Train Accuracy: 0.9213, Validation Accuracy: 0.9104, Loss: 0.0675
    Epoch  13 Batch   30/269 - Train Accuracy: 0.9294, Validation Accuracy: 0.9203, Loss: 0.0598
    Epoch  13 Batch   35/269 - Train Accuracy: 0.9220, Validation Accuracy: 0.9031, Loss: 0.0726
    Epoch  13 Batch   40/269 - Train Accuracy: 0.9151, Validation Accuracy: 0.9201, Loss: 0.0653
    Epoch  13 Batch   45/269 - Train Accuracy: 0.9321, Validation Accuracy: 0.9193, Loss: 0.0696
    Epoch  13 Batch   50/269 - Train Accuracy: 0.9100, Validation Accuracy: 0.9138, Loss: 0.0690
    Epoch  13 Batch   55/269 - Train Accuracy: 0.9372, Validation Accuracy: 0.9211, Loss: 0.0539
    Epoch  13 Batch   60/269 - Train Accuracy: 0.9208, Validation Accuracy: 0.9209, Loss: 0.0580
    Epoch  13 Batch   65/269 - Train Accuracy: 0.9403, Validation Accuracy: 0.9244, Loss: 0.0560
    Epoch  13 Batch   70/269 - Train Accuracy: 0.9188, Validation Accuracy: 0.9205, Loss: 0.0614
    Epoch  13 Batch   75/269 - Train Accuracy: 0.9259, Validation Accuracy: 0.9213, Loss: 0.0642
    Epoch  13 Batch   80/269 - Train Accuracy: 0.9086, Validation Accuracy: 0.9076, Loss: 0.0617
    Epoch  13 Batch   85/269 - Train Accuracy: 0.9222, Validation Accuracy: 0.9186, Loss: 0.0566
    Epoch  13 Batch   90/269 - Train Accuracy: 0.9198, Validation Accuracy: 0.9178, Loss: 0.0594
    Epoch  13 Batch   95/269 - Train Accuracy: 0.9367, Validation Accuracy: 0.9203, Loss: 0.0537
    Epoch  13 Batch  100/269 - Train Accuracy: 0.9275, Validation Accuracy: 0.9201, Loss: 0.0593
    Epoch  13 Batch  105/269 - Train Accuracy: 0.9178, Validation Accuracy: 0.9140, Loss: 0.0553
    Epoch  13 Batch  110/269 - Train Accuracy: 0.9128, Validation Accuracy: 0.9128, Loss: 0.0574
    Epoch  13 Batch  115/269 - Train Accuracy: 0.9169, Validation Accuracy: 0.9149, Loss: 0.0624
    Epoch  13 Batch  120/269 - Train Accuracy: 0.9287, Validation Accuracy: 0.9181, Loss: 0.0605
    Epoch  13 Batch  125/269 - Train Accuracy: 0.9296, Validation Accuracy: 0.9264, Loss: 0.0562
    Epoch  13 Batch  130/269 - Train Accuracy: 0.9153, Validation Accuracy: 0.9175, Loss: 0.0571
    Epoch  13 Batch  135/269 - Train Accuracy: 0.9179, Validation Accuracy: 0.9227, Loss: 0.0579
    Epoch  13 Batch  140/269 - Train Accuracy: 0.9182, Validation Accuracy: 0.9175, Loss: 0.0652
    Epoch  13 Batch  145/269 - Train Accuracy: 0.9354, Validation Accuracy: 0.9206, Loss: 0.0533
    Epoch  13 Batch  150/269 - Train Accuracy: 0.9210, Validation Accuracy: 0.9203, Loss: 0.0578
    Epoch  13 Batch  155/269 - Train Accuracy: 0.9349, Validation Accuracy: 0.9264, Loss: 0.0535
    Epoch  13 Batch  160/269 - Train Accuracy: 0.9235, Validation Accuracy: 0.9161, Loss: 0.0582
    Epoch  13 Batch  165/269 - Train Accuracy: 0.9176, Validation Accuracy: 0.9227, Loss: 0.0568
    Epoch  13 Batch  170/269 - Train Accuracy: 0.9169, Validation Accuracy: 0.9221, Loss: 0.0524
    Epoch  13 Batch  175/269 - Train Accuracy: 0.9114, Validation Accuracy: 0.9284, Loss: 0.0695
    Epoch  13 Batch  180/269 - Train Accuracy: 0.9358, Validation Accuracy: 0.9242, Loss: 0.0522
    Epoch  13 Batch  185/269 - Train Accuracy: 0.9332, Validation Accuracy: 0.9216, Loss: 0.0547
    Epoch  13 Batch  190/269 - Train Accuracy: 0.9269, Validation Accuracy: 0.9222, Loss: 0.0580
    Epoch  13 Batch  195/269 - Train Accuracy: 0.9235, Validation Accuracy: 0.9181, Loss: 0.0576
    Epoch  13 Batch  200/269 - Train Accuracy: 0.9367, Validation Accuracy: 0.9276, Loss: 0.0536
    Epoch  13 Batch  205/269 - Train Accuracy: 0.9295, Validation Accuracy: 0.9184, Loss: 0.0585
    Epoch  13 Batch  210/269 - Train Accuracy: 0.9266, Validation Accuracy: 0.9161, Loss: 0.0541
    Epoch  13 Batch  215/269 - Train Accuracy: 0.9126, Validation Accuracy: 0.9256, Loss: 0.0574
    Epoch  13 Batch  220/269 - Train Accuracy: 0.9299, Validation Accuracy: 0.9210, Loss: 0.0614
    Epoch  13 Batch  225/269 - Train Accuracy: 0.9104, Validation Accuracy: 0.9295, Loss: 0.0596
    Epoch  13 Batch  230/269 - Train Accuracy: 0.9210, Validation Accuracy: 0.9198, Loss: 0.0581
    Epoch  13 Batch  235/269 - Train Accuracy: 0.9448, Validation Accuracy: 0.9150, Loss: 0.0522
    Epoch  13 Batch  240/269 - Train Accuracy: 0.9212, Validation Accuracy: 0.9201, Loss: 0.0623
    Epoch  13 Batch  245/269 - Train Accuracy: 0.9182, Validation Accuracy: 0.9047, Loss: 0.0680
    Epoch  13 Batch  250/269 - Train Accuracy: 0.9083, Validation Accuracy: 0.9093, Loss: 0.1088
    Epoch  13 Batch  255/269 - Train Accuracy: 0.8875, Validation Accuracy: 0.8876, Loss: 0.0845
    Epoch  13 Batch  260/269 - Train Accuracy: 0.8743, Validation Accuracy: 0.8747, Loss: 0.0808
    Epoch  13 Batch  265/269 - Train Accuracy: 0.9157, Validation Accuracy: 0.9065, Loss: 0.0939
    Epoch  14 Batch    5/269 - Train Accuracy: 0.9173, Validation Accuracy: 0.9015, Loss: 0.0839
    Epoch  14 Batch   10/269 - Train Accuracy: 0.9219, Validation Accuracy: 0.9089, Loss: 0.0725
    Epoch  14 Batch   15/269 - Train Accuracy: 0.9201, Validation Accuracy: 0.9150, Loss: 0.0621
    Epoch  14 Batch   20/269 - Train Accuracy: 0.9245, Validation Accuracy: 0.9121, Loss: 0.0619
    Epoch  14 Batch   25/269 - Train Accuracy: 0.9173, Validation Accuracy: 0.9071, Loss: 0.0675
    Epoch  14 Batch   30/269 - Train Accuracy: 0.9223, Validation Accuracy: 0.9197, Loss: 0.0562
    Epoch  14 Batch   35/269 - Train Accuracy: 0.9211, Validation Accuracy: 0.9193, Loss: 0.0716
    Epoch  14 Batch   40/269 - Train Accuracy: 0.9153, Validation Accuracy: 0.9145, Loss: 0.0622
    Epoch  14 Batch   45/269 - Train Accuracy: 0.9268, Validation Accuracy: 0.9228, Loss: 0.0634
    Epoch  14 Batch   50/269 - Train Accuracy: 0.9178, Validation Accuracy: 0.9073, Loss: 0.0640
    Epoch  14 Batch   55/269 - Train Accuracy: 0.9349, Validation Accuracy: 0.9111, Loss: 0.0532
    Epoch  14 Batch   60/269 - Train Accuracy: 0.9267, Validation Accuracy: 0.9222, Loss: 0.0556
    Epoch  14 Batch   65/269 - Train Accuracy: 0.9343, Validation Accuracy: 0.9224, Loss: 0.0537
    Epoch  14 Batch   70/269 - Train Accuracy: 0.9251, Validation Accuracy: 0.9160, Loss: 0.0570
    Epoch  14 Batch   75/269 - Train Accuracy: 0.9295, Validation Accuracy: 0.9233, Loss: 0.0588
    Epoch  14 Batch   80/269 - Train Accuracy: 0.9137, Validation Accuracy: 0.9112, Loss: 0.0564
    Epoch  14 Batch   85/269 - Train Accuracy: 0.9152, Validation Accuracy: 0.9219, Loss: 0.0538
    Epoch  14 Batch   90/269 - Train Accuracy: 0.9241, Validation Accuracy: 0.9252, Loss: 0.0543
    Epoch  14 Batch   95/269 - Train Accuracy: 0.9304, Validation Accuracy: 0.9265, Loss: 0.0522
    Epoch  14 Batch  100/269 - Train Accuracy: 0.9275, Validation Accuracy: 0.9204, Loss: 0.0531
    Epoch  14 Batch  105/269 - Train Accuracy: 0.9302, Validation Accuracy: 0.9206, Loss: 0.0530
    Epoch  14 Batch  110/269 - Train Accuracy: 0.9145, Validation Accuracy: 0.9276, Loss: 0.0549
    Epoch  14 Batch  115/269 - Train Accuracy: 0.9260, Validation Accuracy: 0.9255, Loss: 0.0567
    Epoch  14 Batch  120/269 - Train Accuracy: 0.9287, Validation Accuracy: 0.9274, Loss: 0.0610
    Epoch  14 Batch  125/269 - Train Accuracy: 0.9319, Validation Accuracy: 0.9250, Loss: 0.0526
    Epoch  14 Batch  130/269 - Train Accuracy: 0.9246, Validation Accuracy: 0.9207, Loss: 0.0538
    Epoch  14 Batch  135/269 - Train Accuracy: 0.9199, Validation Accuracy: 0.9243, Loss: 0.0527
    Epoch  14 Batch  140/269 - Train Accuracy: 0.9259, Validation Accuracy: 0.9153, Loss: 0.0613
    Epoch  14 Batch  145/269 - Train Accuracy: 0.9368, Validation Accuracy: 0.9258, Loss: 0.0499
    Epoch  14 Batch  150/269 - Train Accuracy: 0.9246, Validation Accuracy: 0.9237, Loss: 0.0550
    Epoch  14 Batch  155/269 - Train Accuracy: 0.9346, Validation Accuracy: 0.9262, Loss: 0.0509
    Epoch  14 Batch  160/269 - Train Accuracy: 0.9252, Validation Accuracy: 0.9119, Loss: 0.0553
    Epoch  14 Batch  165/269 - Train Accuracy: 0.9241, Validation Accuracy: 0.9324, Loss: 0.0542
    Epoch  14 Batch  170/269 - Train Accuracy: 0.9209, Validation Accuracy: 0.9287, Loss: 0.0524
    Epoch  14 Batch  175/269 - Train Accuracy: 0.9195, Validation Accuracy: 0.9272, Loss: 0.0635
    Epoch  14 Batch  180/269 - Train Accuracy: 0.9349, Validation Accuracy: 0.9270, Loss: 0.0493
    Epoch  14 Batch  185/269 - Train Accuracy: 0.9347, Validation Accuracy: 0.9263, Loss: 0.0521
    Epoch  14 Batch  190/269 - Train Accuracy: 0.9297, Validation Accuracy: 0.9272, Loss: 0.0568
    Epoch  14 Batch  195/269 - Train Accuracy: 0.9248, Validation Accuracy: 0.9331, Loss: 0.0505
    Epoch  14 Batch  200/269 - Train Accuracy: 0.9469, Validation Accuracy: 0.9268, Loss: 0.0499
    Epoch  14 Batch  205/269 - Train Accuracy: 0.9335, Validation Accuracy: 0.9260, Loss: 0.0550
    Epoch  14 Batch  210/269 - Train Accuracy: 0.9319, Validation Accuracy: 0.9260, Loss: 0.0489
    Epoch  14 Batch  215/269 - Train Accuracy: 0.9160, Validation Accuracy: 0.9316, Loss: 0.0526
    Epoch  14 Batch  220/269 - Train Accuracy: 0.9321, Validation Accuracy: 0.9316, Loss: 0.0557
    Epoch  14 Batch  225/269 - Train Accuracy: 0.9183, Validation Accuracy: 0.9249, Loss: 0.0519
    Epoch  14 Batch  230/269 - Train Accuracy: 0.9276, Validation Accuracy: 0.9312, Loss: 0.0519
    Epoch  14 Batch  235/269 - Train Accuracy: 0.9414, Validation Accuracy: 0.9213, Loss: 0.0465
    Epoch  14 Batch  240/269 - Train Accuracy: 0.9220, Validation Accuracy: 0.9227, Loss: 0.0489
    Epoch  14 Batch  245/269 - Train Accuracy: 0.9341, Validation Accuracy: 0.9268, Loss: 0.0491
    Epoch  14 Batch  250/269 - Train Accuracy: 0.9257, Validation Accuracy: 0.9311, Loss: 0.0511
    Epoch  14 Batch  255/269 - Train Accuracy: 0.9243, Validation Accuracy: 0.9257, Loss: 0.0537
    Epoch  14 Batch  260/269 - Train Accuracy: 0.9141, Validation Accuracy: 0.9283, Loss: 0.0563
    Epoch  14 Batch  265/269 - Train Accuracy: 0.9326, Validation Accuracy: 0.9330, Loss: 0.0533
    Epoch  15 Batch    5/269 - Train Accuracy: 0.9162, Validation Accuracy: 0.9189, Loss: 0.0539
    Epoch  15 Batch   10/269 - Train Accuracy: 0.9336, Validation Accuracy: 0.9232, Loss: 0.0495
    Epoch  15 Batch   15/269 - Train Accuracy: 0.9392, Validation Accuracy: 0.9243, Loss: 0.0431
    Epoch  15 Batch   20/269 - Train Accuracy: 0.9319, Validation Accuracy: 0.9188, Loss: 0.0503
    Epoch  15 Batch   25/269 - Train Accuracy: 0.9235, Validation Accuracy: 0.9223, Loss: 0.0591
    Epoch  15 Batch   30/269 - Train Accuracy: 0.9336, Validation Accuracy: 0.9279, Loss: 0.0519
    Epoch  15 Batch   35/269 - Train Accuracy: 0.9278, Validation Accuracy: 0.9208, Loss: 0.0647
    Epoch  15 Batch   40/269 - Train Accuracy: 0.9259, Validation Accuracy: 0.9239, Loss: 0.0547
    Epoch  15 Batch   45/269 - Train Accuracy: 0.9364, Validation Accuracy: 0.9253, Loss: 0.0577
    Epoch  15 Batch   50/269 - Train Accuracy: 0.9250, Validation Accuracy: 0.9254, Loss: 0.0596
    Epoch  15 Batch   55/269 - Train Accuracy: 0.9468, Validation Accuracy: 0.9244, Loss: 0.0472
    Epoch  15 Batch   60/269 - Train Accuracy: 0.9385, Validation Accuracy: 0.9307, Loss: 0.0499
    Epoch  15 Batch   65/269 - Train Accuracy: 0.9449, Validation Accuracy: 0.9222, Loss: 0.0487
    Epoch  15 Batch   70/269 - Train Accuracy: 0.9372, Validation Accuracy: 0.9264, Loss: 0.0534
    Epoch  15 Batch   75/269 - Train Accuracy: 0.9406, Validation Accuracy: 0.9270, Loss: 0.0545
    Epoch  15 Batch   80/269 - Train Accuracy: 0.9217, Validation Accuracy: 0.9151, Loss: 0.0538
    Epoch  15 Batch   85/269 - Train Accuracy: 0.9209, Validation Accuracy: 0.9254, Loss: 0.0520
    Epoch  15 Batch   90/269 - Train Accuracy: 0.9345, Validation Accuracy: 0.9234, Loss: 0.0529
    Epoch  15 Batch   95/269 - Train Accuracy: 0.9429, Validation Accuracy: 0.9283, Loss: 0.0483
    Epoch  15 Batch  100/269 - Train Accuracy: 0.9334, Validation Accuracy: 0.9300, Loss: 0.0538
    Epoch  15 Batch  105/269 - Train Accuracy: 0.9382, Validation Accuracy: 0.9222, Loss: 0.0477
    Epoch  15 Batch  110/269 - Train Accuracy: 0.9229, Validation Accuracy: 0.9320, Loss: 0.0509
    Epoch  15 Batch  115/269 - Train Accuracy: 0.9213, Validation Accuracy: 0.9274, Loss: 0.0548
    Epoch  15 Batch  120/269 - Train Accuracy: 0.9274, Validation Accuracy: 0.9252, Loss: 0.0563
    Epoch  15 Batch  125/269 - Train Accuracy: 0.9343, Validation Accuracy: 0.9333, Loss: 0.0535
    Epoch  15 Batch  130/269 - Train Accuracy: 0.9348, Validation Accuracy: 0.9304, Loss: 0.0537
    Epoch  15 Batch  135/269 - Train Accuracy: 0.9284, Validation Accuracy: 0.9227, Loss: 0.0529
    Epoch  15 Batch  140/269 - Train Accuracy: 0.9195, Validation Accuracy: 0.9241, Loss: 0.0594
    Epoch  15 Batch  145/269 - Train Accuracy: 0.9322, Validation Accuracy: 0.9265, Loss: 0.0468
    Epoch  15 Batch  150/269 - Train Accuracy: 0.9376, Validation Accuracy: 0.9229, Loss: 0.0522
    Epoch  15 Batch  155/269 - Train Accuracy: 0.9297, Validation Accuracy: 0.9243, Loss: 0.0456
    Epoch  15 Batch  160/269 - Train Accuracy: 0.9251, Validation Accuracy: 0.9234, Loss: 0.0511
    Epoch  15 Batch  165/269 - Train Accuracy: 0.9229, Validation Accuracy: 0.9328, Loss: 0.0495
    Epoch  15 Batch  170/269 - Train Accuracy: 0.9253, Validation Accuracy: 0.9300, Loss: 0.0500
    Epoch  15 Batch  175/269 - Train Accuracy: 0.9276, Validation Accuracy: 0.9325, Loss: 0.0648
    Epoch  15 Batch  180/269 - Train Accuracy: 0.9475, Validation Accuracy: 0.9286, Loss: 0.0433
    Epoch  15 Batch  185/269 - Train Accuracy: 0.9450, Validation Accuracy: 0.9312, Loss: 0.0468
    Epoch  15 Batch  190/269 - Train Accuracy: 0.9312, Validation Accuracy: 0.9319, Loss: 0.0522
    Epoch  15 Batch  195/269 - Train Accuracy: 0.9337, Validation Accuracy: 0.9267, Loss: 0.0476
    Epoch  15 Batch  200/269 - Train Accuracy: 0.9406, Validation Accuracy: 0.9330, Loss: 0.0480
    Epoch  15 Batch  205/269 - Train Accuracy: 0.9434, Validation Accuracy: 0.9380, Loss: 0.0511
    Epoch  15 Batch  210/269 - Train Accuracy: 0.9364, Validation Accuracy: 0.9313, Loss: 0.0454
    Epoch  15 Batch  215/269 - Train Accuracy: 0.9305, Validation Accuracy: 0.9233, Loss: 0.0473
    Epoch  15 Batch  220/269 - Train Accuracy: 0.9371, Validation Accuracy: 0.9309, Loss: 0.0518
    Epoch  15 Batch  225/269 - Train Accuracy: 0.9207, Validation Accuracy: 0.9370, Loss: 0.0501
    Epoch  15 Batch  230/269 - Train Accuracy: 0.9352, Validation Accuracy: 0.9325, Loss: 0.0494
    Epoch  15 Batch  235/269 - Train Accuracy: 0.9600, Validation Accuracy: 0.9327, Loss: 0.0423
    Epoch  15 Batch  240/269 - Train Accuracy: 0.9283, Validation Accuracy: 0.9303, Loss: 0.0440
    Epoch  15 Batch  245/269 - Train Accuracy: 0.9369, Validation Accuracy: 0.9288, Loss: 0.0461
    Epoch  15 Batch  250/269 - Train Accuracy: 0.9259, Validation Accuracy: 0.9363, Loss: 0.0479
    Epoch  15 Batch  255/269 - Train Accuracy: 0.9334, Validation Accuracy: 0.9285, Loss: 0.0495
    Epoch  15 Batch  260/269 - Train Accuracy: 0.9291, Validation Accuracy: 0.9357, Loss: 0.0509
    Epoch  15 Batch  265/269 - Train Accuracy: 0.9391, Validation Accuracy: 0.9332, Loss: 0.0496
    Epoch  16 Batch    5/269 - Train Accuracy: 0.9380, Validation Accuracy: 0.9236, Loss: 0.0513
    Epoch  16 Batch   10/269 - Train Accuracy: 0.9429, Validation Accuracy: 0.9289, Loss: 0.0435
    Epoch  16 Batch   15/269 - Train Accuracy: 0.9408, Validation Accuracy: 0.9301, Loss: 0.0408
    Epoch  16 Batch   20/269 - Train Accuracy: 0.9429, Validation Accuracy: 0.9238, Loss: 0.0477
    Epoch  16 Batch   25/269 - Train Accuracy: 0.9294, Validation Accuracy: 0.9264, Loss: 0.0577
    Epoch  16 Batch   30/269 - Train Accuracy: 0.9432, Validation Accuracy: 0.9314, Loss: 0.0497
    Epoch  16 Batch   35/269 - Train Accuracy: 0.9305, Validation Accuracy: 0.9224, Loss: 0.0616
    Epoch  16 Batch   40/269 - Train Accuracy: 0.9303, Validation Accuracy: 0.9266, Loss: 0.0535
    Epoch  16 Batch   45/269 - Train Accuracy: 0.9417, Validation Accuracy: 0.9246, Loss: 0.0550
    Epoch  16 Batch   50/269 - Train Accuracy: 0.9302, Validation Accuracy: 0.9300, Loss: 0.0568
    Epoch  16 Batch   55/269 - Train Accuracy: 0.9470, Validation Accuracy: 0.9335, Loss: 0.0443
    Epoch  16 Batch   60/269 - Train Accuracy: 0.9371, Validation Accuracy: 0.9363, Loss: 0.0469
    Epoch  16 Batch   65/269 - Train Accuracy: 0.9479, Validation Accuracy: 0.9370, Loss: 0.0460
    Epoch  16 Batch   70/269 - Train Accuracy: 0.9359, Validation Accuracy: 0.9338, Loss: 0.0511
    Epoch  16 Batch   75/269 - Train Accuracy: 0.9422, Validation Accuracy: 0.9337, Loss: 0.0519
    Epoch  16 Batch   80/269 - Train Accuracy: 0.9282, Validation Accuracy: 0.9252, Loss: 0.0493
    Epoch  16 Batch   85/269 - Train Accuracy: 0.9268, Validation Accuracy: 0.9348, Loss: 0.0474
    Epoch  16 Batch   90/269 - Train Accuracy: 0.9362, Validation Accuracy: 0.9288, Loss: 0.0464
    Epoch  16 Batch   95/269 - Train Accuracy: 0.9421, Validation Accuracy: 0.9349, Loss: 0.0437
    Epoch  16 Batch  100/269 - Train Accuracy: 0.9333, Validation Accuracy: 0.9269, Loss: 0.0502
    Epoch  16 Batch  105/269 - Train Accuracy: 0.9361, Validation Accuracy: 0.9322, Loss: 0.0459
    Epoch  16 Batch  110/269 - Train Accuracy: 0.9301, Validation Accuracy: 0.9330, Loss: 0.0469
    Epoch  16 Batch  115/269 - Train Accuracy: 0.9296, Validation Accuracy: 0.9285, Loss: 0.0512
    Epoch  16 Batch  120/269 - Train Accuracy: 0.9361, Validation Accuracy: 0.9361, Loss: 0.0537
    Epoch  16 Batch  125/269 - Train Accuracy: 0.9385, Validation Accuracy: 0.9374, Loss: 0.0474
    Epoch  16 Batch  130/269 - Train Accuracy: 0.9351, Validation Accuracy: 0.9321, Loss: 0.0478
    Epoch  16 Batch  135/269 - Train Accuracy: 0.9303, Validation Accuracy: 0.9252, Loss: 0.0479
    Epoch  16 Batch  140/269 - Train Accuracy: 0.9281, Validation Accuracy: 0.9201, Loss: 0.0551
    Epoch  16 Batch  145/269 - Train Accuracy: 0.9345, Validation Accuracy: 0.9298, Loss: 0.0476
    Epoch  16 Batch  150/269 - Train Accuracy: 0.9324, Validation Accuracy: 0.9264, Loss: 0.0497
    Epoch  16 Batch  155/269 - Train Accuracy: 0.9353, Validation Accuracy: 0.9286, Loss: 0.0459
    Epoch  16 Batch  160/269 - Train Accuracy: 0.9352, Validation Accuracy: 0.9275, Loss: 0.0460
    Epoch  16 Batch  165/269 - Train Accuracy: 0.9312, Validation Accuracy: 0.9386, Loss: 0.0488
    Epoch  16 Batch  170/269 - Train Accuracy: 0.9297, Validation Accuracy: 0.9413, Loss: 0.0466
    Epoch  16 Batch  175/269 - Train Accuracy: 0.9275, Validation Accuracy: 0.9371, Loss: 0.0597
    Epoch  16 Batch  180/269 - Train Accuracy: 0.9454, Validation Accuracy: 0.9379, Loss: 0.0419
    Epoch  16 Batch  185/269 - Train Accuracy: 0.9478, Validation Accuracy: 0.9316, Loss: 0.0440
    Epoch  16 Batch  190/269 - Train Accuracy: 0.9338, Validation Accuracy: 0.9308, Loss: 0.0542
    Epoch  16 Batch  195/269 - Train Accuracy: 0.9333, Validation Accuracy: 0.9273, Loss: 0.0510
    Epoch  16 Batch  200/269 - Train Accuracy: 0.9436, Validation Accuracy: 0.9320, Loss: 0.0427
    Epoch  16 Batch  205/269 - Train Accuracy: 0.9456, Validation Accuracy: 0.9346, Loss: 0.0530
    Epoch  16 Batch  210/269 - Train Accuracy: 0.9386, Validation Accuracy: 0.9312, Loss: 0.1148
    Epoch  16 Batch  215/269 - Train Accuracy: 0.9230, Validation Accuracy: 0.9250, Loss: 0.0556
    Epoch  16 Batch  220/269 - Train Accuracy: 0.9304, Validation Accuracy: 0.9221, Loss: 0.0647
    Epoch  16 Batch  225/269 - Train Accuracy: 0.9178, Validation Accuracy: 0.9192, Loss: 0.0575
    Epoch  16 Batch  230/269 - Train Accuracy: 0.9333, Validation Accuracy: 0.9354, Loss: 0.0545
    Epoch  16 Batch  235/269 - Train Accuracy: 0.9568, Validation Accuracy: 0.9236, Loss: 0.0432
    Epoch  16 Batch  240/269 - Train Accuracy: 0.9266, Validation Accuracy: 0.9256, Loss: 0.0470
    Epoch  16 Batch  245/269 - Train Accuracy: 0.9225, Validation Accuracy: 0.9199, Loss: 0.0561
    Epoch  16 Batch  250/269 - Train Accuracy: 0.9027, Validation Accuracy: 0.8868, Loss: 0.0549
    Epoch  16 Batch  255/269 - Train Accuracy: 0.9173, Validation Accuracy: 0.9168, Loss: 0.0854
    Epoch  16 Batch  260/269 - Train Accuracy: 0.9167, Validation Accuracy: 0.9094, Loss: 0.0673
    Epoch  16 Batch  265/269 - Train Accuracy: 0.9233, Validation Accuracy: 0.9248, Loss: 0.0595
    Epoch  17 Batch    5/269 - Train Accuracy: 0.9263, Validation Accuracy: 0.9127, Loss: 0.0531
    Epoch  17 Batch   10/269 - Train Accuracy: 0.9284, Validation Accuracy: 0.9261, Loss: 0.0500
    Epoch  17 Batch   15/269 - Train Accuracy: 0.9317, Validation Accuracy: 0.9198, Loss: 0.0441
    Epoch  17 Batch   20/269 - Train Accuracy: 0.9353, Validation Accuracy: 0.9301, Loss: 0.0469
    Epoch  17 Batch   25/269 - Train Accuracy: 0.9222, Validation Accuracy: 0.9230, Loss: 0.0546
    Epoch  17 Batch   30/269 - Train Accuracy: 0.9441, Validation Accuracy: 0.9265, Loss: 0.0456
    Epoch  17 Batch   35/269 - Train Accuracy: 0.9292, Validation Accuracy: 0.9209, Loss: 0.0623
    Epoch  17 Batch   40/269 - Train Accuracy: 0.9267, Validation Accuracy: 0.9380, Loss: 0.0507
    Epoch  17 Batch   45/269 - Train Accuracy: 0.9354, Validation Accuracy: 0.9252, Loss: 0.0513
    Epoch  17 Batch   50/269 - Train Accuracy: 0.9333, Validation Accuracy: 0.9253, Loss: 0.0517
    Epoch  17 Batch   55/269 - Train Accuracy: 0.9504, Validation Accuracy: 0.9326, Loss: 0.0448
    Epoch  17 Batch   60/269 - Train Accuracy: 0.9321, Validation Accuracy: 0.9327, Loss: 0.0470
    Epoch  17 Batch   65/269 - Train Accuracy: 0.9478, Validation Accuracy: 0.9355, Loss: 0.0419
    Epoch  17 Batch   70/269 - Train Accuracy: 0.9412, Validation Accuracy: 0.9407, Loss: 0.0456
    Epoch  17 Batch   75/269 - Train Accuracy: 0.9403, Validation Accuracy: 0.9286, Loss: 0.0499
    Epoch  17 Batch   80/269 - Train Accuracy: 0.9326, Validation Accuracy: 0.9254, Loss: 0.0466
    Epoch  17 Batch   85/269 - Train Accuracy: 0.9313, Validation Accuracy: 0.9284, Loss: 0.0446
    Epoch  17 Batch   90/269 - Train Accuracy: 0.9389, Validation Accuracy: 0.9385, Loss: 0.0451
    Epoch  17 Batch   95/269 - Train Accuracy: 0.9462, Validation Accuracy: 0.9347, Loss: 0.0424
    Epoch  17 Batch  100/269 - Train Accuracy: 0.9368, Validation Accuracy: 0.9323, Loss: 0.0463
    Epoch  17 Batch  105/269 - Train Accuracy: 0.9332, Validation Accuracy: 0.9289, Loss: 0.0421
    Epoch  17 Batch  110/269 - Train Accuracy: 0.9257, Validation Accuracy: 0.9366, Loss: 0.0466
    Epoch  17 Batch  115/269 - Train Accuracy: 0.9408, Validation Accuracy: 0.9411, Loss: 0.0480
    Epoch  17 Batch  120/269 - Train Accuracy: 0.9398, Validation Accuracy: 0.9349, Loss: 0.0484
    Epoch  17 Batch  125/269 - Train Accuracy: 0.9408, Validation Accuracy: 0.9389, Loss: 0.0439
    Epoch  17 Batch  130/269 - Train Accuracy: 0.9495, Validation Accuracy: 0.9401, Loss: 0.0454
    Epoch  17 Batch  135/269 - Train Accuracy: 0.9346, Validation Accuracy: 0.9368, Loss: 0.0431
    Epoch  17 Batch  140/269 - Train Accuracy: 0.9323, Validation Accuracy: 0.9292, Loss: 0.0509
    Epoch  17 Batch  145/269 - Train Accuracy: 0.9415, Validation Accuracy: 0.9312, Loss: 0.0406
    Epoch  17 Batch  150/269 - Train Accuracy: 0.9342, Validation Accuracy: 0.9293, Loss: 0.0471
    Epoch  17 Batch  155/269 - Train Accuracy: 0.9335, Validation Accuracy: 0.9268, Loss: 0.0422
    Epoch  17 Batch  160/269 - Train Accuracy: 0.9422, Validation Accuracy: 0.9240, Loss: 0.0449
    Epoch  17 Batch  165/269 - Train Accuracy: 0.9287, Validation Accuracy: 0.9177, Loss: 0.0447
    Epoch  17 Batch  170/269 - Train Accuracy: 0.9176, Validation Accuracy: 0.9279, Loss: 0.0438
    Epoch  17 Batch  175/269 - Train Accuracy: 0.9291, Validation Accuracy: 0.9340, Loss: 0.0574
    Epoch  17 Batch  180/269 - Train Accuracy: 0.9481, Validation Accuracy: 0.9425, Loss: 0.0404
    Epoch  17 Batch  185/269 - Train Accuracy: 0.9461, Validation Accuracy: 0.9347, Loss: 0.0432
    Epoch  17 Batch  190/269 - Train Accuracy: 0.9351, Validation Accuracy: 0.9309, Loss: 0.0470
    Epoch  17 Batch  195/269 - Train Accuracy: 0.9368, Validation Accuracy: 0.9354, Loss: 0.0504
    Epoch  17 Batch  200/269 - Train Accuracy: 0.9502, Validation Accuracy: 0.9250, Loss: 0.0439
    Epoch  17 Batch  205/269 - Train Accuracy: 0.9367, Validation Accuracy: 0.9354, Loss: 0.0452
    Epoch  17 Batch  210/269 - Train Accuracy: 0.9413, Validation Accuracy: 0.9308, Loss: 0.0506
    Epoch  17 Batch  215/269 - Train Accuracy: 0.9240, Validation Accuracy: 0.9339, Loss: 0.0478
    Epoch  17 Batch  220/269 - Train Accuracy: 0.9350, Validation Accuracy: 0.9310, Loss: 0.0491
    Epoch  17 Batch  225/269 - Train Accuracy: 0.9190, Validation Accuracy: 0.9330, Loss: 0.0443
    Epoch  17 Batch  230/269 - Train Accuracy: 0.9426, Validation Accuracy: 0.9388, Loss: 0.0475
    Epoch  17 Batch  235/269 - Train Accuracy: 0.9613, Validation Accuracy: 0.9395, Loss: 0.0418
    Epoch  17 Batch  240/269 - Train Accuracy: 0.9368, Validation Accuracy: 0.9360, Loss: 0.0392
    Epoch  17 Batch  245/269 - Train Accuracy: 0.9389, Validation Accuracy: 0.9443, Loss: 0.0413
    Epoch  17 Batch  250/269 - Train Accuracy: 0.9271, Validation Accuracy: 0.9350, Loss: 0.0436
    Epoch  17 Batch  255/269 - Train Accuracy: 0.9323, Validation Accuracy: 0.9400, Loss: 0.0443
    Epoch  17 Batch  260/269 - Train Accuracy: 0.9310, Validation Accuracy: 0.9400, Loss: 0.0457
    Epoch  17 Batch  265/269 - Train Accuracy: 0.9437, Validation Accuracy: 0.9385, Loss: 0.0428
    Epoch  18 Batch    5/269 - Train Accuracy: 0.9379, Validation Accuracy: 0.9313, Loss: 0.0469
    Epoch  18 Batch   10/269 - Train Accuracy: 0.9371, Validation Accuracy: 0.9378, Loss: 0.0421
    Epoch  18 Batch   15/269 - Train Accuracy: 0.9440, Validation Accuracy: 0.9300, Loss: 0.0358
    Epoch  18 Batch   20/269 - Train Accuracy: 0.9352, Validation Accuracy: 0.9290, Loss: 0.0406
    Epoch  18 Batch   25/269 - Train Accuracy: 0.9336, Validation Accuracy: 0.9303, Loss: 0.0486
    Epoch  18 Batch   30/269 - Train Accuracy: 0.9489, Validation Accuracy: 0.9337, Loss: 0.0405
    Epoch  18 Batch   35/269 - Train Accuracy: 0.9288, Validation Accuracy: 0.9300, Loss: 0.0580
    Epoch  18 Batch   40/269 - Train Accuracy: 0.9378, Validation Accuracy: 0.9316, Loss: 0.0486
    Epoch  18 Batch   45/269 - Train Accuracy: 0.9502, Validation Accuracy: 0.9278, Loss: 0.0485
    Epoch  18 Batch   50/269 - Train Accuracy: 0.9341, Validation Accuracy: 0.9308, Loss: 0.0509
    Epoch  18 Batch   55/269 - Train Accuracy: 0.9501, Validation Accuracy: 0.9403, Loss: 0.0399
    Epoch  18 Batch   60/269 - Train Accuracy: 0.9381, Validation Accuracy: 0.9332, Loss: 0.0424
    Epoch  18 Batch   65/269 - Train Accuracy: 0.9489, Validation Accuracy: 0.9398, Loss: 0.0394
    Epoch  18 Batch   70/269 - Train Accuracy: 0.9512, Validation Accuracy: 0.9358, Loss: 0.0429
    Epoch  18 Batch   75/269 - Train Accuracy: 0.9476, Validation Accuracy: 0.9366, Loss: 0.0462
    Epoch  18 Batch   80/269 - Train Accuracy: 0.9371, Validation Accuracy: 0.9227, Loss: 0.0446
    Epoch  18 Batch   85/269 - Train Accuracy: 0.9297, Validation Accuracy: 0.9271, Loss: 0.0454
    Epoch  18 Batch   90/269 - Train Accuracy: 0.9433, Validation Accuracy: 0.9286, Loss: 0.0449
    Epoch  18 Batch   95/269 - Train Accuracy: 0.9451, Validation Accuracy: 0.9329, Loss: 0.0437
    Epoch  18 Batch  100/269 - Train Accuracy: 0.9403, Validation Accuracy: 0.9282, Loss: 0.0453
    Epoch  18 Batch  105/269 - Train Accuracy: 0.9355, Validation Accuracy: 0.9271, Loss: 0.0392
    Epoch  18 Batch  110/269 - Train Accuracy: 0.9374, Validation Accuracy: 0.9386, Loss: 0.0425
    Epoch  18 Batch  115/269 - Train Accuracy: 0.9354, Validation Accuracy: 0.9397, Loss: 0.0463
    Epoch  18 Batch  120/269 - Train Accuracy: 0.9458, Validation Accuracy: 0.9349, Loss: 0.0449
    Epoch  18 Batch  125/269 - Train Accuracy: 0.9449, Validation Accuracy: 0.9381, Loss: 0.0421
    Epoch  18 Batch  130/269 - Train Accuracy: 0.9479, Validation Accuracy: 0.9398, Loss: 0.0425
    Epoch  18 Batch  135/269 - Train Accuracy: 0.9327, Validation Accuracy: 0.9376, Loss: 0.0418
    Epoch  18 Batch  140/269 - Train Accuracy: 0.9442, Validation Accuracy: 0.9288, Loss: 0.0496
    Epoch  18 Batch  145/269 - Train Accuracy: 0.9443, Validation Accuracy: 0.9276, Loss: 0.0396
    Epoch  18 Batch  150/269 - Train Accuracy: 0.9338, Validation Accuracy: 0.9283, Loss: 0.0449
    Epoch  18 Batch  155/269 - Train Accuracy: 0.9362, Validation Accuracy: 0.9413, Loss: 0.0401
    Epoch  18 Batch  160/269 - Train Accuracy: 0.9375, Validation Accuracy: 0.9356, Loss: 0.0422
    Epoch  18 Batch  165/269 - Train Accuracy: 0.9388, Validation Accuracy: 0.9429, Loss: 0.0425
    Epoch  18 Batch  170/269 - Train Accuracy: 0.9338, Validation Accuracy: 0.9355, Loss: 0.0419
    Epoch  18 Batch  175/269 - Train Accuracy: 0.9413, Validation Accuracy: 0.9387, Loss: 0.0514
    Epoch  18 Batch  180/269 - Train Accuracy: 0.9429, Validation Accuracy: 0.9403, Loss: 0.0379
    Epoch  18 Batch  185/269 - Train Accuracy: 0.9526, Validation Accuracy: 0.9432, Loss: 0.0422
    Epoch  18 Batch  190/269 - Train Accuracy: 0.9385, Validation Accuracy: 0.9352, Loss: 0.0456
    Epoch  18 Batch  195/269 - Train Accuracy: 0.9383, Validation Accuracy: 0.9430, Loss: 0.0450
    Epoch  18 Batch  200/269 - Train Accuracy: 0.9527, Validation Accuracy: 0.9302, Loss: 0.0392
    Epoch  18 Batch  205/269 - Train Accuracy: 0.9530, Validation Accuracy: 0.9382, Loss: 0.0401
    Epoch  18 Batch  210/269 - Train Accuracy: 0.9450, Validation Accuracy: 0.9384, Loss: 0.0412
    Epoch  18 Batch  215/269 - Train Accuracy: 0.9407, Validation Accuracy: 0.9461, Loss: 0.0430
    Epoch  18 Batch  220/269 - Train Accuracy: 0.9390, Validation Accuracy: 0.9363, Loss: 0.0459
    Epoch  18 Batch  225/269 - Train Accuracy: 0.9252, Validation Accuracy: 0.9405, Loss: 0.0414
    Epoch  18 Batch  230/269 - Train Accuracy: 0.9422, Validation Accuracy: 0.9353, Loss: 0.0416
    Epoch  18 Batch  235/269 - Train Accuracy: 0.9708, Validation Accuracy: 0.9362, Loss: 0.0344
    Epoch  18 Batch  240/269 - Train Accuracy: 0.9422, Validation Accuracy: 0.9375, Loss: 0.0376
    Epoch  18 Batch  245/269 - Train Accuracy: 0.9469, Validation Accuracy: 0.9370, Loss: 0.0379
    Epoch  18 Batch  250/269 - Train Accuracy: 0.9417, Validation Accuracy: 0.9379, Loss: 0.0414
    Epoch  18 Batch  255/269 - Train Accuracy: 0.9373, Validation Accuracy: 0.9442, Loss: 0.0443
    Epoch  18 Batch  260/269 - Train Accuracy: 0.9388, Validation Accuracy: 0.9345, Loss: 0.0434
    Epoch  18 Batch  265/269 - Train Accuracy: 0.9451, Validation Accuracy: 0.9490, Loss: 0.0409
    Epoch  19 Batch    5/269 - Train Accuracy: 0.9426, Validation Accuracy: 0.9379, Loss: 0.0417
    Epoch  19 Batch   10/269 - Train Accuracy: 0.9350, Validation Accuracy: 0.9304, Loss: 0.0369
    Epoch  19 Batch   15/269 - Train Accuracy: 0.9455, Validation Accuracy: 0.9444, Loss: 0.0333
    Epoch  19 Batch   20/269 - Train Accuracy: 0.9498, Validation Accuracy: 0.9319, Loss: 0.0393
    Epoch  19 Batch   25/269 - Train Accuracy: 0.9286, Validation Accuracy: 0.9316, Loss: 0.0472
    Epoch  19 Batch   30/269 - Train Accuracy: 0.9416, Validation Accuracy: 0.9316, Loss: 0.0409
    Epoch  19 Batch   35/269 - Train Accuracy: 0.9346, Validation Accuracy: 0.9357, Loss: 0.0561
    Epoch  19 Batch   40/269 - Train Accuracy: 0.9424, Validation Accuracy: 0.9378, Loss: 0.0462
    Epoch  19 Batch   45/269 - Train Accuracy: 0.9509, Validation Accuracy: 0.9328, Loss: 0.0446
    Epoch  19 Batch   50/269 - Train Accuracy: 0.9243, Validation Accuracy: 0.9334, Loss: 0.0475
    Epoch  19 Batch   55/269 - Train Accuracy: 0.9553, Validation Accuracy: 0.9352, Loss: 0.0395
    Epoch  19 Batch   60/269 - Train Accuracy: 0.9506, Validation Accuracy: 0.9355, Loss: 0.0427
    Epoch  19 Batch   65/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9413, Loss: 0.0376
    Epoch  19 Batch   70/269 - Train Accuracy: 0.9438, Validation Accuracy: 0.9395, Loss: 0.0435
    Epoch  19 Batch   75/269 - Train Accuracy: 0.9422, Validation Accuracy: 0.9329, Loss: 0.0445
    Epoch  19 Batch   80/269 - Train Accuracy: 0.9410, Validation Accuracy: 0.9316, Loss: 0.0436
    Epoch  19 Batch   85/269 - Train Accuracy: 0.9240, Validation Accuracy: 0.9337, Loss: 0.0483
    Epoch  19 Batch   90/269 - Train Accuracy: 0.9324, Validation Accuracy: 0.9335, Loss: 0.0463
    Epoch  19 Batch   95/269 - Train Accuracy: 0.9549, Validation Accuracy: 0.9442, Loss: 0.0423
    Epoch  19 Batch  100/269 - Train Accuracy: 0.9418, Validation Accuracy: 0.9381, Loss: 0.0470
    Epoch  19 Batch  105/269 - Train Accuracy: 0.9405, Validation Accuracy: 0.9350, Loss: 0.0424
    Epoch  19 Batch  110/269 - Train Accuracy: 0.9370, Validation Accuracy: 0.9382, Loss: 0.0405
    Epoch  19 Batch  115/269 - Train Accuracy: 0.9370, Validation Accuracy: 0.9361, Loss: 0.0438
    Epoch  19 Batch  120/269 - Train Accuracy: 0.9440, Validation Accuracy: 0.9378, Loss: 0.0440
    Epoch  19 Batch  125/269 - Train Accuracy: 0.9451, Validation Accuracy: 0.9329, Loss: 0.0404
    Epoch  19 Batch  130/269 - Train Accuracy: 0.9462, Validation Accuracy: 0.9422, Loss: 0.0417
    Epoch  19 Batch  135/269 - Train Accuracy: 0.9387, Validation Accuracy: 0.9390, Loss: 0.0411
    Epoch  19 Batch  140/269 - Train Accuracy: 0.9355, Validation Accuracy: 0.9303, Loss: 0.0490
    Epoch  19 Batch  145/269 - Train Accuracy: 0.9426, Validation Accuracy: 0.9300, Loss: 0.0412
    Epoch  19 Batch  150/269 - Train Accuracy: 0.9362, Validation Accuracy: 0.9300, Loss: 0.0437
    Epoch  19 Batch  155/269 - Train Accuracy: 0.9272, Validation Accuracy: 0.9359, Loss: 0.0392
    Epoch  19 Batch  160/269 - Train Accuracy: 0.9376, Validation Accuracy: 0.9321, Loss: 0.0454
    Epoch  19 Batch  165/269 - Train Accuracy: 0.9366, Validation Accuracy: 0.9408, Loss: 0.0437
    Epoch  19 Batch  170/269 - Train Accuracy: 0.9293, Validation Accuracy: 0.9331, Loss: 0.0431
    Epoch  19 Batch  175/269 - Train Accuracy: 0.9240, Validation Accuracy: 0.9402, Loss: 0.0554
    Epoch  19 Batch  180/269 - Train Accuracy: 0.9465, Validation Accuracy: 0.9428, Loss: 0.0386
    Epoch  19 Batch  185/269 - Train Accuracy: 0.9488, Validation Accuracy: 0.9466, Loss: 0.0402
    Epoch  19 Batch  190/269 - Train Accuracy: 0.9397, Validation Accuracy: 0.9293, Loss: 0.0462
    Epoch  19 Batch  195/269 - Train Accuracy: 0.9353, Validation Accuracy: 0.9395, Loss: 0.0394
    Epoch  19 Batch  200/269 - Train Accuracy: 0.9528, Validation Accuracy: 0.9370, Loss: 0.0367
    Epoch  19 Batch  205/269 - Train Accuracy: 0.9447, Validation Accuracy: 0.9367, Loss: 0.0410
    Epoch  19 Batch  210/269 - Train Accuracy: 0.9422, Validation Accuracy: 0.9371, Loss: 0.0402
    Epoch  19 Batch  215/269 - Train Accuracy: 0.9283, Validation Accuracy: 0.9442, Loss: 0.0423
    Epoch  19 Batch  220/269 - Train Accuracy: 0.9378, Validation Accuracy: 0.9321, Loss: 0.0426
    Epoch  19 Batch  225/269 - Train Accuracy: 0.9265, Validation Accuracy: 0.9414, Loss: 0.0410
    Epoch  19 Batch  230/269 - Train Accuracy: 0.9446, Validation Accuracy: 0.9363, Loss: 0.0393
    Epoch  19 Batch  235/269 - Train Accuracy: 0.9682, Validation Accuracy: 0.9355, Loss: 0.0325
    Epoch  19 Batch  240/269 - Train Accuracy: 0.9303, Validation Accuracy: 0.9387, Loss: 0.0361
    Epoch  19 Batch  245/269 - Train Accuracy: 0.9452, Validation Accuracy: 0.9414, Loss: 0.0352
    Epoch  19 Batch  250/269 - Train Accuracy: 0.9358, Validation Accuracy: 0.9347, Loss: 0.0384
    Epoch  19 Batch  255/269 - Train Accuracy: 0.9363, Validation Accuracy: 0.9326, Loss: 0.0392
    Epoch  19 Batch  260/269 - Train Accuracy: 0.9361, Validation Accuracy: 0.9411, Loss: 0.0447
    Epoch  19 Batch  265/269 - Train Accuracy: 0.9428, Validation Accuracy: 0.9415, Loss: 0.0412
    Epoch  20 Batch    5/269 - Train Accuracy: 0.9414, Validation Accuracy: 0.9347, Loss: 0.0409
    Epoch  20 Batch   10/269 - Train Accuracy: 0.9455, Validation Accuracy: 0.9423, Loss: 0.0356
    Epoch  20 Batch   15/269 - Train Accuracy: 0.9517, Validation Accuracy: 0.9425, Loss: 0.0350
    Epoch  20 Batch   20/269 - Train Accuracy: 0.9425, Validation Accuracy: 0.9358, Loss: 0.0401
    Epoch  20 Batch   25/269 - Train Accuracy: 0.9354, Validation Accuracy: 0.9355, Loss: 0.0457
    Epoch  20 Batch   30/269 - Train Accuracy: 0.9517, Validation Accuracy: 0.9294, Loss: 0.0389
    Epoch  20 Batch   35/269 - Train Accuracy: 0.9410, Validation Accuracy: 0.9366, Loss: 0.0521
    Epoch  20 Batch   40/269 - Train Accuracy: 0.9457, Validation Accuracy: 0.9365, Loss: 0.0426
    Epoch  20 Batch   45/269 - Train Accuracy: 0.9502, Validation Accuracy: 0.9339, Loss: 0.0432
    Epoch  20 Batch   50/269 - Train Accuracy: 0.9351, Validation Accuracy: 0.9368, Loss: 0.0472
    Epoch  20 Batch   55/269 - Train Accuracy: 0.9520, Validation Accuracy: 0.9434, Loss: 0.0369
    Epoch  20 Batch   60/269 - Train Accuracy: 0.9442, Validation Accuracy: 0.9396, Loss: 0.0384
    Epoch  20 Batch   65/269 - Train Accuracy: 0.9592, Validation Accuracy: 0.9442, Loss: 0.0354
    Epoch  20 Batch   70/269 - Train Accuracy: 0.9492, Validation Accuracy: 0.9403, Loss: 0.0392
    Epoch  20 Batch   75/269 - Train Accuracy: 0.9505, Validation Accuracy: 0.9365, Loss: 0.0453
    Epoch  20 Batch   80/269 - Train Accuracy: 0.9457, Validation Accuracy: 0.9325, Loss: 0.0401
    Epoch  20 Batch   85/269 - Train Accuracy: 0.9367, Validation Accuracy: 0.9402, Loss: 0.0440
    Epoch  20 Batch   90/269 - Train Accuracy: 0.9041, Validation Accuracy: 0.9020, Loss: 0.1027
    Epoch  20 Batch   95/269 - Train Accuracy: 0.9395, Validation Accuracy: 0.9369, Loss: 0.0979
    Epoch  20 Batch  100/269 - Train Accuracy: 0.9294, Validation Accuracy: 0.9216, Loss: 0.0813
    Epoch  20 Batch  105/269 - Train Accuracy: 0.9248, Validation Accuracy: 0.9293, Loss: 0.0544
    Epoch  20 Batch  110/269 - Train Accuracy: 0.9252, Validation Accuracy: 0.9232, Loss: 0.0514
    Epoch  20 Batch  115/269 - Train Accuracy: 0.9420, Validation Accuracy: 0.9247, Loss: 0.0525
    Epoch  20 Batch  120/269 - Train Accuracy: 0.9408, Validation Accuracy: 0.9373, Loss: 0.0480
    Epoch  20 Batch  125/269 - Train Accuracy: 0.9448, Validation Accuracy: 0.9394, Loss: 0.0433
    Epoch  20 Batch  130/269 - Train Accuracy: 0.9500, Validation Accuracy: 0.9380, Loss: 0.0469
    Epoch  20 Batch  135/269 - Train Accuracy: 0.9287, Validation Accuracy: 0.9286, Loss: 0.0406
    Epoch  20 Batch  140/269 - Train Accuracy: 0.9323, Validation Accuracy: 0.9351, Loss: 0.0526
    Epoch  20 Batch  145/269 - Train Accuracy: 0.9455, Validation Accuracy: 0.9355, Loss: 0.0403
    Epoch  20 Batch  150/269 - Train Accuracy: 0.9405, Validation Accuracy: 0.9421, Loss: 0.0474
    Epoch  20 Batch  155/269 - Train Accuracy: 0.9417, Validation Accuracy: 0.9440, Loss: 0.0395
    Epoch  20 Batch  160/269 - Train Accuracy: 0.9514, Validation Accuracy: 0.9396, Loss: 0.0377
    Epoch  20 Batch  165/269 - Train Accuracy: 0.9413, Validation Accuracy: 0.9413, Loss: 0.0388
    Epoch  20 Batch  170/269 - Train Accuracy: 0.9382, Validation Accuracy: 0.9396, Loss: 0.0386
    Epoch  20 Batch  175/269 - Train Accuracy: 0.9404, Validation Accuracy: 0.9464, Loss: 0.0512
    Epoch  20 Batch  180/269 - Train Accuracy: 0.9451, Validation Accuracy: 0.9372, Loss: 0.0379
    Epoch  20 Batch  185/269 - Train Accuracy: 0.9576, Validation Accuracy: 0.9454, Loss: 0.0379
    Epoch  20 Batch  190/269 - Train Accuracy: 0.9486, Validation Accuracy: 0.9388, Loss: 0.0422
    Epoch  20 Batch  195/269 - Train Accuracy: 0.9399, Validation Accuracy: 0.9415, Loss: 0.0406
    Epoch  20 Batch  200/269 - Train Accuracy: 0.9561, Validation Accuracy: 0.9347, Loss: 0.0375
    Epoch  20 Batch  205/269 - Train Accuracy: 0.9509, Validation Accuracy: 0.9289, Loss: 0.0409
    Epoch  20 Batch  210/269 - Train Accuracy: 0.9436, Validation Accuracy: 0.9421, Loss: 0.0494
    Epoch  20 Batch  215/269 - Train Accuracy: 0.9268, Validation Accuracy: 0.9399, Loss: 0.0424
    Epoch  20 Batch  220/269 - Train Accuracy: 0.9406, Validation Accuracy: 0.9328, Loss: 0.0447
    Epoch  20 Batch  225/269 - Train Accuracy: 0.9216, Validation Accuracy: 0.9412, Loss: 0.0417
    Epoch  20 Batch  230/269 - Train Accuracy: 0.9476, Validation Accuracy: 0.9379, Loss: 0.0409
    Epoch  20 Batch  235/269 - Train Accuracy: 0.9668, Validation Accuracy: 0.9388, Loss: 0.0324
    Epoch  20 Batch  240/269 - Train Accuracy: 0.9352, Validation Accuracy: 0.9395, Loss: 0.0364
    Epoch  20 Batch  245/269 - Train Accuracy: 0.9479, Validation Accuracy: 0.9355, Loss: 0.0374
    Epoch  20 Batch  250/269 - Train Accuracy: 0.9412, Validation Accuracy: 0.9480, Loss: 0.0364
    Epoch  20 Batch  255/269 - Train Accuracy: 0.9442, Validation Accuracy: 0.9428, Loss: 0.0404
    Epoch  20 Batch  260/269 - Train Accuracy: 0.9365, Validation Accuracy: 0.9347, Loss: 0.0402
    Epoch  20 Batch  265/269 - Train Accuracy: 0.9474, Validation Accuracy: 0.9490, Loss: 0.0386
    Epoch  21 Batch    5/269 - Train Accuracy: 0.9545, Validation Accuracy: 0.9441, Loss: 0.0393
    Epoch  21 Batch   10/269 - Train Accuracy: 0.9494, Validation Accuracy: 0.9455, Loss: 0.0340
    Epoch  21 Batch   15/269 - Train Accuracy: 0.9508, Validation Accuracy: 0.9421, Loss: 0.0314
    Epoch  21 Batch   20/269 - Train Accuracy: 0.9442, Validation Accuracy: 0.9423, Loss: 0.0379
    Epoch  21 Batch   25/269 - Train Accuracy: 0.9409, Validation Accuracy: 0.9397, Loss: 0.0426
    Epoch  21 Batch   30/269 - Train Accuracy: 0.9529, Validation Accuracy: 0.9391, Loss: 0.0383
    Epoch  21 Batch   35/269 - Train Accuracy: 0.9359, Validation Accuracy: 0.9370, Loss: 0.0494
    Epoch  21 Batch   40/269 - Train Accuracy: 0.9464, Validation Accuracy: 0.9416, Loss: 0.0423
    Epoch  21 Batch   45/269 - Train Accuracy: 0.9473, Validation Accuracy: 0.9342, Loss: 0.0426
    Epoch  21 Batch   50/269 - Train Accuracy: 0.9299, Validation Accuracy: 0.9385, Loss: 0.0459
    Epoch  21 Batch   55/269 - Train Accuracy: 0.9592, Validation Accuracy: 0.9406, Loss: 0.0350
    Epoch  21 Batch   60/269 - Train Accuracy: 0.9473, Validation Accuracy: 0.9458, Loss: 0.0372
    Epoch  21 Batch   65/269 - Train Accuracy: 0.9617, Validation Accuracy: 0.9442, Loss: 0.0342
    Epoch  21 Batch   70/269 - Train Accuracy: 0.9458, Validation Accuracy: 0.9428, Loss: 0.0434
    Epoch  21 Batch   75/269 - Train Accuracy: 0.9528, Validation Accuracy: 0.9405, Loss: 0.0472
    Epoch  21 Batch   80/269 - Train Accuracy: 0.9379, Validation Accuracy: 0.9391, Loss: 0.0403
    Epoch  21 Batch   85/269 - Train Accuracy: 0.9344, Validation Accuracy: 0.9450, Loss: 0.0373
    Epoch  21 Batch   90/269 - Train Accuracy: 0.9420, Validation Accuracy: 0.9442, Loss: 0.0394
    Epoch  21 Batch   95/269 - Train Accuracy: 0.9537, Validation Accuracy: 0.9394, Loss: 0.0357
    Epoch  21 Batch  100/269 - Train Accuracy: 0.9455, Validation Accuracy: 0.9381, Loss: 0.0411
    Epoch  21 Batch  105/269 - Train Accuracy: 0.9489, Validation Accuracy: 0.9383, Loss: 0.0362
    Epoch  21 Batch  110/269 - Train Accuracy: 0.9514, Validation Accuracy: 0.9399, Loss: 0.0381
    Epoch  21 Batch  115/269 - Train Accuracy: 0.9471, Validation Accuracy: 0.9334, Loss: 0.0416
    Epoch  21 Batch  120/269 - Train Accuracy: 0.9490, Validation Accuracy: 0.9494, Loss: 0.0405
    Epoch  21 Batch  125/269 - Train Accuracy: 0.9495, Validation Accuracy: 0.9407, Loss: 0.0382
    Epoch  21 Batch  130/269 - Train Accuracy: 0.9588, Validation Accuracy: 0.9414, Loss: 0.0387
    Epoch  21 Batch  135/269 - Train Accuracy: 0.9404, Validation Accuracy: 0.9359, Loss: 0.0387
    Epoch  21 Batch  140/269 - Train Accuracy: 0.9418, Validation Accuracy: 0.9333, Loss: 0.0436
    Epoch  21 Batch  145/269 - Train Accuracy: 0.9440, Validation Accuracy: 0.9328, Loss: 0.0408
    Epoch  21 Batch  150/269 - Train Accuracy: 0.9372, Validation Accuracy: 0.9354, Loss: 0.0417
    Epoch  21 Batch  155/269 - Train Accuracy: 0.9403, Validation Accuracy: 0.9389, Loss: 0.0347
    Epoch  21 Batch  160/269 - Train Accuracy: 0.9490, Validation Accuracy: 0.9387, Loss: 0.0351
    Epoch  21 Batch  165/269 - Train Accuracy: 0.9462, Validation Accuracy: 0.9466, Loss: 0.0382
    Epoch  21 Batch  170/269 - Train Accuracy: 0.9390, Validation Accuracy: 0.9424, Loss: 0.0362
    Epoch  21 Batch  175/269 - Train Accuracy: 0.9378, Validation Accuracy: 0.9463, Loss: 0.0480
    Epoch  21 Batch  180/269 - Train Accuracy: 0.9448, Validation Accuracy: 0.9370, Loss: 0.0354
    Epoch  21 Batch  185/269 - Train Accuracy: 0.9597, Validation Accuracy: 0.9363, Loss: 0.0363
    Epoch  21 Batch  190/269 - Train Accuracy: 0.9492, Validation Accuracy: 0.9438, Loss: 0.0409
    Epoch  21 Batch  195/269 - Train Accuracy: 0.9382, Validation Accuracy: 0.9450, Loss: 0.0370
    Epoch  21 Batch  200/269 - Train Accuracy: 0.9590, Validation Accuracy: 0.9398, Loss: 0.0324
    Epoch  21 Batch  205/269 - Train Accuracy: 0.9558, Validation Accuracy: 0.9394, Loss: 0.0359
    Epoch  21 Batch  210/269 - Train Accuracy: 0.9525, Validation Accuracy: 0.9440, Loss: 0.0362
    Epoch  21 Batch  215/269 - Train Accuracy: 0.9475, Validation Accuracy: 0.9455, Loss: 0.0389
    Epoch  21 Batch  220/269 - Train Accuracy: 0.9442, Validation Accuracy: 0.9402, Loss: 0.0394
    Epoch  21 Batch  225/269 - Train Accuracy: 0.9419, Validation Accuracy: 0.9403, Loss: 0.0375
    Epoch  21 Batch  230/269 - Train Accuracy: 0.9505, Validation Accuracy: 0.9414, Loss: 0.0366
    Epoch  21 Batch  235/269 - Train Accuracy: 0.9728, Validation Accuracy: 0.9403, Loss: 0.0321
    Epoch  21 Batch  240/269 - Train Accuracy: 0.9392, Validation Accuracy: 0.9397, Loss: 0.0352
    Epoch  21 Batch  245/269 - Train Accuracy: 0.9484, Validation Accuracy: 0.9387, Loss: 0.0342
    Epoch  21 Batch  250/269 - Train Accuracy: 0.9337, Validation Accuracy: 0.9405, Loss: 0.0356
    Epoch  21 Batch  255/269 - Train Accuracy: 0.9395, Validation Accuracy: 0.9368, Loss: 0.0374
    Epoch  21 Batch  260/269 - Train Accuracy: 0.9426, Validation Accuracy: 0.9436, Loss: 0.0402
    Epoch  21 Batch  265/269 - Train Accuracy: 0.9493, Validation Accuracy: 0.9513, Loss: 0.0376
    Epoch  22 Batch    5/269 - Train Accuracy: 0.9494, Validation Accuracy: 0.9441, Loss: 0.0361
    Epoch  22 Batch   10/269 - Train Accuracy: 0.9466, Validation Accuracy: 0.9490, Loss: 0.0321
    Epoch  22 Batch   15/269 - Train Accuracy: 0.9513, Validation Accuracy: 0.9403, Loss: 0.0282
    Epoch  22 Batch   20/269 - Train Accuracy: 0.9405, Validation Accuracy: 0.9432, Loss: 0.0365
    Epoch  22 Batch   25/269 - Train Accuracy: 0.9388, Validation Accuracy: 0.9379, Loss: 0.0439
    Epoch  22 Batch   30/269 - Train Accuracy: 0.9513, Validation Accuracy: 0.9340, Loss: 0.0368
    Epoch  22 Batch   35/269 - Train Accuracy: 0.9445, Validation Accuracy: 0.9359, Loss: 0.0498
    Epoch  22 Batch   40/269 - Train Accuracy: 0.9456, Validation Accuracy: 0.9327, Loss: 0.0422
    Epoch  22 Batch   45/269 - Train Accuracy: 0.9509, Validation Accuracy: 0.9352, Loss: 0.0414
    Epoch  22 Batch   50/269 - Train Accuracy: 0.9351, Validation Accuracy: 0.9372, Loss: 0.0438
    Epoch  22 Batch   55/269 - Train Accuracy: 0.9608, Validation Accuracy: 0.9437, Loss: 0.0337
    Epoch  22 Batch   60/269 - Train Accuracy: 0.9467, Validation Accuracy: 0.9434, Loss: 0.0366
    Epoch  22 Batch   65/269 - Train Accuracy: 0.9565, Validation Accuracy: 0.9458, Loss: 0.0353
    Epoch  22 Batch   70/269 - Train Accuracy: 0.9523, Validation Accuracy: 0.9465, Loss: 0.0380
    Epoch  22 Batch   75/269 - Train Accuracy: 0.9520, Validation Accuracy: 0.9434, Loss: 0.0410
    Epoch  22 Batch   80/269 - Train Accuracy: 0.9435, Validation Accuracy: 0.9393, Loss: 0.0395
    Epoch  22 Batch   85/269 - Train Accuracy: 0.9303, Validation Accuracy: 0.9410, Loss: 0.0434
    Epoch  22 Batch   90/269 - Train Accuracy: 0.9460, Validation Accuracy: 0.9451, Loss: 0.0417
    Epoch  22 Batch   95/269 - Train Accuracy: 0.9582, Validation Accuracy: 0.9379, Loss: 0.0359
    Epoch  22 Batch  100/269 - Train Accuracy: 0.9388, Validation Accuracy: 0.9349, Loss: 0.0398
    Epoch  22 Batch  105/269 - Train Accuracy: 0.9472, Validation Accuracy: 0.9446, Loss: 0.0357
    Epoch  22 Batch  110/269 - Train Accuracy: 0.9479, Validation Accuracy: 0.9462, Loss: 0.0387
    Epoch  22 Batch  115/269 - Train Accuracy: 0.9473, Validation Accuracy: 0.9416, Loss: 0.0428
    Epoch  22 Batch  120/269 - Train Accuracy: 0.9464, Validation Accuracy: 0.9412, Loss: 0.0408
    Epoch  22 Batch  125/269 - Train Accuracy: 0.9525, Validation Accuracy: 0.9334, Loss: 0.0386
    Epoch  22 Batch  130/269 - Train Accuracy: 0.9566, Validation Accuracy: 0.9459, Loss: 0.0362
    Epoch  22 Batch  135/269 - Train Accuracy: 0.9432, Validation Accuracy: 0.9374, Loss: 0.0365
    Epoch  22 Batch  140/269 - Train Accuracy: 0.9403, Validation Accuracy: 0.9263, Loss: 0.0429
    Epoch  22 Batch  145/269 - Train Accuracy: 0.9467, Validation Accuracy: 0.9331, Loss: 0.0354
    Epoch  22 Batch  150/269 - Train Accuracy: 0.9443, Validation Accuracy: 0.9392, Loss: 0.0401
    Epoch  22 Batch  155/269 - Train Accuracy: 0.9409, Validation Accuracy: 0.9396, Loss: 0.0357
    Epoch  22 Batch  160/269 - Train Accuracy: 0.9540, Validation Accuracy: 0.9423, Loss: 0.0321
    Epoch  22 Batch  165/269 - Train Accuracy: 0.9479, Validation Accuracy: 0.9534, Loss: 0.0355
    Epoch  22 Batch  170/269 - Train Accuracy: 0.9406, Validation Accuracy: 0.9445, Loss: 0.0365
    Epoch  22 Batch  175/269 - Train Accuracy: 0.9435, Validation Accuracy: 0.9495, Loss: 0.0455
    Epoch  22 Batch  180/269 - Train Accuracy: 0.9471, Validation Accuracy: 0.9429, Loss: 0.0352
    Epoch  22 Batch  185/269 - Train Accuracy: 0.9608, Validation Accuracy: 0.9420, Loss: 0.0345
    Epoch  22 Batch  190/269 - Train Accuracy: 0.9520, Validation Accuracy: 0.9387, Loss: 0.0418
    Epoch  22 Batch  195/269 - Train Accuracy: 0.9366, Validation Accuracy: 0.9488, Loss: 0.0363
    Epoch  22 Batch  200/269 - Train Accuracy: 0.9572, Validation Accuracy: 0.9463, Loss: 0.0319
    Epoch  22 Batch  205/269 - Train Accuracy: 0.9623, Validation Accuracy: 0.9347, Loss: 0.0337
    Epoch  22 Batch  210/269 - Train Accuracy: 0.9504, Validation Accuracy: 0.9453, Loss: 0.0335
    Epoch  22 Batch  215/269 - Train Accuracy: 0.9408, Validation Accuracy: 0.9436, Loss: 0.0345
    Epoch  22 Batch  220/269 - Train Accuracy: 0.9453, Validation Accuracy: 0.9440, Loss: 0.0383
    Epoch  22 Batch  225/269 - Train Accuracy: 0.9300, Validation Accuracy: 0.9454, Loss: 0.0364
    Epoch  22 Batch  230/269 - Train Accuracy: 0.9524, Validation Accuracy: 0.9482, Loss: 0.0349
    Epoch  22 Batch  235/269 - Train Accuracy: 0.9727, Validation Accuracy: 0.9553, Loss: 0.0301
    Epoch  22 Batch  240/269 - Train Accuracy: 0.9400, Validation Accuracy: 0.9374, Loss: 0.0339
    Epoch  22 Batch  245/269 - Train Accuracy: 0.9572, Validation Accuracy: 0.9434, Loss: 0.0338
    Epoch  22 Batch  250/269 - Train Accuracy: 0.9346, Validation Accuracy: 0.9377, Loss: 0.0360
    Epoch  22 Batch  255/269 - Train Accuracy: 0.9450, Validation Accuracy: 0.9422, Loss: 0.0355
    Epoch  22 Batch  260/269 - Train Accuracy: 0.9443, Validation Accuracy: 0.9429, Loss: 0.0382
    Epoch  22 Batch  265/269 - Train Accuracy: 0.9463, Validation Accuracy: 0.9487, Loss: 0.0355
    Epoch  23 Batch    5/269 - Train Accuracy: 0.9513, Validation Accuracy: 0.9512, Loss: 0.0350
    Epoch  23 Batch   10/269 - Train Accuracy: 0.9515, Validation Accuracy: 0.9449, Loss: 0.0300
    Epoch  23 Batch   15/269 - Train Accuracy: 0.9548, Validation Accuracy: 0.9526, Loss: 0.0271
    Epoch  23 Batch   20/269 - Train Accuracy: 0.9472, Validation Accuracy: 0.9506, Loss: 0.0357
    Epoch  23 Batch   25/269 - Train Accuracy: 0.9438, Validation Accuracy: 0.9356, Loss: 0.0395
    Epoch  23 Batch   30/269 - Train Accuracy: 0.9579, Validation Accuracy: 0.9370, Loss: 0.0351
    Epoch  23 Batch   35/269 - Train Accuracy: 0.9419, Validation Accuracy: 0.9451, Loss: 0.0475
    Epoch  23 Batch   40/269 - Train Accuracy: 0.9478, Validation Accuracy: 0.9336, Loss: 0.0389
    Epoch  23 Batch   45/269 - Train Accuracy: 0.9521, Validation Accuracy: 0.9401, Loss: 0.0389
    Epoch  23 Batch   50/269 - Train Accuracy: 0.9344, Validation Accuracy: 0.9384, Loss: 0.0424
    Epoch  23 Batch   55/269 - Train Accuracy: 0.9606, Validation Accuracy: 0.9308, Loss: 0.0314
    Epoch  23 Batch   60/269 - Train Accuracy: 0.9463, Validation Accuracy: 0.9393, Loss: 0.0350
    Epoch  23 Batch   65/269 - Train Accuracy: 0.9580, Validation Accuracy: 0.9364, Loss: 0.0330
    Epoch  23 Batch   70/269 - Train Accuracy: 0.9527, Validation Accuracy: 0.9434, Loss: 0.0368
    Epoch  23 Batch   75/269 - Train Accuracy: 0.9477, Validation Accuracy: 0.9464, Loss: 0.0416
    Epoch  23 Batch   80/269 - Train Accuracy: 0.9474, Validation Accuracy: 0.9371, Loss: 0.0355
    Epoch  23 Batch   85/269 - Train Accuracy: 0.9390, Validation Accuracy: 0.9426, Loss: 0.0360
    Epoch  23 Batch   90/269 - Train Accuracy: 0.9372, Validation Accuracy: 0.9428, Loss: 0.0369
    Epoch  23 Batch   95/269 - Train Accuracy: 0.9569, Validation Accuracy: 0.9480, Loss: 0.0331
    Epoch  23 Batch  100/269 - Train Accuracy: 0.9447, Validation Accuracy: 0.9379, Loss: 0.0406
    Epoch  23 Batch  105/269 - Train Accuracy: 0.9414, Validation Accuracy: 0.9382, Loss: 0.0354
    Epoch  23 Batch  110/269 - Train Accuracy: 0.9423, Validation Accuracy: 0.9482, Loss: 0.0348
    Epoch  23 Batch  115/269 - Train Accuracy: 0.9494, Validation Accuracy: 0.9377, Loss: 0.0387
    Epoch  23 Batch  120/269 - Train Accuracy: 0.9506, Validation Accuracy: 0.9467, Loss: 0.0387
    Epoch  23 Batch  125/269 - Train Accuracy: 0.9553, Validation Accuracy: 0.9426, Loss: 0.0357
    Epoch  23 Batch  130/269 - Train Accuracy: 0.9503, Validation Accuracy: 0.9473, Loss: 0.0389
    Epoch  23 Batch  135/269 - Train Accuracy: 0.9419, Validation Accuracy: 0.9536, Loss: 0.0364
    Epoch  23 Batch  140/269 - Train Accuracy: 0.9425, Validation Accuracy: 0.9352, Loss: 0.0427
    Epoch  23 Batch  145/269 - Train Accuracy: 0.9487, Validation Accuracy: 0.9317, Loss: 0.0348
    Epoch  23 Batch  150/269 - Train Accuracy: 0.9444, Validation Accuracy: 0.9416, Loss: 0.0394
    Epoch  23 Batch  155/269 - Train Accuracy: 0.9388, Validation Accuracy: 0.9430, Loss: 0.0363
    Epoch  23 Batch  160/269 - Train Accuracy: 0.9561, Validation Accuracy: 0.9432, Loss: 0.0332
    Epoch  23 Batch  165/269 - Train Accuracy: 0.9488, Validation Accuracy: 0.9429, Loss: 0.0332
    Epoch  23 Batch  170/269 - Train Accuracy: 0.9409, Validation Accuracy: 0.9391, Loss: 0.0353
    Epoch  23 Batch  175/269 - Train Accuracy: 0.9398, Validation Accuracy: 0.9486, Loss: 0.0469
    Epoch  23 Batch  180/269 - Train Accuracy: 0.9397, Validation Accuracy: 0.9346, Loss: 0.0352
    Epoch  23 Batch  185/269 - Train Accuracy: 0.9581, Validation Accuracy: 0.9466, Loss: 0.0353
    Epoch  23 Batch  190/269 - Train Accuracy: 0.9438, Validation Accuracy: 0.9424, Loss: 0.0454
    Epoch  23 Batch  195/269 - Train Accuracy: 0.9384, Validation Accuracy: 0.9430, Loss: 0.0381
    Epoch  23 Batch  200/269 - Train Accuracy: 0.9621, Validation Accuracy: 0.9333, Loss: 0.0362
    Epoch  23 Batch  205/269 - Train Accuracy: 0.9510, Validation Accuracy: 0.9311, Loss: 0.0383
    Epoch  23 Batch  210/269 - Train Accuracy: 0.9408, Validation Accuracy: 0.9323, Loss: 0.0411
    Epoch  23 Batch  215/269 - Train Accuracy: 0.9398, Validation Accuracy: 0.9424, Loss: 0.0363
    Epoch  23 Batch  220/269 - Train Accuracy: 0.9434, Validation Accuracy: 0.9470, Loss: 0.0423
    Epoch  23 Batch  225/269 - Train Accuracy: 0.9368, Validation Accuracy: 0.9474, Loss: 0.0353
    Epoch  23 Batch  230/269 - Train Accuracy: 0.9491, Validation Accuracy: 0.9423, Loss: 0.0362
    Epoch  23 Batch  235/269 - Train Accuracy: 0.9788, Validation Accuracy: 0.9370, Loss: 0.0288
    Epoch  23 Batch  240/269 - Train Accuracy: 0.9532, Validation Accuracy: 0.9433, Loss: 0.0320
    Epoch  23 Batch  245/269 - Train Accuracy: 0.9542, Validation Accuracy: 0.9443, Loss: 0.0321
    Epoch  23 Batch  250/269 - Train Accuracy: 0.9440, Validation Accuracy: 0.9466, Loss: 0.0335
    Epoch  23 Batch  255/269 - Train Accuracy: 0.9423, Validation Accuracy: 0.9363, Loss: 0.0364
    Epoch  23 Batch  260/269 - Train Accuracy: 0.9468, Validation Accuracy: 0.9411, Loss: 0.0372
    Epoch  23 Batch  265/269 - Train Accuracy: 0.9445, Validation Accuracy: 0.9456, Loss: 0.0358
    Epoch  24 Batch    5/269 - Train Accuracy: 0.9523, Validation Accuracy: 0.9461, Loss: 0.0353
    Epoch  24 Batch   10/269 - Train Accuracy: 0.9532, Validation Accuracy: 0.9474, Loss: 0.0300
    Epoch  24 Batch   15/269 - Train Accuracy: 0.9477, Validation Accuracy: 0.9467, Loss: 0.0261
    Epoch  24 Batch   20/269 - Train Accuracy: 0.9514, Validation Accuracy: 0.9453, Loss: 0.0344
    Epoch  24 Batch   25/269 - Train Accuracy: 0.9417, Validation Accuracy: 0.9383, Loss: 0.0402
    Epoch  24 Batch   30/269 - Train Accuracy: 0.9568, Validation Accuracy: 0.9339, Loss: 0.0366
    Epoch  24 Batch   35/269 - Train Accuracy: 0.9347, Validation Accuracy: 0.9435, Loss: 0.0599
    Epoch  24 Batch   40/269 - Train Accuracy: 0.9316, Validation Accuracy: 0.9260, Loss: 0.0561
    Epoch  24 Batch   45/269 - Train Accuracy: 0.9337, Validation Accuracy: 0.9260, Loss: 0.0488
    Epoch  24 Batch   50/269 - Train Accuracy: 0.9279, Validation Accuracy: 0.9331, Loss: 0.0482
    Epoch  24 Batch   55/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9414, Loss: 0.0409
    Epoch  24 Batch   60/269 - Train Accuracy: 0.9392, Validation Accuracy: 0.9366, Loss: 0.0416
    Epoch  24 Batch   65/269 - Train Accuracy: 0.9517, Validation Accuracy: 0.9450, Loss: 0.0382
    Epoch  24 Batch   70/269 - Train Accuracy: 0.9505, Validation Accuracy: 0.9490, Loss: 0.0400
    Epoch  24 Batch   75/269 - Train Accuracy: 0.9435, Validation Accuracy: 0.9439, Loss: 0.0433
    Epoch  24 Batch   80/269 - Train Accuracy: 0.9448, Validation Accuracy: 0.9415, Loss: 0.0381
    Epoch  24 Batch   85/269 - Train Accuracy: 0.9385, Validation Accuracy: 0.9380, Loss: 0.0394
    Epoch  24 Batch   90/269 - Train Accuracy: 0.9446, Validation Accuracy: 0.9485, Loss: 0.0369
    Epoch  24 Batch   95/269 - Train Accuracy: 0.9540, Validation Accuracy: 0.9474, Loss: 0.0359
    Epoch  24 Batch  100/269 - Train Accuracy: 0.9434, Validation Accuracy: 0.9342, Loss: 0.0364
    Epoch  24 Batch  105/269 - Train Accuracy: 0.9481, Validation Accuracy: 0.9439, Loss: 0.0358
    Epoch  24 Batch  110/269 - Train Accuracy: 0.9475, Validation Accuracy: 0.9372, Loss: 0.0338
    Epoch  24 Batch  115/269 - Train Accuracy: 0.9473, Validation Accuracy: 0.9426, Loss: 0.0422
    Epoch  24 Batch  120/269 - Train Accuracy: 0.9447, Validation Accuracy: 0.9492, Loss: 0.0370
    Epoch  24 Batch  125/269 - Train Accuracy: 0.9466, Validation Accuracy: 0.9446, Loss: 0.0378
    Epoch  24 Batch  130/269 - Train Accuracy: 0.9535, Validation Accuracy: 0.9436, Loss: 0.0353
    Epoch  24 Batch  135/269 - Train Accuracy: 0.9509, Validation Accuracy: 0.9482, Loss: 0.0313
    Epoch  24 Batch  140/269 - Train Accuracy: 0.9452, Validation Accuracy: 0.9346, Loss: 0.0406
    Epoch  24 Batch  145/269 - Train Accuracy: 0.9465, Validation Accuracy: 0.9369, Loss: 0.0360
    Epoch  24 Batch  150/269 - Train Accuracy: 0.9507, Validation Accuracy: 0.9465, Loss: 0.0378
    Epoch  24 Batch  155/269 - Train Accuracy: 0.9490, Validation Accuracy: 0.9518, Loss: 0.0324
    Epoch  24 Batch  160/269 - Train Accuracy: 0.9587, Validation Accuracy: 0.9540, Loss: 0.0313
    Epoch  24 Batch  165/269 - Train Accuracy: 0.9538, Validation Accuracy: 0.9529, Loss: 0.0342
    Epoch  24 Batch  170/269 - Train Accuracy: 0.9444, Validation Accuracy: 0.9508, Loss: 0.0327
    Epoch  24 Batch  175/269 - Train Accuracy: 0.9588, Validation Accuracy: 0.9526, Loss: 0.0461
    Epoch  24 Batch  180/269 - Train Accuracy: 0.9463, Validation Accuracy: 0.9419, Loss: 0.0297
    Epoch  24 Batch  185/269 - Train Accuracy: 0.9588, Validation Accuracy: 0.9450, Loss: 0.0338
    Epoch  24 Batch  190/269 - Train Accuracy: 0.9476, Validation Accuracy: 0.9427, Loss: 0.0380
    Epoch  24 Batch  195/269 - Train Accuracy: 0.9381, Validation Accuracy: 0.9393, Loss: 0.0364
    Epoch  24 Batch  200/269 - Train Accuracy: 0.9551, Validation Accuracy: 0.9230, Loss: 0.0326
    Epoch  24 Batch  205/269 - Train Accuracy: 0.9510, Validation Accuracy: 0.9429, Loss: 0.0376
    Epoch  24 Batch  210/269 - Train Accuracy: 0.9448, Validation Accuracy: 0.9527, Loss: 0.0325
    Epoch  24 Batch  215/269 - Train Accuracy: 0.9313, Validation Accuracy: 0.9412, Loss: 0.0366
    Epoch  24 Batch  220/269 - Train Accuracy: 0.9386, Validation Accuracy: 0.9502, Loss: 0.0363
    Epoch  24 Batch  225/269 - Train Accuracy: 0.9358, Validation Accuracy: 0.9514, Loss: 0.0331
    Epoch  24 Batch  230/269 - Train Accuracy: 0.9506, Validation Accuracy: 0.9542, Loss: 0.0333
    Epoch  24 Batch  235/269 - Train Accuracy: 0.9726, Validation Accuracy: 0.9476, Loss: 0.0301
    Epoch  24 Batch  240/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9474, Loss: 0.0296
    Epoch  24 Batch  245/269 - Train Accuracy: 0.9611, Validation Accuracy: 0.9450, Loss: 0.0303
    Epoch  24 Batch  250/269 - Train Accuracy: 0.9450, Validation Accuracy: 0.9449, Loss: 0.0343
    Epoch  24 Batch  255/269 - Train Accuracy: 0.9538, Validation Accuracy: 0.9476, Loss: 0.0360
    Epoch  24 Batch  260/269 - Train Accuracy: 0.9414, Validation Accuracy: 0.9411, Loss: 0.0394
    Epoch  24 Batch  265/269 - Train Accuracy: 0.9458, Validation Accuracy: 0.9418, Loss: 0.0331
    Epoch  25 Batch    5/269 - Train Accuracy: 0.9587, Validation Accuracy: 0.9475, Loss: 0.0332
    Epoch  25 Batch   10/269 - Train Accuracy: 0.9553, Validation Accuracy: 0.9501, Loss: 0.0286
    Epoch  25 Batch   15/269 - Train Accuracy: 0.9500, Validation Accuracy: 0.9520, Loss: 0.0256
    Epoch  25 Batch   20/269 - Train Accuracy: 0.9512, Validation Accuracy: 0.9479, Loss: 0.0338
    Epoch  25 Batch   25/269 - Train Accuracy: 0.9519, Validation Accuracy: 0.9412, Loss: 0.0385
    Epoch  25 Batch   30/269 - Train Accuracy: 0.9527, Validation Accuracy: 0.9374, Loss: 0.0334
    Epoch  25 Batch   35/269 - Train Accuracy: 0.9465, Validation Accuracy: 0.9426, Loss: 0.0472
    Epoch  25 Batch   40/269 - Train Accuracy: 0.9493, Validation Accuracy: 0.9494, Loss: 0.0402
    Epoch  25 Batch   45/269 - Train Accuracy: 0.9553, Validation Accuracy: 0.9417, Loss: 0.0367
    Epoch  25 Batch   50/269 - Train Accuracy: 0.9344, Validation Accuracy: 0.9411, Loss: 0.0403
    Epoch  25 Batch   55/269 - Train Accuracy: 0.9617, Validation Accuracy: 0.9398, Loss: 0.0317
    Epoch  25 Batch   60/269 - Train Accuracy: 0.9556, Validation Accuracy: 0.9340, Loss: 0.0351
    Epoch  25 Batch   65/269 - Train Accuracy: 0.9562, Validation Accuracy: 0.9457, Loss: 0.0310
    Epoch  25 Batch   70/269 - Train Accuracy: 0.9534, Validation Accuracy: 0.9432, Loss: 0.0354
    Epoch  25 Batch   75/269 - Train Accuracy: 0.9501, Validation Accuracy: 0.9427, Loss: 0.0416
    Epoch  25 Batch   80/269 - Train Accuracy: 0.9475, Validation Accuracy: 0.9354, Loss: 0.0339
    Epoch  25 Batch   85/269 - Train Accuracy: 0.9422, Validation Accuracy: 0.9365, Loss: 0.0356
    Epoch  25 Batch   90/269 - Train Accuracy: 0.9383, Validation Accuracy: 0.9460, Loss: 0.0358
    Epoch  25 Batch   95/269 - Train Accuracy: 0.9576, Validation Accuracy: 0.9529, Loss: 0.0306
    Epoch  25 Batch  100/269 - Train Accuracy: 0.9475, Validation Accuracy: 0.9380, Loss: 0.0355
    Epoch  25 Batch  105/269 - Train Accuracy: 0.9536, Validation Accuracy: 0.9463, Loss: 0.0344
    Epoch  25 Batch  110/269 - Train Accuracy: 0.9476, Validation Accuracy: 0.9502, Loss: 0.0327
    Epoch  25 Batch  115/269 - Train Accuracy: 0.9470, Validation Accuracy: 0.9385, Loss: 0.0410
    Epoch  25 Batch  120/269 - Train Accuracy: 0.9551, Validation Accuracy: 0.9458, Loss: 0.0392
    Epoch  25 Batch  125/269 - Train Accuracy: 0.9482, Validation Accuracy: 0.9420, Loss: 0.0336
    Epoch  25 Batch  130/269 - Train Accuracy: 0.9541, Validation Accuracy: 0.9493, Loss: 0.0342
    Epoch  25 Batch  135/269 - Train Accuracy: 0.9457, Validation Accuracy: 0.9434, Loss: 0.0306
    Epoch  25 Batch  140/269 - Train Accuracy: 0.9448, Validation Accuracy: 0.9371, Loss: 0.0398
    Epoch  25 Batch  145/269 - Train Accuracy: 0.9457, Validation Accuracy: 0.9412, Loss: 0.0327
    Epoch  25 Batch  150/269 - Train Accuracy: 0.9462, Validation Accuracy: 0.9412, Loss: 0.0400
    Epoch  25 Batch  155/269 - Train Accuracy: 0.9553, Validation Accuracy: 0.9438, Loss: 0.0325
    Epoch  25 Batch  160/269 - Train Accuracy: 0.9579, Validation Accuracy: 0.9419, Loss: 0.0355
    Epoch  25 Batch  165/269 - Train Accuracy: 0.9524, Validation Accuracy: 0.9542, Loss: 0.0338
    Epoch  25 Batch  170/269 - Train Accuracy: 0.9501, Validation Accuracy: 0.9487, Loss: 0.0312
    Epoch  25 Batch  175/269 - Train Accuracy: 0.9472, Validation Accuracy: 0.9532, Loss: 0.0438
    Epoch  25 Batch  180/269 - Train Accuracy: 0.9488, Validation Accuracy: 0.9457, Loss: 0.0307
    Epoch  25 Batch  185/269 - Train Accuracy: 0.9611, Validation Accuracy: 0.9461, Loss: 0.0324
    Epoch  25 Batch  190/269 - Train Accuracy: 0.9446, Validation Accuracy: 0.9492, Loss: 0.0366
    Epoch  25 Batch  195/269 - Train Accuracy: 0.9329, Validation Accuracy: 0.9381, Loss: 0.0360
    Epoch  25 Batch  200/269 - Train Accuracy: 0.9566, Validation Accuracy: 0.9292, Loss: 0.0340
    Epoch  25 Batch  205/269 - Train Accuracy: 0.9539, Validation Accuracy: 0.9345, Loss: 0.0348
    Epoch  25 Batch  210/269 - Train Accuracy: 0.9449, Validation Accuracy: 0.9472, Loss: 0.0340
    Epoch  25 Batch  215/269 - Train Accuracy: 0.9353, Validation Accuracy: 0.9454, Loss: 0.0373
    Epoch  25 Batch  220/269 - Train Accuracy: 0.9428, Validation Accuracy: 0.9484, Loss: 0.0369
    Epoch  25 Batch  225/269 - Train Accuracy: 0.9358, Validation Accuracy: 0.9372, Loss: 0.0334
    Epoch  25 Batch  230/269 - Train Accuracy: 0.9501, Validation Accuracy: 0.9399, Loss: 0.0337
    Epoch  25 Batch  235/269 - Train Accuracy: 0.9760, Validation Accuracy: 0.9425, Loss: 0.0295
    Epoch  25 Batch  240/269 - Train Accuracy: 0.9464, Validation Accuracy: 0.9448, Loss: 0.0343
    Epoch  25 Batch  245/269 - Train Accuracy: 0.9556, Validation Accuracy: 0.9447, Loss: 0.0296
    Epoch  25 Batch  250/269 - Train Accuracy: 0.9435, Validation Accuracy: 0.9463, Loss: 0.0363
    Epoch  25 Batch  255/269 - Train Accuracy: 0.9474, Validation Accuracy: 0.9424, Loss: 0.0350
    Epoch  25 Batch  260/269 - Train Accuracy: 0.9405, Validation Accuracy: 0.9434, Loss: 0.0362
    Epoch  25 Batch  265/269 - Train Accuracy: 0.9464, Validation Accuracy: 0.9490, Loss: 0.0324
    Epoch  26 Batch    5/269 - Train Accuracy: 0.9600, Validation Accuracy: 0.9492, Loss: 0.0311
    Epoch  26 Batch   10/269 - Train Accuracy: 0.9608, Validation Accuracy: 0.9471, Loss: 0.0281
    Epoch  26 Batch   15/269 - Train Accuracy: 0.9468, Validation Accuracy: 0.9513, Loss: 0.0261
    Epoch  26 Batch   20/269 - Train Accuracy: 0.9513, Validation Accuracy: 0.9476, Loss: 0.0323
    Epoch  26 Batch   25/269 - Train Accuracy: 0.9407, Validation Accuracy: 0.9396, Loss: 0.0362
    Epoch  26 Batch   30/269 - Train Accuracy: 0.9573, Validation Accuracy: 0.9371, Loss: 0.0338
    Epoch  26 Batch   35/269 - Train Accuracy: 0.9474, Validation Accuracy: 0.9435, Loss: 0.0438
    Epoch  26 Batch   40/269 - Train Accuracy: 0.9574, Validation Accuracy: 0.9418, Loss: 0.0367
    Epoch  26 Batch   45/269 - Train Accuracy: 0.9491, Validation Accuracy: 0.9457, Loss: 0.0388
    Epoch  26 Batch   50/269 - Train Accuracy: 0.9316, Validation Accuracy: 0.9429, Loss: 0.0410
    Epoch  26 Batch   55/269 - Train Accuracy: 0.9634, Validation Accuracy: 0.9432, Loss: 0.0294
    Epoch  26 Batch   60/269 - Train Accuracy: 0.9497, Validation Accuracy: 0.9445, Loss: 0.0326
    Epoch  26 Batch   65/269 - Train Accuracy: 0.9570, Validation Accuracy: 0.9521, Loss: 0.0310
    Epoch  26 Batch   70/269 - Train Accuracy: 0.9552, Validation Accuracy: 0.9458, Loss: 0.0339
    Epoch  26 Batch   75/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9548, Loss: 0.0383
    Epoch  26 Batch   80/269 - Train Accuracy: 0.9554, Validation Accuracy: 0.9506, Loss: 0.0314
    Epoch  26 Batch   85/269 - Train Accuracy: 0.9330, Validation Accuracy: 0.9421, Loss: 0.0361
    Epoch  26 Batch   90/269 - Train Accuracy: 0.9471, Validation Accuracy: 0.9462, Loss: 0.0357
    Epoch  26 Batch   95/269 - Train Accuracy: 0.9495, Validation Accuracy: 0.9384, Loss: 0.0427
    Epoch  26 Batch  100/269 - Train Accuracy: 0.9297, Validation Accuracy: 0.9289, Loss: 0.0550
    Epoch  26 Batch  105/269 - Train Accuracy: 0.9273, Validation Accuracy: 0.9355, Loss: 0.0493
    Epoch  26 Batch  110/269 - Train Accuracy: 0.9156, Validation Accuracy: 0.9141, Loss: 0.0439
    Epoch  26 Batch  115/269 - Train Accuracy: 0.9315, Validation Accuracy: 0.9336, Loss: 0.0480
    Epoch  26 Batch  120/269 - Train Accuracy: 0.9365, Validation Accuracy: 0.9379, Loss: 0.0449
    Epoch  26 Batch  125/269 - Train Accuracy: 0.9535, Validation Accuracy: 0.9399, Loss: 0.0387
    Epoch  26 Batch  130/269 - Train Accuracy: 0.9563, Validation Accuracy: 0.9399, Loss: 0.0396
    Epoch  26 Batch  135/269 - Train Accuracy: 0.9396, Validation Accuracy: 0.9433, Loss: 0.0386
    Epoch  26 Batch  140/269 - Train Accuracy: 0.9448, Validation Accuracy: 0.9462, Loss: 0.0445
    Epoch  26 Batch  145/269 - Train Accuracy: 0.9433, Validation Accuracy: 0.9377, Loss: 0.0361
    Epoch  26 Batch  150/269 - Train Accuracy: 0.9422, Validation Accuracy: 0.9442, Loss: 0.0408
    Epoch  26 Batch  155/269 - Train Accuracy: 0.9497, Validation Accuracy: 0.9524, Loss: 0.0343
    Epoch  26 Batch  160/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9397, Loss: 0.0313
    Epoch  26 Batch  165/269 - Train Accuracy: 0.9461, Validation Accuracy: 0.9514, Loss: 0.0345
    Epoch  26 Batch  170/269 - Train Accuracy: 0.9421, Validation Accuracy: 0.9494, Loss: 0.0316
    Epoch  26 Batch  175/269 - Train Accuracy: 0.9495, Validation Accuracy: 0.9536, Loss: 0.0410
    Epoch  26 Batch  180/269 - Train Accuracy: 0.9502, Validation Accuracy: 0.9408, Loss: 0.0296
    Epoch  26 Batch  185/269 - Train Accuracy: 0.9539, Validation Accuracy: 0.9429, Loss: 0.0316
    Epoch  26 Batch  190/269 - Train Accuracy: 0.9532, Validation Accuracy: 0.9447, Loss: 0.0380
    Epoch  26 Batch  195/269 - Train Accuracy: 0.9389, Validation Accuracy: 0.9484, Loss: 0.0309
    Epoch  26 Batch  200/269 - Train Accuracy: 0.9582, Validation Accuracy: 0.9347, Loss: 0.0270
    Epoch  26 Batch  205/269 - Train Accuracy: 0.9519, Validation Accuracy: 0.9411, Loss: 0.0311
    Epoch  26 Batch  210/269 - Train Accuracy: 0.9493, Validation Accuracy: 0.9513, Loss: 0.0302
    Epoch  26 Batch  215/269 - Train Accuracy: 0.9390, Validation Accuracy: 0.9474, Loss: 0.0328
    Epoch  26 Batch  220/269 - Train Accuracy: 0.9443, Validation Accuracy: 0.9470, Loss: 0.0378
    Epoch  26 Batch  225/269 - Train Accuracy: 0.9388, Validation Accuracy: 0.9499, Loss: 0.0327
    Epoch  26 Batch  230/269 - Train Accuracy: 0.9552, Validation Accuracy: 0.9481, Loss: 0.0350
    Epoch  26 Batch  235/269 - Train Accuracy: 0.9764, Validation Accuracy: 0.9522, Loss: 0.0289
    Epoch  26 Batch  240/269 - Train Accuracy: 0.9602, Validation Accuracy: 0.9496, Loss: 0.0299
    Epoch  26 Batch  245/269 - Train Accuracy: 0.9594, Validation Accuracy: 0.9446, Loss: 0.0330
    Epoch  26 Batch  250/269 - Train Accuracy: 0.9437, Validation Accuracy: 0.9460, Loss: 0.0323
    Epoch  26 Batch  255/269 - Train Accuracy: 0.9561, Validation Accuracy: 0.9401, Loss: 0.0332
    Epoch  26 Batch  260/269 - Train Accuracy: 0.9511, Validation Accuracy: 0.9490, Loss: 0.0335
    Epoch  26 Batch  265/269 - Train Accuracy: 0.9437, Validation Accuracy: 0.9537, Loss: 0.0341
    Epoch  27 Batch    5/269 - Train Accuracy: 0.9567, Validation Accuracy: 0.9577, Loss: 0.0309
    Epoch  27 Batch   10/269 - Train Accuracy: 0.9539, Validation Accuracy: 0.9560, Loss: 0.0263
    Epoch  27 Batch   15/269 - Train Accuracy: 0.9501, Validation Accuracy: 0.9510, Loss: 0.0235
    Epoch  27 Batch   20/269 - Train Accuracy: 0.9443, Validation Accuracy: 0.9480, Loss: 0.0316
    Epoch  27 Batch   25/269 - Train Accuracy: 0.9421, Validation Accuracy: 0.9486, Loss: 0.0368
    Epoch  27 Batch   30/269 - Train Accuracy: 0.9557, Validation Accuracy: 0.9538, Loss: 0.0338
    Epoch  27 Batch   35/269 - Train Accuracy: 0.9457, Validation Accuracy: 0.9493, Loss: 0.0459
    Epoch  27 Batch   40/269 - Train Accuracy: 0.9503, Validation Accuracy: 0.9448, Loss: 0.0366
    Epoch  27 Batch   45/269 - Train Accuracy: 0.9520, Validation Accuracy: 0.9440, Loss: 0.0371
    Epoch  27 Batch   50/269 - Train Accuracy: 0.9343, Validation Accuracy: 0.9446, Loss: 0.0396
    Epoch  27 Batch   55/269 - Train Accuracy: 0.9552, Validation Accuracy: 0.9441, Loss: 0.0305
    Epoch  27 Batch   60/269 - Train Accuracy: 0.9498, Validation Accuracy: 0.9470, Loss: 0.0308
    Epoch  27 Batch   65/269 - Train Accuracy: 0.9577, Validation Accuracy: 0.9548, Loss: 0.0305
    Epoch  27 Batch   70/269 - Train Accuracy: 0.9543, Validation Accuracy: 0.9450, Loss: 0.0329
    Epoch  27 Batch   75/269 - Train Accuracy: 0.9487, Validation Accuracy: 0.9507, Loss: 0.0381
    Epoch  27 Batch   80/269 - Train Accuracy: 0.9491, Validation Accuracy: 0.9416, Loss: 0.0321
    Epoch  27 Batch   85/269 - Train Accuracy: 0.9441, Validation Accuracy: 0.9465, Loss: 0.0349
    Epoch  27 Batch   90/269 - Train Accuracy: 0.9402, Validation Accuracy: 0.9398, Loss: 0.0345
    Epoch  27 Batch   95/269 - Train Accuracy: 0.9515, Validation Accuracy: 0.9445, Loss: 0.0332
    Epoch  27 Batch  100/269 - Train Accuracy: 0.9480, Validation Accuracy: 0.9419, Loss: 0.0353
    Epoch  27 Batch  105/269 - Train Accuracy: 0.9486, Validation Accuracy: 0.9471, Loss: 0.0310
    Epoch  27 Batch  110/269 - Train Accuracy: 0.9480, Validation Accuracy: 0.9542, Loss: 0.0338
    Epoch  27 Batch  115/269 - Train Accuracy: 0.9612, Validation Accuracy: 0.9435, Loss: 0.0373
    Epoch  27 Batch  120/269 - Train Accuracy: 0.9449, Validation Accuracy: 0.9512, Loss: 0.0348
    Epoch  27 Batch  125/269 - Train Accuracy: 0.9551, Validation Accuracy: 0.9500, Loss: 0.0331
    Epoch  27 Batch  130/269 - Train Accuracy: 0.9581, Validation Accuracy: 0.9535, Loss: 0.0328
    Epoch  27 Batch  135/269 - Train Accuracy: 0.9443, Validation Accuracy: 0.9428, Loss: 0.0302
    Epoch  27 Batch  140/269 - Train Accuracy: 0.9530, Validation Accuracy: 0.9450, Loss: 0.0396
    Epoch  27 Batch  145/269 - Train Accuracy: 0.9542, Validation Accuracy: 0.9391, Loss: 0.0329
    Epoch  27 Batch  150/269 - Train Accuracy: 0.9543, Validation Accuracy: 0.9496, Loss: 0.0359
    Epoch  27 Batch  155/269 - Train Accuracy: 0.9577, Validation Accuracy: 0.9494, Loss: 0.0313
    Epoch  27 Batch  160/269 - Train Accuracy: 0.9558, Validation Accuracy: 0.9514, Loss: 0.0286
    Epoch  27 Batch  165/269 - Train Accuracy: 0.9440, Validation Accuracy: 0.9545, Loss: 0.0343
    Epoch  27 Batch  170/269 - Train Accuracy: 0.9496, Validation Accuracy: 0.9522, Loss: 0.0318
    Epoch  27 Batch  175/269 - Train Accuracy: 0.9515, Validation Accuracy: 0.9538, Loss: 0.0409
    Epoch  27 Batch  180/269 - Train Accuracy: 0.9528, Validation Accuracy: 0.9393, Loss: 0.0271
    Epoch  27 Batch  185/269 - Train Accuracy: 0.9653, Validation Accuracy: 0.9588, Loss: 0.0311
    Epoch  27 Batch  190/269 - Train Accuracy: 0.9519, Validation Accuracy: 0.9462, Loss: 0.0350
    Epoch  27 Batch  195/269 - Train Accuracy: 0.9435, Validation Accuracy: 0.9495, Loss: 0.0330
    Epoch  27 Batch  200/269 - Train Accuracy: 0.9604, Validation Accuracy: 0.9433, Loss: 0.0295
    Epoch  27 Batch  205/269 - Train Accuracy: 0.9610, Validation Accuracy: 0.9392, Loss: 0.0289
    Epoch  27 Batch  210/269 - Train Accuracy: 0.9506, Validation Accuracy: 0.9499, Loss: 0.0325
    Epoch  27 Batch  215/269 - Train Accuracy: 0.9478, Validation Accuracy: 0.9471, Loss: 0.0315
    Epoch  27 Batch  220/269 - Train Accuracy: 0.9426, Validation Accuracy: 0.9555, Loss: 0.0360
    Epoch  27 Batch  225/269 - Train Accuracy: 0.9457, Validation Accuracy: 0.9542, Loss: 0.0316
    Epoch  27 Batch  230/269 - Train Accuracy: 0.9502, Validation Accuracy: 0.9486, Loss: 0.0305
    Epoch  27 Batch  235/269 - Train Accuracy: 0.9762, Validation Accuracy: 0.9500, Loss: 0.0275
    Epoch  27 Batch  240/269 - Train Accuracy: 0.9477, Validation Accuracy: 0.9497, Loss: 0.0288
    Epoch  27 Batch  245/269 - Train Accuracy: 0.9631, Validation Accuracy: 0.9466, Loss: 0.0294
    Epoch  27 Batch  250/269 - Train Accuracy: 0.9386, Validation Accuracy: 0.9475, Loss: 0.0295
    Epoch  27 Batch  255/269 - Train Accuracy: 0.9561, Validation Accuracy: 0.9436, Loss: 0.0314
    Epoch  27 Batch  260/269 - Train Accuracy: 0.9490, Validation Accuracy: 0.9458, Loss: 0.0328
    Epoch  27 Batch  265/269 - Train Accuracy: 0.9449, Validation Accuracy: 0.9534, Loss: 0.0313
    Epoch  28 Batch    5/269 - Train Accuracy: 0.9570, Validation Accuracy: 0.9486, Loss: 0.0307
    Epoch  28 Batch   10/269 - Train Accuracy: 0.9535, Validation Accuracy: 0.9576, Loss: 0.0266
    Epoch  28 Batch   15/269 - Train Accuracy: 0.9534, Validation Accuracy: 0.9569, Loss: 0.0229
    Epoch  28 Batch   20/269 - Train Accuracy: 0.9564, Validation Accuracy: 0.9517, Loss: 0.0296
    Epoch  28 Batch   25/269 - Train Accuracy: 0.9463, Validation Accuracy: 0.9493, Loss: 0.0342
    Epoch  28 Batch   30/269 - Train Accuracy: 0.9580, Validation Accuracy: 0.9431, Loss: 0.0288
    Epoch  28 Batch   35/269 - Train Accuracy: 0.9554, Validation Accuracy: 0.9498, Loss: 0.0406
    Epoch  28 Batch   40/269 - Train Accuracy: 0.9454, Validation Accuracy: 0.9429, Loss: 0.0351
    Epoch  28 Batch   45/269 - Train Accuracy: 0.9511, Validation Accuracy: 0.9506, Loss: 0.0361
    Epoch  28 Batch   50/269 - Train Accuracy: 0.9376, Validation Accuracy: 0.9451, Loss: 0.0362
    Epoch  28 Batch   55/269 - Train Accuracy: 0.9580, Validation Accuracy: 0.9419, Loss: 0.0291
    Epoch  28 Batch   60/269 - Train Accuracy: 0.9540, Validation Accuracy: 0.9491, Loss: 0.0304
    Epoch  28 Batch   65/269 - Train Accuracy: 0.9611, Validation Accuracy: 0.9557, Loss: 0.0277
    Epoch  28 Batch   70/269 - Train Accuracy: 0.9565, Validation Accuracy: 0.9510, Loss: 0.0309
    Epoch  28 Batch   75/269 - Train Accuracy: 0.9579, Validation Accuracy: 0.9525, Loss: 0.0348
    Epoch  28 Batch   80/269 - Train Accuracy: 0.9536, Validation Accuracy: 0.9482, Loss: 0.0319
    Epoch  28 Batch   85/269 - Train Accuracy: 0.9392, Validation Accuracy: 0.9491, Loss: 0.0332
    Epoch  28 Batch   90/269 - Train Accuracy: 0.9431, Validation Accuracy: 0.9433, Loss: 0.0331
    Epoch  28 Batch   95/269 - Train Accuracy: 0.9480, Validation Accuracy: 0.9531, Loss: 0.0318
    Epoch  28 Batch  100/269 - Train Accuracy: 0.9501, Validation Accuracy: 0.9354, Loss: 0.0349
    Epoch  28 Batch  105/269 - Train Accuracy: 0.9447, Validation Accuracy: 0.9434, Loss: 0.0328
    Epoch  28 Batch  110/269 - Train Accuracy: 0.9502, Validation Accuracy: 0.9442, Loss: 0.0325
    Epoch  28 Batch  115/269 - Train Accuracy: 0.9561, Validation Accuracy: 0.9518, Loss: 0.0378
    Epoch  28 Batch  120/269 - Train Accuracy: 0.9439, Validation Accuracy: 0.9498, Loss: 0.0328
    Epoch  28 Batch  125/269 - Train Accuracy: 0.9501, Validation Accuracy: 0.9420, Loss: 0.0323
    Epoch  28 Batch  130/269 - Train Accuracy: 0.9560, Validation Accuracy: 0.9499, Loss: 0.0348
    Epoch  28 Batch  135/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9516, Loss: 0.0305
    Epoch  28 Batch  140/269 - Train Accuracy: 0.9454, Validation Accuracy: 0.9389, Loss: 0.0379
    Epoch  28 Batch  145/269 - Train Accuracy: 0.9493, Validation Accuracy: 0.9387, Loss: 0.0312
    Epoch  28 Batch  150/269 - Train Accuracy: 0.9502, Validation Accuracy: 0.9463, Loss: 0.0373
    Epoch  28 Batch  155/269 - Train Accuracy: 0.9497, Validation Accuracy: 0.9495, Loss: 0.0305
    Epoch  28 Batch  160/269 - Train Accuracy: 0.9619, Validation Accuracy: 0.9450, Loss: 0.0323
    Epoch  28 Batch  165/269 - Train Accuracy: 0.9534, Validation Accuracy: 0.9517, Loss: 0.0320
    Epoch  28 Batch  170/269 - Train Accuracy: 0.9465, Validation Accuracy: 0.9506, Loss: 0.0307
    Epoch  28 Batch  175/269 - Train Accuracy: 0.9481, Validation Accuracy: 0.9521, Loss: 0.0400
    Epoch  28 Batch  180/269 - Train Accuracy: 0.9488, Validation Accuracy: 0.9395, Loss: 0.0271
    Epoch  28 Batch  185/269 - Train Accuracy: 0.9566, Validation Accuracy: 0.9504, Loss: 0.0287
    Epoch  28 Batch  190/269 - Train Accuracy: 0.9540, Validation Accuracy: 0.9411, Loss: 0.0337
    Epoch  28 Batch  195/269 - Train Accuracy: 0.9367, Validation Accuracy: 0.9490, Loss: 0.0313
    Epoch  28 Batch  200/269 - Train Accuracy: 0.9605, Validation Accuracy: 0.9425, Loss: 0.0299
    Epoch  28 Batch  205/269 - Train Accuracy: 0.9533, Validation Accuracy: 0.9349, Loss: 0.0329
    Epoch  28 Batch  210/269 - Train Accuracy: 0.9484, Validation Accuracy: 0.9439, Loss: 0.0338
    Epoch  28 Batch  215/269 - Train Accuracy: 0.9406, Validation Accuracy: 0.9355, Loss: 0.0305
    Epoch  28 Batch  220/269 - Train Accuracy: 0.9408, Validation Accuracy: 0.9395, Loss: 0.0360
    Epoch  28 Batch  225/269 - Train Accuracy: 0.9379, Validation Accuracy: 0.9408, Loss: 0.0337
    Epoch  28 Batch  230/269 - Train Accuracy: 0.9540, Validation Accuracy: 0.9491, Loss: 0.0310
    Epoch  28 Batch  235/269 - Train Accuracy: 0.9780, Validation Accuracy: 0.9497, Loss: 0.0263
    Epoch  28 Batch  240/269 - Train Accuracy: 0.9500, Validation Accuracy: 0.9409, Loss: 0.0306
    Epoch  28 Batch  245/269 - Train Accuracy: 0.9555, Validation Accuracy: 0.9478, Loss: 0.0341
    Epoch  28 Batch  250/269 - Train Accuracy: 0.9396, Validation Accuracy: 0.9439, Loss: 0.0331
    Epoch  28 Batch  255/269 - Train Accuracy: 0.9357, Validation Accuracy: 0.9387, Loss: 0.0446
    Epoch  28 Batch  260/269 - Train Accuracy: 0.9484, Validation Accuracy: 0.9450, Loss: 0.0405
    Epoch  28 Batch  265/269 - Train Accuracy: 0.9449, Validation Accuracy: 0.9430, Loss: 0.0381
    Epoch  29 Batch    5/269 - Train Accuracy: 0.9554, Validation Accuracy: 0.9504, Loss: 0.0373
    Epoch  29 Batch   10/269 - Train Accuracy: 0.9525, Validation Accuracy: 0.9471, Loss: 0.0305
    Epoch  29 Batch   15/269 - Train Accuracy: 0.9547, Validation Accuracy: 0.9464, Loss: 0.0280
    Epoch  29 Batch   20/269 - Train Accuracy: 0.9558, Validation Accuracy: 0.9490, Loss: 0.0327
    Epoch  29 Batch   25/269 - Train Accuracy: 0.9543, Validation Accuracy: 0.9496, Loss: 0.0373
    Epoch  29 Batch   30/269 - Train Accuracy: 0.9508, Validation Accuracy: 0.9554, Loss: 0.0378
    Epoch  29 Batch   35/269 - Train Accuracy: 0.9538, Validation Accuracy: 0.9497, Loss: 0.0466
    Epoch  29 Batch   40/269 - Train Accuracy: 0.9517, Validation Accuracy: 0.9547, Loss: 0.0362
    Epoch  29 Batch   45/269 - Train Accuracy: 0.9520, Validation Accuracy: 0.9475, Loss: 0.0341
    Epoch  29 Batch   50/269 - Train Accuracy: 0.9317, Validation Accuracy: 0.9434, Loss: 0.0395
    Epoch  29 Batch   55/269 - Train Accuracy: 0.9555, Validation Accuracy: 0.9395, Loss: 0.0296
    Epoch  29 Batch   60/269 - Train Accuracy: 0.9522, Validation Accuracy: 0.9436, Loss: 0.0295
    Epoch  29 Batch   65/269 - Train Accuracy: 0.9540, Validation Accuracy: 0.9452, Loss: 0.0287
    Epoch  29 Batch   70/269 - Train Accuracy: 0.9498, Validation Accuracy: 0.9429, Loss: 0.0338
    Epoch  29 Batch   75/269 - Train Accuracy: 0.9475, Validation Accuracy: 0.9466, Loss: 0.0396
    Epoch  29 Batch   80/269 - Train Accuracy: 0.9533, Validation Accuracy: 0.9455, Loss: 0.0302
    Epoch  29 Batch   85/269 - Train Accuracy: 0.9408, Validation Accuracy: 0.9445, Loss: 0.0336
    Epoch  29 Batch   90/269 - Train Accuracy: 0.9507, Validation Accuracy: 0.9506, Loss: 0.0330
    Epoch  29 Batch   95/269 - Train Accuracy: 0.9566, Validation Accuracy: 0.9513, Loss: 0.0297
    Epoch  29 Batch  100/269 - Train Accuracy: 0.9530, Validation Accuracy: 0.9464, Loss: 0.0329
    Epoch  29 Batch  105/269 - Train Accuracy: 0.9566, Validation Accuracy: 0.9442, Loss: 0.0305
    Epoch  29 Batch  110/269 - Train Accuracy: 0.9462, Validation Accuracy: 0.9556, Loss: 0.0318
    Epoch  29 Batch  115/269 - Train Accuracy: 0.9554, Validation Accuracy: 0.9447, Loss: 0.0375
    Epoch  29 Batch  120/269 - Train Accuracy: 0.9498, Validation Accuracy: 0.9476, Loss: 0.0309
    Epoch  29 Batch  125/269 - Train Accuracy: 0.9573, Validation Accuracy: 0.9521, Loss: 0.0308
    Epoch  29 Batch  130/269 - Train Accuracy: 0.9552, Validation Accuracy: 0.9479, Loss: 0.0328
    Epoch  29 Batch  135/269 - Train Accuracy: 0.9449, Validation Accuracy: 0.9560, Loss: 0.0286
    Epoch  29 Batch  140/269 - Train Accuracy: 0.9465, Validation Accuracy: 0.9423, Loss: 0.0371
    Epoch  29 Batch  145/269 - Train Accuracy: 0.9554, Validation Accuracy: 0.9467, Loss: 0.0289
    Epoch  29 Batch  150/269 - Train Accuracy: 0.9524, Validation Accuracy: 0.9602, Loss: 0.0350
    Epoch  29 Batch  155/269 - Train Accuracy: 0.9489, Validation Accuracy: 0.9601, Loss: 0.0271
    Epoch  29 Batch  160/269 - Train Accuracy: 0.9667, Validation Accuracy: 0.9530, Loss: 0.0287
    Epoch  29 Batch  165/269 - Train Accuracy: 0.9555, Validation Accuracy: 0.9561, Loss: 0.0290
    Epoch  29 Batch  170/269 - Train Accuracy: 0.9521, Validation Accuracy: 0.9487, Loss: 0.0301
    Epoch  29 Batch  175/269 - Train Accuracy: 0.9521, Validation Accuracy: 0.9537, Loss: 0.0391
    Epoch  29 Batch  180/269 - Train Accuracy: 0.9545, Validation Accuracy: 0.9517, Loss: 0.0283
    Epoch  29 Batch  185/269 - Train Accuracy: 0.9559, Validation Accuracy: 0.9513, Loss: 0.0307
    Epoch  29 Batch  190/269 - Train Accuracy: 0.9484, Validation Accuracy: 0.9548, Loss: 0.0346
    Epoch  29 Batch  195/269 - Train Accuracy: 0.9379, Validation Accuracy: 0.9540, Loss: 0.0333
    Epoch  29 Batch  200/269 - Train Accuracy: 0.9129, Validation Accuracy: 0.9094, Loss: 0.0788
    Epoch  29 Batch  205/269 - Train Accuracy: 0.9119, Validation Accuracy: 0.8811, Loss: 0.1763
    Epoch  29 Batch  210/269 - Train Accuracy: 0.8837, Validation Accuracy: 0.8778, Loss: 0.0909
    Epoch  29 Batch  215/269 - Train Accuracy: 0.8909, Validation Accuracy: 0.8784, Loss: 0.0886
    Epoch  29 Batch  220/269 - Train Accuracy: 0.9033, Validation Accuracy: 0.9023, Loss: 0.0691
    Epoch  29 Batch  225/269 - Train Accuracy: 0.9148, Validation Accuracy: 0.8966, Loss: 0.0571
    Epoch  29 Batch  230/269 - Train Accuracy: 0.9227, Validation Accuracy: 0.9272, Loss: 0.0458
    Epoch  29 Batch  235/269 - Train Accuracy: 0.9575, Validation Accuracy: 0.9330, Loss: 0.0433
    Epoch  29 Batch  240/269 - Train Accuracy: 0.9355, Validation Accuracy: 0.9312, Loss: 0.0438
    Epoch  29 Batch  245/269 - Train Accuracy: 0.9432, Validation Accuracy: 0.9374, Loss: 0.0394
    Epoch  29 Batch  250/269 - Train Accuracy: 0.9346, Validation Accuracy: 0.9436, Loss: 0.0398
    Epoch  29 Batch  255/269 - Train Accuracy: 0.9521, Validation Accuracy: 0.9412, Loss: 0.0409
    Epoch  29 Batch  260/269 - Train Accuracy: 0.9440, Validation Accuracy: 0.9442, Loss: 0.0400
    Epoch  29 Batch  265/269 - Train Accuracy: 0.9453, Validation Accuracy: 0.9442, Loss: 0.0371
    Epoch  30 Batch    5/269 - Train Accuracy: 0.9513, Validation Accuracy: 0.9490, Loss: 0.0323
    Epoch  30 Batch   10/269 - Train Accuracy: 0.9427, Validation Accuracy: 0.9566, Loss: 0.0313
    Epoch  30 Batch   15/269 - Train Accuracy: 0.9492, Validation Accuracy: 0.9522, Loss: 0.0245
    Epoch  30 Batch   20/269 - Train Accuracy: 0.9524, Validation Accuracy: 0.9399, Loss: 0.0328
    Epoch  30 Batch   25/269 - Train Accuracy: 0.9412, Validation Accuracy: 0.9459, Loss: 0.0395
    Epoch  30 Batch   30/269 - Train Accuracy: 0.9513, Validation Accuracy: 0.9426, Loss: 0.0340
    Epoch  30 Batch   35/269 - Train Accuracy: 0.9424, Validation Accuracy: 0.9486, Loss: 0.0453
    Epoch  30 Batch   40/269 - Train Accuracy: 0.9414, Validation Accuracy: 0.9502, Loss: 0.0345
    Epoch  30 Batch   45/269 - Train Accuracy: 0.9468, Validation Accuracy: 0.9464, Loss: 0.0354
    Epoch  30 Batch   50/269 - Train Accuracy: 0.9360, Validation Accuracy: 0.9402, Loss: 0.0357
    Epoch  30 Batch   55/269 - Train Accuracy: 0.9542, Validation Accuracy: 0.9480, Loss: 0.0293
    Epoch  30 Batch   60/269 - Train Accuracy: 0.9514, Validation Accuracy: 0.9489, Loss: 0.0300
    Epoch  30 Batch   65/269 - Train Accuracy: 0.9555, Validation Accuracy: 0.9501, Loss: 0.0283
    Epoch  30 Batch   70/269 - Train Accuracy: 0.9473, Validation Accuracy: 0.9480, Loss: 0.0321
    Epoch  30 Batch   75/269 - Train Accuracy: 0.9495, Validation Accuracy: 0.9446, Loss: 0.0360
    Epoch  30 Batch   80/269 - Train Accuracy: 0.9455, Validation Accuracy: 0.9471, Loss: 0.0297
    Epoch  30 Batch   85/269 - Train Accuracy: 0.9374, Validation Accuracy: 0.9468, Loss: 0.0295
    Epoch  30 Batch   90/269 - Train Accuracy: 0.9357, Validation Accuracy: 0.9515, Loss: 0.0311
    Epoch  30 Batch   95/269 - Train Accuracy: 0.9585, Validation Accuracy: 0.9580, Loss: 0.0287
    Epoch  30 Batch  100/269 - Train Accuracy: 0.9493, Validation Accuracy: 0.9549, Loss: 0.0316
    Epoch  30 Batch  105/269 - Train Accuracy: 0.9546, Validation Accuracy: 0.9514, Loss: 0.0292
    Epoch  30 Batch  110/269 - Train Accuracy: 0.9526, Validation Accuracy: 0.9584, Loss: 0.0291
    Epoch  30 Batch  115/269 - Train Accuracy: 0.9491, Validation Accuracy: 0.9510, Loss: 0.0349
    Epoch  30 Batch  120/269 - Train Accuracy: 0.9516, Validation Accuracy: 0.9545, Loss: 0.0312
    Epoch  30 Batch  125/269 - Train Accuracy: 0.9552, Validation Accuracy: 0.9520, Loss: 0.0283
    Epoch  30 Batch  130/269 - Train Accuracy: 0.9519, Validation Accuracy: 0.9554, Loss: 0.0293
    Epoch  30 Batch  135/269 - Train Accuracy: 0.9478, Validation Accuracy: 0.9504, Loss: 0.0278
    Epoch  30 Batch  140/269 - Train Accuracy: 0.9533, Validation Accuracy: 0.9450, Loss: 0.0323
    Epoch  30 Batch  145/269 - Train Accuracy: 0.9539, Validation Accuracy: 0.9551, Loss: 0.0279
    Epoch  30 Batch  150/269 - Train Accuracy: 0.9515, Validation Accuracy: 0.9541, Loss: 0.0327
    Epoch  30 Batch  155/269 - Train Accuracy: 0.9565, Validation Accuracy: 0.9577, Loss: 0.0282
    Epoch  30 Batch  160/269 - Train Accuracy: 0.9622, Validation Accuracy: 0.9493, Loss: 0.0257
    Epoch  30 Batch  165/269 - Train Accuracy: 0.9563, Validation Accuracy: 0.9529, Loss: 0.0279
    Epoch  30 Batch  170/269 - Train Accuracy: 0.9485, Validation Accuracy: 0.9554, Loss: 0.0279
    Epoch  30 Batch  175/269 - Train Accuracy: 0.9516, Validation Accuracy: 0.9592, Loss: 0.0381
    Epoch  30 Batch  180/269 - Train Accuracy: 0.9631, Validation Accuracy: 0.9558, Loss: 0.0270
    Epoch  30 Batch  185/269 - Train Accuracy: 0.9559, Validation Accuracy: 0.9480, Loss: 0.0269
    Epoch  30 Batch  190/269 - Train Accuracy: 0.9615, Validation Accuracy: 0.9577, Loss: 0.0307
    Epoch  30 Batch  195/269 - Train Accuracy: 0.9466, Validation Accuracy: 0.9596, Loss: 0.0278
    Epoch  30 Batch  200/269 - Train Accuracy: 0.9636, Validation Accuracy: 0.9509, Loss: 0.0245
    Epoch  30 Batch  205/269 - Train Accuracy: 0.9594, Validation Accuracy: 0.9415, Loss: 0.0264
    Epoch  30 Batch  210/269 - Train Accuracy: 0.9482, Validation Accuracy: 0.9495, Loss: 0.0278
    Epoch  30 Batch  215/269 - Train Accuracy: 0.9486, Validation Accuracy: 0.9455, Loss: 0.0284
    Epoch  30 Batch  220/269 - Train Accuracy: 0.9416, Validation Accuracy: 0.9455, Loss: 0.0329
    Epoch  30 Batch  225/269 - Train Accuracy: 0.9434, Validation Accuracy: 0.9589, Loss: 0.0296
    Epoch  30 Batch  230/269 - Train Accuracy: 0.9534, Validation Accuracy: 0.9531, Loss: 0.0286
    Epoch  30 Batch  235/269 - Train Accuracy: 0.9742, Validation Accuracy: 0.9478, Loss: 0.0236
    Epoch  30 Batch  240/269 - Train Accuracy: 0.9510, Validation Accuracy: 0.9538, Loss: 0.0271
    Epoch  30 Batch  245/269 - Train Accuracy: 0.9558, Validation Accuracy: 0.9454, Loss: 0.0261
    Epoch  30 Batch  250/269 - Train Accuracy: 0.9522, Validation Accuracy: 0.9554, Loss: 0.0287
    Epoch  30 Batch  255/269 - Train Accuracy: 0.9590, Validation Accuracy: 0.9548, Loss: 0.0296
    Epoch  30 Batch  260/269 - Train Accuracy: 0.9571, Validation Accuracy: 0.9474, Loss: 0.0286
    Epoch  30 Batch  265/269 - Train Accuracy: 0.9485, Validation Accuracy: 0.9545, Loss: 0.0308
    Epoch  31 Batch    5/269 - Train Accuracy: 0.9557, Validation Accuracy: 0.9582, Loss: 0.0305
    Epoch  31 Batch   10/269 - Train Accuracy: 0.9462, Validation Accuracy: 0.9577, Loss: 0.0239
    Epoch  31 Batch   15/269 - Train Accuracy: 0.9560, Validation Accuracy: 0.9548, Loss: 0.0207
    Epoch  31 Batch   20/269 - Train Accuracy: 0.9582, Validation Accuracy: 0.9513, Loss: 0.0282
    Epoch  31 Batch   25/269 - Train Accuracy: 0.9397, Validation Accuracy: 0.9519, Loss: 0.0333
    Epoch  31 Batch   30/269 - Train Accuracy: 0.9477, Validation Accuracy: 0.9534, Loss: 0.0287
    Epoch  31 Batch   35/269 - Train Accuracy: 0.9558, Validation Accuracy: 0.9511, Loss: 0.0409
    Epoch  31 Batch   40/269 - Train Accuracy: 0.9481, Validation Accuracy: 0.9490, Loss: 0.0340
    Epoch  31 Batch   45/269 - Train Accuracy: 0.9476, Validation Accuracy: 0.9506, Loss: 0.0343
    Epoch  31 Batch   50/269 - Train Accuracy: 0.9410, Validation Accuracy: 0.9492, Loss: 0.0331
    Epoch  31 Batch   55/269 - Train Accuracy: 0.9585, Validation Accuracy: 0.9481, Loss: 0.0277
    Epoch  31 Batch   60/269 - Train Accuracy: 0.9451, Validation Accuracy: 0.9478, Loss: 0.0300
    Epoch  31 Batch   65/269 - Train Accuracy: 0.9593, Validation Accuracy: 0.9506, Loss: 0.0261
    Epoch  31 Batch   70/269 - Train Accuracy: 0.9576, Validation Accuracy: 0.9523, Loss: 0.0311
    Epoch  31 Batch   75/269 - Train Accuracy: 0.9564, Validation Accuracy: 0.9584, Loss: 0.0337
    Epoch  31 Batch   80/269 - Train Accuracy: 0.9570, Validation Accuracy: 0.9453, Loss: 0.0279
    Epoch  31 Batch   85/269 - Train Accuracy: 0.9382, Validation Accuracy: 0.9499, Loss: 0.0298
    Epoch  31 Batch   90/269 - Train Accuracy: 0.9379, Validation Accuracy: 0.9548, Loss: 0.0316
    Epoch  31 Batch   95/269 - Train Accuracy: 0.9597, Validation Accuracy: 0.9587, Loss: 0.0274
    Epoch  31 Batch  100/269 - Train Accuracy: 0.9527, Validation Accuracy: 0.9522, Loss: 0.0288
    Epoch  31 Batch  105/269 - Train Accuracy: 0.9513, Validation Accuracy: 0.9539, Loss: 0.0250
    Epoch  31 Batch  110/269 - Train Accuracy: 0.9555, Validation Accuracy: 0.9601, Loss: 0.0290
    Epoch  31 Batch  115/269 - Train Accuracy: 0.9503, Validation Accuracy: 0.9541, Loss: 0.0312
    Epoch  31 Batch  120/269 - Train Accuracy: 0.9509, Validation Accuracy: 0.9482, Loss: 0.0283
    Epoch  31 Batch  125/269 - Train Accuracy: 0.9574, Validation Accuracy: 0.9469, Loss: 0.0280
    Epoch  31 Batch  130/269 - Train Accuracy: 0.9530, Validation Accuracy: 0.9520, Loss: 0.0291
    Epoch  31 Batch  135/269 - Train Accuracy: 0.9566, Validation Accuracy: 0.9598, Loss: 0.0255
    Epoch  31 Batch  140/269 - Train Accuracy: 0.9575, Validation Accuracy: 0.9575, Loss: 0.0312
    Epoch  31 Batch  145/269 - Train Accuracy: 0.9551, Validation Accuracy: 0.9556, Loss: 0.0273
    Epoch  31 Batch  150/269 - Train Accuracy: 0.9534, Validation Accuracy: 0.9483, Loss: 0.0322
    Epoch  31 Batch  155/269 - Train Accuracy: 0.9606, Validation Accuracy: 0.9586, Loss: 0.0264
    Epoch  31 Batch  160/269 - Train Accuracy: 0.9655, Validation Accuracy: 0.9506, Loss: 0.0264
    Epoch  31 Batch  165/269 - Train Accuracy: 0.9585, Validation Accuracy: 0.9551, Loss: 0.0261
    Epoch  31 Batch  170/269 - Train Accuracy: 0.9485, Validation Accuracy: 0.9566, Loss: 0.0267
    Epoch  31 Batch  175/269 - Train Accuracy: 0.9564, Validation Accuracy: 0.9582, Loss: 0.0351
    Epoch  31 Batch  180/269 - Train Accuracy: 0.9603, Validation Accuracy: 0.9573, Loss: 0.0253
    Epoch  31 Batch  185/269 - Train Accuracy: 0.9633, Validation Accuracy: 0.9525, Loss: 0.0252
    Epoch  31 Batch  190/269 - Train Accuracy: 0.9586, Validation Accuracy: 0.9545, Loss: 0.0315
    Epoch  31 Batch  195/269 - Train Accuracy: 0.9488, Validation Accuracy: 0.9612, Loss: 0.0270
    Epoch  31 Batch  200/269 - Train Accuracy: 0.9662, Validation Accuracy: 0.9575, Loss: 0.0235
    Epoch  31 Batch  205/269 - Train Accuracy: 0.9572, Validation Accuracy: 0.9409, Loss: 0.0273
    Epoch  31 Batch  210/269 - Train Accuracy: 0.9519, Validation Accuracy: 0.9505, Loss: 0.0264
    Epoch  31 Batch  215/269 - Train Accuracy: 0.9502, Validation Accuracy: 0.9449, Loss: 0.0287
    Epoch  31 Batch  220/269 - Train Accuracy: 0.9474, Validation Accuracy: 0.9520, Loss: 0.0337
    Epoch  31 Batch  225/269 - Train Accuracy: 0.9468, Validation Accuracy: 0.9561, Loss: 0.0278
    Epoch  31 Batch  230/269 - Train Accuracy: 0.9562, Validation Accuracy: 0.9485, Loss: 0.0273
    Epoch  31 Batch  235/269 - Train Accuracy: 0.9812, Validation Accuracy: 0.9506, Loss: 0.0214
    Epoch  31 Batch  240/269 - Train Accuracy: 0.9528, Validation Accuracy: 0.9528, Loss: 0.0281
    Epoch  31 Batch  245/269 - Train Accuracy: 0.9541, Validation Accuracy: 0.9476, Loss: 0.0268
    Epoch  31 Batch  250/269 - Train Accuracy: 0.9483, Validation Accuracy: 0.9530, Loss: 0.0267
    Epoch  31 Batch  255/269 - Train Accuracy: 0.9546, Validation Accuracy: 0.9411, Loss: 0.0306
    Epoch  31 Batch  260/269 - Train Accuracy: 0.9592, Validation Accuracy: 0.9542, Loss: 0.0305
    Epoch  31 Batch  265/269 - Train Accuracy: 0.9498, Validation Accuracy: 0.9496, Loss: 0.0302
    Epoch  32 Batch    5/269 - Train Accuracy: 0.9537, Validation Accuracy: 0.9544, Loss: 0.0289
    Epoch  32 Batch   10/269 - Train Accuracy: 0.9611, Validation Accuracy: 0.9561, Loss: 0.0248
    Epoch  32 Batch   15/269 - Train Accuracy: 0.9532, Validation Accuracy: 0.9562, Loss: 0.0225
    Epoch  32 Batch   20/269 - Train Accuracy: 0.9561, Validation Accuracy: 0.9502, Loss: 0.0267
    Epoch  32 Batch   25/269 - Train Accuracy: 0.9456, Validation Accuracy: 0.9548, Loss: 0.0318
    Epoch  32 Batch   30/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9477, Loss: 0.0282
    Epoch  32 Batch   35/269 - Train Accuracy: 0.9482, Validation Accuracy: 0.9485, Loss: 0.0387
    Epoch  32 Batch   40/269 - Train Accuracy: 0.9397, Validation Accuracy: 0.9411, Loss: 0.0315
    Epoch  32 Batch   45/269 - Train Accuracy: 0.9477, Validation Accuracy: 0.9513, Loss: 0.0337
    Epoch  32 Batch   50/269 - Train Accuracy: 0.9426, Validation Accuracy: 0.9515, Loss: 0.0388
    Epoch  32 Batch   55/269 - Train Accuracy: 0.9484, Validation Accuracy: 0.9341, Loss: 0.0258
    Epoch  32 Batch   60/269 - Train Accuracy: 0.9517, Validation Accuracy: 0.9484, Loss: 0.0288
    Epoch  32 Batch   65/269 - Train Accuracy: 0.9582, Validation Accuracy: 0.9591, Loss: 0.0258
    Epoch  32 Batch   70/269 - Train Accuracy: 0.9570, Validation Accuracy: 0.9545, Loss: 0.0282
    Epoch  32 Batch   75/269 - Train Accuracy: 0.9521, Validation Accuracy: 0.9508, Loss: 0.0323
    Epoch  32 Batch   80/269 - Train Accuracy: 0.9568, Validation Accuracy: 0.9478, Loss: 0.0267
    Epoch  32 Batch   85/269 - Train Accuracy: 0.9387, Validation Accuracy: 0.9561, Loss: 0.0297
    Epoch  32 Batch   90/269 - Train Accuracy: 0.9437, Validation Accuracy: 0.9561, Loss: 0.0307
    Epoch  32 Batch   95/269 - Train Accuracy: 0.9580, Validation Accuracy: 0.9580, Loss: 0.0279
    Epoch  32 Batch  100/269 - Train Accuracy: 0.9555, Validation Accuracy: 0.9483, Loss: 0.0286
    Epoch  32 Batch  105/269 - Train Accuracy: 0.9608, Validation Accuracy: 0.9561, Loss: 0.0249
    Epoch  32 Batch  110/269 - Train Accuracy: 0.9554, Validation Accuracy: 0.9549, Loss: 0.0283
    Epoch  32 Batch  115/269 - Train Accuracy: 0.9608, Validation Accuracy: 0.9543, Loss: 0.0336
    Epoch  32 Batch  120/269 - Train Accuracy: 0.9510, Validation Accuracy: 0.9497, Loss: 0.0294
    Epoch  32 Batch  125/269 - Train Accuracy: 0.9644, Validation Accuracy: 0.9574, Loss: 0.0287
    Epoch  32 Batch  130/269 - Train Accuracy: 0.9574, Validation Accuracy: 0.9535, Loss: 0.0282
    Epoch  32 Batch  135/269 - Train Accuracy: 0.9600, Validation Accuracy: 0.9535, Loss: 0.0259
    Epoch  32 Batch  140/269 - Train Accuracy: 0.9609, Validation Accuracy: 0.9482, Loss: 0.0326
    Epoch  32 Batch  145/269 - Train Accuracy: 0.9557, Validation Accuracy: 0.9480, Loss: 0.0261
    Epoch  32 Batch  150/269 - Train Accuracy: 0.9561, Validation Accuracy: 0.9527, Loss: 0.0325
    Epoch  32 Batch  155/269 - Train Accuracy: 0.9599, Validation Accuracy: 0.9569, Loss: 0.0258
    Epoch  32 Batch  160/269 - Train Accuracy: 0.9673, Validation Accuracy: 0.9574, Loss: 0.0247
    Epoch  32 Batch  165/269 - Train Accuracy: 0.9484, Validation Accuracy: 0.9534, Loss: 0.0292
    Epoch  32 Batch  170/269 - Train Accuracy: 0.9513, Validation Accuracy: 0.9545, Loss: 0.0263
    Epoch  32 Batch  175/269 - Train Accuracy: 0.9564, Validation Accuracy: 0.9491, Loss: 0.0390
    Epoch  32 Batch  180/269 - Train Accuracy: 0.9691, Validation Accuracy: 0.9543, Loss: 0.0248
    Epoch  32 Batch  185/269 - Train Accuracy: 0.9592, Validation Accuracy: 0.9426, Loss: 0.0265
    Epoch  32 Batch  190/269 - Train Accuracy: 0.9568, Validation Accuracy: 0.9580, Loss: 0.0297
    Epoch  32 Batch  195/269 - Train Accuracy: 0.9470, Validation Accuracy: 0.9593, Loss: 0.0250
    Epoch  32 Batch  200/269 - Train Accuracy: 0.9624, Validation Accuracy: 0.9479, Loss: 0.0257
    Epoch  32 Batch  205/269 - Train Accuracy: 0.9619, Validation Accuracy: 0.9440, Loss: 0.0293
    Epoch  32 Batch  210/269 - Train Accuracy: 0.9487, Validation Accuracy: 0.9511, Loss: 0.0312
    Epoch  32 Batch  215/269 - Train Accuracy: 0.9461, Validation Accuracy: 0.9447, Loss: 0.0279
    Epoch  32 Batch  220/269 - Train Accuracy: 0.9518, Validation Accuracy: 0.9540, Loss: 0.0323
    Epoch  32 Batch  225/269 - Train Accuracy: 0.9487, Validation Accuracy: 0.9543, Loss: 0.0279
    Epoch  32 Batch  230/269 - Train Accuracy: 0.9545, Validation Accuracy: 0.9535, Loss: 0.0293
    Epoch  32 Batch  235/269 - Train Accuracy: 0.9754, Validation Accuracy: 0.9498, Loss: 0.0236
    Epoch  32 Batch  240/269 - Train Accuracy: 0.9503, Validation Accuracy: 0.9488, Loss: 0.0284
    Epoch  32 Batch  245/269 - Train Accuracy: 0.9555, Validation Accuracy: 0.9519, Loss: 0.0279
    Epoch  32 Batch  250/269 - Train Accuracy: 0.9461, Validation Accuracy: 0.9518, Loss: 0.0286
    Epoch  32 Batch  255/269 - Train Accuracy: 0.9519, Validation Accuracy: 0.9521, Loss: 0.0327
    Epoch  32 Batch  260/269 - Train Accuracy: 0.9597, Validation Accuracy: 0.9503, Loss: 0.0290
    Epoch  32 Batch  265/269 - Train Accuracy: 0.9480, Validation Accuracy: 0.9524, Loss: 0.0284
    Epoch  33 Batch    5/269 - Train Accuracy: 0.9636, Validation Accuracy: 0.9580, Loss: 0.0302
    Epoch  33 Batch   10/269 - Train Accuracy: 0.9583, Validation Accuracy: 0.9607, Loss: 0.0241
    Epoch  33 Batch   15/269 - Train Accuracy: 0.9570, Validation Accuracy: 0.9559, Loss: 0.0232
    Epoch  33 Batch   20/269 - Train Accuracy: 0.9583, Validation Accuracy: 0.9586, Loss: 0.0286
    Epoch  33 Batch   25/269 - Train Accuracy: 0.9426, Validation Accuracy: 0.9493, Loss: 0.0302
    Epoch  33 Batch   30/269 - Train Accuracy: 0.9610, Validation Accuracy: 0.9521, Loss: 0.0279
    Epoch  33 Batch   35/269 - Train Accuracy: 0.9527, Validation Accuracy: 0.9524, Loss: 0.0391
    Epoch  33 Batch   40/269 - Train Accuracy: 0.9503, Validation Accuracy: 0.9523, Loss: 0.0315
    Epoch  33 Batch   45/269 - Train Accuracy: 0.9434, Validation Accuracy: 0.9517, Loss: 0.0324
    Epoch  33 Batch   50/269 - Train Accuracy: 0.9460, Validation Accuracy: 0.9497, Loss: 0.0345
    Epoch  33 Batch   55/269 - Train Accuracy: 0.9608, Validation Accuracy: 0.9470, Loss: 0.0254
    Epoch  33 Batch   60/269 - Train Accuracy: 0.9501, Validation Accuracy: 0.9513, Loss: 0.0297
    Epoch  33 Batch   65/269 - Train Accuracy: 0.9581, Validation Accuracy: 0.9577, Loss: 0.0268
    Epoch  33 Batch   70/269 - Train Accuracy: 0.9602, Validation Accuracy: 0.9510, Loss: 0.0280
    Epoch  33 Batch   75/269 - Train Accuracy: 0.9625, Validation Accuracy: 0.9506, Loss: 0.0326
    Epoch  33 Batch   80/269 - Train Accuracy: 0.9628, Validation Accuracy: 0.9521, Loss: 0.0275
    Epoch  33 Batch   85/269 - Train Accuracy: 0.9430, Validation Accuracy: 0.9483, Loss: 0.0292
    Epoch  33 Batch   90/269 - Train Accuracy: 0.9443, Validation Accuracy: 0.9584, Loss: 0.0309
    Epoch  33 Batch   95/269 - Train Accuracy: 0.9591, Validation Accuracy: 0.9554, Loss: 0.0277
    Epoch  33 Batch  100/269 - Train Accuracy: 0.9480, Validation Accuracy: 0.9445, Loss: 0.0311
    Epoch  33 Batch  105/269 - Train Accuracy: 0.9595, Validation Accuracy: 0.9484, Loss: 0.0307
    Epoch  33 Batch  110/269 - Train Accuracy: 0.9512, Validation Accuracy: 0.9538, Loss: 0.0310
    Epoch  33 Batch  115/269 - Train Accuracy: 0.9580, Validation Accuracy: 0.9490, Loss: 0.0344
    Epoch  33 Batch  120/269 - Train Accuracy: 0.9525, Validation Accuracy: 0.9510, Loss: 0.0325
    Epoch  33 Batch  125/269 - Train Accuracy: 0.9554, Validation Accuracy: 0.9474, Loss: 0.0281
    Epoch  33 Batch  130/269 - Train Accuracy: 0.9589, Validation Accuracy: 0.9513, Loss: 0.0306
    Epoch  33 Batch  135/269 - Train Accuracy: 0.9595, Validation Accuracy: 0.9566, Loss: 0.0269
    Epoch  33 Batch  140/269 - Train Accuracy: 0.9510, Validation Accuracy: 0.9507, Loss: 0.0341
    Epoch  33 Batch  145/269 - Train Accuracy: 0.9589, Validation Accuracy: 0.9581, Loss: 0.0277
    Epoch  33 Batch  150/269 - Train Accuracy: 0.9499, Validation Accuracy: 0.9479, Loss: 0.0335
    Epoch  33 Batch  155/269 - Train Accuracy: 0.9566, Validation Accuracy: 0.9612, Loss: 0.0259
    Epoch  33 Batch  160/269 - Train Accuracy: 0.9633, Validation Accuracy: 0.9536, Loss: 0.0255
    Epoch  33 Batch  165/269 - Train Accuracy: 0.9563, Validation Accuracy: 0.9598, Loss: 0.0269
    Epoch  33 Batch  170/269 - Train Accuracy: 0.9510, Validation Accuracy: 0.9627, Loss: 0.0291
    Epoch  33 Batch  175/269 - Train Accuracy: 0.9552, Validation Accuracy: 0.9561, Loss: 0.0341
    Epoch  33 Batch  180/269 - Train Accuracy: 0.9667, Validation Accuracy: 0.9533, Loss: 0.0270
    Epoch  33 Batch  185/269 - Train Accuracy: 0.9563, Validation Accuracy: 0.9401, Loss: 0.0252
    Epoch  33 Batch  190/269 - Train Accuracy: 0.9566, Validation Accuracy: 0.9596, Loss: 0.0302
    Epoch  33 Batch  195/269 - Train Accuracy: 0.9445, Validation Accuracy: 0.9595, Loss: 0.0251
    Epoch  33 Batch  200/269 - Train Accuracy: 0.9649, Validation Accuracy: 0.9511, Loss: 0.0230
    Epoch  33 Batch  205/269 - Train Accuracy: 0.9644, Validation Accuracy: 0.9416, Loss: 0.0259
    Epoch  33 Batch  210/269 - Train Accuracy: 0.9547, Validation Accuracy: 0.9495, Loss: 0.0267
    Epoch  33 Batch  215/269 - Train Accuracy: 0.9502, Validation Accuracy: 0.9521, Loss: 0.0282
    Epoch  33 Batch  220/269 - Train Accuracy: 0.9529, Validation Accuracy: 0.9499, Loss: 0.0296
    Epoch  33 Batch  225/269 - Train Accuracy: 0.9535, Validation Accuracy: 0.9479, Loss: 0.0357
    Epoch  33 Batch  230/269 - Train Accuracy: 0.9489, Validation Accuracy: 0.9468, Loss: 0.0283
    Epoch  33 Batch  235/269 - Train Accuracy: 0.9567, Validation Accuracy: 0.9327, Loss: 0.0455
    Epoch  33 Batch  240/269 - Train Accuracy: 0.9218, Validation Accuracy: 0.9211, Loss: 0.0458
    Epoch  33 Batch  245/269 - Train Accuracy: 0.9311, Validation Accuracy: 0.9396, Loss: 0.0626
    Epoch  33 Batch  250/269 - Train Accuracy: 0.9247, Validation Accuracy: 0.9265, Loss: 0.0432
    Epoch  33 Batch  255/269 - Train Accuracy: 0.9395, Validation Accuracy: 0.9386, Loss: 0.0493
    Epoch  33 Batch  260/269 - Train Accuracy: 0.9413, Validation Accuracy: 0.9419, Loss: 0.0467
    Epoch  33 Batch  265/269 - Train Accuracy: 0.9335, Validation Accuracy: 0.9360, Loss: 0.0457
    Epoch  34 Batch    5/269 - Train Accuracy: 0.9496, Validation Accuracy: 0.9380, Loss: 0.0342
    Epoch  34 Batch   10/269 - Train Accuracy: 0.9442, Validation Accuracy: 0.9390, Loss: 0.0304
    Epoch  34 Batch   15/269 - Train Accuracy: 0.9603, Validation Accuracy: 0.9487, Loss: 0.0257
    Epoch  34 Batch   20/269 - Train Accuracy: 0.9579, Validation Accuracy: 0.9513, Loss: 0.0328
    Epoch  34 Batch   25/269 - Train Accuracy: 0.9351, Validation Accuracy: 0.9532, Loss: 0.0337
    Epoch  34 Batch   30/269 - Train Accuracy: 0.9562, Validation Accuracy: 0.9555, Loss: 0.0302
    Epoch  34 Batch   35/269 - Train Accuracy: 0.9503, Validation Accuracy: 0.9467, Loss: 0.0410
    Epoch  34 Batch   40/269 - Train Accuracy: 0.9455, Validation Accuracy: 0.9529, Loss: 0.0332
    Epoch  34 Batch   45/269 - Train Accuracy: 0.9478, Validation Accuracy: 0.9545, Loss: 0.0327
    Epoch  34 Batch   50/269 - Train Accuracy: 0.9427, Validation Accuracy: 0.9535, Loss: 0.0347
    Epoch  34 Batch   55/269 - Train Accuracy: 0.9573, Validation Accuracy: 0.9436, Loss: 0.0270
    Epoch  34 Batch   60/269 - Train Accuracy: 0.9507, Validation Accuracy: 0.9432, Loss: 0.0285
    Epoch  34 Batch   65/269 - Train Accuracy: 0.9560, Validation Accuracy: 0.9491, Loss: 0.0257
    Epoch  34 Batch   70/269 - Train Accuracy: 0.9507, Validation Accuracy: 0.9556, Loss: 0.0308
    Epoch  34 Batch   75/269 - Train Accuracy: 0.9562, Validation Accuracy: 0.9512, Loss: 0.0342
    Epoch  34 Batch   80/269 - Train Accuracy: 0.9562, Validation Accuracy: 0.9479, Loss: 0.0271
    Epoch  34 Batch   85/269 - Train Accuracy: 0.9480, Validation Accuracy: 0.9508, Loss: 0.0277
    Epoch  34 Batch   90/269 - Train Accuracy: 0.9510, Validation Accuracy: 0.9435, Loss: 0.0289
    Epoch  34 Batch   95/269 - Train Accuracy: 0.9594, Validation Accuracy: 0.9562, Loss: 0.0282
    Epoch  34 Batch  100/269 - Train Accuracy: 0.9506, Validation Accuracy: 0.9498, Loss: 0.0303
    Epoch  34 Batch  105/269 - Train Accuracy: 0.9625, Validation Accuracy: 0.9522, Loss: 0.0294
    Epoch  34 Batch  110/269 - Train Accuracy: 0.9504, Validation Accuracy: 0.9540, Loss: 0.0282
    Epoch  34 Batch  115/269 - Train Accuracy: 0.9555, Validation Accuracy: 0.9464, Loss: 0.0317
    Epoch  34 Batch  120/269 - Train Accuracy: 0.9523, Validation Accuracy: 0.9538, Loss: 0.0287
    Epoch  34 Batch  125/269 - Train Accuracy: 0.9601, Validation Accuracy: 0.9540, Loss: 0.0264
    Epoch  34 Batch  130/269 - Train Accuracy: 0.9574, Validation Accuracy: 0.9521, Loss: 0.0313
    Epoch  34 Batch  135/269 - Train Accuracy: 0.9524, Validation Accuracy: 0.9447, Loss: 0.0264
    Epoch  34 Batch  140/269 - Train Accuracy: 0.9525, Validation Accuracy: 0.9517, Loss: 0.0315
    Epoch  34 Batch  145/269 - Train Accuracy: 0.9590, Validation Accuracy: 0.9500, Loss: 0.0247
    Epoch  34 Batch  150/269 - Train Accuracy: 0.9541, Validation Accuracy: 0.9541, Loss: 0.0321
    Epoch  34 Batch  155/269 - Train Accuracy: 0.9624, Validation Accuracy: 0.9518, Loss: 0.0264
    Epoch  34 Batch  160/269 - Train Accuracy: 0.9664, Validation Accuracy: 0.9531, Loss: 0.0258
    Epoch  34 Batch  165/269 - Train Accuracy: 0.9590, Validation Accuracy: 0.9575, Loss: 0.0267
    Epoch  34 Batch  170/269 - Train Accuracy: 0.9458, Validation Accuracy: 0.9513, Loss: 0.0285
    Epoch  34 Batch  175/269 - Train Accuracy: 0.9591, Validation Accuracy: 0.9545, Loss: 0.0333
    Epoch  34 Batch  180/269 - Train Accuracy: 0.9589, Validation Accuracy: 0.9550, Loss: 0.0244
    Epoch  34 Batch  185/269 - Train Accuracy: 0.9616, Validation Accuracy: 0.9455, Loss: 0.0267
    Epoch  34 Batch  190/269 - Train Accuracy: 0.9619, Validation Accuracy: 0.9528, Loss: 0.0288
    Epoch  34 Batch  195/269 - Train Accuracy: 0.9472, Validation Accuracy: 0.9666, Loss: 0.0263
    Epoch  34 Batch  200/269 - Train Accuracy: 0.9719, Validation Accuracy: 0.9561, Loss: 0.0217
    Epoch  34 Batch  205/269 - Train Accuracy: 0.9613, Validation Accuracy: 0.9521, Loss: 0.0256
    Epoch  34 Batch  210/269 - Train Accuracy: 0.9448, Validation Accuracy: 0.9630, Loss: 0.0256
    Epoch  34 Batch  215/269 - Train Accuracy: 0.9473, Validation Accuracy: 0.9459, Loss: 0.0284
    Epoch  34 Batch  220/269 - Train Accuracy: 0.9503, Validation Accuracy: 0.9541, Loss: 0.0305
    Epoch  34 Batch  225/269 - Train Accuracy: 0.9453, Validation Accuracy: 0.9535, Loss: 0.0276
    Epoch  34 Batch  230/269 - Train Accuracy: 0.9558, Validation Accuracy: 0.9529, Loss: 0.0287
    Epoch  34 Batch  235/269 - Train Accuracy: 0.9780, Validation Accuracy: 0.9532, Loss: 0.0204
    Epoch  34 Batch  240/269 - Train Accuracy: 0.9552, Validation Accuracy: 0.9582, Loss: 0.0262
    Epoch  34 Batch  245/269 - Train Accuracy: 0.9576, Validation Accuracy: 0.9569, Loss: 0.0266
    Epoch  34 Batch  250/269 - Train Accuracy: 0.9415, Validation Accuracy: 0.9569, Loss: 0.0269
    Epoch  34 Batch  255/269 - Train Accuracy: 0.9587, Validation Accuracy: 0.9507, Loss: 0.0301
    Epoch  34 Batch  260/269 - Train Accuracy: 0.9626, Validation Accuracy: 0.9499, Loss: 0.0288
    Epoch  34 Batch  265/269 - Train Accuracy: 0.9525, Validation Accuracy: 0.9532, Loss: 0.0287
    Epoch  35 Batch    5/269 - Train Accuracy: 0.9603, Validation Accuracy: 0.9501, Loss: 0.0263
    Epoch  35 Batch   10/269 - Train Accuracy: 0.9638, Validation Accuracy: 0.9585, Loss: 0.0227
    Epoch  35 Batch   15/269 - Train Accuracy: 0.9607, Validation Accuracy: 0.9569, Loss: 0.0196
    Epoch  35 Batch   20/269 - Train Accuracy: 0.9628, Validation Accuracy: 0.9500, Loss: 0.0274
    Epoch  35 Batch   25/269 - Train Accuracy: 0.9424, Validation Accuracy: 0.9553, Loss: 0.0312
    Epoch  35 Batch   30/269 - Train Accuracy: 0.9607, Validation Accuracy: 0.9586, Loss: 0.0310
    Epoch  35 Batch   35/269 - Train Accuracy: 0.9504, Validation Accuracy: 0.9586, Loss: 0.0393
    Epoch  35 Batch   40/269 - Train Accuracy: 0.9500, Validation Accuracy: 0.9480, Loss: 0.0341
    Epoch  35 Batch   45/269 - Train Accuracy: 0.9583, Validation Accuracy: 0.9465, Loss: 0.0332
    Epoch  35 Batch   50/269 - Train Accuracy: 0.9461, Validation Accuracy: 0.9543, Loss: 0.0334
    Epoch  35 Batch   55/269 - Train Accuracy: 0.9566, Validation Accuracy: 0.9498, Loss: 0.0258
    Epoch  35 Batch   60/269 - Train Accuracy: 0.9546, Validation Accuracy: 0.9550, Loss: 0.0285
    Epoch  35 Batch   65/269 - Train Accuracy: 0.9544, Validation Accuracy: 0.9547, Loss: 0.0258
    Epoch  35 Batch   70/269 - Train Accuracy: 0.9561, Validation Accuracy: 0.9513, Loss: 0.0302
    Epoch  35 Batch   75/269 - Train Accuracy: 0.9604, Validation Accuracy: 0.9575, Loss: 0.0311
    Epoch  35 Batch   80/269 - Train Accuracy: 0.9557, Validation Accuracy: 0.9466, Loss: 0.0266
    Epoch  35 Batch   85/269 - Train Accuracy: 0.9485, Validation Accuracy: 0.9516, Loss: 0.0269
    Epoch  35 Batch   90/269 - Train Accuracy: 0.9419, Validation Accuracy: 0.9490, Loss: 0.0296
    Epoch  35 Batch   95/269 - Train Accuracy: 0.9591, Validation Accuracy: 0.9581, Loss: 0.0256
    Epoch  35 Batch  100/269 - Train Accuracy: 0.9577, Validation Accuracy: 0.9515, Loss: 0.0284
    Epoch  35 Batch  105/269 - Train Accuracy: 0.9586, Validation Accuracy: 0.9561, Loss: 0.0261
    Epoch  35 Batch  110/269 - Train Accuracy: 0.9577, Validation Accuracy: 0.9603, Loss: 0.0294
    Epoch  35 Batch  115/269 - Train Accuracy: 0.9627, Validation Accuracy: 0.9530, Loss: 0.0309
    Epoch  35 Batch  120/269 - Train Accuracy: 0.9546, Validation Accuracy: 0.9534, Loss: 0.0283
    Epoch  35 Batch  125/269 - Train Accuracy: 0.9627, Validation Accuracy: 0.9480, Loss: 0.0259
    Epoch  35 Batch  130/269 - Train Accuracy: 0.9585, Validation Accuracy: 0.9599, Loss: 0.0291
    Epoch  35 Batch  135/269 - Train Accuracy: 0.9609, Validation Accuracy: 0.9518, Loss: 0.0262
    Epoch  35 Batch  140/269 - Train Accuracy: 0.9483, Validation Accuracy: 0.9543, Loss: 0.0324
    Epoch  35 Batch  145/269 - Train Accuracy: 0.9589, Validation Accuracy: 0.9478, Loss: 0.0256
    Epoch  35 Batch  150/269 - Train Accuracy: 0.9610, Validation Accuracy: 0.9543, Loss: 0.0326
    Epoch  35 Batch  155/269 - Train Accuracy: 0.9496, Validation Accuracy: 0.9582, Loss: 0.0254
    Epoch  35 Batch  160/269 - Train Accuracy: 0.9664, Validation Accuracy: 0.9537, Loss: 0.0226
    Epoch  35 Batch  165/269 - Train Accuracy: 0.9615, Validation Accuracy: 0.9521, Loss: 0.0240
    Epoch  35 Batch  170/269 - Train Accuracy: 0.9517, Validation Accuracy: 0.9561, Loss: 0.0281
    Epoch  35 Batch  175/269 - Train Accuracy: 0.9534, Validation Accuracy: 0.9535, Loss: 0.0330
    Epoch  35 Batch  180/269 - Train Accuracy: 0.9575, Validation Accuracy: 0.9566, Loss: 0.0232
    Epoch  35 Batch  185/269 - Train Accuracy: 0.9631, Validation Accuracy: 0.9548, Loss: 0.0243
    Epoch  35 Batch  190/269 - Train Accuracy: 0.9577, Validation Accuracy: 0.9529, Loss: 0.0276
    Epoch  35 Batch  195/269 - Train Accuracy: 0.9525, Validation Accuracy: 0.9569, Loss: 0.0250
    Epoch  35 Batch  200/269 - Train Accuracy: 0.9607, Validation Accuracy: 0.9522, Loss: 0.0228
    Epoch  35 Batch  205/269 - Train Accuracy: 0.9617, Validation Accuracy: 0.9476, Loss: 0.0237
    Epoch  35 Batch  210/269 - Train Accuracy: 0.9596, Validation Accuracy: 0.9471, Loss: 0.0238
    Epoch  35 Batch  215/269 - Train Accuracy: 0.9505, Validation Accuracy: 0.9553, Loss: 0.0273
    Epoch  35 Batch  220/269 - Train Accuracy: 0.9530, Validation Accuracy: 0.9537, Loss: 0.0314
    Epoch  35 Batch  225/269 - Train Accuracy: 0.9386, Validation Accuracy: 0.9519, Loss: 0.0266
    Epoch  35 Batch  230/269 - Train Accuracy: 0.9577, Validation Accuracy: 0.9547, Loss: 0.0263
    Epoch  35 Batch  235/269 - Train Accuracy: 0.9805, Validation Accuracy: 0.9574, Loss: 0.0183
    Epoch  35 Batch  240/269 - Train Accuracy: 0.9546, Validation Accuracy: 0.9568, Loss: 0.0257
    Epoch  35 Batch  245/269 - Train Accuracy: 0.9607, Validation Accuracy: 0.9568, Loss: 0.0230
    Epoch  35 Batch  250/269 - Train Accuracy: 0.9527, Validation Accuracy: 0.9490, Loss: 0.0264
    Epoch  35 Batch  255/269 - Train Accuracy: 0.9516, Validation Accuracy: 0.9523, Loss: 0.0304
    Epoch  35 Batch  260/269 - Train Accuracy: 0.9527, Validation Accuracy: 0.9500, Loss: 0.0299
    Epoch  35 Batch  265/269 - Train Accuracy: 0.9570, Validation Accuracy: 0.9517, Loss: 0.0288
    Epoch  36 Batch    5/269 - Train Accuracy: 0.9652, Validation Accuracy: 0.9568, Loss: 0.0262
    Epoch  36 Batch   10/269 - Train Accuracy: 0.9596, Validation Accuracy: 0.9577, Loss: 0.0243
    Epoch  36 Batch   15/269 - Train Accuracy: 0.9629, Validation Accuracy: 0.9566, Loss: 0.0187
    Epoch  36 Batch   20/269 - Train Accuracy: 0.9633, Validation Accuracy: 0.9481, Loss: 0.0257
    Epoch  36 Batch   25/269 - Train Accuracy: 0.9490, Validation Accuracy: 0.9545, Loss: 0.0317
    Epoch  36 Batch   30/269 - Train Accuracy: 0.9645, Validation Accuracy: 0.9584, Loss: 0.0268
    Epoch  36 Batch   35/269 - Train Accuracy: 0.9572, Validation Accuracy: 0.9545, Loss: 0.0400
    Epoch  36 Batch   40/269 - Train Accuracy: 0.9546, Validation Accuracy: 0.9408, Loss: 0.0306
    Epoch  36 Batch   45/269 - Train Accuracy: 0.9511, Validation Accuracy: 0.9470, Loss: 0.0311
    Epoch  36 Batch   50/269 - Train Accuracy: 0.9302, Validation Accuracy: 0.9409, Loss: 0.0357
    Epoch  36 Batch   55/269 - Train Accuracy: 0.9605, Validation Accuracy: 0.9347, Loss: 0.0289
    Epoch  36 Batch   60/269 - Train Accuracy: 0.9565, Validation Accuracy: 0.9460, Loss: 0.0308
    Epoch  36 Batch   65/269 - Train Accuracy: 0.9555, Validation Accuracy: 0.9472, Loss: 0.0256
    Epoch  36 Batch   70/269 - Train Accuracy: 0.9634, Validation Accuracy: 0.9504, Loss: 0.0302
    Epoch  36 Batch   75/269 - Train Accuracy: 0.9530, Validation Accuracy: 0.9482, Loss: 0.0329
    Epoch  36 Batch   80/269 - Train Accuracy: 0.9583, Validation Accuracy: 0.9545, Loss: 0.0263
    Epoch  36 Batch   85/269 - Train Accuracy: 0.9461, Validation Accuracy: 0.9492, Loss: 0.0253
    Epoch  36 Batch   90/269 - Train Accuracy: 0.9437, Validation Accuracy: 0.9509, Loss: 0.0319
    Epoch  36 Batch   95/269 - Train Accuracy: 0.9553, Validation Accuracy: 0.9556, Loss: 0.0277
    Epoch  36 Batch  100/269 - Train Accuracy: 0.9528, Validation Accuracy: 0.9470, Loss: 0.0302
    Epoch  36 Batch  105/269 - Train Accuracy: 0.9581, Validation Accuracy: 0.9500, Loss: 0.0271
    Epoch  36 Batch  110/269 - Train Accuracy: 0.9477, Validation Accuracy: 0.9574, Loss: 0.0299
    Epoch  36 Batch  115/269 - Train Accuracy: 0.9563, Validation Accuracy: 0.9474, Loss: 0.0325
    Epoch  36 Batch  120/269 - Train Accuracy: 0.9464, Validation Accuracy: 0.9564, Loss: 0.0273
    Epoch  36 Batch  125/269 - Train Accuracy: 0.9561, Validation Accuracy: 0.9526, Loss: 0.0281
    Epoch  36 Batch  130/269 - Train Accuracy: 0.9556, Validation Accuracy: 0.9482, Loss: 0.0319
    Epoch  36 Batch  135/269 - Train Accuracy: 0.9562, Validation Accuracy: 0.9569, Loss: 0.0267
    Epoch  36 Batch  140/269 - Train Accuracy: 0.9460, Validation Accuracy: 0.9483, Loss: 0.0343
    Epoch  36 Batch  145/269 - Train Accuracy: 0.9612, Validation Accuracy: 0.9458, Loss: 0.0271
    Epoch  36 Batch  150/269 - Train Accuracy: 0.9572, Validation Accuracy: 0.9532, Loss: 0.0344
    Epoch  36 Batch  155/269 - Train Accuracy: 0.9527, Validation Accuracy: 0.9478, Loss: 0.0281
    Epoch  36 Batch  160/269 - Train Accuracy: 0.9676, Validation Accuracy: 0.9458, Loss: 0.0254
    Epoch  36 Batch  165/269 - Train Accuracy: 0.9625, Validation Accuracy: 0.9523, Loss: 0.0263
    Epoch  36 Batch  170/269 - Train Accuracy: 0.9501, Validation Accuracy: 0.9526, Loss: 0.0288
    Epoch  36 Batch  175/269 - Train Accuracy: 0.9596, Validation Accuracy: 0.9516, Loss: 0.0370
    Epoch  36 Batch  180/269 - Train Accuracy: 0.9556, Validation Accuracy: 0.9536, Loss: 0.0252
    Epoch  36 Batch  185/269 - Train Accuracy: 0.9579, Validation Accuracy: 0.9419, Loss: 0.0276
    Epoch  36 Batch  190/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9455, Loss: 0.0295
    Epoch  36 Batch  195/269 - Train Accuracy: 0.9543, Validation Accuracy: 0.9512, Loss: 0.0251
    Epoch  36 Batch  200/269 - Train Accuracy: 0.9690, Validation Accuracy: 0.9530, Loss: 0.0219
    Epoch  36 Batch  205/269 - Train Accuracy: 0.9575, Validation Accuracy: 0.9525, Loss: 0.0266
    Epoch  36 Batch  210/269 - Train Accuracy: 0.9579, Validation Accuracy: 0.9528, Loss: 0.0251
    Epoch  36 Batch  215/269 - Train Accuracy: 0.9498, Validation Accuracy: 0.9532, Loss: 0.0280
    Epoch  36 Batch  220/269 - Train Accuracy: 0.9522, Validation Accuracy: 0.9550, Loss: 0.0295
    Epoch  36 Batch  225/269 - Train Accuracy: 0.9538, Validation Accuracy: 0.9578, Loss: 0.0253
    Epoch  36 Batch  230/269 - Train Accuracy: 0.9563, Validation Accuracy: 0.9502, Loss: 0.0265
    Epoch  36 Batch  235/269 - Train Accuracy: 0.9791, Validation Accuracy: 0.9577, Loss: 0.0208
    Epoch  36 Batch  240/269 - Train Accuracy: 0.9575, Validation Accuracy: 0.9551, Loss: 0.0273
    Epoch  36 Batch  245/269 - Train Accuracy: 0.9650, Validation Accuracy: 0.9520, Loss: 0.0251
    Epoch  36 Batch  250/269 - Train Accuracy: 0.9543, Validation Accuracy: 0.9577, Loss: 0.0260
    Epoch  36 Batch  255/269 - Train Accuracy: 0.9544, Validation Accuracy: 0.9581, Loss: 0.0295
    Epoch  36 Batch  260/269 - Train Accuracy: 0.9641, Validation Accuracy: 0.9542, Loss: 0.0275
    Epoch  36 Batch  265/269 - Train Accuracy: 0.9505, Validation Accuracy: 0.9544, Loss: 0.0281
    Epoch  37 Batch    5/269 - Train Accuracy: 0.9673, Validation Accuracy: 0.9636, Loss: 0.0267
    Epoch  37 Batch   10/269 - Train Accuracy: 0.9575, Validation Accuracy: 0.9611, Loss: 0.0236
    Epoch  37 Batch   15/269 - Train Accuracy: 0.9591, Validation Accuracy: 0.9545, Loss: 0.0201
    Epoch  37 Batch   20/269 - Train Accuracy: 0.9609, Validation Accuracy: 0.9538, Loss: 0.0268
    Epoch  37 Batch   25/269 - Train Accuracy: 0.9651, Validation Accuracy: 0.9573, Loss: 0.0281
    Epoch  37 Batch   30/269 - Train Accuracy: 0.9660, Validation Accuracy: 0.9540, Loss: 0.0263
    Epoch  37 Batch   35/269 - Train Accuracy: 0.9548, Validation Accuracy: 0.9526, Loss: 0.0393
    Epoch  37 Batch   40/269 - Train Accuracy: 0.9521, Validation Accuracy: 0.9534, Loss: 0.0284
    Epoch  37 Batch   45/269 - Train Accuracy: 0.9542, Validation Accuracy: 0.9493, Loss: 0.0351
    Epoch  37 Batch   50/269 - Train Accuracy: 0.9461, Validation Accuracy: 0.9600, Loss: 0.0306
    Epoch  37 Batch   55/269 - Train Accuracy: 0.9648, Validation Accuracy: 0.9520, Loss: 0.0260
    Epoch  37 Batch   60/269 - Train Accuracy: 0.9571, Validation Accuracy: 0.9545, Loss: 0.0249
    Epoch  37 Batch   65/269 - Train Accuracy: 0.9488, Validation Accuracy: 0.9592, Loss: 0.0254
    Epoch  37 Batch   70/269 - Train Accuracy: 0.9590, Validation Accuracy: 0.9596, Loss: 0.0270
    Epoch  37 Batch   75/269 - Train Accuracy: 0.9568, Validation Accuracy: 0.9558, Loss: 0.0300
    Epoch  37 Batch   80/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9485, Loss: 0.0261
    Epoch  37 Batch   85/269 - Train Accuracy: 0.9489, Validation Accuracy: 0.9554, Loss: 0.0268
    Epoch  37 Batch   90/269 - Train Accuracy: 0.9408, Validation Accuracy: 0.9438, Loss: 0.0290
    Epoch  37 Batch   95/269 - Train Accuracy: 0.9592, Validation Accuracy: 0.9594, Loss: 0.0258
    Epoch  37 Batch  100/269 - Train Accuracy: 0.9563, Validation Accuracy: 0.9580, Loss: 0.0262
    Epoch  37 Batch  105/269 - Train Accuracy: 0.9598, Validation Accuracy: 0.9561, Loss: 0.0268
    Epoch  37 Batch  110/269 - Train Accuracy: 0.9496, Validation Accuracy: 0.9577, Loss: 0.0266
    Epoch  37 Batch  115/269 - Train Accuracy: 0.9583, Validation Accuracy: 0.9518, Loss: 0.0337
    Epoch  37 Batch  120/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9563, Loss: 0.0271
    Epoch  37 Batch  125/269 - Train Accuracy: 0.9587, Validation Accuracy: 0.9526, Loss: 0.0275
    Epoch  37 Batch  130/269 - Train Accuracy: 0.9568, Validation Accuracy: 0.9430, Loss: 0.0275
    Epoch  37 Batch  135/269 - Train Accuracy: 0.9554, Validation Accuracy: 0.9546, Loss: 0.0256
    Epoch  37 Batch  140/269 - Train Accuracy: 0.9542, Validation Accuracy: 0.9568, Loss: 0.0292
    Epoch  37 Batch  145/269 - Train Accuracy: 0.9674, Validation Accuracy: 0.9588, Loss: 0.0241
    Epoch  37 Batch  150/269 - Train Accuracy: 0.9548, Validation Accuracy: 0.9519, Loss: 0.0288
    Epoch  37 Batch  155/269 - Train Accuracy: 0.9583, Validation Accuracy: 0.9567, Loss: 0.0245
    Epoch  37 Batch  160/269 - Train Accuracy: 0.9693, Validation Accuracy: 0.9488, Loss: 0.0218
    Epoch  37 Batch  165/269 - Train Accuracy: 0.9636, Validation Accuracy: 0.9513, Loss: 0.0258
    Epoch  37 Batch  170/269 - Train Accuracy: 0.9501, Validation Accuracy: 0.9557, Loss: 0.0260
    Epoch  37 Batch  175/269 - Train Accuracy: 0.9580, Validation Accuracy: 0.9566, Loss: 0.0325
    Epoch  37 Batch  180/269 - Train Accuracy: 0.9706, Validation Accuracy: 0.9506, Loss: 0.0235
    Epoch  37 Batch  185/269 - Train Accuracy: 0.9635, Validation Accuracy: 0.9551, Loss: 0.0244
    Epoch  37 Batch  190/269 - Train Accuracy: 0.9581, Validation Accuracy: 0.9576, Loss: 0.0289
    Epoch  37 Batch  195/269 - Train Accuracy: 0.9481, Validation Accuracy: 0.9585, Loss: 0.0256
    Epoch  37 Batch  200/269 - Train Accuracy: 0.9723, Validation Accuracy: 0.9652, Loss: 0.0218
    Epoch  37 Batch  205/269 - Train Accuracy: 0.9618, Validation Accuracy: 0.9548, Loss: 0.0271
    Epoch  37 Batch  210/269 - Train Accuracy: 0.9644, Validation Accuracy: 0.9503, Loss: 0.0281
    Epoch  37 Batch  215/269 - Train Accuracy: 0.9537, Validation Accuracy: 0.9509, Loss: 0.0278
    Epoch  37 Batch  220/269 - Train Accuracy: 0.9517, Validation Accuracy: 0.9475, Loss: 0.0298
    Epoch  37 Batch  225/269 - Train Accuracy: 0.9490, Validation Accuracy: 0.9535, Loss: 0.0245
    Epoch  37 Batch  230/269 - Train Accuracy: 0.9563, Validation Accuracy: 0.9562, Loss: 0.0241
    Epoch  37 Batch  235/269 - Train Accuracy: 0.9803, Validation Accuracy: 0.9529, Loss: 0.0202
    Epoch  37 Batch  240/269 - Train Accuracy: 0.9548, Validation Accuracy: 0.9487, Loss: 0.0246
    Epoch  37 Batch  245/269 - Train Accuracy: 0.9557, Validation Accuracy: 0.9561, Loss: 0.0257
    Epoch  37 Batch  250/269 - Train Accuracy: 0.9593, Validation Accuracy: 0.9482, Loss: 0.0295
    Epoch  37 Batch  255/269 - Train Accuracy: 0.9593, Validation Accuracy: 0.9558, Loss: 0.0289
    Epoch  37 Batch  260/269 - Train Accuracy: 0.9612, Validation Accuracy: 0.9564, Loss: 0.0279
    Epoch  37 Batch  265/269 - Train Accuracy: 0.9561, Validation Accuracy: 0.9567, Loss: 0.0299
    Epoch  38 Batch    5/269 - Train Accuracy: 0.9602, Validation Accuracy: 0.9555, Loss: 0.0250
    Epoch  38 Batch   10/269 - Train Accuracy: 0.9602, Validation Accuracy: 0.9498, Loss: 0.0243
    Epoch  38 Batch   15/269 - Train Accuracy: 0.9685, Validation Accuracy: 0.9519, Loss: 0.0190
    Epoch  38 Batch   20/269 - Train Accuracy: 0.9606, Validation Accuracy: 0.9444, Loss: 0.0268
    Epoch  38 Batch   25/269 - Train Accuracy: 0.9522, Validation Accuracy: 0.9559, Loss: 0.0286
    Epoch  38 Batch   30/269 - Train Accuracy: 0.9606, Validation Accuracy: 0.9563, Loss: 0.0269
    Epoch  38 Batch   35/269 - Train Accuracy: 0.9530, Validation Accuracy: 0.9523, Loss: 0.0364
    Epoch  38 Batch   40/269 - Train Accuracy: 0.9600, Validation Accuracy: 0.9527, Loss: 0.0307
    Epoch  38 Batch   45/269 - Train Accuracy: 0.9524, Validation Accuracy: 0.9580, Loss: 0.0332
    Epoch  38 Batch   50/269 - Train Accuracy: 0.9480, Validation Accuracy: 0.9522, Loss: 0.0340
    Epoch  38 Batch   55/269 - Train Accuracy: 0.9611, Validation Accuracy: 0.9485, Loss: 0.0257
    Epoch  38 Batch   60/269 - Train Accuracy: 0.9537, Validation Accuracy: 0.9498, Loss: 0.0259
    Epoch  38 Batch   65/269 - Train Accuracy: 0.9633, Validation Accuracy: 0.9580, Loss: 0.0248
    Epoch  38 Batch   70/269 - Train Accuracy: 0.9566, Validation Accuracy: 0.9577, Loss: 0.0296
    Epoch  38 Batch   75/269 - Train Accuracy: 0.9494, Validation Accuracy: 0.9398, Loss: 0.0312
    Epoch  38 Batch   80/269 - Train Accuracy: 0.9173, Validation Accuracy: 0.9101, Loss: 0.1500
    Epoch  38 Batch   85/269 - Train Accuracy: 0.8999, Validation Accuracy: 0.9173, Loss: 0.0840
    Epoch  38 Batch   90/269 - Train Accuracy: 0.9170, Validation Accuracy: 0.9153, Loss: 0.0811
    Epoch  38 Batch   95/269 - Train Accuracy: 0.9388, Validation Accuracy: 0.9237, Loss: 0.0612
    Epoch  38 Batch  100/269 - Train Accuracy: 0.9334, Validation Accuracy: 0.9268, Loss: 0.0496
    Epoch  38 Batch  105/269 - Train Accuracy: 0.9378, Validation Accuracy: 0.9454, Loss: 0.0477
    Epoch  38 Batch  110/269 - Train Accuracy: 0.9332, Validation Accuracy: 0.9430, Loss: 0.0438
    Epoch  38 Batch  115/269 - Train Accuracy: 0.9368, Validation Accuracy: 0.9437, Loss: 0.0457
    Epoch  38 Batch  120/269 - Train Accuracy: 0.9424, Validation Accuracy: 0.9455, Loss: 0.0422
    Epoch  38 Batch  125/269 - Train Accuracy: 0.9488, Validation Accuracy: 0.9401, Loss: 0.0355
    Epoch  38 Batch  130/269 - Train Accuracy: 0.9502, Validation Accuracy: 0.9501, Loss: 0.0367
    Epoch  38 Batch  135/269 - Train Accuracy: 0.9493, Validation Accuracy: 0.9534, Loss: 0.0290
    Epoch  38 Batch  140/269 - Train Accuracy: 0.9440, Validation Accuracy: 0.9419, Loss: 0.0351
    Epoch  38 Batch  145/269 - Train Accuracy: 0.9510, Validation Accuracy: 0.9474, Loss: 0.0269
    Epoch  38 Batch  150/269 - Train Accuracy: 0.9584, Validation Accuracy: 0.9607, Loss: 0.0308
    Epoch  38 Batch  155/269 - Train Accuracy: 0.9516, Validation Accuracy: 0.9538, Loss: 0.0285
    Epoch  38 Batch  160/269 - Train Accuracy: 0.9648, Validation Accuracy: 0.9561, Loss: 0.0254
    Epoch  38 Batch  165/269 - Train Accuracy: 0.9634, Validation Accuracy: 0.9528, Loss: 0.0258
    Epoch  38 Batch  170/269 - Train Accuracy: 0.9462, Validation Accuracy: 0.9536, Loss: 0.0274
    Epoch  38 Batch  175/269 - Train Accuracy: 0.9597, Validation Accuracy: 0.9591, Loss: 0.0357
    Epoch  38 Batch  180/269 - Train Accuracy: 0.9710, Validation Accuracy: 0.9629, Loss: 0.0212
    Epoch  38 Batch  185/269 - Train Accuracy: 0.9613, Validation Accuracy: 0.9433, Loss: 0.0253
    Epoch  38 Batch  190/269 - Train Accuracy: 0.9586, Validation Accuracy: 0.9514, Loss: 0.0274
    Epoch  38 Batch  195/269 - Train Accuracy: 0.9527, Validation Accuracy: 0.9625, Loss: 0.0247
    Epoch  38 Batch  200/269 - Train Accuracy: 0.9763, Validation Accuracy: 0.9473, Loss: 0.0214
    Epoch  38 Batch  205/269 - Train Accuracy: 0.9520, Validation Accuracy: 0.9525, Loss: 0.0234
    Epoch  38 Batch  210/269 - Train Accuracy: 0.9493, Validation Accuracy: 0.9541, Loss: 0.0248
    Epoch  38 Batch  215/269 - Train Accuracy: 0.9528, Validation Accuracy: 0.9529, Loss: 0.0282
    Epoch  38 Batch  220/269 - Train Accuracy: 0.9496, Validation Accuracy: 0.9542, Loss: 0.0279
    Epoch  38 Batch  225/269 - Train Accuracy: 0.9496, Validation Accuracy: 0.9550, Loss: 0.0251
    Epoch  38 Batch  230/269 - Train Accuracy: 0.9590, Validation Accuracy: 0.9599, Loss: 0.0240
    Epoch  38 Batch  235/269 - Train Accuracy: 0.9808, Validation Accuracy: 0.9580, Loss: 0.0194
    Epoch  38 Batch  240/269 - Train Accuracy: 0.9520, Validation Accuracy: 0.9603, Loss: 0.0248
    Epoch  38 Batch  245/269 - Train Accuracy: 0.9641, Validation Accuracy: 0.9533, Loss: 0.0223
    Epoch  38 Batch  250/269 - Train Accuracy: 0.9513, Validation Accuracy: 0.9578, Loss: 0.0245
    Epoch  38 Batch  255/269 - Train Accuracy: 0.9623, Validation Accuracy: 0.9569, Loss: 0.0277
    Epoch  38 Batch  260/269 - Train Accuracy: 0.9643, Validation Accuracy: 0.9545, Loss: 0.0254
    Epoch  38 Batch  265/269 - Train Accuracy: 0.9476, Validation Accuracy: 0.9612, Loss: 0.0276
    Epoch  39 Batch    5/269 - Train Accuracy: 0.9594, Validation Accuracy: 0.9603, Loss: 0.0260
    Epoch  39 Batch   10/269 - Train Accuracy: 0.9596, Validation Accuracy: 0.9600, Loss: 0.0217
    Epoch  39 Batch   15/269 - Train Accuracy: 0.9631, Validation Accuracy: 0.9591, Loss: 0.0188
    Epoch  39 Batch   20/269 - Train Accuracy: 0.9540, Validation Accuracy: 0.9611, Loss: 0.0251
    Epoch  39 Batch   25/269 - Train Accuracy: 0.9562, Validation Accuracy: 0.9501, Loss: 0.0295
    Epoch  39 Batch   30/269 - Train Accuracy: 0.9536, Validation Accuracy: 0.9616, Loss: 0.0247
    Epoch  39 Batch   35/269 - Train Accuracy: 0.9647, Validation Accuracy: 0.9603, Loss: 0.0349
    Epoch  39 Batch   40/269 - Train Accuracy: 0.9571, Validation Accuracy: 0.9549, Loss: 0.0284
    Epoch  39 Batch   45/269 - Train Accuracy: 0.9440, Validation Accuracy: 0.9575, Loss: 0.0289
    Epoch  39 Batch   50/269 - Train Accuracy: 0.9435, Validation Accuracy: 0.9526, Loss: 0.0293
    Epoch  39 Batch   55/269 - Train Accuracy: 0.9584, Validation Accuracy: 0.9515, Loss: 0.0244
    Epoch  39 Batch   60/269 - Train Accuracy: 0.9499, Validation Accuracy: 0.9493, Loss: 0.0261
    Epoch  39 Batch   65/269 - Train Accuracy: 0.9583, Validation Accuracy: 0.9563, Loss: 0.0232
    Epoch  39 Batch   70/269 - Train Accuracy: 0.9615, Validation Accuracy: 0.9508, Loss: 0.0249
    Epoch  39 Batch   75/269 - Train Accuracy: 0.9492, Validation Accuracy: 0.9545, Loss: 0.0295
    Epoch  39 Batch   80/269 - Train Accuracy: 0.9528, Validation Accuracy: 0.9442, Loss: 0.0215
    Epoch  39 Batch   85/269 - Train Accuracy: 0.9464, Validation Accuracy: 0.9478, Loss: 0.0238
    Epoch  39 Batch   90/269 - Train Accuracy: 0.9506, Validation Accuracy: 0.9452, Loss: 0.0271
    Epoch  39 Batch   95/269 - Train Accuracy: 0.9598, Validation Accuracy: 0.9597, Loss: 0.0226
    Epoch  39 Batch  100/269 - Train Accuracy: 0.9531, Validation Accuracy: 0.9613, Loss: 0.0271
    Epoch  39 Batch  105/269 - Train Accuracy: 0.9668, Validation Accuracy: 0.9553, Loss: 0.0260
    Epoch  39 Batch  110/269 - Train Accuracy: 0.9515, Validation Accuracy: 0.9565, Loss: 0.0256
    Epoch  39 Batch  115/269 - Train Accuracy: 0.9551, Validation Accuracy: 0.9504, Loss: 0.0281
    Epoch  39 Batch  120/269 - Train Accuracy: 0.9617, Validation Accuracy: 0.9568, Loss: 0.0243
    Epoch  39 Batch  125/269 - Train Accuracy: 0.9651, Validation Accuracy: 0.9569, Loss: 0.0250
    Epoch  39 Batch  130/269 - Train Accuracy: 0.9557, Validation Accuracy: 0.9538, Loss: 0.0301
    Epoch  39 Batch  135/269 - Train Accuracy: 0.9589, Validation Accuracy: 0.9526, Loss: 0.0236
    Epoch  39 Batch  140/269 - Train Accuracy: 0.9538, Validation Accuracy: 0.9544, Loss: 0.0277
    Epoch  39 Batch  145/269 - Train Accuracy: 0.9650, Validation Accuracy: 0.9628, Loss: 0.0246
    Epoch  39 Batch  150/269 - Train Accuracy: 0.9601, Validation Accuracy: 0.9569, Loss: 0.0296
    Epoch  39 Batch  155/269 - Train Accuracy: 0.9603, Validation Accuracy: 0.9570, Loss: 0.0239
    Epoch  39 Batch  160/269 - Train Accuracy: 0.9679, Validation Accuracy: 0.9539, Loss: 0.0225
    Epoch  39 Batch  165/269 - Train Accuracy: 0.9641, Validation Accuracy: 0.9570, Loss: 0.0254
    Epoch  39 Batch  170/269 - Train Accuracy: 0.9499, Validation Accuracy: 0.9634, Loss: 0.0245
    Epoch  39 Batch  175/269 - Train Accuracy: 0.9590, Validation Accuracy: 0.9576, Loss: 0.0331
    Epoch  39 Batch  180/269 - Train Accuracy: 0.9599, Validation Accuracy: 0.9566, Loss: 0.0237
    Epoch  39 Batch  185/269 - Train Accuracy: 0.9649, Validation Accuracy: 0.9584, Loss: 0.0246
    Epoch  39 Batch  190/269 - Train Accuracy: 0.9581, Validation Accuracy: 0.9576, Loss: 0.0275
    Epoch  39 Batch  195/269 - Train Accuracy: 0.9527, Validation Accuracy: 0.9590, Loss: 0.0227
    Epoch  39 Batch  200/269 - Train Accuracy: 0.9713, Validation Accuracy: 0.9617, Loss: 0.0199
    Epoch  39 Batch  205/269 - Train Accuracy: 0.9658, Validation Accuracy: 0.9549, Loss: 0.0223
    Epoch  39 Batch  210/269 - Train Accuracy: 0.9604, Validation Accuracy: 0.9483, Loss: 0.0216
    Epoch  39 Batch  215/269 - Train Accuracy: 0.9515, Validation Accuracy: 0.9522, Loss: 0.0244
    Epoch  39 Batch  220/269 - Train Accuracy: 0.9507, Validation Accuracy: 0.9508, Loss: 0.0264
    Epoch  39 Batch  225/269 - Train Accuracy: 0.9498, Validation Accuracy: 0.9541, Loss: 0.0253
    Epoch  39 Batch  230/269 - Train Accuracy: 0.9617, Validation Accuracy: 0.9572, Loss: 0.0232
    Epoch  39 Batch  235/269 - Train Accuracy: 0.9853, Validation Accuracy: 0.9616, Loss: 0.0194
    Epoch  39 Batch  240/269 - Train Accuracy: 0.9592, Validation Accuracy: 0.9573, Loss: 0.0238
    Epoch  39 Batch  245/269 - Train Accuracy: 0.9626, Validation Accuracy: 0.9543, Loss: 0.0210
    Epoch  39 Batch  250/269 - Train Accuracy: 0.9488, Validation Accuracy: 0.9603, Loss: 0.0253
    Epoch  39 Batch  255/269 - Train Accuracy: 0.9573, Validation Accuracy: 0.9622, Loss: 0.0275
    Epoch  39 Batch  260/269 - Train Accuracy: 0.9662, Validation Accuracy: 0.9541, Loss: 0.0246
    Epoch  39 Batch  265/269 - Train Accuracy: 0.9576, Validation Accuracy: 0.9655, Loss: 0.0263
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
      Word Ids:      [10, 104, 122, 209, 168, 171, 110]
      English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']
    
    Prediction
      Word Ids:      [284, 35, 1]
      French Words: paris . <EOS>
    

## Imperfect Translation
You might notice that some sentences translate better than others.  Since the dataset you're using only has a vocabulary of 227 English words of the thousands that you use, you're only going to see good results using these words.  For this project, you don't need a perfect translation. However, if you want to create a better translation model, you'll need better data.

You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.  However, this will take you days to train, so make sure you've a GPU and the neural network is performing well on dataset we provided.  Just make sure you play with the WMT10 corpus after you've submitted this project.
## Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_language_translation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
