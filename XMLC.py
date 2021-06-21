#!/usr/bin/env python
# coding: utf-8

import os
import collections
import bert
import bert.modeling
import bert.optimization
import pandas
import tensorflow
from datetime import datetime


TRAIN_DATA ='C:/Users/kgpaz/Desktop/project/data/eco100.csv'
TEST_DATA ='C:/Users/kgpaz/Desktop/project/data/eco100.csv'

ID = 'id'
DATA_COLUMN = 'title'
LABEL_COLUMNS = ['labels','fold']


BERT_VOCAB = 'C:/Users/kgpaz/Desktop/project/bertconfig/vocab.txt'
BERT_CHECKPOINT = 'C:/Users/kgpaz/Desktop/project/bertconfig/bert_model.ckpt'
BERT_CONFIG = 'C:/Users/kgpaz/Desktop/project/bertconfig/bert_config.json'

MAX_SEQ_TOKENS = 128

MODEL_DIR = 'C:/Users/kgpaz/Desktop/project/model'

train = pandas.read_csv(TRAIN_DATA)
test = pandas.read_csv(TEST_DATA)

train.head()

bert.bert_tokenization.validate_case_matches_checkpoint(True,BERT_CHECKPOINT)

tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file=BERT_VOCAB, do_lower_case=True)


class InputExample(object):
    def __init__(self, text1, text2=None, labels=None):
        self.text1 = text1  # text for the first sequence
        self.text2 = text2  # text for the second sequence used for sequence pairs
        self.labels = labels  # the label of the example used for training


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_ids, is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids,
        self.is_real_example=is_real_example

# Creates examples for the training 
def create_examples(df):
    examples = []
    for (i, row) in enumerate(df.values):
        text1 = row[1]  # text is located at the second column on csv
        labels = [0,0]
        examples.append(
            InputExample(text1=text1, labels=labels))
    return examples

# 90% of examples for training 10% for test
TRAIN_VAL_RATIO = 0.9 
LEN = train.shape[0]
TRAIN_SIZE = int(TRAIN_VAL_RATIO*LEN)

x_train = train[:TRAIN_SIZE]
x_val = train[TRAIN_SIZE:]

train_examples = create_examples(x_train)

def convert_examples_to_features(examples,  MAX_SEQ_TOKENS, tokenizer):
    # InputFeatures from a data file

    features = []
    for (ex_index, example) in enumerate(examples):
        print(example.text1)
        tokens1 = tokenizer.tokenize(example.text1)

        tokens2 = None
        if example.text2:
            tokens2 = tokenizer.tokenize(example.text2)
            _truncate_seq_pair(tokens1, tokens2, MAX_SEQ_TOKENS - 3)
        else:
            if len(tokens1) > MAX_SEQ_TOKENS - 2:
                tokens1 = tokens1[:(MAX_SEQ_TOKENS - 2)]

        tokens = ["[CLS]"] + tokens1 + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens2:
            tokens += tokens2 + ["[SEP]"]
            segment_ids += [1] * (len(tokens2) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. 
        input_mask = [1] * len(input_ids)

        # Zero padding up to length of the sequence.
        padding = [0] * (MAX_SEQ_TOKENS - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == MAX_SEQ_TOKENS
        assert len(input_mask) == MAX_SEQ_TOKENS
        assert len(segment_ids) == MAX_SEQ_TOKENS
        
        labels_ids = []
        for label in example.labels:
            labels_ids.append(int(label))

        if ex_index < 0:
            logger.info("---- Example ----")
            logger.info("Tokens:\t%s" % " ".join([str(x) for x in tokens]))
            logger.info("Input IDs:\t%s" % " ".join([str(x) for x in input_ids]))
            logger.info("Input Mask:\t%s" % " ".join([str(x) for x in input_mask]))
            logger.info("Segment IDs:\t%s" % " ".join([str(x) for x in segment_ids]))
            logger.info("Labels:\t%s" % (example.labels))
            logger.info("Label IDs:\t%s" % (labels_ids))

        features.append(
                            InputFeatures(
                                              input_ids=input_ids,
                                              input_mask=input_mask,
                                              segment_ids=segment_ids,
                                              label_ids=labels_ids
                                         )
                       )
    return features


# Create Bert Classification Model  
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    model = bert.modeling.BertModel(
                                        config=bert_config,
                                        is_training=is_training,
                                        input_ids=input_ids,
                                        input_mask=input_mask,
                                        token_type_ids=segment_ids,
                                        use_one_hot_embeddings=use_one_hot_embeddings
                                   )


    # model.get_sequence_output()
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tensorflow.get_variable(
                                "output_weights", [num_labels, hidden_size],
                                    initializer=tensorflow.truncated_normal_initializer(stddev=0.02)
                                                )

    output_bias = tensorflow.get_variable("output_bias", [num_labels],
                                         initializer=tensorflow.zeros_initializer())

    with tensorflow.variable_scope("loss"):
        if is_training:
            output_layer = tensorflow.nn.dropout(output_layer, keep_prob=0.9) # 0.1 dropout

        logits = tensorflow.matmul(output_layer, output_weights, transpose_b=True)
        logits = tensorflow.nn.bias_add(logits, output_bias)
        
        # multi-label case
        probabilities = tensorflow.nn.sigmoid(logits)
        
        labels = tensorflow.cast(labels, tensorflow.float32)
        tensorflow.logging.info("num_labels:{};logits:{};labels:{}".format(num_labels, logits, labels))
        per_example_loss = tensorflow.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        loss = tensorflow.reduce_mean(per_example_loss)

        ## multiclass case
        # probabilities = tensorflow.nn.softmax(logits, axis=-1)
        # log_probs = tensorflow.nn.log_softmax(logits, axis=-1)
        #
        # one_hot_labels = tensorflow.one_hot(labels, depth=num_labels, dtype=tensorflow.float32)
        #
        # per_example_loss = -tensorflow.reduce_sum(one_hot_labels * log_probs, axis=-1)
        # loss = tensorflow.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


# model_fn closure for TPUEstimator
def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):

    def model_fn(features, labels, mode, params):  

        #tensorflow.logging.info("---- Features ----")
        #for name in sorted(features.keys()):
        #    tensorflow.logging.info("\tname = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
             is_real_example = tensorflow.cast(features["is_real_example"], dtype=tensorflow.float32)
        else:
             is_real_example = tensorflow.ones(tensorflow.shape(label_ids), dtype=tensorflow.float32)

        is_training = (mode == tensorflow.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tensorflow.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names ) = bert.modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tensorflow.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tensorflow.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tensorflow.train.init_from_checkpoint(init_checkpoint, assignment_map)

        # tensorflow.logging.info("---- Trainable Variables ----")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *FROM_CHECKPOINT*"
        #     tensorflow.logging.info("  name = %s, shape = %s%s", var.name, var.shape,init_string)

        output_spec = None
        if mode == tensorflow.estimator.ModeKeys.TRAIN:

            train_op = bert.optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tensorflow.estimator.EstimatorSpec(
                                                                mode=mode,
                                                                loss=total_loss,
                                                                train_op=train_op,
                                                                scaffold=scaffold_fn
                                                            )
        elif mode == tensorflow.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, probabilities, is_real_example):

                logits_split = tensorflow.split(probabilities, num_labels, axis=-1)
                label_ids_split = tensorflow.split(label_ids, num_labels, axis=-1)
                # metrics change to auc of every class
                eval_dict = {}
                for j, logits in enumerate(logits_split):
                    label_id_ = tensorflow.cast(label_ids_split[j], dtype=tensorflow.int32)
                    current_auc, update_op_auc = tensorflow.metrics.auc(label_id_, logits)
                    eval_dict[str(j)] = (current_auc, update_op_auc)
                eval_dict['eval_loss'] = tensorflow.metrics.mean(values=per_example_loss)
                return eval_dict

            eval_metrics = metric_fn(per_example_loss, label_ids, probabilities, is_real_example)
            output_spec = tensorflow.estimator.EstimatorSpec(
                                                                mode=mode,
                                                                loss=total_loss,
                                                                eval_metric_ops=eval_metrics,
                                                                scaffold=scaffold_fn
                                                            )
        else:
            print("mode:", mode,"probabilities:", probabilities)
            output_spec = tensorflow.estimator.EstimatorSpec(
                                                                mode=mode,
                                                                predictions={"probabilities": probabilities},
                                                                scaffold=scaffold_fn
                                                            )
        return output_spec

    return model_fn

# Train and warmup steps from batch size
BATCH_SIZE = 32
LEARNING_RATE = 2e-5 #  1/50000
NUM_TRAIN_EPOCHS = 2.0

# Warmup is a period of time where hte learning rate is small and gradually increases--usually helps training.
WARMUP_PROPORTION = 0.1
# Model Configs
SAVE_CHECKPOINTS_STEPS = 1000
SAVE_SUMMARY_STEPS = 500

run_config = tensorflow.estimator.RunConfig(
    model_dir=MODEL_DIR,
    save_summary_steps=SAVE_SUMMARY_STEPS,
    keep_checkpoint_max=1,
    save_checkpoints_steps=SAVE_CHECKPOINTS_STEPS)
    
    
# Preprocessing according to Bert https://arxiv.org/abs/1810.04805


# input_fn closure for TPUEstimator
def input_fn_builder(features, seq_length, is_training, drop_remainder):

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_ids)

    def input_fn(params):
        batch_size = params["batch_size"]

        num_examples = len(features)

        d = tensorflow.data.Dataset.from_tensor_slices({
            "input_mask":
                tensorflow.constant(
                                all_input_mask,
                                shape=[num_examples, seq_length],
                                dtype=tensorflow.int32
                            ),
            "input_ids":
                tensorflow.constant(
                                all_input_ids, 
                                shape=[num_examples, seq_length],
                                dtype=tensorflow.int32
                           ),
            "segment_ids":
                tensorflow.constant(
                                all_segment_ids,
                                shape=[num_examples, seq_length],
                                dtype=tensorflow.int32
                            ),
            "label_ids":
                tensorflow.constant(
                                all_label_ids, 
                                shape=[num_examples, 
                                len(LABEL_COLUMNS)], 
                                dtype=tensorflow.int32
                           ),
        })

        if is_training:
          d = d.repeat()
          d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


class NoneInputExamplePadding(object):
    """
    Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. We use this class instead of `None` because it could cause silent errors.
    """

# Convert InputExample to InputFeatures
def convert_single_example(ex_index, example, MAX_SEQ_TOKENS, tokenizer):

    if isinstance(example, NoneInputExamplePadding):
        return InputFeatures(
            input_ids=[0] * MAX_SEQ_TOKENS,
            input_mask=[0] * MAX_SEQ_TOKENS,
            segment_ids=[0] * MAX_SEQ_TOKENS,
            label_ids=0,
            is_real_example=False)
            
    tokens1 = tokenizer.tokenize(example.text1)
    tokens2 = None
    if example.text2:
        tokens2 = tokenizer.tokenize(example.text2)

    if tokens2:
        # Modifies tokens1 and tokens2 so that the total length 
        # is less than the specified length.
        _truncate_seq_pair(tokens1, tokens2, MAX_SEQ_TOKENS - 3)
    else:
        if len(tokens1) > MAX_SEQ_TOKENS - 2:
            tokens1 = tokens1[0:(MAX_SEQ_TOKENS - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens1:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens2:
        for token in tokens2:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < MAX_SEQ_TOKENS:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == MAX_SEQ_TOKENS
    assert len(input_mask) == MAX_SEQ_TOKENS
    assert len(segment_ids) == MAX_SEQ_TOKENS

    labels_ids = []
    for label in example.labels:
        labels_ids.append(int(label))


    features = InputFeatures(
                                input_ids=input_ids,
                                input_mask=input_mask,
                                label_ids=labels_ids,
                                segment_ids=segment_ids,
                                is_real_example=True
                           )
    return features


# A set of InputExample to TFRecord file
def file_based_convert_examples_to_features(
        examples, MAX_SEQ_TOKENS, tokenizer, output_file):

    writer = tensorflow.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):

        feature = convert_single_example(
                                            ex_index, 
                                            example,
                                            MAX_SEQ_TOKENS, 
                                            tokenizer
                                        )

        def create_int_feature(values):
            return tensorflow.train.Feature(int64_list=tensorflow.train.Int64List(value=list(values)))

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        if isinstance(feature.label_ids, list):
            label_ids = feature.label_ids
        else:
            label_ids = feature.label_ids[0]
        features["label_ids"] = create_int_feature(label_ids)

        tf_example = tensorflow.train.Example(features=tensorflow.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


# input_fn closure for TPUEstimator."""
def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder):

    name_to_features = {
        "input_ids": tensorflow.FixedLenFeature([seq_length], tensorflow.int64),
        "input_mask": tensorflow.FixedLenFeature([seq_length], tensorflow.int64),
        "segment_ids": tensorflow.FixedLenFeature([seq_length], tensorflow.int64),
        "label_ids": tensorflow.FixedLenFeature([2], tensorflow.int64),
        "is_real_example": tensorflow.FixedLenFeature([], tensorflow.int64),
    }

    # Decodes a record to a TensorFlow example.
    def _decode_record(record, name_to_features):
        example = tensorflow.parse_single_example(record, name_to_features)

        # tensorflow.Example only supports tensorflow.int64, but the TPU only supports tensorflow.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tensorflow.int64:
                t = tensorflow.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tensorflow.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tensorflow.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


# Simple heuristic which will always truncate the longer sequence one token at a time.
# This makes more sense than truncating an equal percent of tokens from each, 
# since if one sequence is very short then each token that's truncated likely 
# contains more information than a longer sequence.
def _truncate_seq_pair(tokens1, tokens2, max_length):
    while True:
        total_length = len(tokens1) + len(tokens2)
        if total_length <= max_length:
            break
        if len(tokens1) > len(tokens2):
            tokens1.pop()
        else:
            tokens2.pop()

#   TRAINING

train_file = os.path.join(MODEL_DIR, "train.record")

if not os.path.exists(train_file):
    open(train_file, 'w').close()

num_train_steps = int(len(train_examples) / BATCH_SIZE * NUM_TRAIN_EPOCHS)
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

file_based_convert_examples_to_features(
            train_examples, MAX_SEQ_TOKENS, tokenizer, train_file)
tensorflow.logging.info("---- Running training ----")
tensorflow.logging.info("Examples = %d", len(train_examples))
tensorflow.logging.info("Batch Size = %d", BATCH_SIZE)
tensorflow.logging.info("Steps = %d", num_train_steps)

train_input_fn = file_based_input_fn_builder(
                                                input_file=train_file,
                                                seq_length=MAX_SEQ_TOKENS,
                                                is_training=True,
                                                drop_remainder=True
                                            )


bert_config = bert.modeling.BertConfig.from_json_file(BERT_CONFIG)

model_fn = model_fn_builder(
                              bert_config=bert_config,
                              num_labels= len(LABEL_COLUMNS),
                              init_checkpoint=BERT_CHECKPOINT,
                              learning_rate=LEARNING_RATE,
                              num_train_steps=num_train_steps,
                              num_warmup_steps=num_warmup_steps,
                              use_tpu=False,
                              use_one_hot_embeddings=False
                           )

estimator = tensorflow.estimator.Estimator(
                                              model_fn=model_fn,
                                              config=run_config,
                                              params={"batch_size": BATCH_SIZE}
                                          )
                                          
print('Training Started!')
current_time = datetime.now()
estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
print('Training completed in ', datetime.now() - current_time)

# EVALUATION 

eval_file = os.path.join(MODEL_DIR, "eval.record")

if not os.path.exists(eval_file):
    open(eval_file, 'w').close()

eval_examples = create_examples(x_val)
file_based_convert_examples_to_features(
    eval_examples, MAX_SEQ_TOKENS, tokenizer, eval_file)

eval_steps = None

eval_drop_remainder = False
eval_input_fn = file_based_input_fn_builder(
                                                input_file=eval_file,
                                                seq_length=MAX_SEQ_TOKENS,
                                                is_training=False,
                                                drop_remainder=False
                                            )
print('Evalution Started!')
current_time = datetime.now()
result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
print('Evalution completed in ', datetime.now() - current_time)

output_eval_file = os.path.join(MODEL_DIR, "eval_results.txt")
with tensorflow.gfile.GFile(output_eval_file, "w") as writer:
    tensorflow.logging.info("---- Eval results ----")
    for key in sorted(result.keys()):
        tensorflow.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
