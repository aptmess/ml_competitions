import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Input,
    Bidirectional,
    LSTM,
    concatenate,
    Dropout
)
from tensorflow.keras.models import Model
from transformers import (
    TFDistilBertModel,
    TFRobertaModel,
    TFBertModel
)


def distilbert_model(input_shape,
                     model_name,
                     transformer_model,
                     output_shape=15,
                     output_activation='softmax',
                     optimizer='Adam',
                     optimizer_params=None,
                     loss='categorical_crossentropy',
                     metrics=None,
                     experiment=False):
    if optimizer_params is None:
        optimizer_params = {'lr': 1e-5}
    input_ids = Input((input_shape,), dtype=tf.int32)
    input_mask = Input((input_shape,), dtype=tf.int32)
    if model_name == 'DISTILBERT':
        transformer_encoder = TFDistilBertModel.from_pretrained(
            transformer_model,
            from_pt=True,
            output_hidden_states=True
        )
        outputs = transformer_encoder.distilbert(input_ids,
                                                 attention_mask=input_mask)
    elif model_name == 'ROBERTA':
        transformer_encoder = TFRobertaModel.from_pretrained(
            transformer_model,
            from_pt=True,
            output_hidden_states=True
        )
        outputs = transformer_encoder.roberta(input_ids,
                                              attention_mask=input_mask)
    elif model_name == 'BERT':
        transformer_encoder = TFBertModel.from_pretrained(
            transformer_model,
            from_pt=True,
            output_hidden_states=True
        )
        outputs = transformer_encoder.bert(input_ids,
                                           attention_mask=input_mask)
    elif model_name == 'PRROBERTA':
        transformer_encoder = TFRobertaModel.from_pretrained(
            "DeepPavlov/rubert-base-cased-sentence",
            from_pt=True,
            output_hidden_states=True
        )
        outputs = transformer_encoder.roberta(input_ids,
                                              attention_mask=input_mask)

    else:
        raise ValueError(f'unknown model type {model_name}')
    if not experiment:
        x = outputs[0]
        x = GlobalAveragePooling1D()(x)
        output = Dense(output_shape,
                       activation=output_activation)(x)

        model = Model(inputs=[input_ids, input_mask],
                      outputs=output)
        model.compile(loss=loss,
                      metrics=metrics,
                      optimizer=getattr(optimizers, optimizer)(
                          **optimizer_params)
                      )
    else:
        transformer_encoder.trainable = False
        x = outputs[0]
        bi_lstm = Bidirectional(LSTM(64, return_sequences=True))(x)
        avg_pool = GlobalAveragePooling1D()(bi_lstm)
        max_pool = GlobalMaxPooling1D()(bi_lstm)
        concat = concatenate([avg_pool, max_pool])
        dropout = Dropout(0.3)(concat)
        output = Dense(output_shape,
                       activation=output_activation)(dropout)

        model = Model(inputs=[input_ids, input_mask],
                      outputs=output)
        model.compile(loss=loss,
                      metrics=metrics,
                      optimizer=getattr(optimizers, optimizer)(
                          **optimizer_params)
                      )
    return model
