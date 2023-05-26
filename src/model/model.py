# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BART model, ported from the fairseq repo."""

import math
import random
import warnings
import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transformers.activations import ACT2FN
from transformers import BartConfig

from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPast,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
import logging

from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt import solvers
solvers.options['show_progress'] = False

tlen_for_abl = 0

logger = logging.getLogger(__name__)

_CONFIG_FOR_DOC = "BartConfig"
_TOKENIZER_FOR_DOC = "BartTokenizer"

BART_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/bart-base",
    "facebook/bart-large",
    "facebook/bart-large-mnli",
    "facebook/bart-large-cnn",
    "facebook/bart-large-xsum",
    "facebook/mbart-large-en-ro",
]
# This list is incomplete. See all BART models at https://huggingface.co/models?filter=bart


BART_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matters related to general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.

"""
BART_GENERATION_EXAMPLE = r"""
    Summarization example::

        from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

        # see ``examples/summarization/bart/run_eval.py`` for a longer example
        model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

        ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        inputs = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')

        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
               Indices of input sequence tokens in the vocabulary. Use BartTokenizer.encode to produce them.
            Padding will be ignored by default should you provide it.
            Indices can be obtained using :class:`transformers.BartTokenizer.encode(text)`.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`, defaults to :obj:`None`):
            Tuple consists of (`last_hidden_state`, `optional`: `hidden_states`, `optional`: `attentions`)
            `last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`) is a sequence of hidden-states at the output of the last layer of the encoder.
            Used in the cross-attention of the decoder.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for translation and summarization training. By default, the model will create this tensor by shifting the input_ids right, following the paper.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
            Default behavior: generate a tensor that ignores pad tokens in decoder_input_ids. Causal mask will also be used by default.
            If you want to change padding behavior, you should read :func:`~transformers.modeling_bart._prepare_decoder_inputs` and modify.
            See diagram 1 in the paper for more info on the default strategy
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains pre-computed key and value hidden-states of the attention blocks.
            Can be used to speed up decoding.
            If ``past_key_values`` are used, the user can optionally input only the last
            ``decoder_input_ids`` (those that don't have their past key value states given to this model) of shape
            :obj:`(batch_size, 1)` instead of all ``decoder_input_ids`` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If `use_cache` is True, ``past_key_values`` are returned and can be used to speed up decoding (see
            ``past_key_values``).
        output_attentions (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the attentions tensors of all attention layers are returned. See ``attentions`` under returned tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the hidden states of all layers are returned. See ``hidden_states`` under returned tensors for more detail.
        return_dict (:obj:`bool`, `optional`, defaults to :obj:`None`):
            If set to ``True``, the model will return a :class:`~transformers.file_utils.ModelOutput` instead of a
            plain tuple.
"""


def invert_mask(attention_mask):
    """Turns 1->0, 0->1, False->True, True-> False"""
    assert attention_mask.dim() == 2, f"Attn mask: {attention_mask.shape}"
    return attention_mask.eq(0)


def _prepare_bart_decoder_inputs(
        config, input_ids, decoder_input_ids=None, decoder_padding_mask=None, causal_mask_dtype=torch.float32
):
    """Prepare masks that ignore padding tokens in the decoder and a causal mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()
    if decoder_padding_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
    else:
        decoder_padding_mask = invert_mask(decoder_padding_mask)
    if decoder_padding_mask is not None and decoder_padding_mask.shape[1] > 1:
        # never mask leading token, even if it is pad
        decoder_padding_mask[:, 0] = decoder_padding_mask[:, 1]
    causal_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1).to(
        dtype=causal_mask_dtype, device=decoder_input_ids.device
    )
    return decoder_input_ids, decoder_padding_mask, causal_mask


class PretrainedBartModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, SinusoidalPositionalEmbedding):
            pass
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


# Helper Modules


class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = Attention(self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout)
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, output_attentions=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights = self.self_attn(query=x, key=x, key_padding_mask=encoder_padding_mask, output_attentions=output_attentions)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights



class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens
        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
            self.padding_idx,
            2,
        )
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim) if config.normalize_embedding else nn.Identity()
        # mbart has one extra layer_norm
        self.layer_norm = LayerNorm(config.d_model) if config.normalize_before else None

    def forward(
            self, input_ids, emo_labels=None,attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            BaseModelOutput or Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (tuple(torch.FloatTensor)): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *output_hidden_states:* is True.
                - **all_attentions** (tuple(torch.FloatTensor)): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            attention_mask = invert_mask(attention_mask)

        inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = [] if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for encoder_layer in self.layers:
            if output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask, output_attentions=output_attentions)

            if output_attentions:
                all_attentions = all_attentions + (attn,)

        if self.layer_norm:
            x = self.layer_norm(x)
        if output_hidden_states:
            encoder_states.append(x)
            # T x B x C -> B x T x C
            encoder_states = tuple(hidden_state.transpose(0, 1) for hidden_state in encoder_states)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if not return_dict:
            return tuple(v for v in [x, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(last_hidden_state=x, hidden_states=encoder_states, attentions=all_attentions)


class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = Attention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention='src_decoder',
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.knowl_attn = Attention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention='knowl_decoder',  # fix this.
        )
        self.knowl_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.kno_gate = nn.Parameter(torch.randn(1))

        self.knowl_gate_w = nn.Linear(self.embed_dim * 2, 1)

    def forward(
            self,
            x,
            encoder_hidden_states,
            lid,
            encoder_attn_mask=None,
            layer_state=None,
            causal_mask=None,
            decoder_padding_mask=None,
            output_attentions=False,
    ):
        src_hidden_states, knowl_hidden_states = encoder_hidden_states
        src_hidden_attn_mask, knowl_attn_mask = encoder_attn_mask

        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        # print("cross attn", x.shape, src_hidden_states.shape, src_hidden_attn_mask.shape)
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, _ = self.encoder_attn(
            query=x,
            key=src_hidden_states,
            key_padding_mask=src_hidden_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Knowledge attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.knowl_attn_layer_norm(x)

        # shared the weight
        x, kno_attn_weight = self.knowl_attn(
            query=x,
            key=knowl_hidden_states,
            key_padding_mask=knowl_attn_mask,
            layer_state=layer_state,  # mutates layer state
            output_attentions=True
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        weight = self.knowl_gate_w(torch.cat([x, residual], dim=-1)).sigmoid()
        x = (1 - weight) * residual + weight * x
        # x = residual + x

        if not self.normalize_before:
            x = self.knowl_attn_layer_norm(x)
        # print(kno_attn_weight.shape)
        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            kno_attn_weight,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
        self.embed_tokens = embed_tokens
        self.embed_positions = LearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.d_model,
                self.padding_idx,
                2,
            )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model) if config.normalize_embedding else nn.Identity()
        self.layer_norm = LayerNorm(config.d_model) if config.add_final_layer_norm else None

    def forward(
            self,
            input_ids,
            encoder_hidden_states,
            encoder_padding_mask,
            decoder_padding_mask,
            decoder_causal_mask,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
            **unused,
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            past_key_values (dict or None): dictionary used for storing state during generation

        Returns:
            BaseModelOutputWithPast or tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - the cache
                - hidden states


                - attentions
        """
        if "decoder_cached_states" in unused:
            warnings.warn(
                "The `decoder_cached_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = unused.pop("decoder_past_key_values")

        # check attention mask and invert
        for i in range(len(encoder_padding_mask)):
            if encoder_padding_mask[i] is not None:
                # print("decoder phase", encoder_padding_mask[i].shape)
                encoder_padding_mask[i] = invert_mask(encoder_padding_mask[i])
        # embed positions
        # logger.info(input_ids.shape)
        positions = self.embed_positions(input_ids, use_cache=use_cache)

        if use_cache:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            # assert input_ids.ne(self.padding_idx).any()
        # logger.info(input_ids.shape)
        x = self.embed_tokens(input_ids) * self.embed_scale
        # print(x.shape, positions.shape)
        x += positions
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Convert to Bart output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        x = x.transpose(0, 1)
        encoder_hidden_states = [ehc.transpose(0, 1) for ehc in encoder_hidden_states]
        # encoder_hidden_states = encoder_hidden_states.transpose(0, 1)
        # logger.info(x.shape)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = []

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (x,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = past_key_values[idx] if past_key_values is not None else None
            # logger.info(x.shape)
            x, layer_self_attn, layer_kno_attn, layer_past = decoder_layer(
                x,
                encoder_hidden_states,
                idx + 1,
                encoder_attn_mask=encoder_padding_mask,
                decoder_padding_mask=decoder_padding_mask,
                layer_state=layer_state,
                causal_mask=decoder_causal_mask,
                output_attentions=output_attentions,
            )
            # print(layer_kno_attn.shape)
            if use_cache:
                next_decoder_cache.append(layer_past.copy())

            if self.layer_norm and (idx == len(self.layers) - 1):  # if config.add_final_layer_norm (mBART)
                x = self.layer_norm(x)
            if output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert to standard output format: (seq_len, BS, model_dim) -> (BS, seq_len, model_dim)
        if output_hidden_states:
            all_hidden_states = tuple(hidden_state.transpose(0, 1) for hidden_state in all_hidden_states)
        x = x.transpose(0, 1)
        # encoder_hidden_states = [ehc.transpose(0, 1) for ehc in encoder_hidden_states]
        # encoder_hidden_states = encoder_hidden_states.transpose(0, 1)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [x, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=x, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns
        )


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim,
            num_heads,
            dropout=0.0,
            bias=True,
            encoder_decoder_attention='self',  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = self.encoder_decoder_attention
        # self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, seq_len, bsz):
        return tensor.contiguous().view(seq_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
            self,
            query,
            key: Optional[Tensor],
            key_padding_mask: Optional[Tensor] = None,
            layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
            attn_mask: Optional[Tensor] = None,
            output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = 'self' not in self.encoder_decoder_attention

        # static_kv: bool = self.encoder_decoder_attention != 'self'
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state and static_kv:
                # previous time steps are cached - no need to recompute key and value if they are static
                key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        # if key_padding_mask is not None:
        #     print(key_padding_mask.size(), bsz, src_len, k.shape, v.shape, self.cache_key)

        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz,src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.masked_fill(reshaped, -10000.0)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get("prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return k, v, new_key_padding_mask


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
            self,
            input_dim,
            inner_dim,
            num_classes,
            pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int, offset):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models dont have this hack
        self.offset = offset
        assert padding_idx is not None
        num_embeddings += offset
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions + self.offset)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


# Public API
def _get_shape(t):
    return getattr(t, "shape", None)


# @add_start_docstrings(
#     "The bare BART Model outputting raw hidden-states without any specific head on top.",
#     BART_START_DOCSTRING,
# )
class BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
        # self.kno_emb = nn.Embedding(100, config.d_model)
        self.encoder = BartEncoder(config, self.shared)
        # method 1
        # self.gatew = nn.Linear(config.d_model * 2, config.d_model, bias=False)
        # self.gatev = nn.Linear(config.d_model, 1)
        # method 2
        self.gatew = nn.Linear(config.d_model * 2, 1)
        self.queryw = nn.Linear(config.d_model, config.d_model, bias=True)
        self.knowl_encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.init_weights()

    def load_addi_weights(self):
        logger.info("Init the params of knowledge encoder with utterance encoder.")
        encoder_dict = self.encoder.state_dict()
        knowl_encoder_dict = self.knowl_encoder.state_dict()
        for name in knowl_encoder_dict:
            knowl_encoder_dict[name].copy_(encoder_dict[name])
        # self.knowl_encoder.load_state_dict(self.encoder.state_dict())
        state_dict = self.decoder.state_dict()
        for name in state_dict:
            if 'knowl_attn' in name:
                pretrain_name = name.replace('knowl_attn', 'encoder_attn')
                state_dict[name].copy_(state_dict[pretrain_name])


    def add_knowl_pos(self, knowl, src_outputs, num_kno, emo_labels, emo_ln, know_single_bundle):
        """
        knowl.shape=(bsz * num_kno, kno_len, d_model)
        """
        bsz_mul_num_kno, kno_len, d_model = knowl.shape
        bsz = bsz_mul_num_kno // num_kno
        # print("emo_labels",emo_labels)

        knowl = knowl.reshape(bsz, num_kno, kno_len, d_model)

        # if emo_ln.weight.requires_grad :
        try:
            emo_ln.requires_grad_(True)
            if emo_labels.dim() == 2:
                emo_labels = emo_labels.squeeze(1)
            # print(emo_ln.weight)
            # print(know_single_bundle[0].shape)
            pred_utter = emo_ln(src_outputs[:,0])
            label_ctx_emo = torch.argmax(pred_utter, dim=1)
            # random.shuffle(know_single_bundle)
            # know_single_bundle1 = copy.deepcopy(know_single_bundle) +  copy.deepcopy(know_single_bundle)+  copy.deepcopy(know_single_bundle)+  copy.deepcopy(know_single_bundle)
            # print("random.shuffle!!!!!!!!!!!!!!!!!!")
            pred_li = [emo_ln(item[:,0,:].reshape(bsz,d_model) ) for item in know_single_bundle ]
            loss_li = []
            grad_li = []
            for i,item in enumerate(pred_li):
                criterion = nn.CrossEntropyLoss()
                criterion.requires_grad = True
                loss_li.append(criterion(item, emo_labels)) # emo_labels
                loss_li[-1].backward()
                grad_li.append(emo_ln.weight.grad.cpu().detach().numpy())
            grad_out,lambda_ = cal_grad(grad_list=grad_li,cost_list=loss_li,m=24576,size_in=768,size_out=32)
            grad_ = torch.from_numpy(grad_out).type(torch.float32)
            grad_mean = torch.mean(grad_,0).cuda()
            # print("lambda_",grad_.shape,lambda_,grad_mean.shape) # torch.Size([32, 768])

            emo_ln.zero_grad() # clear the grad trunk

            # emo_logits = emo_ln(knowl[:,0,0,:].reshape(bsz,d_model) )
            # print(type(emo_logits),type(emo_labels))
            utter = knowl[:, :, 0, :].unsqueeze(-2).expand(bsz, num_kno, kno_len, d_model)
            knowl = torch.add(knowl, grad_mean[ None,None,None,:])

        # emo_loss = nn.CrossEntropyLoss()(emo_logits, emo_labels)

        # print("loss_li",loss_li)
        except:
            with torch.set_grad_enabled(True):
                # emo_ln_copy = copy.deepcopy(emo_ln)
                know_single_bundle = self.know_single_bundle
                # print(id(emo_ln_copy),id(emo_ln))
                emo_ln.requires_grad_(True)
                if emo_labels.dim() == 2:
                    emo_labels = emo_labels.squeeze(1)
                # print("emo_labels",emo_labels.shape)
                # print(know_single_bundle[0].shape)

                # utter = knowl[:, :, 0, :].unsqueeze(-2).expand(bsz, num_kno, kno_len, d_model)
                # print("src_outputs.shape",src_outputs.shape)
                # fsadfasf
                pred_utter = emo_ln(src_outputs[:,0])
                label_ctx_emo = torch.argmax(pred_utter, dim=1)
                # print("pred_utter",label_ctx_emo)
                # print("emo_label", emo_labels)

                pred_li = [emo_ln(item[:,0,:].reshape(bsz,d_model) ) for item in know_single_bundle ]
                # print(pred_li[-1])
                loss_li = []
                grad_li = []

                for i,item in enumerate(pred_li):
                    criterion = nn.CrossEntropyLoss()
                    criterion.requires_grad = True
                    loss_li.append(criterion(item, label_ctx_emo))
                    loss_li[-1].backward()
                    grad_li.append(emo_ln.weight.grad.cpu().detach().numpy())
                # print(pred_li[-1],loss_li[-1])
                grad_out,lambda_ = cal_grad(grad_list=grad_li,cost_list=loss_li,m=24576,size_in=768,size_out=32)
                grad_ = torch.from_numpy(grad_out).type(torch.float32)
                # ()
                grad_mean = torch.mean(grad_,0).cuda()
                # print("lambda_",grad_.shape,lambda_,grad_mean.shape) # torch.Size([32, 768])
                emo_ln.zero_grad() # clear the grad trunk

                # emo_logits = emo_ln(knowl[:,0,0,:].reshape(bsz,d_model) )
                # print(type(emo_logits),type(emo_labels))
                utter = knowl[:, :, 0, :].unsqueeze(-2).expand(bsz, num_kno, kno_len, d_model)
                knowl = torch.add(knowl, grad_mean[ None,None,None,:]) # knowl + grad_mean

        coef = self.gatew(torch.cat([knowl, utter], dim=-1)).sigmoid()
        # print(knowl.shape,utter.shape, coef.shape) #torch.Size([10, 5, 38, 768]) torch.Size([10, 5, 38, 768])
        # (bsz, num_kno, kno_len, d_model)
        knowl = coef * knowl + (1 - coef) * utter # knowl + grad_mean
        # print(knowl.shape)
        # knowl_for_rnn = knowl.reshape(bsz, num_kno * kno_len, d_model)
        # out, _ = self.lstm(knowl_for_rnn)
        knowl = knowl.reshape(bsz * num_kno, kno_len, d_model)
        return knowl

    def add_knowl_pos_inferr(self, knowl, num_kno, src_outputs, emo_ln, know_single_bundle):
        """
        knowl.shape=(bsz * num_kno, kno_len, d_model)
        """
        bsz_mul_num_kno, kno_len, d_model = knowl.shape
        bsz = bsz_mul_num_kno // num_kno
        # print("emo_labels",emo_labels)

        knowl = knowl.reshape(bsz, num_kno, kno_len, d_model)
        # if emo_ln.weight.requires_grad :

        with torch.set_grad_enabled(True):
            # emo_ln_copy = copy.deepcopy(emo_ln)
            know_single_bundle = self.know_single_bundle
            # print(id(emo_ln_copy),id(emo_ln))
            emo_ln.requires_grad_(True)

            pred_utter = emo_ln(src_outputs[:,0])
            label_ctx_emo = torch.argmax(pred_utter, dim=1)


            pred_li = [emo_ln(item[:,0,:].reshape(bsz,d_model) ) for item in know_single_bundle ]
            # emo_labels = [F.softmax(emo_ln(item[:,0,:].reshape(bsz,d_model) )) for item in know_single_bundle ]
            # print("emo_labels",emo_labels,)
            loss_li = []
            grad_li = []
            for i,item in enumerate(pred_li):
                criterion = nn.CrossEntropyLoss()
                criterion.requires_grad = True
                loss_li.append(criterion(item, label_ctx_emo)) # emo_labels
                loss_li[-1].backward()
                grad_li.append(emo_ln.weight.grad.cpu().detach().numpy())
            grad_out,lambda_ = cal_grad(grad_list=grad_li,cost_list=loss_li,m=24576,size_in=768,size_out=32)
            grad_ = torch.from_numpy(grad_out).type(torch.float32)
            # ()
            grad_mean = torch.mean(grad_,0).cuda()
            # print("lambda_",grad_.shape,lambda_,grad_mean.shape) # torch.Size([32, 768])

            emo_ln.zero_grad() # clear the grad trunk

            utter = knowl[:, :, 0, :].unsqueeze(-2).expand(bsz, num_kno, kno_len, d_model)
            knowl = torch.add(knowl, grad_mean[ None,None,None,:]) # knowl + grad_mean

        coef = self.gatew(torch.cat([knowl, utter], dim=-1)).sigmoid()

        knowl = coef * knowl + (1 - coef) * utter

        knowl = knowl.reshape(bsz * num_kno, kno_len, d_model)
        return knowl

    def add_knowl_pos_infer(self, knowl, num_kno):
        """
        knowl.shape=(bsz * num_kno, kno_len, d_model)
        """
        bsz_mul_num_kno, kno_len, d_model = knowl.shape
        bsz = bsz_mul_num_kno // num_kno

        knowl = knowl.reshape(bsz, num_kno, kno_len, d_model)

        utter = knowl[:, :, 0, :].unsqueeze(-2).expand(bsz, num_kno, kno_len, d_model)

        coef = self.gatew(torch.cat([knowl, utter], dim=-1)).sigmoid()
        knowl = coef * knowl + (1 - coef) * utter
        knowl = knowl.reshape(bsz * num_kno, kno_len, d_model)

        return knowl

    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_input_ids=None,
            encoder_outputs: Optional[Tuple] = None,
            decoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            emo_labels=None,
            emo_ln = None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")

        if decoder_input_ids is None:
            use_cache = False

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # make masks if user doesn't supply
        if not use_cache:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_padding_mask=decoder_attention_mask,
                causal_mask_dtype=self.shared.weight.dtype,
            )
        else:
            decoder_padding_mask, causal_mask = None, None

        assert decoder_input_ids is not None
        if input_ids is not None:
            bsz, max_num_knowl, max_knowl_len = input_ids[1].shape
            knowl_bsz = bsz
        else:
            knowl_bsz = encoder_outputs[1]["last_hidden_state"].shape[0]
            bsz = encoder_outputs[0]["last_hidden_state"].shape[0]
            max_knowl_len = encoder_outputs[1]["last_hidden_state"].shape[1]
            max_num_knowl = knowl_bsz // bsz
        kno_weight = None
        if encoder_outputs is None:
            src_outputs = self.encoder(
                input_ids=input_ids[0],
                attention_mask=attention_mask[0],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # print("emo_labels",emo_labels)
            # input_ids[1] (batch_size, max_num_knowl, max_knowl_len) -> (batch_size * max_num_knowl, max_knowl_len)
            # print("input_ids[1]",input_ids[1][:, 0, :].shape)
            knowl_outputs = self.knowl_encoder(
                input_ids=input_ids[1].reshape(-1, max_knowl_len),
                attention_mask=attention_mask[1].reshape(-1, max_knowl_len),
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            self.know_single_bundle = [self.knowl_encoder(
                input_ids=input_ids[1][:, i, :],
                attention_mask=attention_mask[1][:, i, :],
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )["last_hidden_state"] for i in range(5) ]

            self.name_space = ['X_intent','X_need','X_want','X_effect','X_react']

            bsz_mul_num_kno, kno_len, d_model = knowl_outputs["last_hidden_state"].shape

            if emo_labels.dim() == 2: emo_labels = emo_labels.squeeze(1)
            # print("self.know_single_bundle ",self.know_single_bundle[0].shape,emo_labels )
            self.name_space_new = []
            for i in range(bsz):
                pred_li = [emo_ln(item[i,0,:].reshape(1,d_model) )
                                    for item in self.know_single_bundle ]
                criterion = nn.CrossEntropyLoss()
                loss_li = [criterion(qq, torch.unsqueeze(emo_labels[i],dim=-1)).item() for qq in pred_li ]
                loss_li_copy = copy.deepcopy(loss_li)
                loss_li_copy.sort(reverse=True)
                list_index = [ int(loss_li.index(item)) for item in loss_li_copy  ]
                self.name_space_new.append(list_index)
                # self.name_space_new.append([ self.name_space[num] for num in list_index ])
                # print(self.name_space_new)
                # print("emo_labels[i]",emo_labels[i].reshape(-1,1),)
                # while len(pred_li) > 1:
                #     loss_li = [criterion(qq, torch.unsqueeze(emo_labels[i],dim=-1)).item() for qq in pred_li ]
                #     print(loss_li)
                #     loss_argmax = loss_li.index(max(loss_li))

                #     print(loss_argmax,self.name_space[int(loss_argmax)])
                #     pred_li.pop(loss_argmax)



            # print("input_ids[1][:, i, :]",input_ids[1][:, 0, :].shape,input_ids[1][:, 1, :].shape)
            # knowl_outputs_1 = self.knowl_encoder(
            #     input_ids=input_ids[1][:, 0, :],
            #     attention_mask=attention_mask[1][:, 0, :],
            #     output_attentions=output_attentions,
            #     output_hidden_states=output_hidden_states,
            #     return_dict=return_dict,
            # )
            # (bsz, d_model)
            # print('knowl_outputs_1["last_hidden_state"]',knowl_outputs_1["last_hidden_state"].shape)

            kno_last_hidden_state = self.add_knowl_pos(knowl_outputs["last_hidden_state"], src_outputs[0],
                                                        max_num_knowl, emo_labels, emo_ln, self.know_single_bundle)
            knowl_outputs["last_hidden_state"] = kno_last_hidden_state
            encoder_outputs = (src_outputs, knowl_outputs)
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOuput when return_dict=False
        elif return_dict and not isinstance(encoder_outputs[0], BaseModelOutput):
            assert 0

        src_hidden = encoder_outputs[0]["last_hidden_state"]
        knowl_pool_hidden = encoder_outputs[1]["last_hidden_state"].reshape(bsz, -1, src_hidden.shape[-1])
        attention_mask = [attention_mask[0], attention_mask[1].reshape(bsz, -1)]

        decoder_outputs = self.decoder(
            decoder_input_ids,
            [src_hidden, knowl_pool_hidden],
            # encoder_outputs[0],
            attention_mask,
            decoder_padding_mask,
            decoder_causal_mask=causal_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # logger.info(decoder_outputs.last_hidden_state.shape)
        # if not return_dict: print(not return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        # print(encoder_outputs[0].attentions.shape, encoder_outputs[1].attentions.shape)
        # assert 0
        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            encoder_last_hidden_state=[encoder_outputs[0].last_hidden_state, encoder_outputs[1].last_hidden_state],
            encoder_hidden_states=[encoder_outputs[0].hidden_states, encoder_outputs[1].hidden_states],
            encoder_attentions=[encoder_outputs[0].attentions, encoder_outputs[1].attentions],

        ),self.name_space_new

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


# @add_start_docstrings(
#     "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
# )


class SinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions, embedding_dim, padding_idx=None):
        super().__init__(num_positions, embedding_dim)
        if embedding_dim % 2 != 0:
            raise NotImplementedError(f"odd embedding_dim {embedding_dim} not supported")
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """Identical to the XLM create_sinusoidal_embeddings except features are not interleaved.
        The cos features are in the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
        )
        out[:, 0: dim // 2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))  # This line breaks for odd n_pos
        out[:, dim // 2:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        out.requires_grad = False
        return out

    @torch.no_grad()
    def forward(self, input_ids, use_cache=False):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input_ids.shape[:2]
        if use_cache:
            positions = input_ids.data.new(1, 1).fill_(seq_len - 1)  # called before slicing
        else:
            # starts at 0, ends at 1-seq_len
            positions = torch.arange(seq_len, dtype=torch.long, device=self.weight.device)
        return super().forward(positions)



def get_Gh(grad_list,cost_list_,m):
    cost_list = [cost_list_[i].cpu().detach().numpy() for i in range(len(cost_list_)) ]
#     print(cost_list)
    N = len(cost_list)
    G = np.zeros([N,m])
    b = []

    for i in range(N):
#         print(grad_list[i][0])
        g = grad_list[i].flatten()
#         print(g.shape)
        G[i][:] = g
#         G[i][-1] = -1.0
        b.append(float(cost_list[i])) # add cost

    b = np.array(b)
#     print(b)
    GG = matrix(G)
    hh = matrix(b)
    # print(GG)
    return GG,hh


def cal_grad(grad_list, cost_list,m,size_in,size_out):

    N = len(cost_list)

    GG,hh = get_Gh(grad_list,cost_list,m)
    P = matrix(GG)*matrix(GG).T

    q = -matrix(hh)
    # print(P,q)
    G = matrix(-np.eye(N))
    h = matrix(np.zeros(N))
    A = matrix(np.ones([1,N]))
    b = matrix(np.ones([1]))

#     print(0)
    res = qp(P,q,G=G,h=h,A=A,b=b)
    # print(1)
    d = -np.array(GG).T.dot(np.array(res['x']))[:,0].reshape(size_out,size_in)

    return d,np.array(res['x'])
