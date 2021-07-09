import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq import utils
from fairseq.models import (
    FairseqDecoder,
    FairseqLanguageModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
    TransformerSentenceEncoder,
    IRTransformerSentenceEncoder,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params

@register_model('irbert')
class IRBertModel(FairseqLanguageModel):


    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--encoder-layers', type=int, metavar='L',
                            help='num encoder layers')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='H',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='F',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='A',
                            help='num encoder attention heads')
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--embedding-normalize', action='store_true',
                            help='add layernorm after the embedding layer')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN')
        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--max-positions', type=int,
                            help='number of positional embeddings to learn')
        parser.add_argument('--load-checkpoint-heads', action='store_true',
                            help='(re-)register and load heads when loading checkpoints')
        parser.add_argument('--moco-queue-length', type=int, help='MoCo queue length')
        parser.add_argument('--moco-projection-dim', type=int, help='MoCo projection head output dim')
        parser.add_argument('--moco-temperature', type=float, help='MoCo temperature')
        parser.add_argument('--moco-momentum', type=float, help='MoCo momentum')
        parser.add_argument('--smallbert-num-encoder-layers', type=int, help='SmallBERT encoder layers')
        parser.add_argument('--smallbert-num-attention-heads', type=int, help='SmallBERT attention layers')
        parser.add_argument('--smallbert-insts-per-group', type=int, help='SmallBERT instructions per group')
        parser.add_argument('--no-pooling', action='store_true')
        parser.add_argument('--no-pce', action='store_true')

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.function_length

        encoder = IRBertEncoder(args, task.instruction_dictionary, task.state_dictionary)
        return cls(args, encoder)

    def forward(self, src, features_only=False, return_all_hiddens=False, classification_head_name=None,
        classification_head_pooling_indices=None, moco_head=True, has_state=True, **kwargs):
        
        x, extra = self.decoder(src, features_only, return_all_hiddens, moco_head=moco_head, has_state=has_state,
            has_pce=not self.args.no_pce, pooling_instruction=not self.args.no_pooling, **kwargs)

        classification_logits = None
        if classification_head_name is not None:
            feature = x[2] if features_only else x[3]
            if classification_head_pooling_indices is not None:
                feature = torch.stack([feature[x].sum(0) for x in classification_head_pooling_indices])
            classification_logits = self.classification_heads[classification_head_name](feature)
        return x, extra, classification_logits

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = IRBertClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout
        )

    @property
    def supported_targets(self):
        return {'self'}

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != '' else ''
        current_head_names = [] if not hasattr(self, 'classification_heads') else \
            self.classification_heads.keys()

        # Handle new classification heads present in the state dict.
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + 'classification_heads.') or 'amsoftmax_head' in k:
                continue

            head_name = k[len(prefix + 'classification_heads.'):].split('.')[0]
            num_classes = state_dict[prefix + 'classification_heads.' + head_name + '.out_proj.weight'].size(0)

            if getattr(self.args, 'load_checkpoint_heads', False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes)
            else:
                if head_name not in current_head_names:
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'not present in current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                ):
                    print(
                        'WARNING: deleting classification head ({}) from checkpoint '
                        'with different dimensions than current model: {}'.format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, 'classification_heads'):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + 'classification_heads.' + k not in state_dict:
                    print('Overwriting', prefix + 'classification_heads.' + k)
                    state_dict[prefix + 'classification_heads.' + k] = v

    def remove_momentum_encoder(self):
        self.decoder.remove_momentum_encoder()
    
    def remove_lm_head(self):
        self.decoder.remove_lm_head()

    def remove_state(self):
        self.decoder.remove_state()

class IRBertLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features, masked_tokens=None, **kwargs):
        # Only project the unmasked tokens while training,
        # saves both memory and computation
        if masked_tokens is not None:
            features = features[masked_tokens, :]

        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x


class IRBertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = features  # take <s> token (equiv. to [CLS])
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def copy_weights(m1, m2):
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        p2.data[:] = p1.data[:]

class IRBertEncoder(FairseqDecoder):
    """BERT encoder.

    Implements the :class:`~fairseq.models.FairseqDecoder` interface required
    by :class:`~fairseq.models.FairseqLanguageModel`.
    """

    def __init__(self, args, instruction_dictionary, state_dictionary):
        super().__init__((instruction_dictionary, state_dictionary))
        self.args = args
        def create_encoder():
            return IRTransformerSentenceEncoder(
                inst_cls_idx=instruction_dictionary.bos(),
                state_cls_idx=state_dictionary.bos(),
                inst_padding_idx=instruction_dictionary.pad(),
                inst_vocab_size=len(instruction_dictionary),
                state_padding_idx=state_dictionary.pad(),
                state_vocab_size=len(state_dictionary),
                num_encoder_layers=args.encoder_layers,
                embedding_dim=args.encoder_embed_dim,
                ffn_embedding_dim=args.encoder_ffn_embed_dim,
                num_attention_heads=args.encoder_attention_heads,
                dropout=args.dropout,
                attention_dropout=args.attention_dropout,
                activation_dropout=args.activation_dropout,
                max_seq_len=args.max_positions,
                encoder_normalize_before=args.encoder_normalize_before,
                embedding_normalize=args.embedding_normalize,
                apply_bert_init=True,
                activation_fn=args.activation_fn,
                smallbert_num_encoder_layers=self.args.smallbert_num_encoder_layers,
                smallbert_num_attention_heads=self.args.smallbert_num_attention_heads,
                smallbert_insts_per_input=self.args.smallbert_insts_per_group,
                smallbert_states_per_input=self.args.smallbert_insts_per_group,
            )
        self.sentence_encoder = create_encoder()
        self.sentence_encoder_momentum = create_encoder()
        copy_weights(self.sentence_encoder, self.sentence_encoder_momentum)
        def set_nograd(m):
            m.requires_grad_(False)
        self.sentence_encoder_momentum.apply(set_nograd)
        self.lm_head_inst = IRBertLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(instruction_dictionary),
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.inst_bert.embed_tokens.weight,
        )
        self.lm_head_state = IRBertLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(state_dictionary),
            activation_fn=args.activation_fn,
            weight=self.sentence_encoder.state_bert.embed_tokens.weight,
        )
        self.moco_head = MoCoHead(args, self.sentence_encoder, self.sentence_encoder_momentum)

    def remove_lm_head(self):
        self.lm_head_inst = nn.Identity()
        self.lm_head_state = nn.Identity()
        self.sentence_encoder.inst_bert_layers = nn.Identity()
        self.sentence_encoder.state_bert_layers = nn.Identity()

    def remove_momentum_encoder(self):
        self.moco_head = nn.Identity()
        self.sentence_encoder_momentum = nn.Identity()

    def remove_state(self):
        self.sentence_encoder.state_bert_layers = nn.Identity()
        self.sentence_encoder.state_bert = nn.Identity()
        self.lm_head_state = nn.Identity()
        if isinstance(self.sentence_encoder_momentum, IRLengthProjectionTransformerSentenceEncoder):
            self.sentence_encoder_momentum.state_bert_layers = nn.Identity()
            self.sentence_encoder_momentum.state_bert = nn.Identity()

    def forward(self, src, features_only=False, return_all_hiddens=False, masked_tokens=None, moco_head=True,
        lm_head=True, moco_head_only_proj=False, has_state=True, has_pce=True, pooling_instruction=True, **unused):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states.
        """
        x, extra = self.extract_features(src[0], self.sentence_encoder, return_all_hiddens, lm_head=lm_head and not features_only,
            has_state=has_state, has_pce=has_pce, pooling_instruction=pooling_instruction)
        x_m = None
        extra_m = None
        if moco_head and not moco_head_only_proj:
            x_m, extra_m = self.extract_features(src[1], self.sentence_encoder_momentum, return_all_hiddens,
                lm_head=lm_head and not features_only, has_state=has_state, has_pce=has_pce, pooling_instruction=pooling_instruction)
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens, moco_head=moco_head, lm_head=lm_head, moco_head_only_proj=moco_head_only_proj, features_m=x_m, has_state=has_state)
        return x, extra

    def extract_features(self, src, encoder, return_all_hiddens=False, lm_head=True, has_state=True, has_pce=True,
        pooling_instruction=True, **unused):
        inner_states, inst, state, cls_inst, cls_state = encoder(
            src[0], src[1], src[2], src[3], src[4],
            last_state_only=not return_all_hiddens,
            padding_mask=src[5],
            lm_head=lm_head,
            has_state=has_state,
            has_pce=has_pce,
            pooling_instruction=pooling_instruction,
        )
        features = (inst, state, cls_inst, cls_state)
        return features, {'inner_states': inner_states if return_all_hiddens else None}

    def output_layer(self, features, masked_tokens=None, moco_head=True, lm_head=True, moco_head_only_proj=False, features_m=None, has_state=True, **unused):
        if lm_head:
            inst = self.lm_head_inst(features[0], masked_tokens[0] if masked_tokens is not None else None)
            if has_state:
                state = self.lm_head_state(features[1], masked_tokens[1] if masked_tokens is not None else None)
            else:
                state = None
        else:
            inst = None
            state = None
        if moco_head:
            moco_output = self.moco_head(features[2], features_m[2] if features_m is not None else None, only_proj=moco_head_only_proj)
        else:
            moco_output = None
        return (inst, state, moco_output, features[2])

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions


@register_model_architecture('irbert', 'irbert')
def base_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 768)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 3072)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 12)
    args.smallbert_num_encoder_layers = getattr(args, 'smallbert_num_encoder_layers', 3)
    args.smallbert_num_attention_heads = getattr(args, 'smallbert_num_attention_heads', 12)
    args.smallbert_insts_per_group = getattr(args, 'smallbert_insts_per_group', 4)

    args.activation_fn = getattr(args, 'activation_fn', 'gelu')

    args.dropout = getattr(args, 'dropout', 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.0)
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.1)
    args.no_pce = getattr(args, 'no_pce', False)
    args.no_pooling = getattr(args, 'no_pooling', False)

    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.embedding_normalize = getattr(args, 'embedding_normalize', False)


@register_model_architecture('irbert', 'irbert_base')
def bert_base_architecture(args):
    base_architecture(args)


@register_model_architecture('irbert', 'irbert_large')
def bert_large_architecture(args):
    args.encoder_layers = getattr(args, 'encoder_layers', 24)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    base_architecture(args)

class MoCoHead(nn.Module):
    def __init__(self, args, encoder, encoder_momentum):
        super().__init__()
        self.fp16 = args.fp16
        self.encoder_embed_dim = args.encoder_embed_dim
        self.projection_dim = args.moco_projection_dim
        self.momentum = args.moco_momentum
        self.temperature = args.moco_temperature
        self.batch_size = args.max_sentences
        try:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        except:
            self.rank = 0
            self.world_size = 1
        self.update_freq = args.update_freq[0]
        self.current_update = 0
        self.queue_length = args.moco_queue_length
        self.total_batch_size = self.batch_size * self.world_size * self.update_freq
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.projection_head_layer1 = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)
        self.projection_head_layer2 = nn.Linear(self.encoder_embed_dim, self.projection_dim)
        self.projection_head_layer1_m = nn.Linear(self.encoder_embed_dim, self.encoder_embed_dim)
        self.projection_head_layer2_m = nn.Linear(self.encoder_embed_dim, self.projection_dim)
        def set_nograd(m):
            m.requires_grad_(False)
        self.projection_head_layer1_m.apply(set_nograd)
        self.projection_head_layer2_m.apply(set_nograd)
        copy_weights(self.projection_head_layer1, self.projection_head_layer1_m)
        copy_weights(self.projection_head_layer2, self.projection_head_layer2_m)
        self.encoders = (encoder, encoder_momentum) # not submodules
        self.register_buffer('queue', torch.zeros((self.queue_length, self.projection_dim)))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.batch_embedding = torch.cuda.FloatTensor(self.update_freq, self.world_size, self.batch_size, self.projection_dim).fill_(0)
        self.batch_has_data = torch.cuda.LongTensor(self.update_freq, self.world_size, self.batch_size).fill_(0)

    def update(self):
        dist.all_reduce(self.batch_embedding)
        dist.all_reduce(self.batch_has_data)
        total_batch_size = self.batch_has_data.sum().item()
        start = self.queue_ptr[0]
        end = (start + total_batch_size) % self.queue_length
        new_embedding = self.batch_embedding[self.batch_has_data.bool()]
        if end > start:
            self.queue[start:end] = new_embedding
        else:
            self.queue[start:] = new_embedding[:self.queue_length-start]
            self.queue[:total_batch_size-(self.queue_length-start)] = new_embedding[self.queue_length-start:]
        self.queue_ptr[0] = end
        self.batch_embedding[:] = 0
        self.batch_has_data[:] = 0

        def update_param(m1, m2):
            for p1, p2 in zip(m1.parameters(), m2.parameters()):
                p2.data = self.momentum * p2.data + (1 - self.momentum) * p1.data

        encoder, encoder_momentum = self.encoders
        update_param(encoder, encoder_momentum)
        update_param(self.projection_head_layer1, self.projection_head_layer1_m)
        update_param(self.projection_head_layer2, self.projection_head_layer2_m)

    def forward(self, x, x_m, only_proj=False):
        q = self.projection_head_layer1(x)
        q = self.activation_fn(q)
        q = self.projection_head_layer2(q)
        if only_proj:
            return q
        k = self.projection_head_layer1_m(x_m)
        k = self.activation_fn(k)
        k = self.projection_head_layer2_m(k)
        self.batch_embedding[self.current_update, self.rank, :k.size(0), :] = k
        self.batch_has_data[self.current_update, self.rank, :k.size(0)] = 1
        q_float = q.float()
        logits_pos = F.cosine_similarity(q_float, k.float())
        new_shape = (q.shape[0], self.queue.shape[0], self.projection_dim)
        queue = self.queue.unsqueeze(0).expand(new_shape)
        q_float = q_float.unsqueeze(1).expand(new_shape)
        logits_neg = F.cosine_similarity(q_float, queue.float(), 2)
        logits = torch.cat((logits_pos.unsqueeze(1), logits_neg), 1) / self.temperature
        self.current_update = (self.current_update + 1) % self.update_freq
        labels = torch.zeros(x.shape[0], dtype=torch.long)
        return logits, labels