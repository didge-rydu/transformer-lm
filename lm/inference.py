import json
from pathlib import Path
from typing import List, Tuple

import sentencepiece as spm
import torch
import numpy as np

from .model import Model, HParams
from .common import END_OF_LINE, END_OF_TEXT

from torch.nn.functional import softmax

class ModelWrapper:
    END_OF_LINE = END_OF_LINE
    END_OF_TEXT = END_OF_TEXT

    def __init__(self, model: Model, sp_model: spm.SentencePieceProcessor):
        self.model = model
        self.sp_model = sp_model

    def to(self, device):
        self.device = device
        self.model.to(device)

    @classmethod
    def load(cls, root: Path):
        sp_model = spm.SentencePieceProcessor()
        sp_model.load(str(root / 'sp.model'))
        hparams = json.loads((root / 'params.json').read_text())['hparams']
        hparams.setdefault('n_hidden', hparams['n_embed'])
        model = Model(HParams(**hparams))
        state = torch.load(root / 'model.pt', map_location='cpu')
        state_dict = fixed_state_dict(state['state_dict'])
        model.load_state_dict(state_dict)

        tensor_list = list(state_dict.items())
        for layer_tensor_name, tensor in tensor_list:
            print("Layer %-42s: %9d elements" % (layer_tensor_name, torch.numel(tensor)))
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print ("Total # params: %d" % pytorch_total_params)

        return cls(model, sp_model)

    def tokenize(self, s: str) -> List[str]:
        return [self.token_to_id(x) for x in self.sp_model.EncodeAsPieces(s)]

    def token_to_id(self, token: str) -> int:
        return self.sp_model.PieceToId(token)

    def id_to_token(self, token_id: int) -> str:
        return self.sp_model.IdToPiece(int(token_id))

    def get_log_probs(self, tokens: List[str]) -> torch.Tensor:
        """ Return a tensor with shape (len(tokens), len(self.sp_model)),
        with log-probabilities for tokens after each token in tokens.
        If this is a start of the text, you may want to prepend END_OF_TEXT:
        model.get_log_probs([model.END_OF_TEXT] + tokens).
        Use model.tokenize to obtain tokens.
        """
        assert len(tokens) <= self.model.hparams.n_ctx  # TODO
        ids = [self.token_to_id(t) for t in tokens]
        ctx = torch.LongTensor(ids).unsqueeze(0)
        with torch.no_grad():
            logits = self.model(ctx)['logits'].squeeze(0)
            return torch.log_softmax(logits, dim=1)

    def get_occurred_log_probs(
            self, tokens: List[str]) -> List[Tuple[float, str]]:
        """ Return a list of log probs of actually occurred tokens,
        starting from the second.
        """
        log_probs = self.get_log_probs(tokens)
        out = []
        for idx, token in enumerate(tokens[1:]):
            out.append((float(log_probs[idx, self.token_to_id(token)]), token))
        return out

    def get_next_top_k(
            self, tokens: List[str], top_k: int) -> List[Tuple[float, str]]:
        """ Return a list of top k tuples of log prob and token,
        for what would come after the last token.
        """
        next_log_probs = self.get_log_probs(tokens)[-1]
        return sorted([(float(next_log_probs[i]), self.id_to_token(i))
                       for i in next_log_probs.argsort()[-top_k:]],
                      reverse=True)


    """
    Batch generation for sequences with same length

    param tokens_prefix: input sequences, size (batch_size, seq_length)

    """
    def generate_tokens(self, tokens_prefix: torch.LongTensor, tokens_to_generate: int, top_k: int, temperature:float = 0.1) -> List[str]:

        input_length = tokens_prefix.size(1)

        tokens = torch.zeros((tokens_prefix.size(0), input_length + tokens_to_generate), dtype=torch.long, device=self.device)
        tokens[:,:tokens_prefix.size(1)] = tokens_prefix

        for i in range(tokens_to_generate):
            pred = self.model(tokens[:,:input_length+i])['logits'][:,-1,:]
            probs_batch = softmax(pred / temperature, -1)
            for batch_idx, probs in enumerate(probs_batch):
                pred = torch.multinomial(probs,1)
                tokens[batch_idx,input_length+i] = pred

        return tokens

    def detokenize(self, tokens):
        return self.sp_model.DecodePieces([self.id_to_token(token) for token in tokens])

    def tokenize_batch(self, batch):
        batch = [self.sp_model.EncodeAsPieces(prefix) for prefix in batch]
        max_len = max(len(x) for x in batch)
        batch = [[END_OF_LINE] * (max_len - len(prefix)) + prefix for prefix in batch]
        tokens = [[self.token_to_id(x) for x in prefix] for prefix in batch]
        return tokens


def fixed_state_dict(state_dict):
    if all(k.startswith('module.') for k in state_dict):
        # legacy multi-GPU format
        state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
    return state_dict

def gen_main(model_path, prefixes, tokens_to_generate=5, top_k=8):

    device = torch.device("cpu")

    print("loading model from %s" % model_path)
    mw = ModelWrapper.load(Path(model_path))
    mw.to(device)

    print("generating text for prefix %s" % prefixes)
    tokens = mw.tokenize_batch(prefixes)
    tokens = torch.tensor(tokens, dtype=torch.long, device=mw.device)

    tokens_gen = mw.generate_tokens(tokens, tokens_to_generate, top_k)
    for tokens in tokens_gen:
        print(mw.detokenize(tokens))
