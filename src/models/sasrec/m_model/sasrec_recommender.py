from functools import partial
from typing import Optional, Union

import numpy as np
import torch

from .sasrec_module import SASRec
from ..utils import get_torch_device, topidx


class InvalidInputData(Exception): pass

class SASRecModel:
    def __init__(self, config: dict, n_items: int):
        self.n_items = n_items
        self.config = config
        self.device = get_torch_device(self.config.pop('device', None))
        self._model = SASRec(self.config, self.n_items).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)
        self.optimizer = torch.optim.Adam(
            self._model.parameters(), lr=self.config['learning_rate'], betas=(0.9, 0.98)
        )
        self.sampler = None
        self.n_batches = None
    
    @property
    def model(self):
        return self._model

    def recommend(self, seen_seq: Union[list, np.ndarray], topn: int, *, user: Optional[int] = None):
        '''Given an item sequence, predict top-n candidates for the next item.'''
        predictions = self.predict(seen_seq, user=user)
        np.put(predictions, seen_seq, -np.inf)
        predicted_items = topidx(predictions, topn)
        return predicted_items

    def recommend_sequential(
        self,
        target_seq: Union[list, np.ndarray],
        seen_seq: Union[list, np.ndarray],
        topn: int,
        *,
        user: Optional[int] = None
    ):
        '''Given an item sequence and a sequence of next target items,
        predict top-n candidates for each next step in the target sequence.
        '''
        predictions = self.predict_sequential(target_seq[:-1], seen_seq, user=user)
        predictions[:, seen_seq] = -np.inf
        for k in range(1, predictions.shape[0]):
            predictions[k, target_seq[:k]] = -np.inf
        predicted_items = np.apply_along_axis(topidx, 1, predictions, topn)
        return predicted_items

    def train_epoch(self, sampler, n_batches):
        model = self.model
        pad_token = model.pad_token
        criterion, optimizer, device = [
            getattr(self, a) for a in ['criterion', 'optimizer', 'device']
        ]
        l2_emb = self.config['l2_emb']
        as_tensor = partial(torch.as_tensor, dtype=torch.int32, device=device)
        loss = 0
        model.train()

        for _ in range(n_batches):
            _, *seq_data = next(sampler)
            # convert batch data into torch tensors
            seq, pos, neg = [as_tensor(arr) for arr in seq_data]
            pos_logits, neg_logits = model(seq, pos, neg)
            pos_labels = torch.ones(pos_logits.shape, device=device)
            neg_labels = torch.zeros(neg_logits.shape, device=device)
            indices = torch.where(pos != pad_token)
            batch_loss = criterion(pos_logits[indices], pos_labels[indices])
            batch_loss += criterion(neg_logits[indices], neg_labels[indices])
            if l2_emb != 0:
                for param in model.item_emb.parameters():
                    batch_loss += l2_emb * torch.norm(param)**2
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
        model.eval()
        return loss


    def predict(self, seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        with torch.no_grad():
            log_seqs = torch.full([maxlen], model.pad_token, dtype=torch.int64, device=device)
            log_seqs[-len(seq):] = torch.as_tensor(seq[-maxlen:], device=device)
            log_feats = model.log2feats(log_seqs.unsqueeze(0))
            final_feat = log_feats[:, -1, :] # only use last QKV classifier
            item_embs = model.item_emb.weight
            logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits.cpu().numpy().squeeze()

    def predict_sequential(self, target_seq, seen_seq, user):
        model = self.model
        maxlen = self.config['maxlen']
        device = self.device

        n_seen = len(seen_seq)
        n_targets = len(target_seq)
        seq = np.concatenate([seen_seq, target_seq])

        with torch.no_grad():
            pad_seq = torch.as_tensor(
                np.pad(
                    seq, (max(0, maxlen-n_seen), 0),
                    mode = 'constant',
                    constant_values = model.pad_token
                ),
                dtype = torch.int64,
                device = device
            )
            log_seqs = torch.as_strided(pad_seq[-n_targets-maxlen:], (n_targets+1, maxlen), (1, 1))
            log_feats = model.log2feats(log_seqs)
            final_feat = log_feats[:, -1, :] # only use last QKV classifier
            item_embs = model.item_emb.weight
            logits = final_feat.matmul(item_embs.T)
        return logits.cpu().numpy()
        
