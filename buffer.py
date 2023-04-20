import random
import torch
from transformers import AutoTokenizer


class Block:
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    def __init__(self, ids, pos):
        self.ids = ids
        self.pos = pos
        self.relevance = 0
        self.estimation = 0

    def __lt__(self, rhs):
        return self.pos < rhs.pos

    def __ne__(self, rhs):
        return self.pos != rhs.pos

    def __len__(self):
        return len(self.ids)

    def __str__(self):
        return Block.tokenizer.convert_tokens_to_string(Block.tokenizer.convert_ids_to_tokens(self.ids))


class Buffer:

    def __init__(self, summary):
        self.blocks = []
        self.summary = summary

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, key):
        return self.blocks[key]

    def __str__(self):
        return ''.join([str(b) + '\n' for b in self.blocks])

    def clone(self):
        ret = Buffer(self.summary)
        ret.blocks = self.blocks.copy()
        return ret

    def calc_size(self):
        return sum([len(b) for b in self.blocks])

    def block_ends(self):
        t, ret = 0, []
        for b in self.blocks:
            t += len(b)
            ret.append(t)
        return ret

    def insert(self, b, reverse=True):
        if not reverse:
            for index in range(len(self.blocks) + 1):
                if index >= len(self.blocks):
                    self.blocks.insert(index, b)
                    break
        else:
            for index in range(len(self.blocks), -1, -1):
                if index == 0:
                    self.blocks.insert(index, b)
                    break

    def fill(self, buf):
        ret, tmp_buf, tmp_size = [], self.clone(), self.calc_size()
        for blk in buf:
            if tmp_size + len(blk) > 512:
                ret.append(tmp_buf)
                tmp_buf, tmp_size = self.clone(), self.calc_size()
            tmp_buf.blocks.append(blk)
            tmp_size += len(blk)
        ret.append(tmp_buf)
        return ret

    def filtered(self, fltr: 'function blk, index->bool', need_residue=False):
        ret, ret2 = Buffer(self.summary), Buffer(self.summary)
        for i, blk in enumerate(self.blocks):
            if fltr(blk, i):
                ret.blocks.append(blk)
            else:
                ret2.blocks.append(blk)
        if need_residue:
            return ret, ret2
        else:
            return ret

    def random_sample(self, size):
        assert size <= len(self.blocks)
        index = sorted(random.sample(range(len(self.blocks)), size))
        ret = Buffer(self.summary)
        ret.blocks = [self.blocks[i] for i in index]
        return ret

    def sort_(self):
        self.blocks.sort()
        return self

    def export(self, out, device=None):
        ids, att_masks = out
        att_masks.zero_()
        t = 0
        for b in self.blocks:
            ids[t:t + len(b)] = torch.tensor(b.ids, dtype=torch.long, device=device)
            att_masks[t:t + len(b)] = 1
            t += len(b)
        return ids, att_masks

    def export_relevance(self, out):
        relevance = out
        t = 0
        for b in self.blocks:
            if b.relevance >= 1:
                relevance[t: t + len(b)] = 1
            t += len(b)
        return relevance


def buffer_collate(batch):
    return batch
