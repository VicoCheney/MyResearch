import logging
import os
import pickle
import random
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from simcse import SimCSE
from tqdm import tqdm
from transformers import AutoTokenizer
from buffer import Buffer
from nltk.tokenize import sent_tokenize
from numpy import mean
from sklearn.metrics.pairwise import cosine_similarity
from utils import compress
import torch.nn.functional as F
from rouge_score import rouge_scorer


class DataDriver:
    def __init__(self, config):
        self.config = config
        with open(config.train_source, 'rb') as fin:
            logging.info('Loading dataset...')
            self.dataset = pickle.load(fin)
        self.d = {}
        for data_buf in self.dataset:
            for block in data_buf:
                assert block.pos not in self.d
                self.d[block.pos] = block

    def apply_changes(self, tmp_dir):
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('changes'):
                with open(filename, 'r') as fin:
                    for line in fin:
                        tmp = [
                            int(s) if s.isdigit() or s[0] == '-' and s[1:].isdigit() else s
                            for s in line.split()
                        ]
                        setattr(self.d[tmp[0]], tmp[1], tmp[2])
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))

    def collect_estimations(self, tmp_dir):
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('estimations'):
                with open(filename, 'r') as fin:
                    for line in fin:
                        l = line.split()
                        pos, estimation = int(l[0]), float(l[1])
                        self.d[pos].estimation = estimation
                os.replace(filename, os.path.join(tmp_dir, 'backup_' + shortname))

    def build_random_buffer(self, num_samples):
        n0, n1 = [int(s) for s in num_samples.split(',')]
        ret = []
        max_blk_num = 16
        logging.info("⚡⚡⚡⚡⚡⚡⚡⚡⚡Building Random Buffers for Bert-training⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        for data_buf in tqdm(self.dataset):
            # ret_temp = Buffer(data_buf.summary)
            # ret += ret_temp.fill(data_buf)
            for i in range(n0):
                st = random.randint(0, max(0, len(data_buf) - max_blk_num))
                buf = Buffer(data_buf.summary)
                buf.blocks = data_buf.blocks[st + i * max_blk_num:st + (i + 1) * max_blk_num]
                ret.append(buf.sort_())
            pbuf, nbuf = data_buf.filtered(lambda blk, idx: blk.relevance >= 1, need_residue=True)
            for i in range(n1):
                selected_pblks = random.sample(pbuf.blocks, min(max_blk_num, len(pbuf)))
                selected_nblks = random.sample(nbuf.blocks, min(max_blk_num - len(selected_pblks), len(nbuf)))
                buf = Buffer(data_buf.summary)
                buf.blocks = selected_pblks + selected_nblks
                ret.append(buf.sort_())
        return ret

    def build_promising_buffer(self):
        ret = []
        max_blk_num = 32
        logging.info("⚡⚡⚡⚡⚡⚡⚡⚡⚡Building Random Buffers for Bart-training⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        for data_buf in tqdm(self.dataset):
            if len(data_buf) <= 32:
                ret.append(data_buf)
            else:
                estimations = torch.tensor([blk.estimation for blk in data_buf], dtype=torch.float)
                indices = estimations.argsort(descending=True)

                # buf1 = Buffer(data_buf.summary)
                # count = 0
                # nblk = []
                # for i in indices.tolist():
                #     if count == 13:
                #         if len(nblk) >= 3:
                #             break
                #         else:
                #             if data_buf[i].relevance <= 0:
                #                 nblk.append(data_buf[i])
                #     else:
                #         if data_buf[i].relevance > 0:
                #             buf1.blocks.append(data_buf[i])
                #             count += 1
                #         else:
                #             nblk.append(data_buf[i])
                # if count < max_blk_num:
                #     buf1.blocks += nblk[:(max_blk_num-count)]
                # assert len(buf1.blocks) == max_blk_num
                # ret.append(buf1.sort_())

                buf2 = Buffer(data_buf.summary)
                count2 = 0
                for j in indices.tolist():
                    if count2 == 32:
                        break
                    else:
                        buf2.blocks.append(data_buf[j])
                        count2 += 1
                assert len(buf2.blocks) == max_blk_num
                ret.append(buf2.sort_())
        return ret

    def save_relevance(self):
        self.__file = open(os.path.join(self.config.tmp_dir, 'changes.txt'), 'w')
        for data_buf in self.dataset:
            for block in data_buf:
                self.__file.write('{} {} {}\n'.format(block.pos, 'relevance', block.relevance))
        self.__file.close()

    def cross_encoder_initialize(self):
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡Start Initialization⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        model = CrossEncoder('cross-encoder/stsb-roberta-large')
        self._file = open(os.path.join(self.config.tmp_dir, 'initialization.txt'), 'w')
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        for data_buf in tqdm(self.dataset):
            summary = tokenizer.decode(data_buf.summary['input_ids'][0], skip_special_tokens=True)
            sents_pairs = []
            for blk in data_buf:
                sents_pairs.append([summary, str(blk)])
            if len(sents_pairs) != 0:
                scores = model.predict(sents_pairs, batch_size=128)
                max_score = max(scores)
                if max_score > 0:
                    for i, blk in enumerate(data_buf):
                        if 1 - scores[i] / max_score < 0.15:
                            self._file.write('{} {} {}\n'.format(blk.pos, 'relevance', blk.relevance + 1))
        self._file.close()
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡Finish Initialization⚡⚡⚡⚡⚡⚡⚡⚡⚡")

    # def bi_encoder_initialize(self):
    #     print("⚡⚡⚡⚡⚡⚡⚡⚡⚡Start Initialization⚡⚡⚡⚡⚡⚡⚡⚡⚡")
    #     model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    #     self._file = open(os.path.join(self.config.tmp_dir, 'initialization.txt'), 'w')
    #     tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    #     for data_buf in tqdm(self.dataset):
    #
    #         summary = tokenizer.decode(data_buf.summary['input_ids'][0], skip_special_tokens=True)
    #         summary_sentences = sent_tokenize(summary)
    #         sum_sens = []
    #         temp_sen = ""
    #         for i, sen in enumerate(summary_sentences):
    #             if len(temp_sen.split()) < len(summary.split())//2:
    #                 if len(temp_sen.split()) + len(sen.split()) <= len(summary.split())//2:
    #                     temp_sen += sen
    #                 else:
    #                     sum_sens.append(temp_sen)
    #                     sum_sens.append(summary_sentences[i:])
    #         if len(temp_sen.split()) > 0:
    #             sum_sens.append(temp_sen)
    #
    #         blks_np = model.encode([str(blk) for blk in data_buf.blocks])
    #         cos_sim = [F.cosine_similarity(vec1, blk_np, dim=0) for blk_np in blks_np]
    #
    #         summary = tokenizer.decode(data_buf.summary['input_ids'][0], skip_special_tokens=True)
    #         summary_np = np.array(model.encode(summary))
    #         blks_np = np.array(model.encode([str(blk) for blk in data_buf.blocks]))
    #         scores = [summary_np.dot(blk_np) / np.linalg.norm(summary_np) * np.linalg.norm(blk_np) for blk_np in blks_np]
    #         if len(scores) != 0:
    #             max_score = max(scores)
    #             if max_score > 0:
    #                 for i, blk in enumerate(data_buf):
    #                     if 1 - scores[i] / max_score < 0.12:
    #                         self._file.write('{} {} {}\n'.format(blk.pos, 'relevance', blk.relevance + 1))
    #         # indices = torch.tensor(scores).argsort(descending=True)
    #         # if len(indices) < 32:
    #         #     good = indices[:(len(indices) // 2)]
    #         #     bad = indices[-(len(indices) // 2 + 1):]
    #         # else:
    #         #     good = indices[:16]
    #         #     bad = indices[-16:]
    #         # for i, blk in enumerate(data_buf):
    #         #     if i in good:
    #         #         self._file.write('{} {} {}\n'.format(blk.pos, 'relevance', blk.relevance + 1))
    #         #     if i in bad:
    #         #         self._file.write('{} {} {}\n'.format(blk.pos, 'relevance', blk.relevance - 1))
    #     self._file.close()
    #     print("⚡⚡⚡⚡⚡⚡⚡⚡⚡Finish Initialization⚡⚡⚡⚡⚡⚡⚡⚡⚡")

    def rouge_initialize(self):
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡Start Initialization⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        self._file = open(os.path.join(self.config.tmp_dir, 'initialization.txt'), 'w')
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        for data_buf in tqdm(self.dataset):
            summary = tokenizer.decode(data_buf.summary['input_ids'][0], skip_special_tokens=True)
            scores = []
            for blk in data_buf.blocks:
                res = scorer.score(str(blk), summary)
                scores.append(res['rouge1'].fmeasure / 2 + res['rougeL'].fmeasure)
            if len(scores) != 0:
                max_score = max(scores)
                if max_score > 0:
                    for i, blk in enumerate(data_buf):
                        if 1 - scores[i] / max_score < 0.2:
                            self._file.write('{} {} {}\n'.format(blk.pos, 'relevance', blk.relevance + 1))
        self._file.close()
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡Finish Initialization⚡⚡⚡⚡⚡⚡⚡⚡⚡")

    # def SimCSE_initialize(self):
    #     print("⚡⚡⚡⚡⚡⚡⚡⚡⚡Start Initialization⚡⚡⚡⚡⚡⚡⚡⚡⚡")
    #     model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    #     tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    #     self._file = open(os.path.join(self.config.tmp_dir, 'initialization.txt'), 'w')
    #     for data_buf in tqdm(self.dataset):
    #         summary = tokenizer.decode(data_buf.summary['input_ids'][0], skip_special_tokens=True)
    #         summary_sentences = sent_tokenize(summary)
    #         sum_sens = []
    #         temp_sen = ""
    #         for sen in summary_sentences:
    #             if len(temp_sen.split()) < 128:
    #                 if len(temp_sen.split()) + len(sen.split()) <= 128:
    #                     temp_sen += sen
    #                 else:
    #                     sum_sens.append(temp_sen)
    #                     temp_sen = ""
    #         if len(temp_sen.split()) > 0:
    #             sum_sens.append(temp_sen)
    #         scores = []
    #         for blk in data_buf.blocks:
    #             scores.append(mean(model.similarity(str(blk), sum_sens)))
    #         if len(scores) != 0:
    #             max_score = max(scores)
    #             if max_score > 0:
    #                 for i, blk in enumerate(data_buf):
    #                     if 1 - scores[i] / max_score < 0.1:
    #                         self._file.write('{} {} {}\n'.format(blk.pos, 'relevance', blk.relevance + 1))
    #     self._file.close()
    #     print("⚡⚡⚡⚡⚡⚡⚡⚡⚡Finish Initialization⚡⚡⚡⚡⚡⚡⚡⚡⚡")

    def SimCSE_initialize(self):
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡Start Initialization⚡⚡⚡⚡⚡⚡⚡⚡⚡")
        model = SentenceTransformer('../data/simcse_unsupervised_arxiv')
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        self._file = open(os.path.join(self.config.tmp_dir, 'initialization.txt'), 'w')
        for data_buf in tqdm(self.dataset):
            summary = tokenizer.decode(data_buf.summary['input_ids'][0], skip_special_tokens=True)
            summary_sentences = sent_tokenize(summary)
            sum_sens = []
            temp_sen = ""
            for sen in summary_sentences:
                if len(temp_sen.split()) < 128:
                    if len(temp_sen.split()) + len(sen.split()) <= 128:
                        temp_sen += sen
                    else:
                        sum_sens.append(temp_sen)
                        temp_sen = ""
            if len(temp_sen.split()) > 0:
                sum_sens.append(temp_sen)
            scores = []
            sum_embedding = model.encode(sum_sens)
            for blk in data_buf.blocks:
                score = 0
                for i in sum_embedding:
                    score += cosine_similarity(model.encode(str(blk)), i)
                scores.append(score / len(sum_embedding))
            if len(scores) != 0:
                max_score = max(scores)
                if max_score > 0:
                    for i, blk in enumerate(data_buf):
                        if 1 - scores[i] / max_score < 0.1:
                            self._file.write('{} {} {}\n'.format(blk.pos, 'relevance', blk.relevance + 1))
        self._file.close()
        print("⚡⚡⚡⚡⚡⚡⚡⚡⚡Finish Initialization⚡⚡⚡⚡⚡⚡⚡⚡⚡")

    def cosine_similarity(x, y):
        num = x.dot(y.T)
        denom = np.linalg.norm(x) * np.linalg.norm(y)
        return num / denom

    def apply_initialization(self, tmp_dir):
        for shortname in os.listdir(tmp_dir):
            filename = os.path.join(tmp_dir, shortname)
            if shortname.startswith('initialization'):
                with open(filename, 'r') as fin:
                    for line in fin:
                        tmp = [
                            int(s) if s.isdigit() or s[0] == '-' and s[1:].isdigit() else s
                            for s in line.split()
                        ]
                        setattr(self.d[tmp[0]], tmp[1], tmp[2])
                print("initialization used!")

    def build_compressed_dataset(self, bert_model):
        ret = []
        with torch.no_grad():
            for data_buf in tqdm(self.dataset):
                compressed_buf = compress(bert_model.compresser, data_buf, self.config.times, "cuda:0", self.config.batch_size_inference)
                ret.append(compressed_buf)
        return ret
