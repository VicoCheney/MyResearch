import os
import pickle
import re
import datasets
from tqdm import tqdm
from transformers import AutoTokenizer
from buffer import Buffer, Block


def clean(data):
    tmp_doc = []
    for words in data.split():
        if ':' in words or '@' in words or len(words) > 60:
            pass
        else:
            c = re.sub(r'[>|-]', '', words)
            if len(c) > 0:
                tmp_doc.append(c)
    tmp_doc = ' '.join(tmp_doc)
    tmp_doc = re.sub(r'\([A-Za-z \.]*[A-Z][A-Za-z \.]*\) ', '', tmp_doc)
    return tmp_doc


def build_buffer(dataset, dataset_name):
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    data_buffers, cnt = [], 0
    for i in tqdm(range(len(dataset.data)//20)):
        if len(dataset[i]['abstract']) == 0 or len(dataset[i]['article']) == 0:
            continue
        summary = tokenizer(clean(dataset[i]['abstract']), padding="max_length", max_length=1024, return_tensors="pt", truncation=True)
        ret = Buffer(summary)
        d = tokenizer.tokenize(clean(dataset[i]['article']))
        end_tokens = {'\n': 0, '.': 1, '?': 1, '!': 1, ',': 2}
        for k, v in list(end_tokens.items()):
            end_tokens['Ġ' + k] = v
        sen_cost, break_cost = 4, 8
        poses = [(i, end_tokens[tok]) for i, tok in enumerate(d) if tok in end_tokens]
        poses.insert(0, (-1, 0))
        if poses[-1][0] < len(d) - 1:
            poses.append((len(d) - 1, 0))
        x = 0
        while x < len(poses) - 1:
            if poses[x + 1][0] - poses[x][0] > 63:
                poses.insert(x + 1, (poses[x][0] + 63, break_cost))
            x += 1
        best = [(0, 0)]
        for i, (p, cost) in enumerate(poses):
            if i == 0:
                continue
            best.append((-1, 100000))
            for j in range(i - 1, -1, -1):
                if p - poses[j][0] > 63:
                    break
                value = best[j][1] + cost + sen_cost
                if value < best[i][1]:
                    best[i] = (j, value)
            assert best[i][0] >= 0
        intervals, x = [], len(poses) - 1
        while x > 0:
            l = poses[best[x][0]][0]
            intervals.append((l + 1, poses[x][0] + 1))
            x = best[x][0]
        for st, en in reversed(intervals):
            cnt += 1
            tmp = d[st: en] + [tokenizer.sep_token]
            ret.insert(Block(tokenizer.convert_tokens_to_ids(tmp), cnt))
        data_buffers.append(ret)
    with open(os.path.join('data', f'arxiv_mini_{dataset_name}.pkl'), 'wb') as fout:
        pickle.dump(data_buffers, fout)


if __name__ == '__main__':
    # dataset = datasets.load_dataset("ccdv/govreport-summarization")
    # dataset = datasets.load_dataset("ccdv/pubmed-summarization")
    dataset = datasets.load_dataset("scientific_papers", 'arxiv')
    data_train = dataset['train']
    data_validation = dataset['validation']
    data_test = dataset['test']
    print("*********************Dataset Buffers Building Started!*********************")
    build_buffer(data_train, 'train')
    build_buffer(data_validation, 'validation')
    build_buffer(data_test, 'test')
    print("*********************Dataset Buffers Building Finished!*********************")
