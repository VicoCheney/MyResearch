# import datasets
# import numpy
# from sentence_transformers import SentenceTransformer, InputExample
# from sentence_transformers import models, losses
# word_embedding_model = SentenceTransformer('data/simcse_unsupervised_arxiv')
# import torch.nn.functional as F
# import numpy as np
#
# s1 = "what's your age?"
# s2 = "how old are you"
#
# e1=word_embedding_model.encode(s1)
# e2=word_embedding_model.encode(s2)
#
# ee1 = np.linalg.norm(e1)
# ee2 = np.linalg.norm(e2)
# print(np.dot(e1, e2)/(ee1 * ee2))

import torch
print(torch.__version__)
print(torch.cuda.is_available())#cuda是否可用
torch.cuda.device_count()#返回GPU的数量
torch.cuda.get_device_name(0)#返回gpu名字，设备索引默认从0开始
