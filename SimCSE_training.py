import datasets
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

# Define your sentence transformer model using CLS pooling
model_name = 'distilroberta-base'
word_embedding_model = models.Transformer(model_name, max_seq_length=128)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

dataset = datasets.load_dataset("scientific_papers", 'arxiv')

# Define a list with sentences (1k - 100k sentences)
train_sentences = []

for i in tqdm(range(len(dataset['train'].data))):
    papers = dataset['train'][i]['article']
    sentences = sent_tokenize(papers)
    for sentence in sentences:
        if 8 <= len(sentence) <= 128:
            train_sentences.append(sentence)

# Convert train sentences to sentence pairs
train_data = [InputExample(texts=[s, s]) for s in train_sentences]

# DataLoader to batch your data
train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True)

# Use the denoising auto-encoder loss
train_loss = losses.MultipleNegativesRankingLoss(model)

# Call the fit method
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    show_progress_bar=True
)

model.save('data/simcse_unsupervised_arxiv')