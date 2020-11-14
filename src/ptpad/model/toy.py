from torch import nn

class EmbeddingAggClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        """
        Simple aggregation over embedding layer with linear classifier head
        """
        super(EmbeddingAggClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        """
        x should be a batch of LongTensor with batch_dim=0
        """
        embeddings = self.embedding(x)

        aggregated_embeddings = embeddings.mean(dim=1)

        output = self.classifier(aggregated_embeddings)

        return output

