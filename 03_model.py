# %%
# Run after the previous scripts in the same notebook/kernel namespace.
class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, n_hidden1, n_hidden2, n_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Linear(n_hidden2, n_classes),
        )

    def forward(self, x):
        return self.layers(x)


torch.manual_seed(42)
model = ImageClassifier(784, 300, 100, 10).to(device)
xentropy = nn.CrossEntropyLoss()
