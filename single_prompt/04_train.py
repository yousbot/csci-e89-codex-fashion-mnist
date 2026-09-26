# %%
# Run after the previous scripts in the same notebook/kernel namespace.
def evaluate_tm(model, data_loader, metric):
    model_device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    metric.to(model_device)
    metric.reset()
    try:
        with torch.inference_mode():
            for images, labels in data_loader:
                images = images.to(model_device)
                labels = labels.to(model_device)
                metric.update(model(images), labels)
        return metric.compute().item()
    finally:
        metric.reset()
        model.train(was_training)


def train2(model, optimizer, criterion, metric, train_loader, valid_loader,
           n_epochs):
    history = {
        "train_losses": [],
        "train_metrics": [],
        "valid_metrics": [],
    }
    model_device = next(model.parameters()).device
    metric.to(model_device)

    for epoch in range(n_epochs):
        model.train()
        metric.reset()
        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(model_device)
            labels = labels.to(model_device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # The criterion returns the mean loss for this batch.
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            metric.update(logits.detach(), labels)

        train_loss = total_loss / total_samples
        train_metric = metric.compute().item()
        valid_metric = evaluate_tm(model, valid_loader, metric)

        history["train_losses"].append(train_loss)
        history["train_metrics"].append(train_metric)
        history["valid_metrics"].append(valid_metric)
        print(
            f"Epoch {epoch + 1}/{n_epochs} | "
            f"loss: {train_loss:.4f} | "
            f"train accuracy: {train_metric:.4f} | "
            f"valid accuracy: {valid_metric:.4f}"
        )

    return history


optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)
n_epochs = 5
history = train2(
    model, optimizer, xentropy, metric, train_loader, valid_loader, n_epochs
)
