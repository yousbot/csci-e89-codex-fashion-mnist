# %%
# Run after the previous scripts in the same notebook/kernel namespace.
valid_loader = val_loader
images, labels = next(iter(valid_loader))
images, labels = images[:3], labels[:3]

model.eval()
with torch.no_grad():
    logits = model(images.to(device))
    probabilities = F.softmax(logits, dim=1)
    predictions = probabilities.argmax(dim=1)
    top_probabilities, top_indices = probabilities.topk(4, dim=1)

# Move to CPU before rounding, including when predictions were made on MPS.
probabilities = probabilities.cpu()
predictions = predictions.cpu()
top_probabilities = top_probabilities.cpu()
top_indices = top_indices.cpu()
labels = labels.cpu()
rounded_probabilities = np.round(probabilities.numpy(), decimals=3)

print(f"Probability class order: {class_names}")
for i in range(len(images)):
    print(f"\nImage {i + 1}")
    print(f"Predicted: {class_names[predictions[i].item()]}")
    print(f"True: {class_names[labels[i].item()]}")
    print(f"Softmax probabilities: {rounded_probabilities[i]}")
    print("Top-4 classes:")
    for class_index, probability in zip(top_indices[i], top_probabilities[i]):
        print(f"  {class_names[class_index.item()]}: {probability.item():.3f}")
