# %%
# Run after 04_train.py in the same notebook/kernel namespace.
epochs = range(1, len(history["train_metrics"]) + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, history["train_metrics"], marker="o", label="Training accuracy")
plt.plot(epochs, history["valid_metrics"], marker="o", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.xticks(epochs)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
