# %%
# Run after the previous scripts in the same notebook/kernel namespace.
from pathlib import Path

model_path = Path("models/my_fashion_mnist_model.pt")
model_path.parent.mkdir(parents=True, exist_ok=True)

hyperparameters = {
    "n_inputs": model.layers[1].in_features,
    "n_hidden1": model.layers[1].out_features,
    "n_hidden2": model.layers[3].out_features,
    "n_classes": model.layers[5].out_features,
}
torch.save(
    {"state_dict": model.state_dict(), "hyperparameters": hyperparameters},
    model_path,
)

checkpoint = torch.load(model_path, map_location=device, weights_only=True)
loaded_model = ImageClassifier(**checkpoint["hyperparameters"]).to(device)
loaded_model.load_state_dict(checkpoint["state_dict"])

model.eval()
loaded_model.eval()
with torch.no_grad():
    prediction_images = images[:3].to(device)
    original_outputs = model(prediction_images)
    loaded_outputs = loaded_model(prediction_images)

predictions_match = torch.allclose(original_outputs, loaded_outputs)
assert predictions_match, "Reloaded model outputs do not match the original model."
print(f"Saved and reloaded model: {model_path}")
print(f"Predictions match on the 3 images: {predictions_match}")
