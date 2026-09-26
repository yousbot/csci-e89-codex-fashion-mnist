# %%
# Run after 01_setup.py in the same notebook/kernel namespace.
transform = T.Compose([
    T.ToImage(),
    T.ToDtype(torch.float32, scale=True),
])

full_train_dataset = torchvision.datasets.FashionMNIST(
    root="datasets", train=True, download=True, transform=transform
)
test_dataset = torchvision.datasets.FashionMNIST(
    root="datasets", train=False, download=True, transform=transform
)
class_names = full_train_dataset.classes

torch.manual_seed(42)
train_dataset, val_dataset = torch.utils.data.random_split(
    full_train_dataset, [55_000, 5_000]
)

batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
valid_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False
)

sample_image, sample_label = train_dataset[0]
print(f"Shape: {sample_image.shape}")
print(f"Dtype: {sample_image.dtype}")
print(f"Class: {class_names[sample_label]}")
