# %%
# Run after the previous scripts in the same notebook/kernel namespace.
test_accuracy = evaluate_tm(model, test_loader, metric)
print(f"Test-set accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
