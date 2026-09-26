# %%
# Run after the previous scripts in the same notebook/kernel namespace.
def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_hidden = trial.suggest_int("n_hidden", 20, 300)

    torch.manual_seed(42)
    trial_model = ImageClassifier(784, n_hidden, 100, 10).to(device)
    trial_optimizer = torch.optim.SGD(
        trial_model.parameters(), lr=learning_rate
    )
    trial_criterion = nn.CrossEntropyLoss()
    trial_metric = torchmetrics.Accuracy(
        task="multiclass", num_classes=10
    ).to(device)

    for epoch in range(3):
        print(f"Trial {trial.number}, epoch {epoch + 1}/3")
        trial_history = train2(
            trial_model, trial_optimizer, trial_criterion, trial_metric,
            train_loader, valid_loader, n_epochs=1
        )
        validation_accuracy = trial_history["valid_metrics"][-1]
        trial.report(validation_accuracy, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return validation_accuracy


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    # Allow pruning in a two-trial study after one completed reference trial.
    pruner=optuna.pruners.MedianPruner(n_startup_trials=1),
)
study.optimize(objective, n_trials=2)

print(f"best_value: {study.best_value:.4f}")
print(f"best_params: {study.best_params}")
