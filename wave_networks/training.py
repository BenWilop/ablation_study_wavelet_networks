import torch as t
import numpy as np
import copy
from torch.utils.data import TensorDataset, DataLoader
import wandb
from datetime import datetime
from wave_networks.prepare_data import DatasetAudioTimeseries


def evaluate_model(model, loss_function, data, labels):
    model.eval()
    with t.no_grad():
        logits = model(data.unsqueeze(1))  # [B, T] -> [B, 1, T] to add channel
        loss = loss_function(logits, labels)
        accuracy = (t.argmax(logits, 1) == labels).float().mean()
    model.train()
    return loss, accuracy


def train_model(model: t.nn.Module, model_name, dataset_audio_timeseries: DatasetAudioTimeseries, train_parameters, folder) -> t.nn.Module:
    """
    Code is partially from https://github.com/dwromero/wavelet_networks/blob/master/experiments/UrbanSound8K/trainer.py
    """
    wandb.finish()  # In case previous run did not get finished
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_name = f"experiment_{model_name}_{timestamp}"
    wandb.init(
        project="gdl_mini_project",
        name=experiment_name,
        config = train_parameters
    )

    loss_function = t.nn.CrossEntropyLoss()
    best_validation_model_state = _train(model, loss_function, dataset_audio_timeseries, train_parameters)
    model.load_state_dict(best_validation_model_state)
    t.save(best_validation_model_state, folder + "/" + experiment_name)

    # Test
    test_loss, test_accuracy = evaluate_model(model,
                                              loss_function,
                                              dataset_audio_timeseries.test_data, 
                                              dataset_audio_timeseries.test_labels
                                              )
    wandb.log({"test/loss": test_loss, "test/accuracy": test_accuracy})

    wandb.finish()
    return model


def _train(model, loss_function, dataset_audio_timeseries, train_parameters):
    train_dataset = TensorDataset(
        dataset_audio_timeseries.train_data.unsqueeze(1),  # [B, 1, t]
        dataset_audio_timeseries.train_labels,  # [B, n_classes]
    )
    train_loader = DataLoader(train_dataset, batch_size=train_parameters['batch_size'], shuffle=True)

    optimizer = t.optim.Adam(model.parameters(), lr=train_parameters['learning_rate'], weight_decay=train_parameters['weight_decay'])

    best_validation_loss = float('inf')
    best_validation_model_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    patience_counter = 0
    for epoch in range(train_parameters['max_epochs']):
        print("------------- EPOCH {epoch} -------------")
        # Train
        for i, (data, labels) in enumerate(train_loader):
            # if i % 10 == 0:
            print(i)
            optimizer.zero_grad()
            logits = model(data)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

            # Log
            accuracy = (t.argmax(logits, 1) == labels).float().mean()
            wandb.log({"train/loss": loss, "train/accuracy": accuracy})

        # Validation
        validation_loss, validation_accuracy = evaluate_model(model, 
                                                              loss_function,
                                                              dataset_audio_timeseries.validation_data, 
                                                              dataset_audio_timeseries.validation_labels
                                                              )
        if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_model_state = copy.deepcopy(model.state_dict())  # checkpoint best model
                best_epoch = epoch
                patience_counter = 0
        else:
            patience_counter += 1
        
        wandb.log({"validation/loss": validation_loss, "validation/accuracy": validation_accuracy, "validation/best_epoch": best_epoch})

        if patience_counter >= train_parameters['early_stopping_patience']:
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    return best_validation_model_state
        