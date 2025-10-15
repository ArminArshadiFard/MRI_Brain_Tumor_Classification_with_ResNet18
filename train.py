import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.utils import compute_class_weight
from torch.optim import lr_scheduler
import yaml
import time
import copy
import os
from tqdm import tqdm
from data import load_datasets
from model import get_model
import numpy as np


def train_model():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    datasets = load_datasets(config)
    dataloaders = {
        x: torch.utils.data.DataLoader(
            datasets[x],
            batch_size=config['data']['batch_size'],
            shuffle=(x == 'Training'),
            num_workers=config['data']['num_workers']
        ) for x in ['Training', 'Testing']
    }
    dataset_sizes = {x: len(datasets[x]) for x in ['Training', 'Testing']}
    class_names = datasets['Training'].classes

    train_labels = [label for _, label in datasets['Training']]
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_labels),
        y=train_labels
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    model = get_model(
        arch=config['model']['arch'],
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained']
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True
    )

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    patience = 5
    patience_counter = 0

    for epoch in range(config['training']['epochs']):
        print(f'\nEpoch {epoch + 1}/{config["training"]["epochs"]}')
        print('-' * 50)

        for phase in ['Training', 'Testing']:
            if phase == 'Training':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            progress_bar = tqdm(
                dataloaders[phase],
                desc=f'{phase:12}',
                leave=True,
                bar_format='{l_bar}{bar:30}{r_bar}'
            )

            for inputs, labels in progress_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'Training'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'Training':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{batch_acc:.4f}'
                })

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'Training':
                train_losses.append(epoch_loss)
                train_accs.append(epoch_acc.item())
            else:
                val_losses.append(epoch_loss)
                val_accs.append(epoch_acc.item())

            print(f'{phase:12} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | '
                  f'Samples: {dataset_sizes[phase]}')

            if phase == 'Testing':
                scheduler.step(epoch_acc)

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(best_model_wts, config['training']['save_path'])
                    print(f'    🆕 New best model saved! (Acc: {best_acc:.4f})')
                    patience_counter = 0
                else:
                    patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping triggered at epoch {epoch + 1}')
            break

    model.load_state_dict(best_model_wts)

    plt.figure(figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2.5)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2.5)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2.5)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig('plots/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'\n{"=" * 60}')
    print(f' TRAINING COMPLETE!')
    print(f' Best Validation Accuracy: {best_acc:.4f} ({best_acc * 100:.2f}%)')
    print(f'{"=" * 60}')

    return model


if __name__ == "__main__":
    train_model()