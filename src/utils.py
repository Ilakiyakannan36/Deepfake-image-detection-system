import json
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_config(config: dict, config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def plot_confusion_matrix(true_labels, predictions, class_names=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names or ['Real', 'Fake'],
                yticklabels=class_names or ['Real', 'Fake'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt

def save_results(results: dict, filename: str):
    """Save results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)