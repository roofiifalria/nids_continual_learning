import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, save_dir):
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    plt.close()

def plot_class_distribution(y_orig, y_bal, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.countplot(x=y_orig, ax=axes[0])
    axes[0].set_title('Original Distribution')
    sns.countplot(x=y_bal, ax=axes[1])
    axes[1].set_title('Balanced Distribution')
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
    plt.close()

def plot_gan_pca(real_features, fake_features, save_dir):
    pca = PCA(n_components=2)
    all_features = np.vstack((real_features, fake_features))
    pca_result = pca.fit_transform(all_features)
    n_real = len(real_features)
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:n_real, 0], pca_result[:n_real, 1], alpha=0.5, label='Real')
    plt.scatter(pca_result[n_real:, 0], pca_result[n_real:, 1], alpha=0.5, label='Fake')
    plt.legend()
    plt.title('PCA Real vs Fake')
    plt.savefig(os.path.join(save_dir, 'gan_pca.png'))
    plt.close()

def plot_ppo_rewards(rewards, save_dir):
    plt.figure(figsize=(8, 5))
    plt.plot(rewards)
    plt.title('PPO Training Rewards')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.savefig(os.path.join(save_dir, 'ppo_rewards.png'))
    plt.close()