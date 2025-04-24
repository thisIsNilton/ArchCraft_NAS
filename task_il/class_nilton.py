import matplotlib.pyplot as plt
import torch
# import torchvision

def show_task_images(data, task_id=0, split='train', n=10):
    x = data[task_id][split]['x']
    y = data[task_id][split]['y']
    
    mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3,1,1)
    std = torch.tensor([0.2675, 0.2565, 0.2761]).view(3,1,1)

    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    for i in range(n):
        img = x[i]
        label = y[i].item()

        # Desfaz a normalização
        img = img * std + mean
        img = img.permute(1, 2, 0).numpy()  # (C, H, W) → (H, W, C)

        axes[i].imshow(img)
        axes[i].set_title(f'Classe {label}')
        axes[i].axis('off')
    
    plt.suptitle(f'Tarefa {task_id} - {split}', fontsize=16)
    plt.tight_layout()
    plt.show()
