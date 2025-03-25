import os
import matplotlib.pyplot as plt


def tqdm_bar(mode, pbar, target=0.0, current=0.0, epochs=0.0):
    if mode == 'Test':
        pbar.set_description(f"({mode})", refresh=False)
        pbar.refresh()
    else:
        pbar.set_description(
            f"({mode}) Epoch {current}/{epochs}",
            refresh=False)
        pbar.set_postfix(loss=float(target), refresh=False)
        pbar.refresh()

# Draw Training Figure (Loss or Accuracy)
def DrawFigure(mode, path, epochs, train, valid):
    epoch = range(epochs)
    plt.style.use("ggplot")
    plt.figure()

    # Draw Loss Fig.
    if mode == 'loss':
        plt.plot(epoch, train, 'red', label='Training')
        plt.plot(epoch, valid, 'blue', label='Validation')
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        save_path = os.path.join(path, 'Loss.jpg')

    # Draw Accuracy Fig.
    elif mode == 'accuracy':
        plt.plot(epoch, train, 'red', label='Training')
        plt.plot(epoch, valid, 'blue', label='Validation')
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        save_path = os.path.join(path, 'Accuracy.jpg')

    print(f"[DrawFigure] Saved figure to {save_path}")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

# Model size (# parameters)
def weight(model):
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print("# parameters:", total_params)

