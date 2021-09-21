from torch.utils.data import DataLoader
import torch


def get_pseudo_labels(dataset, model, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size*3, shuffle=False, pin_memory=True)
    pseudo_labels = []
    for batch in loader:
        # A batch consists of image data and corresponding labels.
        img, _ = batch

        # Forward the data
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(img.to(model.device))
            pseudo_labels.append(logits.argmax(dim=-1).detach().cpu())
        # Obtain the probability distributions by applying softmax on logits.
    pseudo_labels = torch.cat(pseudo_labels)
    # Update the labels by replacing with pseudo labels.
    for idx, ((img, _), pseudo_label) in enumerate(zip(dataset.samples, pseudo_labels)):
        dataset.samples[idx] = (img, pseudo_label.item())
    return dataset