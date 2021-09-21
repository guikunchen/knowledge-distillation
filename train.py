import torch
import torch.nn as nn

from models.StudentNet import StudentNet
from models.loss import loss_fn_kd
from data.dataloader import get_dataloader


batch_size = 64
lr = 3e-4
weight_decay = 1e-5
n_epochs = 80


def train():
	# "cuda" only when GPUs are available.
	device = "cuda" if torch.cuda.is_available() else "cpu"

	# Initialize a model, and put it on the device specified.
	student_net = StudentNet()
	student_net = student_net.to(device)

	teacher_net = torch.load('./teacher_net.ckpt')
	teacher_net.eval()
	teacher_net = teacher_net.to(device)

	# Whether to do pseudo label.
	do_semi = True

	train_loader, valid_loader, _ = get_dataloader(batch_size, do_semi, teacher_net)

	# For the classification task, we use cross-entropy as the measurement of performance.
	# criterion = nn.CrossEntropyLoss()

	# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
	optimizer = torch.optim.Adam(student_net.parameters(), lr=lr, weight_decay=weight_decay)

	for epoch in range(n_epochs):
		# ---------- Training ----------
		# Make sure the model is in train mode before training.
		student_net.train()

		# These are used to record information in training.
		train_loss = []
		train_accs = []

		# Iterate the training set by batches.
		for batch in train_loader:

			# A batch consists of image data and corresponding labels.
			imgs, labels = batch

			# Forward the data. (Make sure data and model are on the same device.)
			logits = student_net(imgs.to(device))
			# Teacher net will not be updated. And we use torch.no_grad
			# to tell torch do not retain the intermediate values
			# (which are for backpropagation) and save the memory.
			with torch.no_grad():
				soft_labels = teacher_net(imgs.to(device))

			# Calculate the loss in knowledge distillation method.
			loss = loss_fn_kd(logits, labels.to(device), soft_labels)

			# Gradients stored in the parameters in the previous step should be cleared out first.
			optimizer.zero_grad()

			# Compute the gradients for parameters.
			loss.backward()

			# Clip the gradient norms for stable training.
			grad_norm = nn.utils.clip_grad_norm_(student_net.parameters(), max_norm=10)

			# Update the parameters with computed gradients.
			optimizer.step()

			# Compute the accuracy for current batch.
			acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

			# Record the loss and accuracy.
			train_loss.append(loss.item())
			train_accs.append(acc)

		# The average loss and accuracy of the training set is the average of the recorded values.
		train_loss = sum(train_loss) / len(train_loss)
		train_acc = sum(train_accs) / len(train_accs)

		# Print the information.
		print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

		# ---------- Validation ----------
		# Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
		student_net.eval()

		# These are used to record information in validation.
		valid_loss = []
		valid_accs = []

		# Iterate the validation set by batches.
		for batch in valid_loader:

			# A batch consists of image data and corresponding labels.
			imgs, labels = batch

			# We don't need gradient in validation.
			# Using torch.no_grad() accelerates the forward process.
			with torch.no_grad():
				logits = student_net(imgs.to(device))
				soft_labels = teacher_net(imgs.to(device))
			# We can still compute the loss (but not the gradient).
			loss = loss_fn_kd(logits, labels.to(device), soft_labels)

			# Compute the accuracy for current batch.
			acc = (logits.argmax(dim=-1) == labels.to(device)).float().detach().cpu().view(-1).numpy()

			# Record the loss and accuracy.
			valid_loss.append(loss.item())
			valid_accs += list(acc)

		# The average loss and accuracy for entire validation set is the average of the recorded values.
		valid_loss = sum(valid_loss) / len(valid_loss)
		valid_acc = sum(valid_accs) / len(valid_accs)

		# Print the information.
		print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


if __name__ == '__main__':
	train()
