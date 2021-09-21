import torch


def test(test_loader, device):
	# Make sure the model is in eval mode.
	# Some modules like Dropout or BatchNorm affect if the model is in training mode.
	student_net = torch.load('./student_net.ckpt')
	student_net.eval()

	# Initialize a list to store the predictions.
	predictions = []

	# Iterate the testing set by batches.
	for batch in test_loader:
		# A batch consists of image data and corresponding labels.
		# But here the variable "labels" is useless since we do not have the ground-truth.
		# If printing out the labels, you will find that it is always 0.
		# This is because the wrapper (DatasetFolder) returns images and labels for each batch,
		# so we have to create fake labels to make it work normally.
		imgs, labels = batch

	# We don't need gradient in testing, and we don't even have labels to compute loss.
	# Using torch.no_grad() accelerates the forward process.
	with torch.no_grad():
		logits = student_net(imgs.to(device))

	# Take the class with greatest logit as prediction and record it.
	predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

	# Save predictions into the file.
	with open("predict.csv", "w") as f:
		# The first row must be "Id, Category"
		f.write("Id,Category\n")

		# For the rest of the rows, each image id corresponds to a predicted class.
		for i, pred in enumerate(predictions):
			f.write(f"{i},{pred}\n")

if __name__ == '__main__':
	test()