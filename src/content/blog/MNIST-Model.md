---
author: joe-mclean
date: 21/06/2026
title: MNIST model
description: My first model
---

# Making a Simple Feed Forward Network Using PyTorch

Recently, as part of preparation for my research internship, I have been learning PyTorch to get more comfortable with with making and working with neural networks. The first little project I decided to do was make a model that could predict numbers from the MNIST data set. I've heard this is the ML equivalent of 'Hello, World'. 

### Preparing the data

The first thing that I needed to do was load in the MNIST data for my model to learn. Luckily, PyTorch provides a `MNIST()` function that lets you load MNIST data into a dataset.

```python
dataset = MNIST(root='./data', train=True, download=True ,transform=transforms.ToTensor())
```

I had to change the transforms parameter so that each image was in the form of a tensor so pytorch could work with it.

Each entry in the dataset is a tuple of `(image, label)` where `image` is a tensor of shape `(1, 28, 28)` - channels (1 since these are greyscale images), height and width.

I then had to split the data and put it into data loaders. Data loaders are utilities that feed data into a model in batches during training or evaluation.

```python
training_data, validation_data = random_split(dataset, [50000,10000])
batch_size = 128

train_loader = DataLoader(training_data, batch_size, shuffle=True)
val_loader = DataLoader(validation_data, batch_size, shuffle=False)
```

### Making the Model

The model itself has the main goal of performing a good forward pass. So the only functions it contains are its constructor, forward, and then any other function used to train and validate it.

Forward takes a batch of images as 2D tensors, flattens them and passes them through the network. It then returns 10 logits (activations of output layer neurons. These will later become probabilities to allow us to see which image is predicted).

`training_step()` is used to calculate the loss of a model for a batch of data (we use cross entropy since we're doing categorical classification)

`validation_step()` sees how well our model performs on unseen data by returning loss and accuracy of a forward pass on an unseen dataset.

`validation_epoch_end()` computes average loss and accuracy for one epoch, using data obtained from running validation_step() on different batches.

`epoch_end()` shows results of model after an epoch.

```python
# MNIST Model
input_size = 28*28
num_classes = 10

def accuracy(outputs, labels):
	_, preds = torch.max(outputs, dim = 1)
	return(torch.tensor(torch.sum(preds == labels).item()/ len(preds)))

class MnistModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = nn.Linear(input_size, num_classes)
	
	def forward(self, xb):
		xb = xb.reshape(-1,784)
		out = self.linear(xb)
		return out
	
	def training_step(self, batch):
		images, labels = batch
		out = self(images) # Generate predictions
		loss = F.cross_entropy(out, labels) # Calculate the loss
		return(loss)
	
	def validation_step(self,batch):
		images, labels = batch
		out = self(images)
		loss = F.cross_entropy(out, labels)
		acc = accuracy(out, labels)
		return {'val_loss':loss, 'val_acc':acc}
	
	def validation_epoch_end(self, outputs):
		batch_losses = [x['val_loss'] for x in outputs]
		epoch_loss = torch.stack(batch_losses).mean()
		batch_accs = [x['val_acc'] for x in outputs]
		epoch_acc = torch.stack(batch_accs).mean()
		return({'val_loss': epoch_loss.item(), 'val_acc' : epoch_acc.item()})
	
	def epoch_end(self, epoch,result):
		print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))  
```


Then we also need to define functions to fit and evaluate our model.

`evaluate()` is used to compute loss and accuracy of current model as is by using validation set.

`fit()` is used to train and validate the model to be as good as possible. During training we iterate over each batch and load it into model, compute loss, back propagating through the network, updating weights and biases as we go through. We then evaluate the model on unseen data and return accuracy and loss.

```python
def evaluate(model, val_loader):
	outputs = [model.validation_step(batch) for batch in val_loader]
	return(model.validation_epoch_end(outputs))

  

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
	history = []
	optimizer = opt_func(model.parameters(), lr)
	
	for epoch in range(epochs):
		# Training Phase
		for batch in train_loader:
			loss = model.training_step(batch)
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			
		# Validation phase
		result = evaluate(model, val_loader)
		model.epoch_end(epoch, result)
		history.append(result)
	
	return(history)
```

### Fitting the Model

We can now initialise the model and fit it.
```python
model = MnistModel()
  
history = fit(  
	epochs=5,  
	lr=0.001,  
	model=model,  
	train_loader=train_loader,  
	val_loader=val_loader  
)
```

### Testing the Model

For testing the model I defined a `predict` function which takes the activations of output neurons and picks highest value.

```python
def predict_image(img, model):
	xb = img.unsqueeze(0)
	yb = model(xb)
	_, preds = torch.max(yb, dim = 1)
	return(preds[0].item())
```

I then tested the model

```python
test_dataset = MNIST(root = 'data/', train = False, transform = transforms.ToTensor())

img, label = test_dataset[291] # Change index value for different images
plt.imshow(img[0], cmap = 'gray')
print('Label:', label, ', Predicted :', predict_image(img, model))
```

My output was fortunately `2`.