# Implementation-of-Transfer-Learning
## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Include the problem statement and Dataset
</br>
</br>
</br>

## DESIGN STEPS
### STEP 1:
</br>

### STEP 2:
</br>

### STEP 3:

Write your own steps
<br/>

## PROGRAM


### Load Pretrained Model and Modify for Transfer Learning
```
model = models.vgg19(pretrained=True)
```

### Modify the final fully connected layer to match the dataset classes
```
model.classifier[6] = nn.Linear(4096, num_classes)
```

### Include the Loss function and optimizer
```
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### Train the model
```
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # Convert labels to one-hot encoding
            labels = nn.functional.one_hot(labels, num_classes=num_classes).float().to(device) # Change this line
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # Convert labels to one-hot encoding
                labels = nn.functional.one_hot(labels, num_classes=num_classes).float().to(device) # Change this line
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name:Leann Joby Mathew")
    print("Register Number:212222230074")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/b8f91ddf-7faf-4fa8-ab4a-4efa074141af)


### Confusion Matrix
![image](https://github.com/user-attachments/assets/4a4affd0-14f3-4bda-bdec-423363a61707)


### Classification Report
![image](https://github.com/user-attachments/assets/bde7124b-d591-492c-8e67-e1c53a107763)


### New Sample Prediction
![image](https://github.com/user-attachments/assets/0cdce630-260e-407f-956e-a5533f9b1b90)


## RESULT
</br>
</br>
</br>
