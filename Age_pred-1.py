# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

# %%
file_path = r"C:\Users\Essam\Desktop\DL Assignment\Age_Images"

age = []
gender = []
race = []
img_path = []

# Extract age, gender, race, and image path
for file in os.listdir(file_path):
    parts = file.split('_')
    age.append(int(parts[0]))
    gender.append(parts[1])
    race.append(parts[2])
    img_path.append(file)

# Print sample results for verification
print("Age:", age[:5])
print("Gender:", gender[:5])
print("Race:", race[:5])
print("Image Paths:", img_path[:5])


# %%
df = pd.DataFrame({
    'age': age,
    'gender': gender,
    'race': race,
    'img_path': img_path
})
df = df[(df['age'] >= 10) & (df['age'] <= 90)]

# %%
df

# %%
df.isnull().sum()

# %%
df['gender'].unique()

# %%
df['race'].unique()

# %%
print((df['race'] == '20170116174525125.jpg.chip.jpg').sum())
print((df['race'] == '20170109142408075.jpg.chip.jpg').sum())
print((df['race'] == '20170109150557335.jpg.chip.jpg').sum())
print((df['race'] == '0').sum())
print((df['race'] == '1').sum())
print((df['race'] == '2').sum())
print((df['race'] == '3').sum())
print((df['race'] == '4').sum())

# %%
# Filter out rows with specific 'race' values
exclude_values = [
    '20170116174525125.jpg.chip.jpg',
    '20170109142408075.jpg.chip.jpg',
    '20170109150557335.jpg.chip.jpg'
]

# Remove rows with the specified 'race' values
df = df[~df['race'].isin(exclude_values)]

print(f"Remaining rows: {len(df)}")

# %%
df

# %%
df['race'].unique()

# %%
df['gender'].unique()

# %%
base_dir = r'C:\Users\Essam\Desktop\DL Assignment\Age_Images'
df['img_path'] = df['img_path'].apply(lambda x: os.path.join(base_dir, x))

samples_per_race = df.groupby('race').apply(lambda x: x.sample(5, random_state=42)).reset_index(drop=True)

fig, axes = plt.subplots(len(samples_per_race['race'].unique()), 5, figsize=(15, 10))

for i, (race, group) in enumerate(samples_per_race.groupby('race')):
    for j, img_path in enumerate(group['img_path']):
        img = plt.imread(img_path)
        ax = axes[i, j]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Race {race}')
        
plt.tight_layout()
plt.show()


# %%
# Shuffle and split data
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)
train_df, temp_df = train_test_split(df_shuffled, test_size=0.2, random_state=42, stratify=df_shuffled['age'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['age'])

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class AgeDataset(Dataset):
    def __init__(self, dataframe, file_path, transform=None):
        self.dataframe = dataframe
        self.file_path = file_path
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract image, gender, race, and age
        img_name = os.path.join(self.file_path, self.dataframe.iloc[idx]['img_path'])
        gender = self.dataframe.iloc[idx]['gender']
        race = self.dataframe.iloc[idx]['race']
        age = self.dataframe.iloc[idx]['age']

        # Load and transform the image
        image = Image.open(img_name).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Encode categorical features
        gender = torch.tensor(int(gender), dtype=torch.float32)
        race = torch.tensor(int(race), dtype=torch.float32)
        age = torch.tensor(age, dtype=torch.float32)  # Target (age)
        
        # Return a dictionary
        return {
            "image": image,
            "gender": gender,
            "race": race,
            "label": age
        }


# Define image transforms
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# %%
val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# %%
# Create Datasets
train_dataset = AgeDataset(train_df, file_path, transform=train_transforms)
val_dataset = AgeDataset(val_df, file_path, transform=val_test_transforms)
test_dataset = AgeDataset(test_df, file_path, transform=val_test_transforms)

# %%
# DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %%
# Example Usage
for batch in train_loader:
    features = batch  # 'batch' is a dictionary
    images, genders, races = features['image'], features['gender'], features['race']
    labels = features['label']
    
    # Move data to the device
    images, genders, races, labels = images.to(device), genders.to(device), races.to(device), labels.to(device)
    
    # Proceed with forward pass, loss calculation, and optimization


# %%
scaler = torch.cuda.amp.GradScaler()

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + shortcut
        x = F.relu(x)
        return x

class AgePredictionModel(nn.Module):
    def __init__(self):
        super(AgePredictionModel, self).__init__()
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block1 = ResidualBlock(32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.block2 = ResidualBlock(64, 64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.block3 = ResidualBlock(128, 128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.block4 = ResidualBlock(256, 256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)

        self.block5 = ResidualBlock(512, 512)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers for combined features
        self.fc_image = nn.Linear(512, 256)  # For image features
        self.fc_gender = nn.Linear(1, 16)    # For gender feature
        self.fc_race = nn.Linear(1, 16)      # For race feature
        self.fc_combined = nn.Linear(256 + 16 + 16, 128)  # Combine all features

        self.dropout = nn.Dropout(0.3)
        self.fc_output = nn.Linear(128, 1)  # Regression output for age prediction

    def forward(self, image, gender, race):
        # Process image input
        x = self.pool(F.relu(self.bn1(self.conv1(image))))
        x = self.block1(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.block2(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.block3(x)
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = self.block4(x)
        x = self.pool(F.relu(self.bn5(self.conv5(x))))
        x = self.block5(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc_image(x))

        # Process gender and race features
        gender = F.relu(self.fc_gender(gender.unsqueeze(1)))  # Expand dimension for linear layer
        race = F.relu(self.fc_race(race.unsqueeze(1)))

        # Combine all features
        combined = torch.cat((x, gender, race), dim=1)
        combined = F.relu(self.fc_combined(combined))
        combined = self.dropout(combined)
        output = self.fc_output(combined)

        return output


# %%
# Instantiate model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AgePredictionModel().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# %%
# Define helper function to count parameters
def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return trainable_params, non_trainable_params

trainable_params, non_trainable_params = count_parameters(model)
print(f"Trainable Parameters: {trainable_params}")
print(f"Non-Trainable Parameters: {non_trainable_params}")

# %%
import numpy as np  # For numerical operations

num_epochs = 50
patience = 10 
best_val_mae = np.inf
patience_counter = 0

train_losses, val_losses = [], []
train_maes, val_maes = [], []

for epoch in range(num_epochs):
    model.train()
    running_loss, running_mae = 0.0, 0.0

    # Training phase
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images = batch['image'].to(device)
        gender = batch['gender'].to(device)
        race = batch['race'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(images, gender, race)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_mae += torch.sum(torch.abs(outputs.squeeze() - labels.float())).item()

    train_loss = running_loss / len(train_loader.dataset)
    train_mae = running_mae / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_maes.append(train_mae)

    # Validation phase
    model.eval()
    val_loss, val_mae = 0.0, 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images = batch['image'].to(device)
            gender = batch['gender'].to(device)
            race = batch['race'].to(device)
            labels = batch['label'].to(device)

            outputs = model(images, gender, race)
            loss = criterion(outputs.squeeze(), labels.float())

            val_loss += loss.item() * images.size(0)
            val_mae += torch.sum(torch.abs(outputs.squeeze() - labels.float())).item()

    val_loss /= len(val_loader.dataset)
    val_mae /= len(val_loader.dataset)
    val_losses.append(val_loss)
    val_maes.append(val_mae)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f} - Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")

    # Early stopping logic
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        patience_counter = 0  # Reset patience
        print(f"Validation MAE improved to {best_val_mae:.4f}. Saving model...")
        torch.save(model.state_dict(), 'best_model.pth')  # Save best model
    else:
        patience_counter += 1
        print(f"No improvement in MAE for {patience_counter} epoch(s).")

    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

# Plot training and validation metrics
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue', linestyle='--')
plt.plot(val_losses, label='Validation Loss', color='orange', linestyle='-')
plt.title('Training and Validation Losses', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)

# MAE plot
plt.subplot(1, 2, 2)
plt.plot(train_maes, label='Training MAE', color='green', linestyle='--')
plt.plot(val_maes, label='Validation MAE', color='red', linestyle='-')
plt.title('Training and Validation MAE', fontsize=16)
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('MAE', fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.4)

plt.tight_layout()
plt.show()


# %%
# Load the saved model's state dict
model.load_state_dict(torch.load("best_model.pth"))
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully")

# %%
model.eval()
test_loss, test_mae = 0.0, 0.0

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing on Test Dataset"):
        images = batch['image'].to(device)
        gender = batch['gender'].to(device)
        race = batch['race'].to(device)
        labels = batch['label'].to(device)

        # Forward pass
        outputs = model(images, gender, race)
        loss = criterion(outputs.squeeze(), labels.float())

        # Accumulate metrics
        test_loss += loss.item() * images.size(0)
        test_mae += torch.sum(torch.abs(outputs.squeeze() - labels.float())).item()

# Calculate average loss and MAE
test_loss /= len(test_loader.dataset)
test_mae /= len(test_loader.dataset)

print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")



