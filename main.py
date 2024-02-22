import io
import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import itertools


# Fix a seed for reproducibility
seed_value = 42  # You can choose any number as your seed
# Numpy RNG
np.random.seed(seed_value)
# PyTorch RNGs
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)

# Load data
data_path = '/Users/mikky/Downloads/train-00000-of-00001-38cc4fa96c139e86.parquet'
emojis = pd.read_parquet(data_path, engine='pyarrow')
face_df = emojis[emojis['text'].apply(lambda x: 'face' in x.split() and 'clock' not in x.split())]
face_texts = face_df['text']
face_imgs = np.array([np.array(Image.open(io.BytesIO(row['image']['bytes']))) for index, row in face_df.iterrows()])

# Split dataset
from torch.utils.data import random_split
total_length = len(face_df)
train_length = int(total_length * 0.6)
valid_length = int(total_length * 0.2)
test_length = total_length - train_length - valid_length
train_dataset, valid_dataset, test_dataset = random_split(face_imgs, [train_length, valid_length, test_length])


class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (numpy.ndarray): A matrix containing your data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = Image.fromarray(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample


# Augment training data
train_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomAffine(
        degrees=15,  # Rotation: A small degree, since large rotations might make emojis unrecognizable.
        translate=(0.1, 0.1),  # Translation: Shift the image by 10% of its height/width in any direction.
        scale=(0.9, 1.1),  # Scale: Slightly zoom in or out by 10%.
        shear=5  # Shear: Apply a small shearing of 5 degrees.
    ),
    transforms.RandomHorizontalFlip(),  # Horizontally flip the image with a given probability.
    transforms.RandomVerticalFlip(p=0.5),  # Vertically flip the image with a given probability.
    transforms.RandomRotation(20),  # Rotate the image by angle.
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly change the brightness, contrast, and saturation of an image.
    transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor.
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize a tensor image with mean and standard deviation.
])

# Augment validation data: Define light transformations for augmentation
validation_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Mild augmentations
    transforms.RandomHorizontalFlip(),  # Horizontally flip the image with a given probability.
    transforms.RandomVerticalFlip(p=0.5),  # Vertically flip the image with a given probability.
    transforms.RandomRotation(20),  # Rotate the image by angle.
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize a tensor image with mean and standard deviation.
])

# Augment test data: Define light transformations for augmentation
test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    # transforms.ColorJitter(brightness=0.1, contrast=0.1),  # Mild augmentations
    # transforms.RandomHorizontalFlip(),  # Horizontally flip the image with a given probability.
    # transforms.RandomVerticalFlip(p=0.5),  # Vertically flip the image with a given probability.
    # transforms.RandomRotation(20),  # Rotate the image by angle.
    transforms.ToTensor(),
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize a tensor image with mean and standard deviation.
])

params = {
    'batch_size': 16,
    'learning_rate': 0.01,
    'epoch': 100,
    'bottleneck_dim': 12,
    'auto_encoder': 'CNN'
}

training_dataset = CustomDataset(train_dataset, transform=train_transform)
validation_dataset = CustomDataset(valid_dataset, transform=validation_transform)
testing_dataset = CustomDataset(test_dataset, transform=test_transform)
train_loader = DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=True)
valid_loader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False)
test_loader = DataLoader(testing_dataset, batch_size=params['batch_size'], shuffle=False)


def EmojiAutoencoder(structure='Linear'):
    if structure == 'linear':
        return EmojiAutoencoderLinear()
    elif structure == 'CNN':
        return EmojiAutoencoderCNN()


class EmojiAutoencoderLinear(nn.Module):
    def __init__(self):
        super(EmojiAutoencoderLinear, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, params['bottleneck_dim'])
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(params['bottleneck_dim'], 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Sigmoid()  # Use sigmoid to ensure output values are between 0 and 1
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        return x


class EmojiAutoencoderCNN(nn.Module):
    def __init__(self, params):
        super(EmojiAutoencoderCNN, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 8 x 8
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 4x4x128
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 8x8x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32 x 16 x 16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 16 x 32 x 32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 3 x 16 x 16
            nn.ReLU(),
        )
        self.params = params
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.params['learning_rate'])

    def forward(self, x):
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        return x

    def train(self):
        train_losses = []
        validation_losses = []

        for epoch in range(self.params['epoch']):
            train_loss = 0
            for data in train_loader:
                img = data
                if self.params['auto_encoder'] == 'linear':
                    img = img.view(img.size(0), -1)  # Flatten the images to match input dimensions of the network
                self.optimizer.zero_grad()
                outputs = self.forward(img)
                loss = self.criterion(outputs, img)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            validation_loss = 0
            with torch.no_grad():
                for data in valid_loader:
                    img = data
                    if self.params['auto_encoder'] == 'linear':
                        img = img.view(img.size(0), -1)  # Flatten the images to match input dimensions of the network
                    outputs = self.forward(img)
                    loss = self.criterion(outputs, img)
                    validation_loss += loss.item()
            validation_loss /= len(valid_loader)
            validation_losses.append(validation_loss)
            # print(f'Epoch {epoch + 1}/{self.params["epoch"]}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}')

        # Plotting the training and validation loss curves
        plt.figure(figsize=[8, 6])
        plt.plot(train_losses, label='Training Loss')
        plt.plot(validation_losses, label='Validation Loss')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.title('Training and Validation Loss Curves', fontsize=16)
        plt.legend()
        fig_name = f'{self.params["auto_encoder"]}_learning_curve_{self.params["batch_size"]}_{self.params["epoch"]}_{self.params["learning_rate"]}.jpg'
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        # plt.show()

    def test(self):
        total_mse_error = 0.0
        total_samples = 0
        with torch.no_grad():
            for data in test_loader:
                img = data
                if params['auto_encoder'] == 'linear':
                    img = img.view(img.size(0), -1)  # Flatten images
                reconstructed = self.forward(img)
                mse_error = self.criterion(reconstructed, img)
                total_mse_error += mse_error.item() * img.size(0)  # Multiply by batch size to accumulate error correctly
                total_samples += img.size(0)
            average_mse_error = total_mse_error / total_samples
            print(f'Test Average MSE Error: {average_mse_error:.4f}')
            self.visual_inspection()
            return average_mse_error

    def visual_inspection(self):
        # Visual inspect the difference between original images and reconstructed images
        dataiter = iter(test_loader)  # Convert DataLoader to an iterator
        images = next(dataiter)  # Get the next batch of images; no '.next()' directly on DataLoader

        if self.params['auto_encoder'] == 'linear':
            images = images.view(images.size(0), -1)  # Flatten images if required by your model
        reconstructed = self.forward(images)  # Obtain reconstructed images
        images = images.view(-1, 3, 64, 64)  # Reshape original images to proper shape if they were flattened
        reconstructed = reconstructed.view(-1, 3, 64, 64)  # Reshape back to original dimensions if needed
        images = images.numpy()  # Convert to numpy array
        reconstructed = reconstructed.detach().numpy()  # Convert reconstructed images to numpy array

        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
        for i in range(5):
            # Display original images
            ax = axes[0, i]
            image_clipped = np.clip(np.transpose(images[i], (1, 2, 0)), 0, 1)  # Ensures all values are within [0, 1]
            ax.imshow(image_clipped)  # Convert from (C, H, W) to (H, W, C)
            ax.set_title('Original')
            ax.axis('off')
            # Display reconstructed images
            ax = axes[1, i]
            reconstructed_clipped = np.clip(np.transpose(reconstructed[i], (1, 2, 0)), 0, 1)  # Ensures all values are within [0, 1]
            ax.imshow(reconstructed_clipped)  # Convert from (C, H, W) to (H, W, C)
            ax.set_title('Reconstructed')
            ax.axis('off')
        fig_name = f'{self.params["auto_encoder"]}_reconstructed_{self.params["batch_size"]}_{self.params["epoch"]}_{self.params["learning_rate"]}.jpg'
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        # plt.show()


def grid_search():
    params = {
        'batch_size': 16,
        'learning_rate': 0.01,
        'epoch': 100,
        'auto_encoder': 'CNN'
    }

    # experiment with different params
    epoch = [100, 200]
    batch_size = [16, 32, 64]
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    param_combinations = list(itertools.product(epoch, batch_size, learning_rate))
    grid_search_data = []
    for combo in param_combinations:
        keys = ['epoch', 'batch_size', 'learning_rate']
        params_update = dict(zip(keys, combo))
        params.update(params_update)
        autoencoder = EmojiAutoencoderCNN(params)
        autoencoder.train()
        params['error_rate'] = autoencoder.test()
        print(params)
        grid_search_data.append(params.copy())
    df = pd.DataFrame(grid_search_data)
    df.to_csv('CNN_params_grid_search.csv', index=True)


if __name__ == "__main__":
    # grid_search()
    params = {
        'batch_size': 16,
        'learning_rate': 0.001,
        'epoch': 200,
        'auto_encoder': 'CNN'
    }
    autoencoder = EmojiAutoencoderCNN(params)
    autoencoder.train()
    autoencoder.test()

