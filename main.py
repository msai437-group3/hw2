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
from torch.utils.data import random_split

# Fix a seed for reproducibility
seed_value = 42  # You can choose any number as your seed
# Numpy RNG
np.random.seed(seed_value)
# PyTorch RNGs
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)


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


class EmojiAutoencoderLinear(nn.Module):
    def __init__(self, params):
        super(EmojiAutoencoderLinear, self).__init__()
        self.params = params
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
        self.params = params
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # Output: 16 x 32 x 32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Output: 32 x 16 x 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Output: 64 x 8 x 8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Output: 128 x 4 x 4
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 8 x 8 x 64
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 16 x 16 x 32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 32 x 32 x 16
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: 64 x 64 x 3
            nn.Sigmoid(),
        )

    def forward(self, x):
        bottleneck = self.encoder(x)
        x = self.decoder(bottleneck)
        return x


class EmojiAutoencoder(nn.Module):
    def __init__(self, params):
        super(EmojiAutoencoder, self).__init__()
        self.params = params
        if self.params['auto_encoder'] == 'linear':
            self.auto_encoder = EmojiAutoencoderLinear(self.params)
        elif self.params['auto_encoder'] == 'CNN':
            self.auto_encoder = EmojiAutoencoderCNN(self.params)
        else:
            self.auto_encoder = EmojiAutoencoderCNN(self.params)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.params['learning_rate'])
        self.train_loader, self.valid_loader, self.test_loader = load_data(self.params)

    def train(self):
        train_losses = []
        validation_losses = []

        for epoch in range(self.params['epoch']):
            train_loss = 0
            for data in self.train_loader:
                img = data
                if self.params['auto_encoder'] == 'linear':
                    img = img.view(img.size(0), -1)  # Flatten the images to match input dimensions of the network
                self.optimizer.zero_grad()
                outputs = self.auto_encoder.forward(img)
                loss = self.criterion(outputs, img)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)

            # Validation
            validation_loss = 0
            with torch.no_grad():
                for data in self.valid_loader:
                    img = data
                    if self.params['auto_encoder'] == 'linear':
                        img = img.view(img.size(0), -1)  # Flatten the images to match input dimensions of the network
                    outputs = self.auto_encoder.forward(img)
                    loss = self.criterion(outputs, img)
                    validation_loss += loss.item()
            validation_loss /= len(self.valid_loader)
            validation_losses.append(validation_loss)
            print(f'Epoch {epoch + 1}/{self.params["epoch"]}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}')

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
            for data in self.test_loader:
                img = data
                if self.params['auto_encoder'] == 'linear':
                    img = img.view(img.size(0), -1)  # Flatten images
                reconstructed = self.auto_encoder.forward(img)
                mse_error = self.criterion(reconstructed, img)
                total_mse_error += mse_error.item() * img.size(0)  # Multiply by batch size to accumulate error correctly
                total_samples += img.size(0)
            average_mse_error = total_mse_error / total_samples
            print(f'Test Average MSE Error: {average_mse_error:.4f}')
            self.visual_inspection()
            return average_mse_error

    def visual_inspection(self):
        # Visual inspect the difference between original images and reconstructed images
        dataiter = iter(self.test_loader)  # Convert DataLoader to an iterator
        images = next(dataiter)  # Get the next batch of images
        if self.params['auto_encoder'] == 'linear':
            images = images.view(images.size(0), -1)
        reconstructed = self.auto_encoder.forward(images)
        images = images.view(-1, 3, 64, 64)  # Reshape original images to proper shape
        reconstructed = reconstructed.view(-1, 3, 64, 64)  # Reshape back to original dimensions
        images = images.numpy()
        reconstructed = reconstructed.detach().numpy()

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

    def _denormalize(self, tensor):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
        denormalized = tensor * std + mean
        # Clip to ensure the denormalized values are within a valid range [0, 1]
        return torch.clamp(denormalized, 0, 1)


def load_data(params):
    # Read data
    data_path = '/Users/mikky/Downloads/train-00000-of-00001-38cc4fa96c139e86.parquet'
    emojis = pd.read_parquet(data_path, engine='pyarrow')
    face_df = emojis[emojis['text'].apply(lambda x: 'face' in x.split() and 'clock' not in x.split())]
    face_imgs = np.array([np.array(Image.open(io.BytesIO(row['image']['bytes']))) for index, row in face_df.iterrows()])

    # Split dataset
    total_length = len(face_df)
    train_length = int(total_length * 0.6)
    valid_length = int(total_length * 0.2)
    test_length = total_length - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = random_split(face_imgs, [train_length, valid_length, test_length])

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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # Randomly change the brightness, contrast, and saturation of an image.
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

    training_dataset = CustomDataset(train_dataset, transform=train_transform)
    validation_dataset = CustomDataset(valid_dataset, transform=validation_transform)
    testing_dataset = CustomDataset(test_dataset, transform=test_transform)
    train_loader = DataLoader(training_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(testing_dataset, batch_size=params['batch_size'], shuffle=False)
    return train_loader, valid_loader, test_loader


def grid_search():
    params = {
        'epoch': None,
        'batch_size': None,
        'learning_rate': None,
        'bottleneck_dim': None,
        'auto_encoder': None
    }
    epoch = [100, 200]
    batch_size = [16, 32, 64]
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    bottleneck_dim = [16, 32, 64]
    CNN_param_combinations = list(itertools.product(epoch, batch_size, learning_rate, ['CNN']))
    linear_param_combinations = list(itertools.product(epoch, batch_size, learning_rate, bottleneck_dim, ['linear']))
    # CNN network
    grid_search_data = []
    for combo in CNN_param_combinations:
        keys = ['epoch', 'batch_size', 'learning_rate', 'auto_encoder']
        params_update = dict(zip(keys, combo))
        params.update(params_update)
        autoencoder = EmojiAutoencoder(params)
        autoencoder.train()
        params['error_rate'] = autoencoder.test()
        print(params)
        grid_search_data.append(params.copy())
    # linear network
    for combo in linear_param_combinations:
        keys = ['epoch', 'batch_size', 'learning_rate','bottleneck_dim', 'auto_encoder']
        params_update = dict(zip(keys, combo))
        params.update(params_update)
        autoencoder = EmojiAutoencoder(params)
        autoencoder.train()
        params['error_rate'] = autoencoder.test()
        print(params)
        grid_search_data.append(params.copy())
    df = pd.DataFrame(grid_search_data)
    df.to_csv('CNN_params_grid_search_final.csv', index=True)


class EmojiAutoencoderWithClassifier(nn.Module):
    def __init__(self, params, num_classes):
        super(EmojiAutoencoderWithClassifier, self).__init__()
        self.params = params
        self.auto_encoder = EmojiAutoencoderCNN(self.params)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),  # Assuming the bottleneck features are of size 128
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout for regularization
            nn.Linear(256, num_classes)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=self.params['learning_rate'])
        self.train_loader, self.valid_loader, self.test_loader = load_data_classification(self.params)
        self.mse_loss_function = nn.MSELoss()
        self.cross_entropy_loss_function = nn.CrossEntropyLoss()


    def forward(self, x):
        bottleneck = self.auto_encoder.encoder(x)
        decoded_features = self.auto_encoder.decoder(bottleneck)
        class_logits = self.classifier(bottleneck.view(bottleneck.size(0), -1))
        return decoded_features, class_logits

    def train(self):
        train_losses = []
        train_accuracies = []
        valid_losses = []
        valid_accuracies = []

        for epoch in range(self.params['epoch']):
            train_loss = 0
            correct_train = 0
            total_train = 0
            for data in self.train_loader:
                imgs, labels = data
                self.optimizer.zero_grad()
                decoded_features, class_logits = self.forward(imgs)
                # Calculate total loss
                reconstruction_loss = self.mse_loss_function(decoded_features, imgs)
                classification_loss = self.cross_entropy_loss_function(class_logits, labels)
                loss = reconstruction_loss + self.params["lambda"] * classification_loss
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                # Calculate accuracy
                _, predicted = torch.max(class_logits.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            # record total loss progress
            train_loss /= len(self.train_loader)
            train_losses.append(train_loss)
            train_accuracies.append(100 * correct_train / total_train)

            # Validation
            valid_loss = 0
            correct_valid = 0
            total_valid = 0
            with torch.no_grad():
                for data in self.valid_loader:
                    imgs, labels = data
                    decoded_features, class_logits = self.forward(imgs)
                    # Calculate total loss
                    reconstruction_loss = self.mse_loss_function(decoded_features, imgs)
                    classification_loss = self.cross_entropy_loss_function(class_logits, labels)
                    loss = reconstruction_loss + self.params["lambda"] * classification_loss
                    valid_loss += loss.item()
                    # Calculate accuracy
                    _, predicted = torch.max(class_logits.data, 1)
                    total_valid += labels.size(0)
                    correct_valid += (predicted == labels).sum().item()
            valid_loss /= len(self.valid_loader)
            valid_losses.append(valid_loss)
            valid_accuracies.append(100 * correct_valid / total_valid)
            print(f'Epoch {epoch + 1}/{self.params["epoch"]}, '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Train Accuracy: {train_accuracies[-1]:.2f}%, '
                  f'Validation Loss: {valid_losses[-1]:.4f}, '
                  f'Validation Accuracy: {valid_accuracies[-1]:.2f}%')

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train MSE Loss')
        plt.plot(valid_losses, label='Validation MSE Loss')
        plt.title('Training and Validation MSE Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(valid_accuracies, label='Validation Accuracy')
        plt.title('Training and Validation Classification Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()
        fig_name = f'learning_curve_{self.params["batch_size"]}_{self.params["epoch"]}_{self.params["learning_rate"]}.jpg'
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        # plt.show()

    def test(self):
        total_mse_error = 0.0
        total_samples = 0
        total_test = 0
        correst_test = 0
        with torch.no_grad():
            for data in self.test_loader:
                imgs, labels = data
                decoded_features, class_logits = self.forward(imgs)
                # Calculate total loss
                reconstruction_loss = self.mse_loss_function(decoded_features, imgs)
                total_mse_error += reconstruction_loss.item() * imgs.size(0)  # Multiply by batch size to accumulate error correctly
                # Calculate accuracy
                _, predicted = torch.max(class_logits.data, 1)
                total_test += labels.size(0)
                correst_test += (predicted == labels).sum().item()
                total_samples += imgs.size(0)
            # performance for autoencoder
            average_mse_error = total_mse_error / total_samples
            print(f'Test Average MSE Error: {average_mse_error:.4f}')
            self.visual_inspection()
            # performance for classifier
            accuracy = 100 * correst_test / total_test
            print(f'Test Average Accuracyr: {accuracy:.4f}')
            return average_mse_error, accuracy

    def visual_inspection(self):
        # Visual inspect the difference between original images and reconstructed images
        dataiter = iter(self.test_loader)  # Convert DataLoader to an iterator
        images, labels = next(dataiter)  # Get the next batch of images
        reconstructed = self.auto_encoder.forward(images)
        images = images.view(-1, 3, 64, 64)  # Reshape original images to proper shape
        reconstructed = reconstructed.view(-1, 3, 64, 64)  # Reshape back to original dimensions
        images = images.numpy()
        reconstructed = reconstructed.detach().numpy()

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
        fig_name = f'reconstructed_{self.params["batch_size"]}_{self.params["epoch"]}_{self.params["learning_rate"]}.jpg'
        plt.savefig(fig_name, dpi=300, bbox_inches='tight')
        # plt.show()


class CustomDatasetClassification(Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        feature = Image.fromarray(feature)
        if self.transform:
            feature = self.transform(feature)
        return feature, self.labels[idx]


def load_data_classification(params):
    # Read data
    data_path = '/Users/mikky/Downloads/train-00000-of-00001-38cc4fa96c139e86.parquet'
    emojis = pd.read_parquet(data_path, engine='pyarrow')
    face_df = emojis[emojis['text'].apply(lambda x: 'face' in x.split() and 'clock' not in x.split())].copy()
    text_label_mapping = {item: label for label, text_list in label_dict.items() for item in text_list}
    face_df['label'] = face_df['text'].replace(text_label_mapping)
    face_features = np.array([Image.open(io.BytesIO(item.get('bytes'))) for item in face_df['image'].values])
    face_labels = np.array(face_df['label'].values)
    unique_labels = set(face_labels)  # Find all unique labels
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    label_indices = np.array([label_to_index[label] for label in face_labels])
    full_dataset = CustomDatasetClassification(face_features, label_indices)  # No transform is applied here

    # Split dataset
    total_length = len(full_dataset)
    train_length = int(total_length * 0.6)
    valid_length = int(total_length * 0.2)
    test_length = total_length - train_length - valid_length
    train_data, valid_data, test_data = random_split(full_dataset, [train_length, valid_length, test_length])

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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        # Randomly change the brightness, contrast, and saturation of an image.
        transforms.ToTensor(),  # Convert a PIL Image or numpy.ndarray to tensor.
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize a tensor image with mean and standard deviation.
    ])

    # Augment validation data: Define light transformations for augmentation
    valid_transform = transforms.Compose([
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

    train_dataset = CustomDatasetClassification(train_data.dataset.features[train_data.indices],
                                  train_data.dataset.labels[train_data.indices], train_transform)
    valid_dataset = CustomDatasetClassification(valid_data.dataset.features[valid_data.indices],
                                  valid_data.dataset.labels[valid_data.indices], valid_transform)
    test_dataset = CustomDatasetClassification(test_data.dataset.features[test_data.indices],
                                 test_data.dataset.labels[test_data.indices], test_transform)
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
    return train_loader, valid_loader, test_loader



label_dict = {
    'animal': ['smiling cat face with open mouth',
               'grinning cat face with smiling eyes',
               'cat face with tears of joy',
               'smiling cat face with heart shaped eyes',
               'cat face with wry smile',
               'kissing cat face with closed eyes',
               'weary cat face',
               'crying cat face',
               'pouting cat face',
               'monkey face',
               'dog face',
               'wolf face',
               'fox face',
               'cat face',
               'lion face',
               'tiger face',
               'horse face',
               'unicorn face',
               'zebra face',
               'cow face',
               'pig face',
               'giraffe face',
               'mouse face',
               'hamster face',
               'rabbit face',
               'bear face',
               'panda face',
               'frog face',
               'dragon face'],
    'human face': ['grinning face',
               'smiling face with open mouth',
               'winking face',
               'smiling face with smiling eyes',
               'smiling face with halo',
               'smiling face with smiling eyes and three hearts',
               'smiling face with heart shaped eyes',
               'grinning face with star eyes',
               'face throwing a kiss',
               'kissing face',
               'white smiling face',
               'kissing face with closed eyes',
               'smiling face with open mouth and smiling eyes',
               'kissing face with smiling eyes',
               'face savouring delicious food',
               'face with stuck out tongue',
               'face with stuck out tongue and winking eye',
               'grinning face with one large and one small eye',
               'face with stuck out tongue and tightly closed eyes',
               'money mouth face',
               'hugging face',
               'smiling face with smiling eyes and hand covering mouth',
               'face with finger covering closed lips',
               'grinning face with smiling eyes',
               'thinking face',
               'zipper mouth face',
               'face with one eyebrow raised',
               'neutral face',
               'expressionless face',
               'face without mouth',
               'smirking face',
               'unamused face',
               'face with rolling eyes',
               'grimacing face',
               'smiling face with open mouth and tightly closed eyes',
               'lying face',
               'relieved face',
               'pensive face',
               'sleepy face',
               'drooling face',
               'sleeping face',
               'face with medical mask',
               'face with thermometer',
               'face with head bandage',
               'nauseated face',
               'smiling face with open mouth and cold sweat',
               'face with open mouth vomiting',
               'sneezing face',
               'overheated face',
               'freezing face',
               'face with uneven eyes and wavy mouth',
               'dizzy face',
               'shocked face with exploding head',
               'face with cowboy hat',
               'face with party horn and party hat',
               'smiling face with sunglasses',
               'nerd face',
               'face with monocle',
               'confused face',
               'worried face',
               'slightly frowning face',
               'white frowning face',
               'face with open mouth',
               'hushed face',
               'astonished face',
               'flushed face',
               'face with tears of joy',
               'face with pleading eyes',
               'frowning face with open mouth',
               'anguished face',
               'fearful face',
               'face with open mouth and cold sweat',
               'disappointed but relieved face',
               'crying face',
               'loudly crying face',
               'face screaming in fear',
               'confounded face',
               'slightly smiling face',
               'persevering face',
               'disappointed face',
               'face with cold sweat',
               'weary face',
               'tired face',
               'face with look of triumph',
               'pouting face',
               'angry face',
               'serious face with symbols covering mouth',
               'smiling face with horns',
               'upside down face'],
    'other': ['robot face',
               'new moon with face',
               'first quarter moon with face',
               'last quarter moon with face',
               'full moon with face',
               'sun with face',
               'wind blowing face',
               'clown face'],
    'person': ['face massage',
               'face massage',
               'face massage',
               'face massage',
               'face massage',
               'face massage',
               'man getting face massage',
               'man getting face massage type 1 2',
               'man getting face massage type 3',
               'man getting face massage type 4',
               'man getting face massage type 5',
               'man getting face massage type 6',
               'woman getting face massage',
               'woman getting face massage type 1 2',
               'woman getting face massage type 3',
               'woman getting face massage type 4',
               'woman getting face massage type 5',
               'woman getting face massage type 6',
               'person with pouting face',
               'person with pouting face',
               'person with pouting face',
               'person with pouting face',
               'person with pouting face',
               'person with pouting face',
               'face with no good gesture',
               'face with no good gesture',
               'face with no good gesture',
               'face with no good gesture',
               'face with no good gesture',
               'face with no good gesture',
               'face with ok gesture',
               'face with ok gesture',
               'face with ok gesture',
               'face with ok gesture',
               'face with ok gesture',
               'face with ok gesture',
               'face palm',
               'face palm',
               'face palm',
               'face palm',
               'face palm',
               'face palm']}

if __name__ == "__main__":
    # grid_search()
    # params = {
    #     'batch_size': 16,
    #     'learning_rate': 0.001,
    #     'epoch': 100,
    #     'auto_encoder': 'CNN'
    # }
    # autoencoder = EmojiAutoencoder(params)
    # autoencoder.train()
    # autoencoder.test()
    params = {
        'batch_size': 16,
        'learning_rate': 0.001,
        'epoch': 100,
        'lambda': 0.5
    }
    classifier = EmojiAutoencoderWithClassifier(params, 4)
    classifier.train()
    classifier.test()

