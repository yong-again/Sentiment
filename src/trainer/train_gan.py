import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model.gan import Generator, Discriminator
from torch.utils.data import DataLoader

import sys
sys.path.append("./dataloaders")
from dataloaders.data_loader_gan import TFIDFDataset

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    # Convert TF-IDF data (X) to PyTorch tensors and move to device
    tfidf_tensor = torch.from_numpy(X).float().to(device)
    
    # Create DataLoader
    dataset = TFIDFDataset(tfidf_tensor)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Create models
    generator = Generator(args.latent_dim, args.tfidf_dim).to(device)
    discriminator = Discriminator(args.tfidf_dim).to(device)

    # Loss and optimizers
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=args.learning_rate)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        for real_data in dataloader:
            current_batch_size = real_data.shape[0]
            real_labels = torch.ones(current_batch_size, 1).to(device)
            fake_labels = torch.zeros(current_batch_size, 1).to(device)
            
            # Train the Discriminator
            optimizer_d.zero_grad()
            real_outputs = discriminator(real_data)
            real_loss = criterion(real_outputs, real_labels)
            real_loss.backward()
            
            noise = torch.randn(current_batch_size, args.latent_dim).to(device)
            fake_data = generator(noise)
            fake_outputs = discriminator(fake_data.detach())
            fake_loss = criterion(fake_outputs, fake_labels)
            fake_loss.backward()
            
            optimizer_d.step()

            # Train the Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_data)
            gen_loss = criterion(outputs, real_labels)
            gen_loss.backward()
            optimizer_g.step()
        
        # Print statistics or save models periodically
        if epoch % 10 == 0:
            print(f"Epoch [{epoch}/{args.epochs}] | D Loss: {real_loss + fake_loss:.4f} | G Loss: {gen_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAN for text data augmentation.")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="Learning rate for the optimizer.")
    parser.add_argument('--latent_dim', type=int, default=100, help="Dimension of the random noise vector.")
    parser.add_argument('--tfidf_dim', type=int, default=1000, help="Dimension of the TF-IDF vector.")
    parser.add_argument('--epochs', type=int, default=1000, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for training.")
    
    args = parser.parse_args()
    main(args)
