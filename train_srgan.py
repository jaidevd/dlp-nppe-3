import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from torchvision.transforms.v2.functional import to_dtype
from torcheval.metricss.functional import peak_signal_noise_ratio
from srgan import Generator, Discriminator
from main import EnhanceDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
op = os.path


def train_srgan(generator, discriminator, train_loader, val_loader, n_epochs, device):
    # Loss functions
    criterion_gan = nn.BCELoss()
    criterion_content = nn.MSELoss()

    # Optimizers
    opt_g = optim.Adam(generator.parameters(), lr=0.001)
    opt_d = optim.Adam(discriminator.parameters(), lr=0.001)
    steppers = [optim.lr_scheduler.StepLR(opt_g, 10, gamma=0.1),
                optim.lr_scheduler.StepLR(opt_d, 10, gamma=0.1)]

    best_psnr_epoch = best_psnr = 0
    n_batches = len(train_loader) + len(val_loader)

    for epoch in range(n_epochs):
        epoch_train_loss = epoch_val_loss = train_psnr = val_psnr = 0
        with tqdm(total=n_batches) as pbar:
            generator.train()
            discriminator.train()
            for lr_images, hr_images in train_loader:
                batch_size = lr_images.size(0)

                # Move to device
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

                # Ground truths
                real_labels = torch.ones(batch_size, 1).to(device)
                fake_labels = torch.zeros(batch_size, 1).to(device)

                opt_g.zero_grad()

                # Generate SR images
                sr_images = generator(lr_images)

                # Adversarial loss
                gen_validity = discriminator(sr_images)
                loss_gan = criterion_gan(gen_validity, real_labels)

                # Content loss
                loss_content = criterion_content(sr_images, hr_images)

                # Total generator loss
                loss_g = loss_content + 1e-3 * loss_gan
                loss_g.backward()
                opt_g.step()

                opt_d.zero_grad()

                # Loss on real images
                real_validity = discriminator(hr_images)
                loss_real = criterion_gan(real_validity, real_labels)

                # Loss on fake images
                fake_validity = discriminator(sr_images.detach())
                loss_fake = criterion_gan(fake_validity, fake_labels)

                # Total discriminator loss
                loss_d = (loss_real + loss_fake) / 2
                loss_d.backward()
                opt_d.step()
                litem = (loss_d.item() + loss_g.item()) / 2
                epoch_train_loss += litem

                train_psnr += peak_signal_noise_ratio(
                    sr_images, hr_images, data_range=1.0
                )

                pbar.set_postfix_str(f"Train loss: {litem:.4f}")
                pbar.update(1)

            generator.eval()
            discriminator.eval()
            for lr_images, hr_images in val_loader:
                batch_size = lr_images.size(0)

                # Move to device
                lr_images = lr_images.to(device)
                hr_images = hr_images.to(device)

                # Generate SR images
                with torch.no_grad():
                    sr_images = generator(lr_images)

                    gen_validity = discriminator(sr_images)
                real_labels = torch.ones(batch_size, 1).to(device)
                loss_gan = criterion_gan(gen_validity, real_labels)

                # Content loss
                loss_content = criterion_content(sr_images, hr_images)

                # Total generator loss
                loss_g = loss_content + 1e-3 * loss_gan
                litem = loss_g.item()
                epoch_val_loss += litem
                val_psnr += peak_signal_noise_ratio(
                    sr_images, hr_images, data_range=1.0
                )
                pbar.set_postfix_str(f"Val loss: {litem:.4f}")
                pbar.update(1)
            [stepper.step() for stepper in steppers]
        epoch_train_loss /= len(train_loader)
        epoch_val_loss /= len(val_loader)
        train_psnr /= len(train_loader)
        val_psnr /= len(val_loader)
        current_psnr = (val_psnr * len(val_loader) + train_psnr * len(train_loader)) / n_batches
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_psnr_epoch = epoch
            print(f'Better model found at epoch {epoch}')  # NOQA: T201
            torch.save(generator.state_dict(), "models/generator-best.pth")
            torch.save(discriminator.state_dict(), "models/discriminator-best.pth")
        elif epoch - best_psnr_epoch > 5:
            print(f'Stopping at epoch {epoch} - best epoch was {best_psnr_epoch}')  # NOQA: T201
            break

        print(  # NOQA: T201
            f"Train: {epoch_train_loss:.4f}; Val: {epoch_val_loss:.4f};",
            f"Train PSNR: {train_psnr:.4f}; Val PSNR: {val_psnr:.4f}",
        )


def collate_sr(batch):
    images, labels = map(torch.stack, zip(*batch))
    images = to_dtype(images, torch.float32, scale=True)
    labels = to_dtype(labels, torch.float32, scale=True)
    return images, labels


def main():
    n_epochs = 40
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator().to(device)
    if op.exists('models/generator-best.pth'):
        generator.load_state_dict(torch.load("models/generator-best.pth", weights_only=True))
    discriminator = Discriminator().to(device)
    if op.exists('models/discriminator-best.pth'):
        discriminator.load_state_dict(torch.load("models/discriminator-best.pth", weights_only=True))

    train_ds = EnhanceDataset("denoised/train/train/", "archive/train/gt/")
    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, pin_memory=True, collate_fn=collate_sr
    )
    val_ds = EnhanceDataset("denoised/val/val/", "archive/val/gt/")
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=True, pin_memory=True, collate_fn=collate_sr
    )

    train_srgan(generator, discriminator, train_loader, val_loader, n_epochs, device)


if __name__ == "__main__":
    main()
