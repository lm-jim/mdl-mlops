import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

class ConvCVAE(pl.LightningModule):
    def __init__(self, latent_dim=20, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.latent_dim = latent_dim
        self.lr = lr

        # Red Encoder
        self.encoder_net = nn.Sequential(
            nn.Conv2d(11, 32, kernel_size=3, stride=2, padding=1), # 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.Flatten(), # 64 * 7 * 7 = 3136
        )
        self.fc_mu = nn.Linear(3136, latent_dim)
        self.fc_logvar = nn.Linear(3136, latent_dim)

        # Red de proyeccion
        self.decoder_input = nn.Linear(latent_dim + 10, 3136)

        # Red Decoder
        self.decoder_net = nn.Sequential(
            nn.Unflatten(1, (64, 7, 7)), # 7x7
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1), # 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 28x28
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # Desviación estándar
        eps = torch.randn_like(std)   # Ruido aleatorio
        return mu + eps * std

    def forward(self, x, y):
        batch_size = x.size(0)

        # Codificamos con One-Hot el tag y lo expandimos a 28x28
        y_oh = F.one_hot(y, num_classes=10).float()
        y_map = y_oh.view(batch_size, 10, 1, 1).expand(batch_size, 10, 28, 28)

        # Aplicamos la red Encoder
        x_cond = torch.cat([x, y_map], dim=1) # 11 canales, 1 para la imagen y 10 para los tags
        x_enc = self.encoder_net(x_cond)
        mu, logvar = self.fc_mu(x_enc), self.fc_logvar(x_enc)

        # Parametrización
        z = self.reparameterize(mu, logvar)

        # Aplicamos la red Decoder
        z_cond = torch.cat([z, y_oh], dim=1) # Vector latente + tags

        # Red de proyección, reescalamos a 64x7x7
        d_in = self.decoder_input(z_cond)
        decoded_x = self.decoder_net(d_in)

        return decoded_x, mu, logvar

    def training_step(self, batch, batch_idx):
        x, y = batch
        recon_x, mu, logvar = self.forward(x, y)

        # Loss de reconstruccion (BCE)
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # Loss de Divergencia KL
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Promedio de ambos
        loss = (recon_loss + kld_loss) / x.size(0)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def generate_number(self, number):
        label = torch.zeros(1, 10)
        label[:, number] = 1
        z = torch.randn(1, self.latent_dim)
        z_cond = torch.cat([z, label], dim=1)

        with torch.no_grad():
            d_in = self.decoder_input(z_cond)
            recon_x = self.decoder_net(d_in)
        return recon_x.squeeze().cpu().numpy()