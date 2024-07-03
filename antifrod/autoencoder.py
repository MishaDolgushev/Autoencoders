from torch.nn import functional as F
import torch
from torch import nn
from tqdm import tqdm


class VariationalAutoencoder(nn.Module):
    def __init__(self, n_input_features, latent_dim, encoder_neuron_list, decoder_neuron_list) -> None:
        super().__init__()
        self.n_input_features = n_input_features
        self.latent_dim = latent_dim
        self.encoder_neuron_list = encoder_neuron_list
        self.decoder_neuron_list = decoder_neuron_list

        self.error = []
        self._build_encoder()
        self._build_decoder()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-3)

    def _build_encoder(self):
        encoder_list = []
        encoder_list.append(nn.Linear(self.n_input_features, self.encoder_neuron_list[0]))
        encoder_list.append(nn.BatchNorm1d(self.encoder_neuron_list[0]))
        encoder_list.append(nn.ReLU())

        for i in range(0, len(self.encoder_neuron_list) - 1):
            encoder_list.append(nn.Linear(self.encoder_neuron_list[i], self.encoder_neuron_list[i+1]))
            encoder_list.append(nn.BatchNorm1d(self.encoder_neuron_list[i+1]))
            encoder_list.append(nn.ReLU())

        encoder_list.append(nn.Linear(self.encoder_neuron_list[-1], 2 * self.latent_dim))
        self.encoder = nn.Sequential(*encoder_list)
        
    def _build_decoder(self):
        decoder_list = []
        decoder_list.append(nn.Linear(self.latent_dim, self.decoder_neuron_list[0]))
        decoder_list.append(nn.BatchNorm1d(self.decoder_neuron_list[0]))
        decoder_list.append(nn.ReLU())

        for i in range(0, len(self.decoder_neuron_list) - 1):
            decoder_list.append(nn.Linear(self.decoder_neuron_list[i], self.decoder_neuron_list[i+1]))
            decoder_list.append(nn.BatchNorm1d(self.decoder_neuron_list[i+1]))
            decoder_list.append(nn.ReLU())

        decoder_list.append(nn.Linear(self.decoder_neuron_list[-1], self.n_input_features))
        decoder_list.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_list)
        
    def reparametrize(self, mu, sigma):
        return mu + sigma * torch.randn_like(sigma)
    
    def get_new_params(self, x):
        params = self.encoder(x)
        mu, log_sigma = torch.chunk(params, 2, dim=1)
        sigma = torch.exp(0.5 * log_sigma)
        return mu, sigma

    def get_latent_vector(self, mu, sigma):
        return mu + torch.randn_like(sigma) * sigma
        
    def get_new_image(self, z):
        return self.decoder(z)
    
    def loss(self, input_vec, reconstructed_vec, mu, sigma):
        BCE = F.binary_cross_entropy(reconstructed_vec.view(-1, self.n_input_features), input_vec.view(-1, self.n_input_features), reduction='sum')
        KLD = -0.5 * torch.sum(1 + torch.log(sigma**2) - mu.pow(2) - sigma.pow(2))
        return BCE + KLD
    
    def fit(self, n_epoch, dataloader, device):
        self.train(True)
        for epoch in tqdm(range(n_epoch)):
            local_error = 0
            for data in dataloader:
                batch, _ = data
                batch = batch.to(device).view(-1, self.n_input_features) 
                
                self.optimizer.zero_grad()
                mu, sigma = self.get_new_params(batch)
                
                z = self.get_latent_vector(mu, sigma)                
                new_image = self.get_new_image(z)
                
                loss = self.loss(batch, new_image, mu, sigma)
                local_error += loss.item()
                loss.backward()
                self.optimizer.step()
            local_error /= len(dataloader)
            self.error.append(local_error)
            print(f'Epoch [{epoch+1}/{n_epoch}], Loss: {local_error:.4f}')

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vae = VariationalAutoencoder(784, 10, [256, 128], [128, 256])
# vae.to(device)