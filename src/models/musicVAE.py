import torch
from torch import nn


class MusicVAE(nn.Module):
    def __init__(self, input_size, 
                 enc_latent_dim = 512, conductor_dim = 512, U = 16,
                 enc_hidden_size = 2048, dec_hidden_size = 1024, conductor_hidden_size = 1024, 
                 enc_num_layers = 2, dec_num_layers = 2, conductor_num_layers = 2):
        super(MusicVAE, self).__init__()
        self.input_size = input_size # 2**num_classes
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.conductor_hidden_size = conductor_hidden_size
        
        self.enc_num_layers = enc_num_layers
        self.dec_num_layers = dec_num_layers
        self.conductor_num_layers = conductor_num_layers
        
        self.latent_dim = enc_latent_dim
        self.conductor_dim = conductor_dim
        
        self.D = 2 if self.enc_num_layers > 1 else 1 # num_direction 
        self.U = U # embedding vector size
        
        # Encoder layers
        self.encoder = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.enc_hidden_size,
                            num_layers=self.enc_num_layers,
                            bidirectional=True)
        
        self.mu = nn.Linear(self.enc_hidden_size * self.enc_num_layers * self.D, self.latent_dim)
        self.std = nn.Linear(self.enc_hidden_size * self.enc_num_layers * self.D, self.latent_dim)
        self.z = nn.Linear(self.latent_dim, self.U * self.input_size)
        
        # Conductor layers
        self.conductor = nn.LSTM(input_size=self.input_size, 
                                 hidden_size=self.conductor_hidden_size, 
                                 num_layers = self.conductor_num_layers,
                                 bidirectional = False
                                 )
        self.conductor_out = nn.Linear(self.conductor_hidden_size, self.conductor_dim)
        
        # Decoder layers
        self.decoder = nn.LSTM(input_size=self.input_size,
                               hidden_size = self.dec_hidden_size,
                               num_layers = self.dec_num_layers,
                               bidirectional = False)
        self.classifier = nn.Linear(self.dec_hidden_size, self.input_size)
    
    def forward(self, x): # x : (batch, seq_len, 2**num_classes)
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Encoder 
        _, (hn, _) = self.encoder(x.transpose(0, 1)) # out : (seq_len, batch_size, enc_hidden_size) / hn, cn : (enc_num_layers * D, batch_size, enc_hidden_size)
        
        h = hn.transpose(0, 1).reshape(-1, self.enc_num_layers * self.D * self.enc_hidden_size)
        mu = self.mu(h)  # Eq.6 mu : (batch_size, latent_dim)
        sigma = torch.log(torch.exp(self.std(h)) + 1) # Eq.7 sigma : (batch_size, latent_dim)
        with torch.no_grad():
            epsilon = torch.rand_like(sigma)
        z = mu + (sigma * epsilon) # Eq.2 parameterize : (batch_size, latent_dim)         
        z = self.z(z).view(-1, self.U, self.input_size) # z : (batch_size, self.U, self.input_size)         
        
        # Decoder 
        subseq_len = seq_len // self.U 
        y_hat_seq = torch.zeros(batch_size, seq_len, self.input_size)
        for idx in range(self.U):
            # conductor
            embed, (hn, cn) = self.conductor(z[:, idx, :].unsqueeze(1)) # embed : (batch_size, 1, conductor_hidden_size)
            embed = self.conductor_out(embed) # Eq.9 embed : (batch_size, 1, conductor_latent_dim)
            
            # decoder
            
            # concatenate embedding with prev output token
            x_embed = torch.cat([embed, x[:, idx * subseq_len:(idx + 1) * subseq_len,:]], dim = 1) # (batch_size, subseq_len, input_size)
            
            x_hat, (hn, cn) = self.decoder(x_embed.transpose(0, 1)) # pred : (subseq_len + 1, batch_size, dec_hidden_size)
            y_hat = self.classifier(x_hat[1:, :, :].transpose(0, 1)) # y_hat : (batch_size, subseq_len, input_size)
            y_hat = torch.softmax(y_hat, dim=-1) # y_hat : (batch_size, subseq_len, input_size)
            
            y_hat_seq[:, idx * subseq_len:(idx + 1) * subseq_len, :] = y_hat
        
        return y_hat_seq, mu, sigma
        