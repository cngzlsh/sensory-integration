import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import itertools
from loguru import logger

class FFNet(nn.Module):
    '''
    Generaic feedforward Neural Network. Intended to be used for auditory processing.
    '''
    def __init__(self, input_dim, hidden_dim, output_dim, device=torch.device('cuda')):
        super(FFNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.layers = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
        self.to(device)

    def forward(self, x):
        return self.layers(x)

class FFVisualModuleEncoder(nn.Module):
    '''
    The visual module encoder receives image and projects to a visual embedding.
    Using pre-trained vgg16 model with last layer re-trained and feature layers frozen
    '''
    def __init__(self, visual_embedding_dim, img_dim, pretrained=False, grayscale=False, device=torch.device('cuda')):
        super(FFVisualModuleEncoder, self).__init__()

        self.c_channel = 1 if grayscale else 3

        if not pretrained:
            self.cnn = nn.Sequential(
                nn.Conv2d(self.c_channel, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.threshold_dims = self.get_threshold_dims(img_dim)
            # print('Threhsold dims: ', self.threshold_dims)
            self.cnn.add_module('flatten', nn.Flatten())
            self.cnn.add_module('linear', nn.Linear(np.prod(self.threshold_dims), visual_embedding_dim))
            self.cnn.add_module('relu', nn.ReLU())
        else:
            raise NotImplementedError
        
        self.visual_embedding_dim = visual_embedding_dim
        self.device = device
        self.to(device)
    
    def get_threshold_dims(self, img_dim):
        w, l = img_dim
        with torch.no_grad():
            return self.cnn.cpu()(torch.rand(1, 3, w, l)).shape[1:]

    def forward(self, img):
        img = torch.as_tensor(img).to(self.device) # input must be normalised
        return self.cnn(img)

class FFVisualModuleDecoder(nn.Module):
    '''
    The visual module decoder receives rnn-processed embeddings and predicts the next step of visual signal.
    '''
    def __init__(self, input_dim, threshold_dims, pretrained=False, grayscale=False, device=torch.device('cuda')):
        super(FFVisualModuleDecoder, self).__init__()
        self.input_dim = input_dim
        self.threshold_dims = threshold_dims
        self.linear = nn.Sequential(
            nn.Linear(input_dim, np.prod(threshold_dims)),
            nn.ReLU(),
            )
        self.c_channels = 1 if grayscale else 3
        
        if pretrained is False:
            self.decnn = nn.Sequential(
                nn.ConvTranspose2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, self.c_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.Sigmoid() # projects to (0, 1) scale
            )
        else:
            raise NotImplementedError
        
        self.to(device)
        
    def forward(self, embedding):
        h = self.linear(embedding)
        h = h.reshape(h.shape[0], self.threshold_dims[0], self.threshold_dims[1], self.threshold_dims[2])
        return self.decnn(h)


class FFVisualModuleEncoderL(nn.Module):
    '''
    The visual module encoder receives an image and projects it to a visual embedding.
    '''
    def __init__(self, visual_embedding_dim, img_dim, pretrained=False, grayscale=False, device=torch.device('cuda')):
        super(FFVisualModuleEncoder, self).__init__()

        self.c_channel = 1 if grayscale else 3

        if not pretrained:
            self.cnn = nn.Sequential(
                nn.Conv2d(self.c_channel, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            )
            self.threshold_dims = self.get_threshold_dims(img_dim)
            self.cnn.add_module('flatten', nn.Flatten())
            self.cnn.add_module('linear', nn.Linear(np.prod(self.threshold_dims), visual_embedding_dim))
            self.cnn.add_module('relu', nn.ReLU())
        else:
            raise NotImplementedError
        
        self.visual_embedding_dim = visual_embedding_dim
        self.device = device
        self.to(device)
    
    def get_threshold_dims(self, img_dim):
        w, l = img_dim
        with torch.no_grad():
            return self.cnn.cpu()(torch.rand(1, self.c_channel, w, l)).shape[1:]

    def forward(self, img):
        img = torch.as_tensor(img).to(self.device) # input must be normalized
        return self.cnn(img)
    

class FFVisualModuleDecoderL(nn.Module):
    '''
    The visual module decoder receives RNN-processed embeddings and predicts the next step of the visual signal.
    '''
    def __init__(self, input_dim, threshold_dims, pretrained=False, grayscale=False, device=torch.device('cuda')):
        super(FFVisualModuleDecoder, self).__init__()
        self.input_dim = input_dim
        self.threshold_dims = threshold_dims
        self.linear = nn.Sequential(
            nn.Linear(input_dim, np.prod(threshold_dims)),
            nn.ReLU(),
            )
        self.c_channels = 1 if grayscale else 3
        
        if not pretrained:
            self.decnn = nn.Sequential(
                nn.ConvTranspose2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, self.c_channels, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid() # projects to (0, 1) scale
            )
        else:
            raise NotImplementedError
        
        self.to(device)
        
    def forward(self, embedding):
        h = self.linear(embedding)
        h = h.reshape(h.shape[0], self.threshold_dims[0], self.threshold_dims[1], self.threshold_dims[2])
        return self.decnn(h)


class RecIntegrationModule(nn.Module):
    '''
    A feedforward integration module processes egocentric visual, auditory, velocity and head direction
    '''
    def __init__(self, auditory_enc, auditory_dec, vision_enc, vision_dec, n_rec_layers, rnn_hidden_dim, integrator_type='lstm', aux_dim=5, use_proj=True, dropout=0.1, verbose=True, device=torch.device('cuda')):
        super(RecIntegrationModule, self).__init__()
        self.auditory_enc = auditory_enc
        self.auditory_dec = auditory_dec
        self.vision_enc = vision_enc
        self.vision_dec = vision_dec
        self.n_rec_layer = n_rec_layers
        self.aux_dim = aux_dim

        self.visual_embedding_dim = self.vision_enc.visual_embedding_dim
        self.auditory_embedding_dim = self.auditory_enc.output_dim

        self.rnn_hidden_dim = rnn_hidden_dim

        self.device = device
        self.verbose = verbose
        self.use_proj = use_proj
        self.integrator_type = integrator_type

        if self.integrator_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.auditory_embedding_dim+self.visual_embedding_dim+aux_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=self.n_rec_layer,
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
                device=device        
            )
        elif self.integrator_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=self.auditory_embedding_dim+self.visual_embedding_dim+aux_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=self.n_rec_layer,
                nonlinearity='relu',
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
                device=device
            )
        else:
            raise NotImplementedError

        if self.use_proj:
            if self.verbose:
                logger.debug('Using projection layers for auditory and visual modules.')
            self.auditory_dec_proj = nn.Linear(self.rnn_hidden_dim, self.auditory_dec.input_dim)
            self.visual_dec_proj = nn.Linear(self.rnn_hidden_dim, self.vision_dec.input_dim)
        else:
            if self.verbose:
                logger.debug('No projection layers for auditory module.')
            assert self.auditory_dec.input_dim == self.rnn_hidden_dim
            self.visual_dec_proj = nn.Linear(self.rnn_hidden_dim, self.vision_dec.input_dim)

        self.to(device)

    def freeze_vision_modules(self):
        for param in self.vision_enc.parameters():
            param.requires_grad = False
        for param in self.vision_dec.parameters():
            param.requires_grad = False
        if self.verbose:
            logger.info('Gradient for vision modules DISABLED.')

    def unfreeze_vision_modules(self):
        for param in self.vision_enc.parameters():
            param.requires_grad = True
        for param in self.vision_dec.parameters():
            param.requires_grad = True
        if self.verbose:
            logger.info('Gradient for vision modules ENABLED.')
    
    def freeze_auditory_modules(self):
        for param in self.auditory_enc.parameters():
            param.requires_grad = False
        for param in self.auditory_dec.parameters():
            param.requires_grad = False
        if self.verbose:
            logger.info('Gradient for auditory modules DISABLED.')
    
    def unfreeze_auditory_modules(self):
        for param in self.auditory_enc.parameters():
            param.requires_grad = True
        for param in self.auditory_dec.parameters():
            param.requires_grad = True
        if self.verbose:
            logger.info('Gradient for auditory modules ENABLED.')

    def forward(self, imgs, audios, auxs, return_hidden=False):
        audios = torch.as_tensor(audios, dtype=torch.float32).to(self.device)                   # np.ndarray: (bs, ts, 2*freq_dims)
        if len(audios.shape) == 4:                                                              # if channels are not separated:
            audios = audios.reshape(audios.shape[0], audios.shape[1], -1)                       # stack the channels at 3rd dim

        imgs = torch.as_tensor(imgs, dtype=torch.float32).to(self.device)                       # np.ndarray: (bs, ts, c, w, l)
        if imgs.shape[-3] >3:                                                                   # changes to (bs, ts, w, l, c)
            imgs = imgs.permute(0,1,4,2,3)

        audios_emb = self.auditory_enc(audios)                                                  # tensor: (bs, ts, aud_emb_dim)
        imgs_emb = torch.stack([self.vision_enc(img) for img in torch.unbind(imgs, dim=1)], dim=1)  # tensor: (bs, ts, vis_emd_dim)
        auxs = torch.as_tensor(auxs, dtype=torch.float32).to(self.device)                           # tensor: (bs, ts, aux_dim)

        if self.integrator_type == 'lstm':
            out, hidden = self.rnn(
                torch.concat([audios_emb, imgs_emb, auxs], dim=-1)                              # tensor: (bs, 1, hidden_dim)
            )
        elif self.integrator_type == 'rnn':
            out, hidden = self.rnn(
                torch.concat([audios_emb, imgs_emb, auxs], dim=-1),                             # tensor: (bs, 1, hidden_dim)                                          
            )

        out = out[:, -1, :]

        if self.use_proj:
            vision_out = self.visual_dec_proj(out.squeeze(1))
            auditory_out = self.auditory_dec_proj(out.squeeze(1))
        else:
            vision_out = out.squeeze(1)
            auditory_out = out.squeeze(1)

        if return_hidden:
            return vision_out, auditory_out, hidden
        else:
            return self.vision_dec(vision_out), self.auditory_dec(auditory_out)
    
    def get_all_weights(self):
        if self.use_proj:
            return {'auditory_enc': self.auditory_enc.state_dict(),
                    'auditory_dec': self.auditory_dec.state_dict(),
                    'vision_enc': self.vision_enc.state_dict(),
                    'vision_dec': self.vision_dec.state_dict(),
                    'rnn': self.rnn.state_dict(),
                    'auditory_dec_proj': self.auditory_dec_proj.state_dict(),
                    'visual_dec_proj': self.visual_dec_proj.state_dict()}
        else:
            return {'auditory_enc': self.auditory_enc.state_dict(),
                    'auditory_dec': self.auditory_dec.state_dict(),
                    'vision_enc': self.vision_enc.state_dict(),
                    'vision_dec': self.vision_dec.state_dict(),
                    'rnn': self.rnn.state_dict()}


class UniModalRecModule(nn.Module):
    '''
    A unimodal recurrent module processes either egocentric visual or auditory signals.
    '''
    def __init__(self, enc, dec, modality, n_rec_layers, rnn_hidden_dim, integrator_type='rnn', aux_dim=5, use_proj=True, dropout=0.1, verbose=True, device=torch.device('cuda')):
        super(UniModalRecModule, self).__init__()
        self.enc = enc
        self.dec = dec
        
        self.modality = modality
        if self.modality[:3] == 'aud':
            assert isinstance(self.enc, FFNet)
            assert isinstance(self.dec, FFNet)
        elif self.modality[:3] == 'vis':
            assert isinstance(self.enc, FFVisualModuleEncoder)
            assert isinstance(self.dec, FFVisualModuleDecoder)
        else:
            raise ValueError('Modality must be either auditory or visual.')
        
        self.n_rec_layer = n_rec_layers
        self.aux_dim = aux_dim

        self.embedding_dim = self.enc.visual_embedding_dim if hasattr(self.enc, 'visual_embedding_dim') else self.enc.output_dim

        self.rnn_hidden_dim = rnn_hidden_dim

        self.device = device
        self.verbose = verbose
        self.use_proj = use_proj
        self.integrator_type = integrator_type

        if self.integrator_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=self.embedding_dim+aux_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=self.n_rec_layer,
                nonlinearity='relu',
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
                device=device        
            )
        elif self.integrator_type == 'rnn':
            self.rnn = nn.RNN(
                input_size=self.embedding_dim+aux_dim,
                hidden_size=self.rnn_hidden_dim,
                num_layers=self.n_rec_layer,
                nonlinearity='relu',
                batch_first=True,
                dropout=dropout,
                bidirectional=False,
                device=device
            )
            
        else:
            raise NotImplementedError

        if self.use_proj:
            if self.verbose:
                logger.debug('Using projection layers for auditory and visual modules.')
            self.dec_proj = nn.Linear(self.rnn_hidden_dim, self.dec.input_dim)
        else:
            if self.verbose:
                logger.debug('No projection layers for auditory module.')
            assert self.dec.input_dim == self.rnn_hidden_dim

        self.to(device)

    def freeze_sensory_modules(self):
        for param in self.enc.parameters():
            param.requires_grad = False
        for param in self.dec.parameters():
            param.requires_grad = False
        if self.verbose:
            logger.info(f'Gradient for {self.modality} modules DISABLED.')

    def unfreeze_sensory_modules(self):
        for param in self.enc.parameters():
            param.requires_grad = True
        for param in self.dec.parameters():
            param.requires_grad = True
        if self.verbose:
            logger.info(f'Gradient for {self.modality} modules ENABLED.')
    
    def forward(self, sensory_src, auxs):
        # unimodel forward pass only uses one type of modality as well as motion signals

        sensory_src = torch.as_tensor(sensory_src, dtype=torch.float32).to(self.device)
        if self.modality[:3]=='aud':                                                                       # np.ndarray: (bs, ts, 2*freq_dims)
            if len(sensory_src.shape) == 4:                                                                 # if channels are not separated: stack the channels at 3rd dim
                sensory_src = sensory_src.reshape(sensory_src.shape[0], sensory_src.shape[1], -1)
            assert len(sensory_src.shape) == 3
            sensory_emb = self.enc(sensory_src)                                                             # tensor: (bs, ts, aud_emb_dim)
        elif self.modality[:3]=='vis':                                                                       # np.ndarray: (bs, ts, c, w, l)
            assert len(sensory_src.shape) == 5
            if sensory_src.shape[-3] >3:                                                                    # changes to (bs, ts, w, l, c)
                sensory_src = sensory_src.permute(0,1,4,2,3)
            sensory_emb = torch.stack([self.enc(img) for img in torch.unbind(sensory_src, dim=1)], dim=1)   # tensor: (bs, ts, vis_emd_dim)
        auxs = torch.as_tensor(auxs, dtype=torch.float32).to(self.device)                                   # tensor: (bs, ts, aux_dim)

        if self.integrator_type == 'lstm':
            out, _ = self.rnn(
                torch.concat([sensory_emb, auxs], dim=-1)                                                    # tensor: (bs, 1, hidden_dim)
            )
        
        elif self.integrator_type == 'rnn':
            out, h_n = self.rnn(
                torch.concat([sensory_emb, auxs], dim=-1),                                                   # tensor: (bs, 1, hidden_dim)                                          
            )
        
        out = out[:, -1, :]

        if self.use_proj:
            sensory_out = self.dec_proj(out.squeeze(1))
        else:
            sensory_out = out.squeeze(1)
        
        return self.dec(sensory_out)
    
    def get_all_weights(self):
        if self.use_proj:
            return {'enc': self.enc.state_dict(),
                    'dec': self.dec.state_dict(),
                    'rnn': self.rnn.state_dict(),
                    'dec_proj': self.dec_proj.state_dict()}
        else:
            return {'enc': self.enc.state_dict(),
                    'dec': self.dec.state_dict(),
                    'rnn': self.rnn.state_dict()}



if __name__ == '__main__':
    auditory_enc = FFNet(input_dim=400*2, hidden_dim=100, output_dim=100)
    auditory_dec = FFNet(input_dim=200, hidden_dim=100, output_dim=400*2)
    visual_enc = FFVisualModuleEncoder(visual_embedding_dim=200, img_dim=(160,120), pretrained=False)
    visual_dec = FFVisualModuleDecoder(visual_embedding_dim=200, threshold_dims=visual_enc.threshold_dims, pretrained=False)
    
    int_module = RecIntegrationModule(auditory_enc, auditory_dec, visual_enc, visual_dec, 1, 200, 4)

    bs = 32
    ts = 10
    imgs = np.random.random(size=(bs, ts, 3, 160, 120)) # np.ndarray: (bs, ts, 3, w, l)
    audios = np.random.random(size=(bs, ts, 800))
    auxs = np.random.random(size=(bs, ts, 4))

    next_img, next_audio = int_module(imgs, audios, auxs)
    
    assert False