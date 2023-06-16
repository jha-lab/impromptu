#modified from https://github.com/singhgautam/slate/blob/master/slate.py

import torch
from utils import *
from dVAE import dVAE
from slot_attn import SlotAttentionEncoder
from transformer import PositionalEncoding, TransformerDecoder
import argparse


class SLATE_encoder(nn.Module):

    def __init__(self, args):
        """
        SLATE encoder
        """
        super().__init__()

        self.num_slots = args.num_slots
        self.slot_size = args.slot_size
        self.vocab_size = args.vocab_size
        self.d_model = args.d_model
        self.H_enc = args.image_size // 4
        self.W_enc = args.image_size // 4
        self.dvae = dVAE(args.vocab_size, args.img_channels)
        self.positional_encoder = PositionalEncoding(1 + (args.image_size // 4) ** 2, args.d_model, args.dropout)
        self.slot_attn = SlotAttentionEncoder(
            args.num_iterations, 
            args.num_slots,
            args.d_model, 
            args.slot_size, 
            args.mlp_hidden_size, 
            args.pos_channels,
            args.num_slot_heads,
        )
        self.dictionary = OneHotDictionary(args.vocab_size + 1, args.d_model)
    
    def forward(self, images, tau, hard=False):

        B, C, H, W = images.size()

        # dvae encode
        z_logits = F.log_softmax(self.dvae.encoder(images), dim=1)
        _, _, H_enc, W_enc = z_logits.size()
        z = gumbel_softmax(z_logits, tau, hard, dim=1)

        # dvae recon
        recon = self.dvae.decoder(z)
        mse = ((images - recon) ** 2).sum() / B
        # hard z
        z_hard = gumbel_softmax(z_logits, tau, True, dim=1).detach()
        z_transformer_target = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        # add BOS token
        z_transformer_input = torch.cat([torch.zeros_like(z_transformer_target[..., :1]), z_transformer_target], dim=-1)
        z_transformer_input = torch.cat([torch.zeros_like(z_transformer_input[..., :1, :]), z_transformer_input], dim=-2)
        z_transformer_input[..., 0, 0] = 1.0 
        # tokens to embeddings
        emb_input = self.dictionary(z_transformer_input)
        emb_input = self.positional_encoder(emb_input)
        slots, attns = self.slot_attn(emb_input[:, 1:,:])
        attns = attns.transpose(-1,-2)
        attns_raw = attns.reshape(-1, self.num_slots, 1, self.H_enc, self.W_enc).repeat_interleave(H // H_enc, dim=-2).repeat_interleave(W // W_enc, dim=-1)
        attns_vis = images.unsqueeze(1)*attns_raw + (1. - attns_raw)
        
        return (
            slots, 
            attns_vis,
            recon,
            mse,
            emb_input,
            z_transformer_target,
        )

    def generate(self, images):

        z_logits = F.log_softmax(self.dvae.encoder(images), dim=1)
        _, _, H_enc, W_enc = z_logits.size()

        # hard z
        z_hard = torch.argmax(z_logits, axis=1)
        z_hard = F.one_hot(z_hard, num_classes=self.vocab_size).permute(0, 3, 1, 2).float()
        one_hot_tokens = z_hard.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        # add BOS token
        one_hot_tokens = torch.cat([torch.zeros_like(one_hot_tokens[..., :1]), one_hot_tokens], dim=-1)
        one_hot_tokens = torch.cat([torch.zeros_like(one_hot_tokens[..., :1, :]), one_hot_tokens], dim=-2)
        one_hot_tokens[:, 0, 0] = 1.0
        
        # tokens to embeddings
        emb_input = self.dictionary(one_hot_tokens)
        emb_input = self.positional_encoder(emb_input)

        slots, _ = self.slot_attn(emb_input[:, 1:,:])
        

        return slots

class SLATE(nn.Module):

    def __init__(self, args):
        '''
        SLATE 
        '''
        super().__init__()

        self.encoder = SLATE_encoder(args)
        self.vocab_size = args.vocab_size
        self.slot_proj = linear(args.slot_size, args.d_model, bias=False)
        self.out = linear(args.d_model, args.vocab_size, bias=False)
        self.gen_len = (args.image_size // 4) ** 2
        self.H_enc = args.image_size // 4
        self.W_enc = args.image_size // 4
        self.tf_dec = TransformerDecoder(
		                args.num_dec_blocks, 
                        (args.image_size // 4) ** 2, 
                        args.d_model, 
                        args.num_heads,
                        args.dropout
                        )
    

    def forward(self, images, tau, hard=False):

        B, C, H, W = images.size()
        slots, attns, recon, mse, emb_input, z_transformer_target = self.encoder(images, tau, hard)
        prompt = self.slot_proj(slots)
        #generate output
        decoder_output = self.tf_dec(emb_input[:,:-1], prompt)
        pred = self.out(decoder_output)
        cross_entropy = -(z_transformer_target*torch.log_softmax(pred, dim=-1)).flatten(start_dim=1).sum(-1).mean()

        return (
                recon.clamp(0., 1.),
                mse,
                cross_entropy,
                attns
            )

    def generate(self, images):

        gen_len = (images.size(-1) // 4) ** 2
        B, C, H, W = images.size()
        slots = self.encoder.generate(images)
        # tokens to embeddings
        prompt = self.slot_proj(slots)
        #generate output
        z_gen = slots.new_zeros(0)
        z_transformer_input = slots.new_zeros(B, 1, self.vocab_size + 1)
        z_transformer_input[..., 0] = 1.0
        for t in range(gen_len):
            
            decoder_output = self.tf_dec(
                self.encoder.positional_encoder(self.encoder.dictionary(z_transformer_input)),
                prompt
                )
            z_next = F.one_hot(self.out(decoder_output)[:, -1:].argmax(dim=-1), self.vocab_size)
            z_gen = torch.cat((z_gen, z_next), dim=1)
            z_transformer_input = torch.cat([
                z_transformer_input,
                torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1)
            ], dim=1)

        z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, self.H_enc, self.W_enc)
        recon_transformer = self.encoder.dvae.decoder(z_gen)

        return recon_transformer.clamp(0., 1.)


class OneHotDictionary(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.dictionary = nn.Embedding(vocab_size, emb_size)

    def forward(self, x):
        """
        x: B, N, vocab_size
        """

        tokens = torch.argmax(x, dim=-1)  # batch_size x N
        token_embs = self.dictionary(tokens)  # batch_size x N x emb_size
        return token_embs


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_dec_blocks', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=64)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--num_slots', type=int, default=2)
    parser.add_argument('--slot_size', type=int, default=192)
    parser.add_argument('--num_slot_heads', type=int, default=4)
    parser.add_argument('--mlp_hidden_size', type=int, default=192)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--pos_channels', type=int, default=4)

    args = parser.parse_args()

    model = SLATE(args)
    model = model.cuda()
    images = torch.randn(50,3,64,64).cuda()

    recon, mse, ce, attns = model(images, 1.0)
    print(recon.shape, attns.shape)
    recon = model.generate(images)
    print(recon.shape)
    print('unit test passed')

if __name__ == '__main__':
    main()

    







