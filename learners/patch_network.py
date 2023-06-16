import torch
from utils import *
import sys
sys.path.append('../utils/')
from dVAE import dVAE_patch
from transformer import PositionalEncoding, TransformerDecoder, ContextEncoder
import argparse


class PatchNetwork(nn.Module):

	def __init__(self, args):
		super().__init__()

		self.num_slots = (args.image_size // 8) ** 2
		self.slot_size = args.slot_size
		self.vocab_size = args.vocab_size
		self.d_model = args.d_model
		self.dvae = dVAE_patch(args.vocab_size, args.img_channels)
		self.positional_encoder = PositionalEncoding(1 + (args.image_size //8) ** 2, args.d_model, args.dropout)
		self.segment_encoder = nn.Embedding(2, args.d_model)
		self.dictionary = OneHotDictionary(args.vocab_size + 1, args.d_model)
		self.slot_proj = linear(args.d_model, args.d_model, bias=False)
		self.context_proj = linear(args.d_model, args.d_model, bias=False)
		self.example_encoder = ExampleEncoding(3, args.d_model, args.dropout)
		self.tf_dec = TransformerDecoder(
			args.num_dec_blocks, (args.image_size // 8) ** 2, args.d_model, args.num_heads, args.dropout)
		self.out = linear(args.d_model, args.vocab_size, bias=False)
		self.H_enc = args.image_size // 8
		self.W_enc = args.image_size // 8
		self.prompt_generator = ContextEncoder(
			args.num_dec_blocks, args.d_model, args.num_heads, args.dropout,
		)


	def forward(self, support, query, tau, hard=True):

		B, N, _, C, H, W = support.size()
		all_inp = torch.cat([support, query.unsqueeze(1)], dim=1).reshape(-1, C, H, W)

		# dvae encode
		z_logits = F.log_softmax(self.dvae.encoder(all_inp), dim=1)
		_, _, H_enc, W_enc = z_logits.size()
		z = gumbel_softmax(z_logits, tau, hard, dim=1)

		# dvae recon
		recon = self.dvae.decoder(z)
		mse = ((all_inp - recon) ** 2).sum() / all_inp.shape[0]

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

		#grab relevant slots, targets and inputs
		emb_input = emb_input.reshape(B, N+1, 2, -1, self.d_model) 
		z_transformer_target = z_transformer_target.reshape(B, N+1, 2, -1, self.vocab_size)
		slots_support = emb_input[:,:N,:,1:] 
		slots_inp = emb_input[:,-1,0,1:]
		emb_input_target = emb_input[:,-1,1]
		z_target = z_transformer_target[:,-1,1]
		
		context = self.context_proj(slots_support)
		context_A = context[:, :, 0, :, :] # B, num_examples, num_slots,d_model 
		context_B = context[:, :, 1, :, :] # B, num_examples, num_slots, d_model

		context_A = self.example_encoder(context_A) # B, num_examples, num_slots, d_model
		context_B = self.example_encoder(context_B) # B, num_examples, num_slots, d_model

		context_A = context_A + \
				self.segment_encoder(torch.zeros(B, N, self.num_slots, device=support.device).long())
		context_B = context_B + \
				self.segment_encoder(torch.ones(B, N, self.num_slots, device=support.device).long())

		encoded_inp = self.slot_proj(slots_inp)
		encoded_context = torch.cat((context_A, context_B), 2).reshape(B, -1, self.d_model)
		#generate prompt
		prompt = self.prompt_generator(encoded_inp, encoded_context)
		#generate output
		decoder_output = self.tf_dec(emb_input_target[:,:-1], prompt)
		pred = self.out(decoder_output)
		cross_entropy = -(z_target*torch.log_softmax(pred, dim=-1)).flatten(start_dim=1).sum(-1).mean()

		return (
				recon.clamp(0., 1.),
				cross_entropy,
				mse,
			)

	def generate(self, support, query,  temp=0.7, topk=8, top_p=0.75):

		gen_len = (query.size(-1) // 8) ** 2
		B, N, _, C, H, W = support.size()

		support = support.reshape(-1, C, H, W)
		all_inp  = torch.cat([support, query], dim=0)
		z_logits = F.log_softmax(self.dvae.encoder(all_inp), dim=1)
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
		emb_input = self.positional_encoder(emb_input)[...,1:,:] # remove BOS token

		# emb_input = emb_input.reshape(B, 3,-1, self.d_model)
		# slots_ = slots.reshape(B, -1, 2, self.num_slots, self.slot_size)
		slots_support = emb_input[:-B].reshape(B,-1, 2, self.num_slots, self.slot_size)
		slots_query = emb_input[-B:]

		# context = self.get_context(slots_support.reshape(-1, self.num_slots, self.slot_size))
		# context = context.reshape(B, -1, self.num_slots, self.d_model)
		context = self.context_proj(slots_support)

		context_A = context[:, :, 0, :, :] # B, num_examples, num_slots, slot_size 
		context_B = context[:, :, 1, :, :] # B, num_examples, num_slots, slot_size

		context_A = self.example_encoder(context_A) # B, num_examples, num_slots, d_model
		context_B = self.example_encoder(context_B) # B, num_examples, num_slots, d_model

		context_A = context_A + \
				self.segment_encoder(torch.zeros(B, N, self.num_slots, device=support.device).long())
		context_B = context_B + \
				self.segment_encoder(torch.ones(B, N, self.num_slots, device=support.device).long())

		encoded_context = torch.cat((context_A, context_B), 1).reshape(B, -1, self.d_model)
		encoded_inp = self.slot_proj(slots_query)
		prompt = self.prompt_generator(encoded_inp, encoded_context)
		#generate output
		z_gen = prompt.new_zeros(0)
		z_transformer_input = prompt.new_zeros(B, 1, self.vocab_size + 1)
		z_transformer_input[..., 0] = 1.0
		for t in range(gen_len):
			decoder_output = self.tf_dec(
				self.positional_encoder(self.dictionary(z_transformer_input)),
				prompt
				)
			logits = self.out(decoder_output)[:, -1] / temp
			filtered_logits = top_k_top_p_filtering(logits, top_k=topk, top_p=top_p)
			probs = F.softmax(filtered_logits, dim=-1)
			z_next = torch.multinomial(probs, num_samples=1)
			#convert to one_hot
			z_next = F.one_hot(z_next, self.vocab_size).float()
			z_gen = torch.cat((z_gen, z_next), dim=1)
			z_transformer_input = torch.cat([
				z_transformer_input,
				torch.cat([torch.zeros_like(z_next[:, :, :1]), z_next], dim=-1)
			], dim=1)

		z_gen = z_gen.transpose(1, 2).float().reshape(B, -1, self.H_enc, self.W_enc)
		recon_transformer = self.dvae.decoder(z_gen)

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
	parser.add_argument('--object_size', type=int, default=64)
	parser.add_argument('--mlp_hidden_size', type=int, default=192)
	parser.add_argument('--img_channels', type=int, default=3)
	parser.add_argument('--pos_channels', type=int, default=4)

	args = parser.parse_args()

	model = PatchNetwork(args)
	images = torch.randn(10,4,2,3,64,64)
	recon, ce, mse = model(images[:,:3],images[:,3],1.0)
	print(recon.shape)
	recon = model.generate(images[:,:3],images[:,3,0])
	print(recon.shape)
	print('unit test passed')


if __name__ == '__main__':
	main()

	







