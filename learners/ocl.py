import torch
import sys
sys.path.append('../utils/')
from utils import *
from SLATE import SLATE_encoder
from transformer import TransformerDecoder, ContextEncoder
import argparse

class OCL(nn.Module):

	def __init__(self, args):
		'''
		In-Context Learning using Object Centric Entities
		'''
		super().__init__()

		self.num_slots = args.num_slots
		self.slot_size = args.slot_size
		self.vocab_size = args.vocab_size
		self.d_model = args.d_model
		self.slate_encoder = SLATE_encoder(args)
		self.slot_proj = linear(args.slot_size, args.d_model, bias=False)
		self.context_proj = linear(args.slot_size, args.d_model, bias=False)
		self.example_encoder = ExampleEncoding(3, args.d_model, args.dropout)
		self.segment_encoder = nn.Embedding(2, args.d_model)
		self.tf_dec = TransformerDecoder(
			args.num_dec_blocks, (args.image_size // 4) ** 2, args.d_model, args.num_heads, args.dropout)
		self.out = linear(args.d_model, args.vocab_size, bias=False)
		self.gen_len = (args.image_size // 4) ** 2
		self.H_enc = args.image_size // 4
		self.W_enc = args.image_size // 4
		self.prompt_generator = ContextEncoder(
			args.num_enc_blocks, args.d_model, args.num_enc_heads, args.dropout,
		)
	
	def forward(self, support, query, tau, hard=False):

		B, N, _, C, H, W = support.size()
		all_inp = torch.cat([support, query.unsqueeze(1)], dim=1).reshape(-1, C, H, W)
		slots, attns_vis, recon, mse, emb_input, z_transformer_target = self.slate_encoder(all_inp, tau, hard)
		slots = slots.reshape(B, N+1, 2, -1, self.slot_size)
		emb_input = emb_input.reshape(B, N+1, 2, -1, self.d_model)
		z_transformer_target = z_transformer_target.reshape(B, N+1, 2, -1, self.vocab_size)
		#grab relevant slots, targets and inputs
		slots_support = slots[:,:N]
		slots_inp = slots[:,-1,0]
		emb_input_target = emb_input[:,-1,1]
		z_target = z_transformer_target[:,-1,1]
		
		# context = self.get_context(slots_support.reshape(-1, self.num_slots, self.slot_size))
		context = self.context_proj(slots_support)
		context_A = context[:, :, 0, :, :] # B, num_examples, num_slots, slot_size 
		context_B = context[:, :, 1, :, :] # B, num_examples, num_slots, slot_size

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
				mse,
				cross_entropy,
				attns_vis,
			)
	


	def generate(self, support, query, temp=0.7, topk=8, top_p=0.75):

		gen_len = (query.size(-1) // 4) ** 2
		B, N, _, C, H, W = support.size()

		support = support.reshape(-1, C, H, W)
		all_inp  = torch.cat([support, query], dim=0)
		slots = self.slate_encoder.generate(all_inp)
		slots_support = slots[:-B].reshape(B,-1, 2, self.num_slots, self.slot_size)
		slots_query = slots[-B:]

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
		z_gen = slots.new_zeros(0)
		z_transformer_input = slots.new_zeros(B, 1, self.vocab_size + 1)
		z_transformer_input[..., 0] = 1.0
		for t in range(gen_len):
			decoder_output = self.tf_dec(
				self.slate_encoder.positional_encoder(self.slate_encoder.dictionary(z_transformer_input)),
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
		recon_transformer = self.slate_encoder.dvae.decoder(z_gen)

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
	parser.add_argument('--num_enc_blocks', type=int, default=4)
	parser.add_argument('--num_enc_heads', type=int, default=4)
	parser.add_argument('--num_slot_heads', type=int, default=1)
	parser.add_argument('--vocab_size', type=int, default=64)
	parser.add_argument('--d_model', type=int, default=192)
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument('--image_size', type=int, default=16)
	parser.add_argument('--num_iterations', type=int, default=3)
	parser.add_argument('--num_slots', type=int, default=2)
	parser.add_argument('--slot_size', type=int, default=192)
	parser.add_argument('--mlp_hidden_size', type=int, default=192)
	parser.add_argument('--img_channels', type=int, default=3)
	parser.add_argument('--pos_channels', type=int, default=4)

	args = parser.parse_args()

	model = OCL(args).cuda()
	images = torch.randn(10,4,2,3,16,16).cuda()
	recon, mse, ce, attns = model(images[:,:3],images[:,3],1.0)
	print(recon.shape,attns.shape)
	recon = model.generate(images[:,:3],images[:,3,0])
	print(recon.shape)
	recon = model.sample(num_samples=5,device='cuda')
	print(recon.shape)
	recon = model.sample_query(images[:,-1,0])
	print(recon.shape)
	recon = model.sample_query_context(images[:,-1,0],images[:,0,1])
	print(recon.shape)
	print('unit test passed')



if __name__ == '__main__':
	main()

	







