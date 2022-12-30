import torch
from torch import nn
from mapper import latent_mappers
from mapper.stylegan2.model import Generator


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	return {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}


class StyleCLIPMapper(nn.Module):

	def __init__(self, opts):
		super(StyleCLIPMapper, self).__init__()
		self.opts = opts
		# Define architecture
		self.mapper = self.set_mapper()
		self.decoder = Generator(self.opts.stylegan_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

		self.decoder.cuda()
		with torch.no_grad():
			self.latent_avg = self.decoder.mean_latent(4096).cuda()

	def set_mapper(self):
		if self.opts.mapper_type == 'SingleMapper':
			mapper = latent_mappers.SingleMapper(self.opts)
		elif self.opts.mapper_type == 'LevelsMapper':
			mapper = latent_mappers.LevelsMapper(self.opts)
		else:
			raise Exception(f'{self.opts.mapper_type} is not a valid mapper')
		return mapper

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print(f'Loading from checkpoint: {self.opts.checkpoint_path}')
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.mapper.load_state_dict(get_keys(ckpt, 'mapper'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
		else:
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		codes = x if input_code else self.mapper(x)
		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is None:
					codes[:, i] = 0

				elif alpha is not None:
					codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
				else:
					codes[:, i] = inject_latent[:, i]
		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		return (images, result_latent) if return_latents else images
