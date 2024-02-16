import torch
import numpy as np

from activeINR import geometry


def scale_input(tensor, transform=None, scale=None):
    if transform is not None:
        t_shape = tensor.shape
        tensor = geometry.transform.transform_3D_grid(
            tensor.view(-1, 3), transform=transform)
        tensor = tensor.view(t_shape)

    if scale is not None:
        tensor = tensor * scale

    return tensor

class FFPositionalEncoding(torch.nn.Module):
    def __init__(self, num_encoding_functions=256, include_input=True, log_sampling=True, normalize=False,
                 input_dim=3, gaussian_pe=True, gaussian_variance=25):#6
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None
        self.embedding_size = 259#186 # 259 #39 for nerfEncoding

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = torch.nn.Parameter(gaussian_variance * torch.randn(num_encoding_functions//2, input_dim),
                                                 requires_grad=False)

        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1/self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.

        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).

        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx]*func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)

class PostionalEncoding(torch.nn.Module):
    def __init__(
        self,
        min_deg=0,
        max_deg=6,
        scale=0.1,
        transform=None,
    ):
        super(PostionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.n_freqs = max_deg - min_deg + 1
        self.scale = scale
        self.transform = transform
        self.gauss_embed = None

        self.dirs = torch.tensor([
            0.8506508, 0, 0.5257311,
            0.809017, 0.5, 0.309017,
            0.5257311, 0.8506508, 0,
            1, 0, 0,
            0.809017, 0.5, -0.309017,
            0.8506508, 0, -0.5257311,
            0.309017, 0.809017, -0.5,
            0, 0.5257311, -0.8506508,
            0.5, 0.309017, -0.809017,
            0, 1, 0,
            -0.5257311, 0.8506508, 0,
            -0.309017, 0.809017, -0.5,
            0, 0.5257311, 0.8506508,
            -0.309017, 0.809017, 0.5,
            0.309017, 0.809017, 0.5,
            0.5, 0.309017, 0.809017,
            0.5, -0.309017, 0.809017,
            0, 0, 1,
            -0.5, 0.309017, 0.809017,
            -0.809017, 0.5, 0.309017,
            -0.809017, 0.5, -0.309017
        ]).reshape(-1, 3).T

        frequency_bands = 2.0 ** np.linspace(
            self.min_deg, self.max_deg, self.n_freqs)
        self.embedding_size = 2 * self.dirs.shape[1] * self.n_freqs + 3

        print(
            "Icosahedron embedding with periods:",
            (2 * np.pi) / (frequency_bands * self.scale),
            " -- embedding size:", self.embedding_size
        )

    def vis_embedding(self):
        x = torch.linspace(0, 5, 640)
        embd = x * self.scale
        if self.gauss_embed is not None:
            frequency_bands = torch.norm(self.B_layer.weight, dim=1)
            frequency_bands = torch.sort(frequency_bands)[0]
        else:
            frequency_bands = 2.0 ** torch.linspace(
                self.min_deg, self.max_deg, self.n_freqs)

        embd = embd[..., None] * frequency_bands
        embd = torch.sin(embd)

        import matplotlib.pylab as plt
        plt.imshow(embd.T, cmap='hot', interpolation='nearest',
                   aspect='auto', extent=[0, 5, 0, embd.shape[1]])
        plt.colorbar()
        plt.xlabel("x values")
        plt.ylabel("embedings")
        plt.show()

    def forward(self, tensor):
        frequency_bands = 2.0 ** torch.linspace(
            self.min_deg, self.max_deg, self.n_freqs,
            dtype=tensor.dtype, device=tensor.device)

        tensor = scale_input(
            tensor, transform=self.transform, scale=self.scale)
        #print(tensor)

        proj = torch.matmul(tensor, self.dirs.to(tensor.device))
        xb = torch.reshape(
            proj[..., None] * frequency_bands,
            list(proj.shape[:-1]) + [-1]
        )
        embedding = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
        embedding = torch.cat([tensor] + [embedding], dim=-1)

        return embedding
