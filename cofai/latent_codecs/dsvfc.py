"""AI M2482 DeepStack-aware Visual Feature Codec for Qwen3-VL slot 9."""

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
)
from compressai.layers import CheckerboardMaskedConv2d, sequential_channel_ramp
from compressai.models.base import CompressionModel
from compressai.models.utils import conv, deconv
from compressai.registry import register_model
from torch import nn

from cofai.engine.registry import register
from cofai.latent_codecs.vit_feature_codec import ChannelGroupsLatentCodecContiguous
from cofai.layers.vit import Block
from cofai.utils.tensor_ops import border_pad


@register_model("DSVFC")
@register("DSVFC")
class DSVFC(CompressionModel):
    """Single-rate codec for one Qwen3-VL block-order token grid.

    The standard CTC interface takes ``h=(1,H*W,C)`` and ``token_res=(H,W)``.
    Each RD point uses its own checkpoint; ``qp`` does not select a rate.
    Transform and entropy-model parameter names match the author's checkpoints.
    """

    def __init__(
        self,
        h_dim=1152,
        y_dim=256,
        z_dim=192,
        groups=16,
        spatial_merge_size=2,
    ):
        super().__init__()
        if isinstance(groups, list):
            self.groups = groups
        elif isinstance(groups, int):
            self.groups = [groups] * (y_dim // groups)
        assert sum(self.groups) == y_dim, "groups must sum to y_dim"

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.spatial_merge_size = spatial_merge_size

        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(h_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(y_dim, y_dim, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(y_dim, y_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(y_dim, h_dim, kernel_size=3, stride=1),
        )

        h_a = nn.Sequential(
            conv(y_dim, z_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim, z_dim * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim * 3 // 2, z_dim * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": nn.Sequential(
                conv(sum(self.groups[:k]), z_dim, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(z_dim, z_dim, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(z_dim, self.groups[k] * 2, kernel_size=5, stride=1),
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + z_dim * 2,
                self.groups[k] * 2,
                min_ch=z_dim * 2,
                num_layers=3,
                interp="linear",
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # Channel groups with space-channel context model (SCCTX):
        self.y_lc = ChannelGroupsLatentCodecContiguous(
            groups=self.groups,
            channel_context=channel_context,
            latent_codec=scctx_latent_codec,
        )
        self.hyper_lc = HyperLatentCodec(
            entropy_bottleneck=EntropyBottleneck(z_dim),
            h_a=h_a,
            h_s=h_s,
            quantizer="ste",
        )

    def _seq_to_grid(self, seq, t, h, w):
        """Convert one image's flat token sequence to a 2D feature map.

        The Qwen3-VL sequence order is ``(t, h//m, w//m, m, m)`` (spatial-merge
        block order). This reverses the permutation to a row-major
        ``(t, C, h, w)`` map suitable for the convolutional transforms.

        Args:
            seq (torch.Tensor): Token features of shape ``(t * h * w, C)``.
            t (int): Number of temporal frames.
            h (int): Patch-grid height.
            w (int): Patch-grid width.

        Returns:
            torch.Tensor: Feature map of shape ``(t, C, h, w)``.
        """
        m = self.spatial_merge_size
        c = seq.shape[-1]
        x = seq.view(t, h // m, w // m, m, m, c)
        # (t, block_row, block_col, intra_row, intra_col, C)
        #  -> (t, C, block_row, intra_row, block_col, intra_col)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        return x.view(t, c, h, w)

    def _grid_to_seq(self, x):
        """Convert a 2D feature map back to Qwen3-VL block-order token sequence.

        Inverse of :meth:`_seq_to_grid`.

        Args:
            x (torch.Tensor): Feature map of shape ``(t, C, h, w)``.

        Returns:
            torch.Tensor: Token features of shape ``(t * h * w, C)``.
        """
        m = self.spatial_merge_size
        t, c, h, w = x.shape
        x = x.view(t, c, h // m, m, w // m, m)
        # (t, C, block_row, intra_row, block_col, intra_col)
        #  -> (t, block_row, block_col, intra_row, intra_col, C)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        return x.view(t * h * w, c)

    @staticmethod
    def _token_grid(h, token_res):
        height, width = map(int, token_res)
        if h.ndim != 3 or h.shape[0] != 1 or h.shape[1] != height * width:
            raise ValueError("DS-VFC expects h=(1,H*W,C) and token_res=(H,W).")
        return height, width

    def forward(self, h, token_res, qp=0, **kwargs):
        """Estimate rate and reconstruct a single image's slot features."""
        height, width = self._token_grid(h, token_res)
        original_dtype = h.dtype
        seq = h[0].to(self.f_a[0].weight.dtype)
        seq = self.pre_vit_blocks(seq.unsqueeze(0)).squeeze(0)
        x = self._seq_to_grid(seq, 1, height, width)
        y = border_pad(self.f_a(x), 4)
        hyper = self.hyper_lc(y)
        coded = self.y_lc(y, hyper["params"])
        x_hat = self.f_s(coded["y_hat"])[:, :, :height, :width]
        h_hat = self.post_vit_blocks(self._grid_to_seq(x_hat).unsqueeze(0))
        return {
            "h_hat": h_hat.to(original_dtype),
            "likelihoods": {
                "y": coded["likelihoods"]["y"].reshape(-1),
                "z": hyper["likelihoods"]["z"].reshape(-1),
            },
        }

    def compress(self, h, token_res, qp=0, **kwargs):
        """Return standard per-unit streams without an extra image dimension."""
        height, width = self._token_grid(h, token_res)
        seq = h[0].to(self.f_a[0].weight.dtype)
        seq = self.pre_vit_blocks(seq.unsqueeze(0)).squeeze(0)
        x = self._seq_to_grid(seq, 1, height, width)
        y = border_pad(self.f_a(x), 4)
        hyper = self.hyper_lc.compress(y)
        coded = self.y_lc.compress(y, hyper["params"])
        return {
            "strings": {"y": coded["strings"], "z": hyper["strings"]},
            "pstate": {
                "y_shape": coded["shape"],
                "z_shape": hyper["shape"],
                "token_res": (height, width),
            },
        }

    def decompress(self, strings, pstate, **kwargs):
        """Decode one grid and crop its latent padding before token synthesis."""
        height, width = pstate["token_res"]
        hyper = self.hyper_lc.decompress(strings["z"], pstate["z_shape"])
        coded = self.y_lc.decompress(strings["y"], pstate["y_shape"], hyper["params"])
        y_hat = coded["y_hat"].to(self.f_a[0].weight.dtype)
        x_hat = self.f_s(y_hat)[:, :, :height, :width]
        h_hat = self.post_vit_blocks(self._grid_to_seq(x_hat).unsqueeze(0))
        return {"h_hat": h_hat}
