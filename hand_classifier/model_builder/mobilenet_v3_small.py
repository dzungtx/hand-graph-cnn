import tensorflow as tf

from .layers import ConvNormAct, Bneck, LastStage
from .utils import _make_divisible, LayerNamespaceWrapper


class MobileNetV3(tf.keras.Model):
    def __init__(
            self,
            num_classes: int = 1001,
            width_multiplier: float = 1.0,
            name: str = "MobileNetV3_Small",
            divisible_by: int = 8,
            l2_reg: float = 1e-5,
    ):
        super().__init__(name=name)

        # First layer
        self.first_layer = ConvNormAct(
            16,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer="bn",
            act_layer="hswish",
            use_bias=False,
            l2_reg=l2_reg,
            name="FirstLayer",
        )

        # Bottleneck layers
        self.bneck_settings = [
            # k   exp   out  SE      NL         s
            [3,  16,   16,  True,   "relu",    2],
            [3,  72,   24,  False,  "relu",    2],
            [3,  88,   24,  False,  "relu",    1],
            [5,  96,   40,  True,   "hswish",  2],
            [5,  240,  40,  True,   "hswish",  1],
            [5,  240,  40,  True,   "hswish",  1],
            [5,  120,  48,  True,   "hswish",  1],
            [5,  144,  48,  True,   "hswish",  1],
            [5,  288,  96,  True,   "hswish",  2],
            [5,  576,  96,  True,   "hswish",  1],
            [5,  576,  96,  True,   "hswish",  1],
        ]

        self.bneck = tf.keras.Sequential(name="Bneck")
        for idx, (k, exp, out, SE, NL, s) in enumerate(self.bneck_settings):
            out_channels = _make_divisible(
                out * width_multiplier, divisible_by)
            exp_channels = _make_divisible(
                exp * width_multiplier, divisible_by)

            self.bneck.add(
                LayerNamespaceWrapper(
                    Bneck(
                        out_channels=out_channels,
                        exp_channels=exp_channels,
                        kernel_size=k,
                        stride=s,
                        use_se=SE,
                        act_layer=NL,
                    ),
                    name=f"Bneck{idx}")
            )

        # Last stage
        penultimate_channels = _make_divisible(
            576 * width_multiplier, divisible_by)
        last_channels = _make_divisible(1_280 * width_multiplier, divisible_by)

        self.last_stage = LastStage(
            penultimate_channels,
            last_channels,
            num_classes,
            l2_reg=l2_reg,
        )

    def call(self, input):
        x = self.first_layer(input)
        x = self.bneck(x)
        x = self.last_stage(x)
        return x
