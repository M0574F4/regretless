from monai.networks.nets import UNet


def MyModel(cfg):
    return UNet(
        spatial_dims=1,
        in_channels=2,
        out_channels=2,
        channels=(64, 128, 256, 512, 512),  # You can change as needed
        strides=(2, 2, 2, 2),           # Must match the number of channels-1
        num_res_units=2,
    ).to('cuda')