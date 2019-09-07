import torch.utils.model_zoo as model_zoo

# Optional list of dependencies required by the package
dependencies = ['torch']


def style_gan(pretrained=False, *args, **kwargs):
    """
    TODO: Document this
    """
    from models import style_gan
    if 'config' not in kwargs or kwargs['config'] is None:
        kwargs['config'] = {}

    model = style_gan.get_style_gan()

    checkpoint = 'https://github.com/ndahlquist/lernapparat/releases/download/0.0.0/karras2019stylegan-ffhq-1024x1024' \
                 '.for_g_all-b418e00b3f4f99b879c3e805b8c55e.pt'
    if pretrained:
        state_dict = model_zoo.load_url(checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
    return model
