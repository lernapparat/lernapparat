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

    checkpoint = 'https://github.com/lernapparat/lernapparat/releases/download/v2019-02-01/karras2019stylegan-ffhq' \
                 '-1024x1024.for_g_all.pt '
    if pretrained:
        state_dict = model_zoo.load_url(checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
    return model
