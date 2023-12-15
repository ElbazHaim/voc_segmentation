from PIL.Image import NEAREST
from torchvision.transforms import Compose, ToTensor, Resize, Lambda

image_transforms = Compose(
    [
        ToTensor(),
        Resize((374, 500), antialias=True),
    ]
)

mask_transforms = Compose(
    [
        ToTensor(),
        Resize((374, 500), interpolation=NEAREST, antialias=True),
        Lambda(lambda x: x.squeeze(0)),
    ]
)
