from torchvision.datasets import ImageFolder
from torchvision import transforms
from models import *
import utils


# load parameters
PATH_PARAMS_JSON = r"params.json"
params = utils.load_json(PATH_PARAMS_JSON)


transform = transforms.Compose(
    [
        transforms.Resize((240,)),
        transforms.RandomCrop((240, 240)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # TODO: find better values
    ]
)


dataset = ImageFolder(r"data/split/train", transform)
model = Toy(
    params["num_classes"],
)

pass