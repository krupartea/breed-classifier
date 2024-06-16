import telebot
import pickle
from pathlib import Path
from os import environ as env
from PIL import Image
import io
import numpy as np
from models import Resnet
import torch
from torchvision import transforms


# load model
model = Resnet()
model.load_state_dict(torch.load(env["MODEL_PATH"]))
model.eval()
transform = transforms.Compose(
    [
        transforms.Resize((240,)),
        transforms.CenterCrop((240, 240)),  # TODO probably its better to resize
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]
)


# set up the working directory
DATA_DIR = Path(env["DATA_DIR"])  # data is logged there
MESSAGES_DIR = DATA_DIR / r"messages"
IMAGES_DIR = DATA_DIR / r"images"

# create the `data` folder if it isn't there already
if DATA_DIR.exists():
    print("`data` folder already exists")
else:
    DATA_DIR.mkdir()
    print("`data` folder created")

# same for `messages`...
if MESSAGES_DIR.exists():
    print("`messages` folder already exists")
else:
    MESSAGES_DIR.mkdir()
    print("`messages` folder created")

# same for `images`...
if IMAGES_DIR.exists():
    print("`images` folder already exists")
else:
    IMAGES_DIR.mkdir()
    print("`images` folder created")


# load class names of the dog breeds
breeds = np.loadtxt(env["CLASS_NAMES_PATH"], dtype=str, delimiter=",")

# enable middleware handlers
telebot.apihelper.ENABLE_MIDDLEWARE = True

# create bot instance
# token must be defined in a .env file like `TOKEN=<TOKEN>`
bot = telebot.TeleBot(env["TOKEN"], parse_mode=None)  


@bot.middleware_handler(update_types=['message'])
def pickle_message(bot_instance, message):
    # create a user's sub-directory if it doesn't exist (new user case)
    user_dir = MESSAGES_DIR / str(message.from_user.id)
    if not user_dir.exists():
        user_dir.mkdir()
    # store the message to the users sub-directory
    path = user_dir / f"{message.id}.pickle"
    with open(path, "wb") as f:
        pickle.dump(message, f)


# will be used in the `process_image` function
def save_photo(message, img):
    # create a user's sub-directory if it doesn't exist (new user case)
    user_dir = IMAGES_DIR / str(message.from_user.id)
    if not user_dir.exists():
        user_dir.mkdir()
    # store the image to the users sub-directory
    path = user_dir / f"{message.photo[-1].file_unique_id}.jpg"
    img.save(path)


@bot.message_handler(commands=["start"])
def welcome(message):
    bot.send_message(message.chat.id,
                     r"Welcome to the What-Dog-Bot! Send me an image of a dog and I'll do my best to guess its breed"
                     )


@bot.message_handler(content_types=["photo"])
def process_image(message):
    # TODO: check if the file is valid (e.g. if the maximum sendable filesize
    # is not exceeded)
    
    # TODO: let the user know how long will the processing take

    # NOTE: the first and only one image is processed.
    
    # NOTE: research on `media_group_id` to find out how to process multiple files
    # if a group is sent, only the first photo is accessed by `message.photo`

    # NOTE: `message.photo` is a list of the same photo on different compression levels
    # the last photo in the list is the original image
    file_info = bot.get_file(message.photo[-1].file_id)
    image_bytes = bot.download_file(file_info.file_path)
    img = Image.open(io.BytesIO(image_bytes))

    # collect data
    save_photo(message, img)

    # inference model
    # pre-process the image
    img = transform(img)
    img = img.unsqueeze(0)  # emualte batch dim
    with torch.no_grad():
        pred = model(img)
    breed = breeds[torch.argmax(pred).item()]
    
    bot.send_message(message.chat.id, breed)


bot.infinity_polling()
