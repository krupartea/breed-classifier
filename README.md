# What Dog Bot
It is an AI-based Telegram-messanger bot which is able to classify dog breeds.

![whatDogBot_demo](https://github.com/krupartea/breed-classifier/assets/48570933/302b8bbe-3166-4b48-87a0-0a36a88dfc0c)

## How can I use it?
You may find a working instance of the bot by this [link](https://t.me/find_dog_breed_bot).

If the bot doesn't respond, then I have, probably, de-hosted it. But you can try cloning the repo and hosting one yourself. You'll need to install PyTorch and pyTelegramBotAPI, and, probably, some other dependencies.

## How it works?
The backend of this bot is a pre-trained ResNet152. It was fine-tuned to classify 120 dog breeds. Full list of breeds that can be correctly classified can be found in `beautiful_class_names.txt`.

The ResNet152 definition can be found in `models.py` along with my hand-made model "BreedClassifier" which reaches 30% of validation accuracy (ResNet152 reaches 90%).

Training / fine-tuning code can be found in `train.py`.

The bot's functionality is implemented in `bot.py`.

To make the repo work, you have to have a `.env` file in the working directory, which must look like this:

```
TOKEN="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
MODEL_PATH = "runs\Resnet\saved_weights\best_model.pt"
DATA_DIR = "data"
CLASS_NAMES_PATH = "beautiful_class_names.txt"
```
