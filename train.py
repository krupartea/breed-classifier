import torch.utils
import torch.utils.data
from torchvision.datasets import ImageFolder
from torchvision import transforms
from models import *
import utils
from tqdm import tqdm
import torch
from pathlib import Path
import sys


# load parameters
PATH_PARAMS_JSON = r"params.json"
params = utils.load_json(PATH_PARAMS_JSON)


# infer subfolders paths
DIR_RUN = Path("runs", params["run_name"])
# change the run name to a unique one if this run name already exists
DIR_RUN = utils.make_unique_dir_name(DIR_RUN)
DIR_LOGS = DIR_RUN / "logs"
DIR_WEIGHTS = DIR_RUN / "saved_weights"
DIR_CODE = DIR_RUN / "code"

# actually setup the run directory: create subfolders
# error if attempt to override an existing run
DIR_RUN.mkdir(exist_ok=False)
DIR_LOGS.mkdir()
DIR_WEIGHTS.mkdir()
DIR_CODE.mkdir()
# copy the current code there
utils.save_code(src_dir=Path("."), dst_dir=DIR_CODE)

# set up logging
PATH_LOG_TRAIN = DIR_LOGS / "log_train.csv"
file_log_train = open(PATH_LOG_TRAIN, "a")
file_log_train.write("epoch,batch,loss,acc\n")
file_log_train.flush()

PATH_LOG_VAL = DIR_LOGS / "log_val.csv"
file_log_val = open(PATH_LOG_VAL, "a")
file_log_val.write("epoch,loss,acc\n")
file_log_val.flush()

# log to both terminal and file
PATH_LOG_STDOUT = DIR_LOGS / "log_stdout.txt"
file_log_stdout = open(PATH_LOG_STDOUT, "a")
sys.stdout = utils.Tee(sys.__stdout__, file_log_stdout)









transform = transforms.Compose(
    [
        transforms.Resize((240,)),
        transforms.CenterCrop((240, 240)),  # TODO: replace with RandomCrop, but not for val
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # TODO: find better values
    ]
)


dataset_train = ImageFolder(r"data/split/train", transform)
dataset_val = ImageFolder(r"data/split/val", transform)
# limit datasets (e.g. for overfitting or debugging)
if params["max_samples"] is not None:
    indices = list(range(params["max_samples"]))  # select first samples
    dataset_train = torch.utils.data.Subset(dataset_train, indices)
    dataset_val = torch.utils.data.Subset(dataset_val, indices)

loader_train = torch.utils.data.DataLoader(
    dataset_train,
    params["batch_size"],
    shuffle=True,
)
loader_val = torch.utils.data.DataLoader(
    dataset_val,
    params["batch_size"],
    shuffle=False,
)


model = Toy(
    params["num_classes"],
).to(params["device"])


optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
criterion = nn.CrossEntropyLoss()


# for epoch in tqdm(
#     range(params["num_epochs"]),
#     "Epoch"
# ):
#     # training loop
#     model.train()
#     for features, target in loader_train:
#         # training step
#         optimizer.zero_grad()
#         features = features.to(params["device"])
#         target = target.to(params["device"])
#         pred = model(features)
#         loss = criterion(pred, target)
#         loss.backward()
#         optimizer.step()
#         print(loss.item())



############### TRAINING ###############################################
min_avg_loss = torch.inf  # to compare for model saving decision
for epoch in tqdm(
    range(params["num_epochs"]),
    desc="Epochs"
):
    # train step
    model.train()
    for batch_idx, (data, target) in tqdm(
        enumerate(loader_train),
        desc="Training batches",
        total=len(loader_train),
        leave=False
    ):
        data = data.to(params["device"])
        target = target.to(params["device"])
        pred = model(data)
        loss = criterion(pred, target)
        acc = (torch.argmax(pred, axis=1) == target).sum() / data.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log
        file_log_train.write(f"{epoch},{batch_idx},{loss.item()},{acc}\n")
        file_log_train.flush()
    
    # val step
    model.eval()
    # init metrics
    cum_loss = 0
    cum_acc = 0
    for batch_idx, (data, target) in tqdm(
        enumerate(loader_val),
        desc="Validation batches",
        total=len(loader_val),
        leave=False
    ):
        data = data.to(params["device"])
        target = target.to(params["device"])
        with torch.no_grad():
            pred = model(data)
    
        cum_loss += criterion(pred, target).item()
        cum_acc += (torch.argmax(pred, axis=1) == target).sum() / data.shape[0]
    avg_loss = cum_loss / len(loader_val)
    avg_acc = cum_acc / len(loader_val)
    
    # log
    file_log_val.write(f"{epoch},{avg_loss},{avg_acc},\n")
    file_log_val.flush()
    
    print("\n___")
    print(f"Epoch: {epoch}")
    print(f"Average validation loss: {avg_loss}")

    # save model
    if avg_loss < min_avg_loss:
        torch.save(model.state_dict(), DIR_WEIGHTS / "best_model.pt")
        print("Best model saved")
        min_avg_loss = avg_loss
    
    print("___\n")
