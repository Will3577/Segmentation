

from Code.utils.metricfunctions import dice_coef,f1
from Code.utils.lossfunctions import *

import os
import tensorflow as tf
from torch.utils.data import DataLoader, random_split
# import segmentation_models_pytorch as smp

from skimage.util.shape import view_as_windows
import json
import cv2
import torch
import torch.nn as nn
from Code.utils.data_loading import *
import glob
from unet import UNet
from Code.utils.dice_score import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with open('./config.json') as config_file:
    config = json.load(config_file)

im_width = config['im_width']
im_height = config['im_height']
patch_width = config['patch_width']
patch_height = config['patch_height']
Epochs = config['Epochs']

batch_size = config['Batch']
epochs = config['Epochs']

TRAIN_PATH_IMAGES = config['TRAIN_PATH_IMAGES']
TRAIN_PATH_GT = config['TRAIN_PATH_GT']
TEST_PATH_IMAGES = config['TEST_PATH_IMAGES']
TEST_PATH_GT = config['TEST_PATH_GT']


ids_train_x = glob.glob(TRAIN_PATH_IMAGES)
ids_train_y = glob.glob(TRAIN_PATH_GT)
print("No. of training images = ", len(ids_train_x))
ids_test_x = glob.glob(TEST_PATH_IMAGES)
ids_test_y = glob.glob(TEST_PATH_GT)
print("No. of testing images = ", len(ids_test_x))

#X_train = np.zeros((len(ids_train_x), im_height, im_width, 3), dtype=np.float32)
#y_train = np.zeros((len(ids_train_y), im_height, im_width, 1), dtype=np.float32)

#X_test = np.zeros((len(ids_test_x), im_height, im_width, 3), dtype=np.float32)
#y_test = np.zeros((len(ids_test_y), im_height, im_width, 1), dtype=np.float32)

X_train = []
y_train = []
X_test = []
y_test = []

print("Loading Training Data")
count =0 
for x in (ids_train_x):
    base=os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(config['TRAIN_PATH_GT']+fn+'*')[0]
    x_img = img_to_array(load_img(x, color_mode='rgb', target_size=[im_width,im_height]))
    x_img = x_img/255.0
    # Load masks
    mask = img_to_array(load_img(y, color_mode='grayscale', target_size=[im_width,im_height]))
    mask = mask/255.0
    #X_train[count] = x_img/255.0
    #y_train[count] = mask/255.0
    new_imgs = view_as_windows(x_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))
    #print("Number of Patches")
    #print(new_imgs.shape)
    for patch in new_imgs:
        X_train.append(patch)
    new_masks = view_as_windows(mask, (patch_width, patch_height, 1), (patch_width//2, patch_height//2, 1))
    for patch in new_masks:
        y_train.append(patch)
    count = count+1



print("Loading Testing Data")
count =0 
for x in (ids_test_x):
    base=os.path.basename(x)
    fn = os.path.splitext(base)[0]
    y = glob.glob(config['TEST_PATH_GT']+fn+'*')[0]
    x_img = img_to_array(load_img(x, color_mode='rgb', target_size=[im_width,im_height]))
    x_img = x_img/255.0
    # Load masks
    mask = img_to_array(load_img(y, color_mode='grayscale', target_size=[im_width,im_height]))
    mask = mask/255.0
    #X_test[count] = x_img/255.0
    #y_test[count] = mask/255.0
    new_imgs = view_as_windows(x_img, (patch_width, patch_height, 3), (patch_width//2, patch_height//2, 3))
    for patch in new_imgs:
        X_test.append(patch)
    new_masks = view_as_windows(mask, (patch_width, patch_height, 1), (patch_width//2, patch_height//2, 1))
    for patch in new_masks:
        y_test.append(patch)
    count = count+1


#print(len(X_train),len(y_train))
#print(len(X_test),len(y_test))
X_train = np.array(X_train) 
y_train = np.array(y_train) 
X_test = np.array(X_test) 
y_test = np.array(y_test)

X_train = X_train.reshape(-1,patch_height,patch_width,3)
# y_train = y_train.reshape(-1,patch_height,patch_width,1)
y_train = y_train.reshape(-1,patch_height,patch_width)

X_test = X_test.reshape(-1,patch_height,patch_width,3)
y_test = y_test.reshape(-1,patch_height,patch_width,1)

X_train = np.transpose(X_train, (0, 3, 1, 2))
# y_train = np.transpose(y_train, (0, 3, 1, 2))
X_test = np.transpose(X_test, (0, 3, 1, 2))
y_test = np.transpose(y_test, (0, 3, 1, 2))

# 1. Create dataset
img_scale = [im_width,im_height]
# dataset = BasicDataset(TRAIN_PATH_IMAGES, TRAIN_PATH_GT, img_scale)
dataset = MyDataset(X_train,y_train)

# 2. Split into train / validation partitions
n_val = 256 #int(len(dataset) * val_percent)
n_train = len(dataset) - n_val
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

# 3. Create data loaders
loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)


# net = smp.Unet(
#         encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
#         encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
#         in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
#         classes=2,                      # model output channels (number of classes in your dataset)
#     )
net = UNet(n_channels=3, n_classes=1, bilinear=True)

if torch.cuda.is_available():
    net.cuda()

from tqdm import tqdm
from torch import optim
import torch.nn.functional as F
from evaluate import evaluate

save_checkpoint = True
dir_checkpoint = '/content/Segmentation/Results/weights/UNET'
try:
    os.makedirs(dir_checkpoint)
except OSError:
    print ("Creation of the directory %s failed" % dir_checkpoint)
else:
    print ("Successfully created the directory %s" % dir_checkpoint)

amp = False
learning_rate = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
criterion = nn.CrossEntropyLoss()
global_step = 0

# 5. Begin training
for epoch in range(epochs):
    net.train()
    epoch_loss = 0
    with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
            images = batch['image']
            true_masks = batch['mask']

            assert images.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)

            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images)
                loss = criterion(masks_pred, true_masks) \
                        + dice_loss(F.softmax(masks_pred, dim=1).float(),
                                    F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),
                                    multiclass=False)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            pbar.update(images.shape[0])
            global_step += 1
            epoch_loss += loss.item()
            # experiment.log({
            #     'train loss': loss.item(),
            #     'step': global_step,
            #     'epoch': epoch
            # })
            pbar.set_postfix(**{'loss (batch)': loss.item()})

            # Evaluation round
            if global_step % (n_train // (10 * batch_size)) == 0:
                histograms = {}
                # for tag, value in net.named_parameters():
                    # tag = tag.replace('/', '.')
                    # histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                    # histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                val_score = evaluate(net, val_loader, device)
                scheduler.step(val_score)

                # logging.info('Validation Dice score: {}'.format(val_score))
                # experiment.log({
                #     'learning rate': optimizer.param_groups[0]['lr'],
                #     'validation Dice': val_score,
                #     'images': wandb.Image(images[0].cpu()),
                #     'masks': {
                #         'true': wandb.Image(true_masks[0].float().cpu()),
                #         'pred': wandb.Image(torch.softmax(masks_pred, dim=1)[0].float().cpu()),
                #     },
                #     'step': global_step,
                #     'epoch': epoch,
                #     **histograms
                # })

if save_checkpoint:
    Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), str(dir_checkpoint+'/' + 'checkpoint_epoch{}.pth'.format(epoch + 1)))
        # logging.info(f'Checkpoint {epoch + 1} saved!')






