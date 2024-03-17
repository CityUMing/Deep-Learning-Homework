# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="8ce61861"
# # Baseline Code for HW2
#
# This is just the baseline code to set up the basic function you need. You need to modify the code yourself to achieve a better result.
#
# About the Dataset
# The dataset used here is national flower, a collection of food images in 9 classes.
#
# The data have been slightly modified by the TA. Please DO NOT access the original fully-labeled training data or testing labels.

# %% [markdown] id="90dcc260"
# ## Import packages you need

# %% id="6844d531"
# Import necessary packages.
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset,Dataset
from torchvision.datasets import DatasetFolder

#If you haven't download the tqdm package, just uncomment the following line.
# #!pip install tqdm
# This is for the progress bar.
from tqdm.auto import tqdm

# %% [markdown] id="6omPBF1AGuLj"
# If you run your code in Colab, you can use the following lines to access the data you put in your google drive.
# If not, just skip this.

# %% colab={"base_uri": "https://localhost:8080/"} id="Ilsx0fnpGlcO" outputId="6e9c78c0-13bb-46e6-ec94-6c5aadadbe28"
# from google.colab import drive
# drive.mount('/content/gdrive')
# os.chdir("/content/gdrive/MyDrive/Colab Notebooks") 

# %% [markdown] id="a92d5876"
# ## Dataset, Data Loader, and Transforms
# Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.
#
# Here, since our data are stored in folders by class labels, we can directly apply torchvision.datasets.DatasetFolder for wrapping data without much effort.
#
# Please refer to PyTorch official website for details about different transforms.

# %% id="dd6625ab"
folder = 'flower'
NUM_CLASSES = 9

# %%
# It is important to do data augmentation in training.
# However, not every augmentation is useful.
# Please think about what kind of augmentation is helpful for flower recognition.
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(distortion_scale=0.5, p=.5),
    transforms.RandomResizedCrop(size=(224,224),antialias=True),
    transforms.RandomRotation(45),
    transforms.ColorJitter(brightness=0.4,contrast=0.1,saturation=0.1,hue=0.1),
    transforms.RandomChoice(
        [transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.SVHN)]
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((299,299)),
    transforms.CenterCrop((224,224)),
 #   transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Batch size for training, validation, and testing.
# A greater batch size usually gives a more stable gradient.
# But the GPU memory is limited, so please adjust it carefully.
batch_size = 300

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = DatasetFolder(folder + "/train/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
valid_set = DatasetFolder(folder + "/val", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)
unlabeled_set = DatasetFolder(folder + "/train/unlabeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
test_set = DatasetFolder(folder + "/test", loader=lambda x: Image.open(x), extensions="jpg", transform=test_tfm)

# Construct data loaders.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)


# %% [markdown] id="26656955"
# ## Model
# The basic model here is simply a stack of convolutional layers followed by some fully-connected layers.
#
# Since there are three channels for a color image (RGB), the input channels of the network must be three. In each convolutional layer, typically the channels of inputs grow, while the height and width shrink (or remain unchanged, according to some hyperparameters like stride and padding).
#
# Before fed into fully-connected layers, the feature map must be flattened into a single one-dimensional vector (for each image). These features are then transformed by the fully-connected layers, and finally, we obtain the "logits" for each class.
#
# WARNING -- You Must Know
# You are free to modify the model architecture here for further improvement. However, if you want to use some well-known architectures such as ResNet50, please make sure NOT to load the pre-trained weights. Using such pre-trained models is considered cheating and therefore you will be punished. Similarly, it is your responsibility to make sure no pre-trained weights are used if you use torch.hub to load any modules.
#
# For example, if you use ResNet-18 as your model:
#
# model = torchvision.models.resnet18(pretrained=False) → This is fine.
#
# model = torchvision.models.resnet18(pretrained=True) → This is NOT allowed.

# %% id="7dabb983"
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # The arguments for commonly used modules:
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)

        # input image size: [3, 128, 128]
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(256, 512, 3, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(512, 1024, 3, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(1024*25, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x = self.cnn_layers(x)

        # The extracted feature map must be flatten before going to fully-connected layers.
        x = x.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        x = self.fc_layers(x)
        return x


# %% [markdown] id="470dfe9a"
# ## Training
# *   You can finish supervised learning by simply running the provided code without any modification.
# *   The function "get_pseudo_labels" is used for semi-supervised learning. It is expected to get better performance if you use unlabeled data for semi-supervised learning. However, you have to implement the function on your own and need to adjust several hyperparameters manually.
#

# %%
class PseudoDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, id):
        return self.x[id][0], self.y[id]
    
def get_pseudo_labels(dataset, model, threshold=0.85):
    # This functions generates pseudo-labels of a dataset using given model.
    # It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
    # You are NOT allowed to use any models trained on external data for pseudo-labeling.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Make sure the model is in eval mode.
    model.eval()
    # Define softmax function.
    softmax = nn.Softmax(dim=-1)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    
    idx = []
    labels = []
    
    for i, batch in enumerate(tqdm(dataloader)):
        img, _ = batch
        with torch.no_grad():
            logits = model(img.to(device))
        probs = softmax(logits)

        for j, x in enumerate(probs):
            if torch.max(x) > threshold:
                idx.append(i * batch_size + j)
                labels.append(int(torch.argmax(x)))

    model.train()
    print ("\nNew data: {:5d}\n".format(len(idx)))
    dataset = PseudoDataset(Subset(dataset, idx), labels)
    return dataset


# %% colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["98a10dcb8d094a6d9a16f40605ca6dfc", "2a86f42c126f4e54acaba54ef3efea60", "d8954902280d4ce29fb7d86129b836b8", "aa771824e1ff49c2bce0f699daa5827e", "0dafd867b7144be2b461a87fb3b80e3a", "74af2eb2b65f4f0181627733ad8d8c01", "b84eb29f13754453a6948aac29e8c2f4", "9386837ddb284bd7b9204e01eb5d5f90", "2ddec30eb56a48e5a232f87aa9eb1de0", "ec43a3d167124ac9916f94ee22adfa75", "c55543acbb044a4a85200b363a7d6770", "4d9a28deb2ae4d6dae26ffdca673619e", "48470e9d807e4e22a1672c2f23ed8af5", "b8940f61b0654aa1ad281014968fcec3", "89b15bd7da354d3fb6d07985886329b8", "0351e330c53943d8a518f29d883b6ad8", "631ac30b18214df197461c3d5840bd76", "c5398fad51864e3cb3fd0d26ecc2d09a", "45cef1c22d5549ceaa7f999fee96a79c", "a2fce2edc5b24c22acd38e7912d7c870", "40c26d636d5a411e882a275d8f3fc228", "9a72f8c24e7c4d8d9358f0a2dfc435d0", "2757ea3a79fe46e6a68e9bc3de7d0e1c", "6e07219734e64e4faa56a5f02a202b57", "5436727041174e719716d64dfe36b2a8", "fecf40c170444f7c9d078be637c18cef", "dd4e4d98b09f49a1b47d46ce09fdb56c", "178ddb93b57448dca20e9719032de035", "c03746b73fc64a589917b8abaef67b00", "86fa1f985b2047bb9a1c810ac1eb2ea8", "e9b3207cd9e4474696485b2e52e9df8d", "40fb966b7cc4426eb3b658d53291dacb", "fd168ffb001f4077ab22af3bef2ec7b9", "8a4ea2d72149462385bee3284e40e19a", "be474085269a4c5da1c9dc8f202d872b", "bd90714d38e1438c80be68b5f097947a", "3a4012fcdd1244d3b68ed02d221b9073", "b6a356f0ff79488b8456a057f4f5477b", "4059580a68c94d4ea53996e54adddad4", "1ac73be6c940464db10b0bacb319b08f", "f0dd545472e54b85b173a9806a5387de", "a934c5afa59748bd9dba0dda6f619d49", "b0668b91acd34c1f8462d7f11fdb9626", "edc55cca3a394e94a1a3d8a2f29d6a27", "17a1856f058840ceb7aa79953d3a04e2", "ffbac9a22309438cbb9ab8a23d3c0689", "980df028e74f4b18be333ef5b1649413", "97a49f31f84742ef943cf1c70b99fef3", "d63318137b904cc8a13d625c86f6526e", "11c3c74127f24cf0971fbc70dd85067e", "239416e5346141dd944c83328ea377f9", "0ff106172f7e49ea82939b96d8c4e484", "1405955752c2482d8ff707f00248b402", "2d551de62ec4499ca5b2cc5382dc01e2", "02570ed1513c46e9a3a7b6b3ebe31241", "b47e98cad5254e799af85980dedfd3d1", "b99dd91d2fe54dfab1887426d1d8c2b4", "2e675391095e4b86af7d933ef1b0a5c5", "01a2887df2f14435909733b941713a44", "495d1ca75bcf465e987fefa3c6feb3c6", "6ff041f8911d42ca992d7b00f374024d", "10c80035f4694745ac4049e86b53e753", "1e6ad19f1d1c493b89862479563c5030", "d990140d137b493aaee38cc7341f90fd", "8eeffdeaef2e49e7935609ee46b57717", "96583cb1d5114473a64ada3ca14fe282", "faf714e2c42b4869919556e3aa015398", "97d2145ed17847c5af0292e0f857dfa8", "a148c64844d94d2b923ea6c7a1fe701c", "b444c3cb918e4f49b271f273519cb6d7", "1cb78a64c3fe41f8ad0b890832225b39", "78977c7536244bfea25c191ad73955ae", "312a8e0929874feaac7d895f4e416786", "704b615935c64e2db2fd4f913073d404", "6a3906135b4f407286e614eb934ffc14", "04f99f052c77478ea3081f0af8a57270", "9e99870be91841119193561608612959", "5f30537edd424d56b36d0a6c43069747", "2896d0a378de475fb0569cdea26e9f69", "2fb11dd650964e0b99e3ec91d2d02ca4", "a544dcf5646e491a879c628a4816d65b", "01e8a23f31004b518f7739e5757a6ddf", "140b24efbeef4e0ebe6ec94621e74ff2", "aad71ebd9cc943ad84ace9c3068240c2", "5a8d14e53a0d4f569c16dbec647c39d4", "188ad8fcd6d24909a0c683dc9f12a2fc", "0de7ee8b3a2d4756a5bf26eb1202c4ce", "296125d9baea4b6cad67dabf374bcdaa", "17466ca2c1ad4c6dbfc00df65fd7ad59", "3bfae81b80aa4012ab5e8543cbcdb193", "8de93816030343bda3abe6021151e683", "a07d38f5493647cca60d20f659acdd63", "4a3784f528a9483c9704e4f4b0c67c12", "4717d419601f453984cb77600ed22a80", "420a358325894b0db84d88484dba1fc4", "beffb314ba134c1ba99f64e9943b6611", "aab7c4b20fba412885f06416a1c2b7e8", "895e5855461e4876a541ad012d0252c6", "813a27daec9d4cfe872e725378cad9c8", "c7cdea45d83c45f4b638145c9af3c134", "3993276d624f4026b2c2154fdc18054d", "b50a9723d5b1462b887c7123537ce7e3", "749f16704ff2431c9f5ef64b035fce13", "89bbfab54b8240408f2225532fcf183f", "4716b7ec41614320925313df1ad7f5b4", "53408c8a8525438d8d54698124a7b376", "3d931c6349b4435ab8d931213c36b7bc", "4b74449d47744488b2c92c2f4fd19604", "79460697051a4e5398cfc8fb05ce0f34", "d97fde650f7343b6a992dace3d477155", "6f7b8188d1de42d18143c9b90282effb", "bca3ed13835a4cd1ace5044226ee0061", "eccb9baa0127454cafa7f96ad3afd038", "a65980ab7446411d8435537aa5cebce4", "cc9ce643589b4111a13e30ca781ff3b0", "fc4c729b7b7e489b834dd2dc05be89c5", "5023202982c54b758971ad615eb10812", "15f02ff495b74f809508eb9e44a33300", "9782f379950743aea4f8e80269323bd2", "e06fd3f9e2d84d1283a2a3a87457f2b1", "10b2e9be44f74a03979c671d7807e066", "da95c58c4a1d410ea5adc62ec4a58e0a", "fe069b2aaad447fe98ef00d3723e5a82", "b0f5940ffc624afbbbdbc9a33711a57c", "9ad5b576aabf42e687acc70139a6b2f0", "5928170fe2024dec85c402a42dec723e", "63e49919532e488ab66a6c12912a486c", "9d5e91230a0547c39bc91f75b1da1130", "47bb219341df481aa18bc902a4693fde", "d6f41a0df72d4c7281e30b1b07572d94", "2eb9150b14ad41db894421b3e01aef3a", "91d3d29abb2844d782d7dc4aa4a4a1dc", "943dab5e699641fa81e7f79de6f25600", "73c579762f924950881b43c614442f08", "45949906f2024877bf094bcb4c4f97ff", "22e929457d5349c5838e92c3091ff12c", "7cbde17c212341b3ba96eed24311cb08", "94b72833510d4c58a836dc0b10604e9d", "ca4128aa6036451495b7b936a40669ae", "cf7de34b5f514b48948ed353eb0f7420", "7cc5dc2b9d6046f591c692669adde086", "a47fd8f93c524b0cbe0a5599676e9236", "e097fff606c746e0a31a21c98a21cb0b", "ff13da22156a4720b16036905ab238e9", "4f327123d6bd4ed9bc574eca84fea73d", "30b7f856dc70437aa5ccbba7e151ede3", "012511793fff49c3a773bb4ce0e00629", "f34269f2bfc04ba5b96a92e787d9cb26", "062ceff86f3140c8927ca5b35f7dc35a", "012be219b85d4966a1718a079a1902c3", "cf4f11fc194d4dfb8bde4629ad19735d", "f84ee239516a421bb3d72c3ecb882e87", "0fd62f0a485a4460a4b7133e806e8e6f", "bb0227d646c848bfb690ea964ae855ac", "2ab507991e46478b84216e9fffbe9034", "e1246f7644504939a82bac66c2be0a93", "1480293e87334131900dd66154911428", "b4a5f78a98f04849b25917525e7c02f5", "ce52c7fd8afa4e90b39ba3d59b86940f", "cd12e52a6bb0406886f35bb833a2dde7", "e3868912a08b440b865e3c3dfba26a61", "ddcd3cf810904489b246bb6b07ceb6bf", "aaa1546b83ff42118bc6a9eca7d263c7", "11be2cc13afb4c4d9405d882015c86ca", "62a1ec15213948239b1febfcfaf90312", "1a3264db668a4a16921f0864b100d30d", "00a5c9de688e457c8dd5d9fe250c3dce", "6773cc6677ce4cf7b98f012004a348cf", "1810060a8c7447f3b9476e4445eba725", "141d0444b94248a08deb704d236fa894", "2d65991415cc40488c855899fa839812", "72ced29efaef480790e79e3a76504e9b", "41fb67ceea304c18a4c22c85d018872c", "47bcad98de8542e6adb6c12d2e793464", "f2bd8011b66d42ee87c77307ca9ce4f4", "377337e4279c4eaaa6a2021bb979bc9b", "e7df9df6155a408a915a807ad899638c", "b0f0b1a12369427c81fa0ff8b8111039", "7a1ad70e05964383a28b08f7ac5603e7", "8e40d2abb28f42e89ecfdfc0a40d9796", "fcdc8a42281c4e65983c819ff60f82ab", "b84a163151b24187acbaad1c58a2a1f8", "2c03a43138994aa993609275c59cd346", "69a4253e69994b63aeb3cd78bbdd8468", "08601ee067e94b2db9f2bf59a23f3ea2", "5d4f828693624be886bbc088bf31de80", "33ddbdec27e443178b60d490f30ada06", "5ecdab18896941a78be7ccfe4969527a", "c921cfa5712841af98f9dbb533d2ccc6", "3175cffcfe8a46b7a7ca0b6bbef89176", "c2d2b42721ea4af49f81e0b83c26954f", "7d3dc093d4594c218e7b01dc3ff2e39e", "24df90d78a854c6bbe7fb01c9e181c9e", "64f2f139a82445419251ff0e59a5104f", "a207bbf342334387b79fb1c5a3e9eeaa", "da1bff3b91744aa2b02a31a2b85cdd4a", "94f8b8f5c20e4c8793fdb28fa3054266", "250b8c9685a34f7a9dc89e1f0a4cedb7", "f644429ecf0f40b68ecc9f3047c99e44", "ae4f3503b10e43d58b66c19d399bd991", "004d16f4322c4b2d896de46086b80bc4", "56de625a01204914bbb4338fb3706bd6", "2c79546feab24cd084f2c245331be193", "85a7ab2fc576446f9e413b03f1febf90", "c40d172f165d42a1ba9f67d542025902", "dbf776fd9afd4952a60882f846533b77", "eee055da41da4f68b9661942ae3204dd", "fc9d6bc1c9cf470988b1d288c58ebf07", "33c9f0a0e6694b01afe18640c5fd64c4", "8e398fb07b844ff0b912ca6ae35e70c0", "959cf6e74e404c7d96dd9beb6f5624c4", "e5833f02ccbd4a07bfa35d1ff20a1be7", "d461be70cf10484d9f3bd488a44df1ae", "3d2a5031f4e2410c99156b020006cce5", "3ba01f9d794648b5b7810df3349476aa", "4f1975c4ba0640debc85e0e129a61e33", "fc85ec6e64a34b82a7a7103d3f419e17", "25259836f45c4049bc3a1bfb9b57d456", "b0343e02ca1a447081bc2c82d372675d", "cc41ef9860f84b45bbcfdd7cc4d4bffd", "cbab3005094141d994be287db8709a24", "c37d641199544836add3d12fe7b92eda", "41fd3a71aed346c88125e627d1d1cc32", "c46f7b82041f40c297b02e5b2065fd32", "e94b906aadd6459abeef4a7c60d1bf71", "9f3ef3e8244d4c65b1b324255868e6f1", "1b2eeb6b28414696aa28951a86ff4d13", "fca6a4303e754bc2bd3ebec74cbaae0a", "ee8d62014f0d44e59b4748826f2cebdb", "27f67a9cff034be6b2f3ff85d5e66813", "e0a036c13aa9458ab346c415106c1709", "d8a23a0d4a084bba92fc53e87be42b5e", "9dc018c64e0e427c9dfe90cd070125af", "bcb37c582ee9433abedde1472ae03271", "ba3fe52296ad4bcdbafd27d37bd86826", "48c691a9090d416482e392ff28cf7891", "03e3e61843eb4837b3753ed342701ba3", "0e640a0284d54849b3dc19bd37e4abfe", "947951afe5674663bcaa63b92cf2a277", "b067bc794d184528b9c4df59867786df", "7705c30f45cb4ee393020c065ab05071", "e19ac5859cb24e0ea670b62203aa2ae3", "7d72d51795b74f3d95d0562bd5afbe36", "570d954d4b414569b33ff627f80d3b59", "516a2e2993284709875edc6438ab9086", "d7e1b7ba97e64db9b90ceb4198e9414f", "dc0e03ee6dd04fcfa1e4c3698436fd2e", "d61d0596608e465a9fe4f5b737af64bf", "c1e2d0669cd14021be26e1235702c9c3", "4923697802924dd6831a1a6a1c918246", "e03c8887ee03479ca6af82ea3b1422e2", "cda0fbee772d471eb44ace24f58a3493", "b3d5ba67ae2e45d6b27326975b37c7fb", "b7bd6e4a9d6f49fc886b59b55af8045d", "034a7c2c13f14eb18d5e7335a249c276", "57fa06d2d06044b5b205f91a8b3744cf", "de75a068c43a4f2d84c159bac6468f3a", "b2f376084c844b72968dd2e2235945fb", "98d1fa0bab6c478ca8200cce6b931cdc", "6f9cbedac23b4268a7dd5651598d1db6", "92378b8373114ba78456c6fd14d8ec2b", "6286ad1d24244d6d87ea03242d5043d7", "45887a4b068b4783960882a3f4d504ac", "a77838b816e74f5bb24f4c6723aeeb3f", "93310b23b3e64625a6606a5aa77d56e5", "a2dda1e4c34e4f39b431b9f394c51722", "5dfadb6237414948b73157118039c510", "389ca7a702fc4913bfc75a7f56b39c87", "d2ee60d1b5854e499fad3762942b53ba", "1733fc8b66de499ba6f2b40fcae5ce61", "4db770cf57f340b799e978d17c0f1154", "876a41234c5b413b84f327aca7e44934", "4b8ef9ce1adc4f11aaa6c05af3391621", "51eaa9ac1532427fa03de8f446b40cac", "e28113f2ec6e4fe19b0ee7e234c18a96", "7948bef30c3a4a3f812e179d9ae82e97", "023de8ca951a4f6d9523f118b2228ffe", "c1453925d1ec41308b4d69c899e1faa7", "e6c4ec521c5545128c58f0915227213c", "4d2b4d2fa1b444059435007129f0aaa1", "d10fc1d41b9e4558a18142bf639ff9b2", "52f7ce47e6894ce89a44a0e920210f44", "e873609215a94df3a2d8bcd2758eb2ad", "acf8e2e5f5624a3bac6f9bfdc6a143e4", "46cb0bfeb63249a3b8cfa2b5d6714dc3", "1b9b8e4752c64ad08b755ad4227dafa6", "7c5753ec0bea4df0a030cfbad1e7bb37", "c0ba402c7c634776b299693cfc086652", "c30577590c144cdc8789454cd710c0ba", "62eeee6c7d6f4e2d85d88f59162d4a6b", "d0d3ae51669740f0992dc8677992270e", "05e64e35aacc46f9a6514aa8fef7e977", "e4b037a667a043ce8f4ee157db3f971a", "cffb2f2daaeb4cdb8edac292254c8394", "69963733aec84ad68583c0cca1784fa5", "ea5410eeee964b2294ac2bc55a857bd7", "9bcb6578bb2146179fbea32dcfb269b1", "b756b62b63c7445985f58640d74ecdf0", "def3c5af99c846e48a9a53b55105ac36", "917fc671f4a8479aa0f9a45f0a277741", "bf11fdf0c25549a290fc143ef70be522", "d63c41ab66ca45d19a9da8901416f2e2", "ee3e45d183db44da96a7d645d445a7a1", "69e6d7b2abc64bf0a6833b190cee6bf8", "30c237e9586b4c90b33da9ffcd9c82e8", "60a9c4fdffcb442bab12aec39e436d70", "1bb7d3089d2a4a15ada4a5cad716c1c2", "9a34723120d541dab026a74bb2d0e6b4", "4b201f256b6d4224a36b6a09a2f9fa15", "34494ab9ddec4fdcbe697fef644e2e4b", "a37f3ea7b5bb4a77a4a4b908d80cc1d1", "4d8f5848b4894cfb845f7c9a3065680e", "67c3983c080e4c4896ecdc398608c105", "bbcbaeeb8b6a4c18aee649d6a238f70e", "1995e25d1ae34cd29706ed0b0d58232e", "c6943c32c16a4bfcb3770246477fc9d0", "44b9b926b31c4aeca5a60d255689912e", "01c772e8e2ff410ebb2169279293e98c", "02597a3343c1429b948309110a5ec862", "d07b345a52a54704a0bb0f1afdd98743", "133c6919dca44a66a4b524a2231046a4", "db284fcbfab84894b2d7f6862fad0ee3", "4fac2770d48047489d5ef2eb3d1d23cc", "993a7b7f2ae244cb84d810ed54415186", "3207b123335b4346ac59bbcb675344af", "c177c85de23e4a7e97dce1afc073e050", "d0f2bfa811f54dbbbddf22f93f3e4b71", "ff58c83f95984331a566b7d6b96af31e", "911adedf21a743f6971d38cc4098378c", "ea80a06d5d874897b82edc4093d43f46", "32a8c21280974a139596913fc0eb042c", "18304b685f2245998412b43cc9cd06e5", "cd83b6ddb1a944cf89ed0632e8e41303", "023f85a71b3e4187baad30bea4b1bdb3", "bf70ab40746045c2bbe016fb0fd12c85", "205413f2cfde4d8bae6b11b94cf0d4d7", "af71f29ad5494a6e962db01ad714704c", "a53ee9aad0654c6392b60621e41c7ba1", "bb33858a06504e1eb8fce5a687e85231", "fa6bb8b3c9034c0a905d2e73ce77b5ec", "51ec4bd625fd441e8dba9eec763b5426", "bdd89ea25b3a4d6d85daa77292505a70", "66c0ad10fe94492898301a126ea2598a", "42c175f53502448fb1acf77bda9adc24", "45314e2f8fe147cfb26c6c6d5c078586", "274566db80354417b59ac32d7666a16b", "712e9bb6b2e54d0d947bc4ca021aef73", "40326031986a433a951f988ea469ecdc", "1e10f6dfe14f4cf1b42d78a4ad1c1594", "c03842008567441496c98a74eb18b04d", "29ea4df8fd034cfe83bf424cf38afb70", "5d3e4b0bfe7c42969cc22bf183e5e9bc", "c4194832be784f77869d8ac192f6e926", "69692a7c33e14fdebf8b728e81a7b74e", "2bbe365c15f94b9da07108099f298886", "ce91a9ddacad45b7874e08cdce4431bd", "03da04fceb9449f3a5878ee428b5fa4f", "155e495106a84f5d9f55d7b47bece960", "e55ee49b72eb4452a105716a6ffebecf", "81ae949b095f4226864bd3d5a631ef67", "197604379fef42e781f2a723b8cd0b72", "741072845c624ebc8736fd981422db10", "21fb8a7f56de4394b2ed63651eec4bb7", "ba635fb0448943ef98bdeb598749ecd6", "dcb7c304aa924fa7bd84ae1d9f2d4de2", "0c6db627e2d94a26af7fd2905cf14d22", "7f5de25241c64927a53e6e93a9ff7b73", "b3303401f53847fcb3e2a350b5f627e7", "e21868ee59194239a82dc9deedd9542a", "f6521bf0d7ae48fbaa586feef69386d5", "11fd0429440d4b56aa9c6e3be1f046b3", "983fc18a63e7453db51895eddd1aeb1c", "42e0fe63c42047e68497c7bc33525150", "a7ebf7ae414b457783015b636458d65c", "95afd361288245009dff2e040387fe98", "83526ec1243342dda0ddd7b1d131fe4a", "ee214f80188e45ae83e57e218c187cf4", "f5b5102c69524179a3a82f6b8152dcd6", "79b575d6f1dd4c918d1b9ccb8f36a313", "8dd535e6132145299227d177c527c7a9", "099c9cbd2605492b9ef0282ea74ac3f2", "5a9ed6521ffa453fbe17d214eeecef34", "6084c9c99cbf4cde90eece64425c5bf8", "c64a364745c7444cb2baf22f9a17899d", "c843e43572374859a909a1c03534e9cc", "e443214d8d104a41a17d600011b1a402", "466c4c79a6de49e6a38e4b28e6581821", "046ca8b5f2bd455ebb3b42bc730c380e", "f01f063913f14197a321bbabadc6f2b0", "ed3576ab6c5a4ee38c14d9f01be40e90", "7ab3176013ba4433b2f5a1274ec437b2", "92e747d802ef4263a2bf6d23b60c3286", "a204d037614142ae81110af0a29e64f3", "88d8e4e14aac46e584f87414f848b768", "bd791c3ab0ec43fcbee35f9bab5d0bb6", "51c089ac26d64c85990694d2f0a126e9", "50f020f9e3694161bc1fad7b10b0ebd6", "172345c7d5544a9ca6edf26fdc25307f", "52bf5ab4f32a422eb8068fc6972206b8", "db19048349424dbdabdee9586183d9c8", "94f90e5727d9432faac3dcbfd449e40b", "0e7844e02dd44563b0e76a44fb6b176d", "26bc96cedc51407faacb40fb1d5a1e81", "e64e83972e9d4460a2666af26e6c7ce9", "4e9fed38240646159677be0202e1408f", "b8d035d2233c4febaf564671c648b7df", "8c5a9f85222340a585f77609aace9348", "be320b3ad3054c41b861f13a3a825224", "51ec03b3025245ab8ff0aa9ec7fe8991", "528561b991554600a019b12f5802a95d", "6a629a66ecce49beb74ac22b73ff3539", "c4199e07672846f29ca4b9433989d209", "a545a7f6ab5d402bac8b4a91480e96b8", "654b54ffeb45420caed80060eab141a1", "816f162d236741f3a68636d8b67b1d23", "0356ecb55c094f77bbfeb0c9c610b508", "f62e6d23c6b945378b2e6718b7acfad2", "6d85583a2e1f43d4a440bb86ef8098e5", "25bf00b17fcb4292bd39757429f72c48", "d5b6af59453a4095b4cef74a4296b7c1", "05942c4d2266451ea53ce51eca440706", "27875984d8b14ba9a843fd12f5d05920", "4e5e69dfcb334ce9b77565cd0bd2ad01", "7f08bc9cb91c4273a9d116a3f6b6372d", "1c6af08113c8405dbac2bc1e25202c79", "e71d61aa87eb4d008487e8ec07cce0a1", "68f6b2940492489d80fe0e18927a582f", "918c589de5fa435e8dd626529e09b28f", "ba40f09fb408400dbbdbccd1b3a1448c", "f5fa4d63556449609e6e6c347b07f2fc", "d349c8c37e9541d4bf79f5259b8b45e8", "e74d5843ecf140d684f071886fb03221", "3f83fcac973c4856abfbaa533f84edc2", "3e587ded13c4488b8cbf13eca25ec5a1", "8912e55c0b574d5e87a73fae7ea268d3", "6e80461965ff48f2a853f1d715270f30", "940892f64ec641cab8befabef4292eaf", "b337e149f4104d69a32bb0dea77c6060", "a9b5c7a87e854c0ca7247fc3329b4542", "cd0de0372bda4f9a9a3d33f72e59e4e2", "f8b7289475724d80a12903919f855942", "0ff93cf6bbae49649aaf447e16e1f122", "8bbe821c6b084a2f8509930bafe6bf1a", "1b05be13b19542fba3bacae61a96acaf", "4d79c7086a0443cdb91722ea3097125d", "b7b18bda45354ca48a9ee5eaecb275aa", "71c4842b218c46f1b547f82002004eab", "805d2aed27a74aa88517e04ebb4fa75c", "6e5903a130904c02924bba21ea920a3e", "8081042b3c8d49c7a4a171112d750cf9", "56d8f4331469402484fa8a3225418332", "892f9c32a7454920be49fdca711ccf2a", "f2790f9fa371475cb28ffe3684555979", "1552e6afff944bb5a54a576ca98de5a1", "02f6c9da9bbc4da595c4afd0b4f2bb78", "8329bc7678d94d2887787e2cd4cbb5b7", "3803d60b35ba41f7b330d8fe0837a138", "78fab5671df74fa28aceb96e3e2814ee", "3b69df88adbb4a77a6c17d8a842c38f1", "8bed40351d4e46ce8e05546bda3f5779", "b92488b9b85a4557acd69275e7acc4d2", "4d22d228661d499f8d90bb20717836f8", "60f2456488644bb2b75ca75d1e667bc6", "07d48265c7514a7486d3fe8293bc7e39", "08567d8dc0254038b6b6dda6d8da52a3", "ca0b07c955b447bcb9e73792f9a12450", "1fe9d3f6f12c417997ca3adab733eed2", "e598d2b683e548a792b033372a566304", "12484740a7bc4dd8bd8028dd0f39bbec", "3eb3733635004970b267f069fff8a0eb", "c56bb55259b6400fa8a5b479d5ea811f", "2a544e9289e74dbf98002a4630191265", "17c7f0bc300a430a865489914eda1218", "a6c7f5d7a33b40dc8380f1ee2fad7b9f", "607bff647ff94a829eeb21b5ce85a376", "cce456b7ad5641b599ce06d8413b7f4b", "0753bb4a510e4e199f019c82fad9cb42", "813ce0646422448d8a05286d83a10b32", "0e7f668e539644669a53ebe2eb67cfba", "acbd499131d243019f26078d34b15575", "ba33495e3cad497d83c78e18df6b9946", "ef63fedbe1594c7a8ea64447aaa5b875", "aa37a031111444c49a7af3e410077495", "706f30c53d36463394746bd1fb8c1d35", "72e1712005ab4cfa87a02fabb50ed96e", "f91f98149975419283310b7fc2c16879", "e91b6e5687e64af6b0271e66a42b84d9", "11e44930247045e69d94c1fea30a1b98", "d16f6e2d28f84bbaa90bb6dfbff50add", "49c3910a9fcf4629b4a001da61dfc668", "defe99834e7e4a2dafcaa57274cbdf46", "7189a0648ef24ddfa9653e6cad57ed63", "4e8387e1c20547f9837d7a79f4b77f79", "b2d17c768a004eb88caa62c5d301ddb0", "2bf94cfb3145406eaaafedfaacf4a3eb", "e3f55ca9c94247cab074b2ac99620967", "225d314e698c485f83f24a181b7bb460", "2ed97cce1b92433ebbba00797005d784", "4e7c557e466e46229f3065b562c06db2", "5767f947906047d48b8d086b67b1bfee", "f0e523a2711c41188e0cc7ec75c85253", "8b6c2dd3917f461da3b8297fd6a13d9a", "611584ca2eae4893bd1f5e02a87c49be", "9fd1a7b3091f4dee86eee8f70f641d24", "9105da96f55a465eb720dc237fc210ab", "1967367a527c437e8ca64ac794064b58", "74acba363a5f4b2caddbe61a6c60115e", "1d66f387b3a540758242fe0cc11b7e71", "683128dd8ba049c2ae4168f4f76e252d", "a004d710684b48bd8aba35bde480d60c", "7358e7b085a64596b6d459f0c5590d54", "78e869951e3e42fc8a5807581908e653", "24d61f6e378346649f2b86c402ab4f38", "037305e5ee4b4ca9b0f966c8bd163d92", "d85f7c17d61947dc94948446e5d1c935", "b84eeb49ca0f46e784ee7e78a3639d1a", "c40e1537dc134e7eacb9a654fcb0bf72", "9468fbfed2f241db9871799fa273fa06", "2128a2bcdd804df8b1a584b11e49d4b2", "b3adee54a59d4cb4bf7c413d230a5579", "3f91d56d01464b00b43b8e8b6cb79de1", "26cc7d0563f7454da58cd507be81bdac", "e0c1c9384ea34a5a81a01aeb5d2d2b10", "f1fa353b626449a287bd746ffb68098d", "68ef69eae05d442b844945c202034a8d", "f5c04781dbd34d8ba5e47cd4c3801591", "82c7fd9db60a4079bf9017f9af82b865", "bdf1ad2c06c54e72b6d00b2530179b19", "d1062ba154a3405995a333d62ab6b2a9", "ffb83c6b823a499282197c328690e187", "3879717e3f0f4fa68c4044fedf3a02ec", "06b6544fe1e64b6887acecdab9213e6b", "f6cdb3f724a64e3599a74f215800e1a4", "e92179c5b6014bf28d9fbafb40a1acf9", "90c0849961364f87843e699513d39be8", "7cbd4180a54a455397b62397ebe4d00a", "c8b5bb43a4854fd5858dcf622ddcecb1", "bce924b4791042b58923da47fdf1fc68", "8d63c009cc194535a8a91a7c6fb18845", "ccd84db5f6184bfd9f655c6989fd389b", "9f7b487c409043eab32f03a1f678f89a", "60ac088c43054ec28f5605398e71c03a", "65dca18b6fbd4d7696e822e514237274", "a2ec5ff124b7474a950f127204fa1336", "0f24f46b650b4620b0fc5f7845828077", "1a1b68a0b2084c158f9bf551266be4a5", "133aa7b3795545058ac22d47a5d2d4c6", "f8043ffd2f0b4b218cc1e2a323b3aafb", "1d23fd46f5ae40df8e8bf31bb36ffca0", "6bdbefa366d745fa95e33d3bd6b939d9", "254a864b52db4c0ebb9ffad92280c05c", "eee8382b6e4a4fdb9e892dfe1257bdd9", "add6f4307baa411e8f3c937069f5d3e9", "bbbea9cf11ef48209cc397ad2cfdc26e", "fe5334c5dae942aab3f134cfd26a0452", "f2aa9c62cf4f481ca70f09422a9f7b11", "a4452ca9ec7d433b90aba92ba98efc35", "00aa2f437d494063bd35ffa75772caae", "f2dac13815b24cc7a3404f41d17c0fab", "3d693c6291a34d819dcfd5f549a61d9c", "aeedbec927c04582b6af84d2377fda32", "65cb32beb69a4f69a94eb1b3cc7fd7e5", "b35c6e485ed24e60bc2f11c85118f831", "515c8bb63256421188e7db8bbd082703", "f9cebf88841b410882a6ab2157a28cc1", "e5905f34bd9740b9881173853690fa7b", "13efcf39943d4b50a6c29a0d777d98dc", "b22580028b4f4a58a9db11bb02584371", "f0f4ea36e6f64055938c4bddc4ed4783", "436eabb288954d5ebf433b9a3dec19f4", "89de008a34d24339bb664ca188932460", "f490be4cfd034bab8fcac1f827322cda", "9c46bed0e2784e33bec848483b998216", "1a1d10313a344125b8a309e92ac2c603", "f69ba3c998304aab81f1c8b4875a8ce5", "60842362aad643c99216d838ed618a19", "fb646b38d85e46f882fdbff55d03b81d", "7f8c7fba5cfb4bd4b085c93e13236e0c", "0a15e9568ae04d8db01c5a3dc295498b", "1421bfe05221483fa9746bfa99cdfdf1", "856159ce9a3245f2ad6d5e9f770c01ed", "9a3d51b917f34c029c945a2074c495d3", "88356d5e494646228aff21fe2f2d79a7", "92c4793db08843628b6d43d39fc80a0f", "46f588985bd74f1c9c9156976befdfb0", "65f14ad936a749f9ab9076fd38f5e6cb", "1a321882a2a34d9dbb4ac8e88d899932", "48b89f8770b44c6d8258e629280e2be4", "640edbec0cf1407a8735d86ad84bea4e", "6ed2c4cb14f04be594a2be3dc5a0ae88", "db4340d52bdb41ebbafd2096a1de9fab", "d26555b6c1f3449bafb90b19bd111852", "bcdb525d8432440a90ff1bcc091d60df", "d0139ef68c664c4b82dfc8657cea0e84", "5e3a6e5a1928494bb66952cd5124b463", "5968971f1c614db5ac0c2fabd9dac029", "2a9240f5531f4fe2adc46d01d89f90af", "27a6e24952a54505a1181b734b7588e1", "b9321652e88c43eea1b9ff587b2850c5", "3396da19ec17419093eae6a8a4cf524d", "7bcdd44f5a274301909d42981ae5de8f", "1ac6f8544ad74325b859c7e83c0a10bb", "1385329a56d645b584570748d74979b8", "e95a33f1e48442c6ac51cb6b4937e29f", "6bcf6bd431934b0ba53673e4fa3773b8", "42bdbf6488d84467a0f061ea373a377f", "2eb2b8664c5b49e6904a964ad03b3c17", "227dec77f90448cd8f363f1f02143148", "657adf9723f24c488289ef0e92b922b7", "43141dbbe6f341ff868fd3f533a2790e", "c44a1c0fab1b446e86341778d910ab64", "84f680a3fa0b46149f65b3087ea83009", "ace1d59b193f498aae67ae218a6ae7cc", "7491e4d718a54cc8807c914337af191e", "d60b0298e04d4077a964ac5b605160dc", "86ac7231c06a4097980016ce4aae21c8", "f705d3b2988c4e2cb9244bdb7f321e16", "d677ba285bc443488b677065278d2dd4", "d9fe0b3af97547d59c2799bc3051b476", "e48dd9bb50e74f12b2794e4316cb1946", "9f15721a3ce1448f9ea94c10a9f10edb", "d2e732388e9e4047aeb987075ec4ccca", "ba7d05a35b4448058283a38cb3af7864", "299deb66961744e3a1167755f76560c7", "85e8b15189ad4d41a1b1f7fbb12ee4ae", "abdd2eca09ad4a02a4d81374ac954ac9", "c68ab033c2f5423182e262664bb726b7", "9933dfce2ef94e0ebb8d023df146dc2c", "44d721cf95494f84b6f2afbcee2e330f", "33238fc6c02c4591ae6110e3ffa89b35", "e30d9bb06ed14de599602b99105b49e6", "c8f9bbab08834c938154b240d1d3e17c", "723219ff41224131af8b529ea6f38c75", "af2e1aa523164236a1d6aab8a82439ed", "3e160cf29db7425197e45249acf8a481", "10d891ed5fa84e76833196e02f689245", "b72436fed0d149b0b9903503423589b3", "1fff73b7b9254db193591b20f47ae306", "83f1911984f14f538618ed57d5d5e25d", "19cd5f49b7794e74be8431d85c0b2aae", "691cee08f23d4e388b279a91cb21ed91", "2c1b59939cf94d5fb912ac75d6f90836", "6f4773c66e0144939e4207b4c675fce5", "85796e5a784646cd888e0c1c9fdcba8a", "73e80b2f9f044389b9407d954cb1ba72", "f74e7d175a0241188dd6cef23b886511", "918788236ceb4a0191b4b92b0bde06c5", "3add1c37f80f473aad70fa361e633b4b", "9f6f86a9027743c087add2ca59a38dd9", "a6a28f354fb54c90896edf01e555fa93", "b4744c1dc81744bc9b0b58dfb463ca6c", "270fadfac8654b23abb9249e2ce17b43", "0b9b459e82bd4985b1dd843146fc4019", "dd7473e3ca394fce8e061b06e9ead39d", "cd67a132de77464ab6de85d5d35794aa", "116897f908b3426b9ac262e9b5f6cb80", "f52fcef260b94af09317b720e079a64b", "e9899cc03aef445db3138e4cf6844d38", "e03c5222a1bb4fc9900eb0d0588ba8be", "d73297653a614d49bc35f43d7cd9ef9d", "2f6e06b023b447f7bdbcd9f1fe5c9c90", "d51ece13d26c4cd4ad8d63cf1a5e702b", "c9909d9ffd794299a8030ca808fa9764", "9045cc26fde54ba7815e7a088f97632b", "c8be62522451408ebc6daf232cfcc479", "169a54eb824e491cb3fc534acbf42132", "8ac95e66fcd142689f883fff2fc73508", "b7ac16568ce848a0a2f54e60e7632bdf", "55f3188c7aaa48b69cfd75668923569d", "3d69aa5118154f0582fb2ddf046712ac", "ee9bcfd9ec0946ee84b079ef3a961063", "692b8208a0ea4559971159d52d2d2abb", "9da394667c884bdd9f27812382d85ca5", "97ddeea4e430453da5fe832dd2583474", "e7f2a8a45dec4abda142ef457e82f6c8", "759f4933e350409ead57fe0205ab890e", "68c2109376c44e46b93e040620b9c3d5", "d7bf70a28acb473bb232dc1a97077faf", "18b7c044ac0042a38d600fabc7880312", "3d56330ed9184b999c69786a8faaa81c", "f90250e301a349c08298406e54066e5f", "53b26436b7be4834b60b193beb268e4b", "463b5ca38f3a46eea21be42cbd24a43f", "a0f8dcaa1e0b44368bad98abb208ed06", "d16c1c355b144723ae0f57700c23a001", "e771b08351354b75883a438a3344c744", "f184dbe9979943b0b1f030e24e01a45e", "3e294bac1d674285ae05b48ea847879e", "5b34c4b8e53844f087184028406e1104", "3bfc4ec1336b40fa87528764e18cb485", "38eadab608eb4ec08c706ca69557ba14", "9af27a62da324038b993301d72586916", "2a9e7d52a3a74bb68999f55dc9f593fc", "9aef2908ab1e4d76ad163802712dab78", "d6ceab58602042aba1522f3802089dac", "5a8d0baae507483c916aac39ffb8c4ad", "bd550f6037bf45fc94e2ec7fabd8ae39", "ffedec58fb0a47108037b4b1fa751e4f", "6c9c88290e7e460a97f3c505b8b29303", "241d626d2f614d09b0eed0f70838e0c8", "20ba4ad6062e49baab077fb2284bc941", "adca9fa1f6d149fba3e982a979b235d0", "8dc6482bd15f4c2e90b25f7112962ba8", "67e96ebf9c5640f89743797ff53b6fb9", "b01bb0f843234624a213ef6dc273516c", "5deff0bc16f441dfb1b770a851299ed2", "a7064141ddf243b792d7f298b0d4ff42", "e48ef8b4e1f14ee4896590831818d240", "a5ded8187c034fcb91eca99407db8c08", "988915d540d94c2988f446def7acfaa4", "d8d3fa12878a4be99d94f9a5a88161c0", "202eabb92d84486ab0644f8d414695ff", "b4952ec471a944c2979bf9bfb56c48d5", "2b4362c2acaf4b1baafeb87d88938170", "017870346cc44ce7bc44ed61233d26f4", "758af6d2b07e4a1aaca45456cf613310", "0a1412ad4c8a45d186b24e14451607e2", "6ab772fef4c24e1882bf5c66a38f1dc2", "e1ef18dd04204c58a49d2225176b9abd", "f0c3df3df6e64b37864614004485e5d4", "bdb9cd312fa24d0e9d9b1b6e83dc3d0a", "7116be87887d47ebb13c9608b6b737f9", "7c8f9d0471f849a4a9b09251a89ab9cc", "1d0d94528f0f4bba933d14bd5b82648b", "00f3e894d9d24de1abbeb14077301a2a", "bedabe719ab144d1b11d61d0bb2ef2cb", "53a5bd347c5d4bc7b9c8a3397749f507", "d73f1d9743df4e7ea2eb2e283df6768e", "64886c07927244ac8718e911b416f5ce", "66f60ac62b2d4ee780491bb16500bb26", "de9763b143b84fbabebf0668eef56668", "00032da3e2624ecb89372de45dc6002b", "3c63af60bf574a54a9efe01afa2f9325", "e165bce72b164835b0a7bef70b81dff6", "c94cd08f84c3467b8d42075162b5440f", "3840db0bd64840d08697b85bf22cce10", "95f014e2fd4b4f96a2602431701634b5", "37013409742248eb8a699586d8381672", "27ef43e324444f78b7f186e5619512fd", "2f289cacdf4e4fe1ad1608ec42429986", "33062ed561444c029545c86068999d5a", "8d739570f20244e3bdb30a4da4e300d9", "09d5dd2ff6ac41ae99bd85d4b67ef1fd", "1de82180b204407588e44198827b5a60", "6b660cc54547429c96fc49934af08d8a", "aebbfef1a712423c91b853c0a81c460c", "c3251f1c2f2b4f9ebc03cc39a2a9a29f", "881a78a43a5d4f43a2dcca588b794be2", "f8ba98c601fd4127be2412e605718cae", "3064b6b578934d47beb3157f12ccd3d6", "f556d4f42d1c48d2835f2cc7628a4c6e", "90b481c53cbf4f71aaea23c5e966bc1b", "850d6050946c48abadc6f6f13fbde08a", "901d951a91b04f0fb3f9ffd04c70bad9", "ab5ed6940c164a4eb50f08cff82e5e80", "181773db6044458190dc6ffa5c80e10e", "549641dd89fd4cadaa309fdf6ac77252", "25e56d97be91406abbf897c9f3e69b2e", "35894a22a5db42969f725d023cecf15b", "ba1e2c274fd14b058af4189c8ce46c0d", "4e611743197c4139badcd8252345a532", "6280c79f0af14ca0bb76c6450ffc10d5", "c8c3511d91ea49b39dd37c6b68b4f6be", "9653cfac8c6d4a2aa80ef1ec774677d9", "7cb38c6aadd147daabb40b924319b2aa", "5cce0ea731db44b08ae0a66e131914b0", "2e71f2ac9d6449bf8b701f45eee52196", "d88584c5a9754a97853a599d27b9b6d8", "617d52729ed04fd5b4b5666dcf6c0199", "4eeb519ad7d64071880725c9bfd5847f", "afcb57f3b96645649d462d239c34c2ef", "a9ce54a2ec894ff386ebca70cd1ae6cd", "5818f13313064d7db785377e7a9661e6", "f1eb6a918fe347f9994247d45a906bbf", "dfc3b48586a441ea92a63964efa960ff", "d21a66ecbea64e1da9c21235f0dca302", "ec70f8b8c46242f8ac8aed9d1bc23469", "846579d515e04873b87ae15c9ddf42b4", "7eb5c0183d9a4f5aa413000b15aec754", "a7853d477d2f4ca0b7f06017663ec8a7", "e18d31697fb74345be1f2937c791ed40", "ea1ebd3e0d40431aa9fb349f80fee1ee", "9ed98db704774b4589dbc33b5966af1a", "8e76e4d9b820475aa9b41c18c2676416", "d595cd58a84a4e84897ecbcb6863be6d", "1dffe66b4e2849e89180fa7c2a7ed67c", "94d794ee8d734b07831f9b69c0403be5", "bbbc7ea9ab56427eb8f06cb3e02b54c2", "df237f76a74a4c86880bde87e2d659be", "9e49d6ddf65143ddbc442c855b28bd6b", "694d5bb92a20440ea7d03b063ea6c5d5", "78f779ecf5034c0ab37ab2b69d1f17f3", "511317a4d92b49de8aea9dc12b8d52b4", "c98f48cbe433432ca8e7383646dc4ad2", "b8363ee6cf5446238d01e78b9d978636", "4f66a1ca20bd485cac6e5be9f6e6649f", "9b75175792364268bae2c8ace83c8fad", "4124ed260bc146b49b10ae396ef8a4cb", "a67718b1450e493cab58418f759b4f18", "ff0481bc349044e8bc0138704db61d63", "893d5d2d141540389039e3beb505f5f3", "14249e36c1e740c0824901092c2f5494", "619db942f25748c28c03d7cd1b61868a", "ba69b83052eb460590f2f3be67b97ced", "b792c86119104ab099bd533645f3bed9", "73df205f9a2449938aeed2e526a32a45", "bf083d8bf5cf4571a31bbec590fcd73f", "b188d73c0c6b467ab3e01775195fe20a", "e8c27d7575424967b2041c0dfbad0042", "1262898e12bb445590fcae74934704a0", "867534d2cfa34dc78a4730f3de422554", "0f42602b30824331a0fde678a32d6319", "e5f2c700347d490e9c8d7080fd254d26", "b62e4ac34caa497b9a527aff0df80dc5", "a054591efacd4bfaad333039568b6339", "d3f2de61c68a48349ad20ca16c3ff032", "485455fc358f4916b2e51489d6ce9d92", "10e36bc8ccb745abb650cfc8cf030c16", "69a6a52ec3334bc683b548a61316150b", "13a4df8cb6cb482092779a1803a1f3c7", "7fdd8643898d42568794e1bcf5c73c28", "c1a9cf3b2be24c359fefec0d3b4a09da", "418d8278ff8e4e2cb6d800d6e33ded04", "4b3d6ba36a174c0eb7f5df08c93074a1", "22853e848c64482f8253c0a3eaa95289", "0543bc76a70247bb933f3220dc69a3fd", "8e72e14a124740d5824c2ceaa46389cd", "888f72c384e942268e5ebb0a950dce99", "4380283b7def4c0fab48d0d7edcd9d53", "2dffcc8a347442b3aaf47303702e9e60", "b0cbd3e4f28043ba8d1ec670ec6d289f", "9dbc1b1628584d9f8130c0c528205119", "257813c33b5b4c8eae4196cf27f25abb", "249a2cceeadc4e5181eba5d19f13fcb3", "26ed1b36779f4cf086d5b329378be63b", "63129a61b24a48e78544918a38b98c3c", "e87b977f9c92429f85812c72e8fea00f", "4d6e694254d94fba9bfe6167bc23657c", "de041cb3c6da4acfbaba4e40dfccf31e", "e1d4b1e652e04036ae466bbd46e6511e", "b30eb1aaeba1418b9511a4210aa3c71b", "38c88995d2e046ceb4132ba4eb3953a4", "d8821d52ff3349a78c0827ed286b9ba6", "46831e770bee4112a23a2f47e567f522", "a8b7bd9f036441a2a0bb2332d8fe82cd", "bf4d9c6c4ef7495cb71912b0d183a2a4", "5aa589e83de1469988fe5ca5638e295e", "f37a8cdbb77843f395f77acc31db107f", "22985bd7cd674e7d8477857242c0edb9", "01c2445f6a3c462c915b5160e8a1ae0e", "d9cf3fbc2e86469f97285084f30ede12", "e14faf16e0684b6e98157323d5fbc90e", "45427d66b93148a1859380b02213442b", "2097d32db8674bc5b560ff1254d721b5", "5c10aeb90f1b43f8a91c4017c8676e7c", "770cf3f8d2ab4fbbaf5655667694b293", "54dfb10fe88146899ab25325a718c3de", "ca016ecfdec5426e9597c3dbf364959c", "04914e0103bc4c7cb6ad94a8f3d031ef", "f265e606c41341a3ba2fb3565c0c7bdf", "6421a255f1f04ce3884b1028397e184b", "88df7e6b2d8b40d68da4043cd417c1ad", "4c210d45b1884cb18ac3e712355f3b3f", "f17a606e6945439597a0a2f46d45c019", "711def32b2a043bc9c05bbb1c58b2b5d", "4173f99496a14665993ece9e97194155", "3739d0aa6bd94636880a2b431dbfb581", "0e118b8b3d174810a47f96554fc9febc", "5cd30ac65c154694823afd91f0629d7e", "b29293c506ab412cadd86862e9a38585", "1e9dcd58d6e24a2c946fc42224fa447a", "a78b4127408348e696d64b292f783e8a", "3bb6e871a02045e48bce8410d31f7cc7", "5e56420684334a49b14f21a888597189", "51bba381971144beafc4e64a5486ad98", "21d88a333722459393c6ffd366df6c99", "8312834b13a74c93abf92da0ab821dbc", "6009adf6f70e422ea354c5a7db92d2ea", "ca17bd9625934dedb95321e344ae4769", "a0c8b12fd88c4a42ac9257a0f0eb3dba", "78b7ff9615314eb48ec39b802e67c2d8", "4f3aafcbb30a41cb8342c48cc3aa4730", "074d24aacf204cf4a4169ec2b340d8e7", "613d71cd3c0547aaacc0a38f14b39899", "53c36de80a2f471e88c0438040f15469", "bfa8e2b736814bdf9a57981470bc7ee2", "b35d3f5c8a20474a9a0c838932c409a7", "a0472783563e478a9c2443d9bfc5da92", "17ef630770e94797bf4b9dc84d1269dd", "dc400527fb8b40f8a8a1048e912ac526", "3e27ed3e04a944e8a5370fc0f333503d", "34232677138a4520be69164312b41584", "3b3dfe0e61264a3398fb1549fad8a93e", "f948983de0d74378af721480906b96b5", "e674fe394f5d4df5ae90da3c8c2e4923", "d8c6e37d95a848719b0b15ac93509939", "2cf463a39b934e78834eb27a0d2bd1a4", "ba68823429134ae49631ffe724ec34d8", "0dcccd5fe0604606a73f758297afb4ee", "a48b6657ea2d44259de863b1841cb33a", "83790eb8b2f0493e82987121de5c1501", "0a55acec5b9d4cf68cee09d310fc3a8c", "8ef268190f494c33bb0b94dd2deefc32", "7e21f1aafc674a46a59df2e7c820ee69", "2ea77c9ea0a24965b0e5b18b9bf78f9a", "bae281ed69394617beb8bc558301dfc3", "5eae67a0d983496e84b1208cacee9ded", "77dcab9dec494e7cb2ec505ad8ad2b60", "5173892f947b4f6e8090cb5827e72192", "55553b2c775445be8385f16de06ffc9c", "51a162a63b004f52b2697cda94e77a3e", "e2f91a924b914e318139270f655f216c", "8bb251ba48144f1f9f76703fa1405cf0", "11c55db908d1473ca4496c4bbf32133f", "4710f81ca5a341539646008362c74a7e", "e75ede3f8ccc4ce48abe939d8dea85f3", "9f9d59e6dd894d319c1374574c655075", "6fe44c466cd748ddb98858c06e6acd81", "6a119d5bd9334821a313055a956e4d89", "378a0cb2ba19445db548e974db5083ea", "b0ada5af9f4443ffacfe697819b14e86", "bc3822e72da747f8b9a1fcd5497cfd8f", "666820bfe62742c798576860b772871f", "835ec49ad8644d669c70fdde5ac65734", "ffee21d5e4fd4028928d53bbcdef9477", "76b3b23f3edb4f6aae39e38265362ed0", "e4fc8ed2965e4517a8cb59d4f4ebf560", "ec0303d6c80f4e1bb812e37540a0af27", "551bf815c7fd463bb6cbfd17097aeb9c", "fe2c4fe92a7f43e1a04121feb64bf351", "e26e9cc5362f4afa92e2402771f3f77e", "44f723c9f459486b8dde753048d94863", "cbec4f9759bc4efdb98335e84f17311a", "8e346ba8d5ad4348870b68601a9f2c46", "af924a153f1c4a4f841cc6e77f55ba6e", "3332a57de6844f4fa97b3ce0cb48644b", "f7af02266a484309b938e48a788a3f92", "73500a2c699f4a3a8537870fd11355cf", "83a9c3604a704e2798439210a4de87d0", "ea630663303b4466a788a0dc4cd634e2", "17211c32ad604248ae63ac2df436a0c9", "c5e6cc760163431291bbfe822675f1a3", "0bdcf7fd1d4b45ff87513979c9c840e7", "ec44eb8df2234bce85e10516060cec13", "cd6be9b3ad964fc4a8d6fd9241693c4a", "ce0eebeecbce42b6a32f6888ee7f4562", "7c69843343284bd397baaa92618c8665", "6776f9e2e0bb415e9b85fa6e606087d0", "7346c9ad5d3d4823a0600414e18ea25e", "e35763bdd8744624b989b97a2393f2cf", "25086be2a1d240d7a667497cd378d3c3", "cfd14c95e0a841e38eac968df841f593", "73bfd57880d043968cac5a9ec5651db5", "19ff03a95aaf4e2d9438e6acb20a65ac", "5a74d2f0242f42b398459355c03455d7", "b946ce79addd4ac1afacce63fb53ed38", "2380fdc037664d6393cd93a12fc61c15", "d667c531df1943398c1a806c55c9d45b", "fed52fe6d07f423ea51dbef92e533f35", "15e713b44d95447bb733cfcfb0e53379", "e6699576910b4f178d1bb61c60196260", "c338c351f32e419a9392f487e8a204a4", "e32c1272d231422cba1c327f883b2851", "1c672692d80249dab14fe583faceb546", "81d2f7734bc94ea5b772075dffcb6d67", "b02ae942976641f882dd2479b39495bb", "6d15c0f154dc45d5be849241b72cffc7", "34bd86aeeb46444b83a6284d0d54556e", "a658db64a039400093da9d050b30ddf5", "ebf68d6ca71f4f9ebfcbe53812b84b2a", "453b1a75375b40579204b86829690ba9", "b2b5a03dd4f5414d80603c9d6c208973", "df6d39733e3542c084a5fb616c6ace73", "5834c7603cb94d8f8de3225989cc7192", "d7caa07b67b04d3ba1c2d827714d1f8d", "c637295a9e7648378071ef2937655f99", "ea1fd893d60c4c0f80864e3d283936db", "f66b27a9db9f4bd982435cd5f0050c07", "269727f5b71947289928e47212845c36", "8a6caa615d5c4d548bd02f66f68b8fa1", "fcfdb3cd340d4e1fb20f0546af4a35ff", "ec63666a359f4a33be536189b6d46629", "cdfa0eeafa06420fbe324eca51a62012", "0d03bf34c88041c392c57819414470b6", "8c22f16e24b5492fa883016bcd6fde57", "e3f3d016ff8045c39ee5cba4ce46d0ec", "2068d35a04ed4c528d0ab6e2f75f5a8a", "2d4a88fb9b804dc487ef8cc1865e0a3f", "ff291b38444c4945a7ba4de66b28f3be", "4021dcd322f9495e8a75df96cada3263", "3b4a15b24238432092af01c5a0cf2b09", "8e5321bbbf6b4c3787d3d09e907a39fb", "f8c4aee6802f4dd090e9f0b7b289150a", "69d4565c81274dad97476d5c899f428f", "0d47e700276e49a68302a632537c9312", "bdff4f93d8ec4b93b550a3ba0e846f9a", "4a5f36f12cb1400ea0c79b29382e5686", "dd26931faac841169179468d25cfed8e", "8079a67aee724e57a9410d94036d4da5", "419508bebb9c41a9a8c157fe7bdf93e2", "cf0dcc32e521498c87bbb045bbaf19b8", "aeba2ca3ca1e41e7b8cd08b4d9248a54", "bb3ee123c77645a5a019bbf120415443", "ab74ac646c2e4796bc2eac62755f365d", "f51151e6f22a46709335d076715636d3", "337f5f8d29134f95858596370055c74c", "e39510af4ad748c7afe3e545db63db01", "f7c8c33cab3942e3b1489f8dbd0eec18", "42489cad46e74ab49f27151d1ca297a1", "f496562e02f94f2da22bd0ad5f384fc1", "47ed875b363c4e7cbf29fd2e10f8d2c8", "f411ce1601454c55af8e1b5e5cc0d1e4", "4f5756234b2f443da43fe8430a22d4d4", "f4be4abc5a884507bd51dc34cc5aa809", "d15f684d31ba49808f28cd7132f578b6", "78ee04a5e97b4939b847d91bbacbf8f2", "1446e34a99e6415699a285a1dbf94d38", "cc732fa1ea3741f2b3fb4c82d413063f", "4f279499f8ca4be68185793ab02ebbe2", "46beee07dfa94a43bc0d882994a316ac", "9de0b0e0db3d42aabb34cc43e3ec0445", "aeae7aa4c3b54b22a739111dc81d7e38", "cc90cb8d5fa3428f874b6980fe50fc42", "e298b031fef142d5a5029ec837b94857", "307a40badeaf44d7a7a6553c39cfe8e8", "cc1d673efe6b4e1fa128a21cd06cc233", "a0784b2dea614f83966fe57a18b441a6", "de5336cf24544a6291bd22a3ea409adc", "3efe1c7c3d444f159c782aa864f003e4", "9481c200ba3849d7b1e3150ed227ccc8", "961a6b1f31be40258510da60558db31a", "5cbab692c2104e8396e5a6b0c0762ccd", "c7e469888a854eb78348a1bed312ae67", "ebd5594cb4b44edaaa2767776b0f4be4", "73d1099679a34165a8afa5f0108b7e39", "51eec5f6098541bbb3ebc3adfa4a739f", "3c2a594630a74769a80e08a8b8147d43", "950abc720afd429a839903a53ed3b46b", "1a583ea0206b4dd9abcb483f622872e1", "dc9b602084c042f3bcefbebc1abaa1fe", "c8dfa515ba4b43a78a1bfe8dfaeb3220", "b97a62b763404538b0ee5f57fd8884fb", "c028f1596854432bb1341e7afe333f3d", "dfbebbd9066648f3a052ea55118500ba", "99d10669636f4942ba4724dfef5616b3", "3ba6be1df845494498526f990c774d17", "7b63555e68c14a7f990a41059806bf36", "04a361443fe5449c8ab68efd9d4a3fe9", "1697588deb5441148a4300d8c3c875d9", "e605a9398fa24088b6c9d91752507ed4", "3898fc8460cb4079bc5de3f0ab201ffb", "33f273122edd47f9a337334954652287", "b0bb35bf0e814d1bbcc694af615facb9", "8cfc107a21554ccdb21e7282b592f83c", "63caacb291bf4208a35d7b5e87d1b9fe", "082c59e722ac48518dfdba5626f67946", "2c88f46286954741be8eac382ad33b11", "e0f665e928874ea3acb060e2d52b9cfd", "af18b6c3d9c64cffaa9cef5bd4384e05", "39ef8dc00cad4ebbaef2b71fe8cef218", "89d05a45d91b483a9b094a0b08825043", "fc68f299796d4d3e8844f914c984f6d6", "cb355ad31d2941dba0adb569dc104e73", "0d35593a14c6440ebd316aec9dbe1074", "7e1fdf8da0364a78a655d81b2b9917aa", "d0cc9315851247db85760ec208cf78f9", "ee2df8af7594436d840d81c2e8f8c82d", "6abf1fcd55974a0e9b15fa7d89ef15a1", "d4504a57f9624e5297a7a28e2ce511b7", "6a3456bbe4aa426ab804f045f96bb211", "fb1c6f771af44e5a9bc9f54676a1ad07", "aa3f957a257f46ca98313fe57bef88a9", "8fa36d123037443da4236abc126cb37f", "e9c496699c64476491b13efa0212487e", "b4e67e4814f14414a1af9f3fadcede02", "369a584d016f459d85685a1ffe18778f", "b3ec5baba9b94047ad4a721016f8f9ef", "ffede9b8afb842abb47b216a57654861", "50a42d4ed8f544ecab882b31259dcc23", "281002ab55e04d4687459fdde40373e5", "4cbafda382f74c099156b068c1b1a509", "9c5d2c5beeda4867ac080f8d7c12657f", "4e5f16b6f13f4fb99c362e84315af98c", "b8d262bff73348928462a948346f054e", "05e255abf6d34911b86c588eca1e3736", "23f779187a564b37adce8a46ae908929", "33808ba7360c49cdb2f8fc7f4ebc8c6c", "6eb20d85f07e4264ad5e93aaffd5e54e", "064d53e7e7524a0584a65c278f3030b1", "9b21271ecb2d4c1c939bb523c25584c1", "0c5ca3c6fe34427d9cf304cb07a5cea2", "edcb055ab75a44ebbdb7ce72cca1ffd6", "f28d4d4ee0bc450d94ddfc0e8cbe8a67", "daf38103a9c140fbb39a34ce80d98a31", "49c72be2c9d34d11aef58879dfc0a910", "1128dca32b984ceda30c818b0af82370", "db93fa2329414e11b8681e5393860f5b", "3c56e99eae28486987cb52fa2ef8d916", "8e3fcf97730f42e1aade4c626d8e083f", "9349cc6e573143609fdcf936ff1a1bc5", "768b7846f7404b50b3da77a44c2eaff5", "117aa85f64fa4a7988e7c5c701febc1e", "0f8ba1a4ab2643faadf932cab1049fec", "f662b3b150e04b5b8056c2c4e0d8398b", "6cafe4557743437f9b11724eef327c2c", "e4c0d73ca3f04b91b600aad097aefdef", "06c3c825491b4bb7affb2ca30280f8b9", "35c20b52c38643deb94e611912daf789", "a96e6d915e6847d1ba3a0da494fa353f", "d33b3ba75948457f8e43e82410155fd9", "c9737bc6a9b9400493566493460b6e31", "d108f8334a5d4dbc97ce7844bd607656", "ce4a9bfb6da144d68450ec4a042d9549", "a8374c452d5e407cb0142ec26d1c6a05", "cd2650f180304a83bd3247e30406bb81", "964571bac9fe4be39978e0615dd999e3", "9e84ba84316f4c99b164630db9994e1a", "15e2167ded654448af51e3292782aaec", "7a27da566b444f6293511bb8165b42b4", "25c881e1333b487aa76628ff02a721be", "f80321fc7b9b496cb9dfb60d349fd994", "3f3da4a8622b4773b37b17847ceb1702", "9716f1ce7aab432aa3984ff08f4f5dfe", "54832b4c0b5548179964ca7e7c7cbad9", "33900b9f293849eeb2f279fee2747219", "eb3464dbbd8347ef86cb26dc977d2e6f", "fea76c65441148d99fab463930cf1d07", "f808371e90fa49d283b3dbc3e6268324", "2cefa0dc183e441a925a222271883dfb", "34c751766acf45d88153d068ba245d16", "70225482a1074a3c9196728e2660611f", "29dc693df4f341f9a744935811458941", "c0df87c50351473bb4efabafd46e8482", "3af5664627eb4b19a15530b75d054df9", "a1b9617cd80b43d8bd8af7c9ebc8650d", "508723cbb7d541958c193f6196b26d93", "df65dfb679ec4c0b846f8ba5294bbcbf", "26437ab365d743119b689fda13c3e988", "3ecce0a95d264cb89f4d7ad711c5aa94", "eaae4dda275d4adfb92a0ec86978049a", "7b5d088ede264e8da1fc233dd386945f", "5006befdd1644925be3064d1f082da02", "f220051ac7f949d98c97810c6ff57a35", "feb024390f0c4656a13fda8233188e8b", "b5b17a722e554350baead7b372bfb89a", "c1343ee674f54bdfaf345276c13dee05", "1a58e87c41f549efb853ec95ec859dde", "6994c1361a2540b48aa6532f8eae1ee6", "11d0791c8acb463f85a18f5bdadd9ecc", "873295daf7e947b1bd2a99d4ff4f82eb", "3f199b6c7a8b46f88fae0284213effdc", "bd725a86c99442fbbd0385a1d078c195", "8bb1ebefbf17439d8580b69b3bfbc926", "f09a068e43e34f5eb2a25d98cbd71c3d", "1d6b4d29ae1345cfbae7ad0a931745d1", "796659e5812648ccb6328b07167283d0", "9e6ca1d367304b1ca683d6da7b12c78e", "b9c3b9822dc444ff8a4536832a34ec69", "50fa353713294fe7826bab5f8c93f6bb", "5c8cf38bb29e4cc1b3d60dc91a27699e", "e32a680196a641079580a593135de4b1", "37202bba5c294b7d8152717b506fc71e", "7d68904864e340018e67a8fb110a78eb", "b0b230c813e2417f8945ca080da89fc5", "69246805ec404ce2b78defcf8807f37a", "79fde91a32914fe69d8b6ac8fa58b84e", "c01ff55fa1714122bc5ccb1334ab089f", "936a94018ad24d85b5d8bd32aba2edbb", "aff7b460a5c64a34b8dd7f555134480b", "ee755a542db24735825fa9f5d4e6bef7", "fc34a95c05f9493798241fb1f5f0a6f8", "22cce126403141539f1edd7fcefa8841", "4ce14a4545da48a9b571ab8b61bfa1e8", "953126af5bbb467db9471f2a7e1fc71f", "e17bab498d714a0282afaeb0c34a7438", "b8002c3a005a43b18fb92eebecde54c8", "4696cd4c05e241a594c9309dd8705668", "45517ede7b854739a34f56be1866e101", "0576f94f4dcc492d836699f1332edfc1", "147525d18f9e4514931aca2b5d5c2d78", "02c8f94757564e3cb30a5ef82c2697cf", "2b3cc9149fe545adae581271901adfa5", "87adb2b3ca3d414fbbecf991f540e0b9", "7427a5ed19db4b8cadcc86e78184b6e8", "285f6325782743e9a308c6aa75c2f191", "4bb197e8309542e7bcc8a5e654ebf230", "a17b1ebd569046bbb675ad7b4fe4278e", "c40d542f318c4ef7a2315169f04eab4e", "e9294808282d46b3bcc047c897240eaa", "46025bacfe8f4cc0813375ccbc592720", "e94d9cc976224056b5ddc05419b140ea", "20a6bfa486ab412b959221becac84428", "b1679d30b71b435ba9bb234fe4c8d5b7", "8875d0c77f7a425fb3eedd946c71bdff", "781abe06d2b349f98f791679f8e546d1", "2b5268b240ea431fa9722cad4e3dd5f9", "b739cd82174b4fb8a14008f84341485c", "20f24a42550e4cbd9e47fe9524c5b60d", "afc1b70e46254ebe83046d1380f551d6", "05377b37d8e04325984039dd2678055a", "cdd1c5af99884fc0ad6ef98d405ac65c", "9844c9f31d24486a964d6f8c3e67e7b6", "2ad28ce23aa9426cbd96ca6c596d9313", "dbce3fad3a464757b1ea724f9d79f1ed", "b73e090b667140c9bf64a70150bb7958", "53c094a9855b438bb5e1df94bb6097e9", "1d4baa1b49234508b653178aa85babdf", "bae8767f3b9741e49564236c552900dd", "439a353c2a574f3fac58be3c5ab0d874", "8ebb67ddce78460187aa06afcfc922d9", "f06aef2fc41b4576982f64fb50607b07", "484f45962c6946bd8a5a90b5da7d7841", "9b52291cf08d4e73b3a4a702e60129fd", "bb7c5423552043a589e3ebac8bdf18e8", "c210374920434b74b4b9e6b27b4ac989", "be0bbda5cc734dc7a17142b5f2733f9f", "45c9cfe9d24f41049a6ccef494c6abb1", "d56a263b303f451abf08cdd83afec074", "e729379fdf74499e8271cd038bb43b54", "ec359796433947b59dae0910c6c6e196", "630a8c11ff584f1184571d0fc90011ce", "a95a6507443e4452893795c848bf6491", "c011b992022a4989923f15563e04935c", "714c76f3157240969a97a19ed30f4a9c", "900bc1e063964b9fa56e92810e1c9e75", "011019bad4434599ba08972b97d968dd", "6cd8c5649131493bb7cce4885fd7e9b4", "f987d8fa30ef451093581bf5afdbba72", "0c08370ab63a474aa6806f68f819bcff", "217b8a5421f24c0b8d40829185287a43", "9bcdf951fb9d4e05b9e953200fa5f34d", "b4fb7d317cfa4a299b4a485435885053", "acd9724b17484e64a46e24cbb083a1d5", "fee6f83b27cd4449927a64cd741f0379", "657acb2354be43af8e69535b07336917", "78461c8ac29e4eccb63866203739f2d9", "64027d68dbac40208382e7caf5a23c02", "e83ca82c349843129407d8a59c114e6f", "19a1bed56b8c481ea9cabf38e944bd8b", "8e7dac30ac7a43198b228d7c56f903cb", "eb26e8e757274bbd8d2795dc6f453ddd", "df505e47d036433d969914c38f3090f7", "7f4c4cb982214911a4ae0a60a84f9e9b", "721d075033874b37bc67447c0ab07003", "6d6dd231191e4c6eb962cda622d32259", "424eb0b877f14a6cac209b5dbb5e7a0c", "a77a5ce389544de9bbd2686fd5937b84", "5bd5ed1fc96242c5a0db732476f736cf", "94df0d6c9ee4489fa50ad42db2843d91", "47c4255e155f4d3cbc7ef68b1574b076", "e3befab39173462f829db125f0c251dd", "42283a6ce04446b195714f6c7c393767", "06779e25ea0845949b87d601396c7116", "574d45823e57449dbd573043e3925046", "8cad010ffc1a47c8b2f3492c07630f01", "93d0d884c7b442a78c81823673ce3012", "ef7dc73f492a43e39d5e5d678ff20078", "ef091973aa8d49e7aa2a6ebe007cb3c7", "95b05c0c287f4e00b35ba10cbc2db502", "f1ee3e8b303148398ba29f3170f979db", "cee79c5a19b74a708d2b106a02f557e9", "4aaaa1fa0fa042afbfde15e5a72400b7", "613c992373604d5a9a6d9407c601d986", "03aa68b6a86047b3815ff1d30070bc6b", "9632056287ca4cf589421f87651dd053", "36df88fdf94a4d4dbdd1fe0d56bf6fab", "a7797ce2317c41b99da96794b9396d87", "bb38cb71ab4944e2ada9f2f77d3ba483", "4361d8baf8b9405f8c5c4f4bdc5dabaf", "6416bebd3acd4a828c0130670a0a2fc2", "6dc1006e347e4a8fa0ba4ae99692a33d", "f42679055eb4420999752741f529f2ed", "b1124a3b2146470ea6f6c6fc69f63e68", "7dd8a8e8615d4b7cabd9312a8258c4c4", "5efc8a0e9cc24f0b9dd8b0d866076ba9", "8e9c5a0ae05a4965a7bebef643be0a64", "e07a8b986d39490f8f6391af46163f2e", "156abe11719941f5ba33843aea860423", "8016c2c3bb844a9ab01fb1a928b0ea9f", "4f4bc1cdd0224609a4824e1633ca3152", "f9e60b2ba0914672bd4cc9c948d67a69", "132783348e4b45ac89fcfaa8e3c241a7", "31066f3c4b8a420ba5a8df1d66295993", "806cd112c8e2496e89498524dca425d2", "f58345c43cd24bc09373e7c6a53feb05", "78380c85032f462f8988dbfe6570ccd0", "170ae7464cdc44a48cd2728700b5978f", "a1c5bf39b7124cf4b552c2e06629e30f", "fa3c89f8621142308fe3e97d0c1f7b0a", "cdc831437ad547e69c6937fc2ca99b70", "1e50e1b355144bb6adfe556f40245cd3", "ede5ed11f7a84facb065f7db06b14d45", "54779fc6f49d4bb2aec81df6f8c6ce1e", "0226a686d83b4e7393e203626d666b7f", "0f92bc39582d4d9a9c8174c16f278ced", "db179758fae84deaa75dc1307c9f3ab2", "6114b8b3b84e450d9dc8cc45f2ca4f62", "86f9cf2ba43d4614af557b3373d3debd", "e8b25fd8f09d48a89f1a22faef116328", "28350086588a4d5fb40a0e9455d76f6a", "fca0ca6ff702447ab8e67660ea10678c", "3771982e74a2419e9bf1ccd18aabad63", "cc30f84dd5904a718bff043df4e1e050", "b2496d4c562b4aeeaf95ea2c6074e9f9", "0985264d41f64445b82406583d17267d", "9b7d477d450a4f47b2f874bd571fb3b7", "983cb4be3def4a6285bb16a3ab2e4003", "5a3db008441f4310815ef912dabcdac8", "59329d488b7f4cd78ada4d29ffba4729", "fab4a76eb73a4f129bfce41e58dde977", "78b5b6a473bd4237abe55f7e3fe5c1bf", "996b719f5473411e80f6e3d2361d252e", "d872e8576666479198261135ec4a2632", "31f545a5837f43a6b4435976ef1e1981", "1426d8e826614671810bc4ed0805184a", "0785fd4db07b439d89e926ff274059cb", "7f7114a58a294b929d69e58789d0012e", "6edd981e8a9f4b26becc607a3da9a06d", "b1b222180f9041ee84368da82c7adf1c", "7874bdc2b3374f2ab7ca181c9f074e7b", "33d15eee08ed437cb65bd81884a7180f", "bce32ef9796f490e8689c867de6fb378", "923b92bf8ab04b248ae2c952358dc9b1", "58cd233c84e64c9086628a01418bfa1e", "27db72e475a44bc7a65c5903693fbfa0", "12cff51aaab84d59844af553e0b38b1c", "9a2409fd0a1a4872aed60d657b21e311", "a1e56f9ddf514f83a6b0e26b0ca2281e", "458ab8dfc9bf4e4f9f478ba551722246", "75fea2aca53c46388f9c4472fbadb4c9", "2fdbbd97272c41dd9be50082620e93fd", "bb5541df161949389623203b3c50e1a8", "4722dadedbc64b779e0de2a3ede2014b", "066ad511017743978ade290d9cd59fa8", "e8d82ee3c11441469df6cb3b0ed5871a", "d8e149ef4526490e9fe40e13936da674", "ead5907e41704b7fabff402d745f5b7f", "cf1da43a77b04a658a440ad6b56b92e5", "fffaa28a8a3c4f4e86a0700a88c9fa04", "8a3228f9a1e544e09bffb5c4b82ba79c", "a334e5f26b0b475db24151539c6a0db9", "cb5bfe0f747449b58ffd8131d44f3e09", "ea51cd02cb7949028f294be030d5b521", "3c0e08af9a2349c4a53056880e733ad6", "cdc172da935a4bd4b829b4e56cb5eb53", "e8738036e9f14036baf7e8239bad369f", "02f756d441884249bc95ed445003ee8e", "ec5d4393d34a4a48b3a33f4f83006b8d", "b1d18d5f8583498486909fd9840ab5ac", "e21d87f4b9ce4e0dbac2c728108565cf", "a977288272d4497e9dfe015dd174a1a9", "7df7bf1675114ca3a545630165a10b58", "f220eaa0054a495c895e37112b5a7dc3", "569c3043e7674e02a3b8db44ab39f6fe", "7dee9134e7054f90a4f8150cd64ef773", "3699b6fbd89947d1a2d7a7503a652000", "5ed8a549c6b344a491405013daff987a", "d64a47d75e4a43bd8366fdcddab9928c", "633002df0c7b41a7a50ec1e70ad06587", "10e43fe04e3b4accbf0fdbecc537c70e", "6f5fd57da45249d5b0c8b42bb84ebc60", "1dfbc959955b4a72b0337cf4a23d7527", "7a094de66f0243ccbd4e51db0d4c9252", "8b40b6dbc084419e827f845c30b570c8", "4c2af4f62cc94c22b6ccaae11d289ba8", "37c3eeac048940c88b5625c42a6560aa", "8ef9aefdafbb45ab80fc0283a6ffa160", "0480b2a2e7614f068c3136595e1ff9dc", "c6a92ec560ee414fb3cc902c451ecc60", "afd897294e414d0e873ce27137e16aee", "b9b351a4173342b48696059c65a34e2b", "50572f356e124fa5b537ddd484507179", "663a242a0ad44d4c99c475dbc4774090", "b2acc2cf37fb40c08635ebade83c2b71", "0b6f7686a83643a2a34e5a7d35ed3d96", "4f6ede66c648491991c1d6e7d4ed7e3c", "ba523f346c3348e9823b43befcad855b", "eb905021d5f74d58aa1f431a13b16a25", "5c431f33ad3b4ed0b51e4c9252f5b1b7", "cb8caed78e684d64b83f2eec1797f8b6", "56b3b13fb41a434db091cc974965d08a", "60fb2116f0b6430e9fb7206b456c05a5", "fcad6b848d1a4041b6eade48d6975f98", "af4c23a5b4ad44d1b054c62d3d27b533", "d4098b6762144e6885d8f3767475780e", "3dfe8588e4484edeb4c2c127ce3c3dd8", "263c3402a4764c09bb1956da271de6b3", "7009bc1c784b47bb9fe50280f021d3ff", "96ddef24a0ea48479027f7f4e10226f6", "34102f3acec742cd81f82b2ca44d5a55", "fc0da01aa3c542669c68980f92a8ce9d", "52e1dce3aeff41f29345d8c0145bf85c", "0d57e0e2724b447b9832e8e7fc234e8a", "8410b4fdf43c43899339d336dda928cb", "1af6fe5f3014411e9e6c42918049ac6c", "22e63b8082b5412c96d9897d72fbe27a", "58291d94d12f467199dae7f8aeee4ac8", "97d099da724a49ab93f03567ee1419b9", "69ec86e7162541e69bc112ae353ac187", "23734160b8d04fdc872e01ddba475a83", "7da7ed28836c4030a7b412b90f0c770d", "b9bd4aa908634c2fbc15190084a876ea", "82ea47c6fe7a4fc7bf0cbf5d103daac4", "5005fbf5339d47f0b920a81e659fde4a", "eff3d55f5c0046939a57c5662371305d", "0cecf488651c49239b5d185d9fc0f9cc", "5abd48849f6242b4b78b2cf78d6b4fe8", "750f6a8c3115410cbc5a1173261e906f", "1639f5a88fb74f58a5086068b5dc1047", "56fa5ff0f4534847a8bcbb846990b293", "28dbebacf85d4273afb79ff546d4ea13", "280d6517ab8e47a8b1636d24f3d2994e", "0b2a7790a3d44966bcceeb6f2c36716a", "59b7949089ec47f6a8517838b8bb78a2", "0a7d342418b243d8a284d6be06537f2e", "7352bd0add59495dbc32604e2cfa4b60", "83d7d7f80285439a9049bcfc81787ac3", "e3172f3aa5e64676b4c0388cab4580d8", "bc32af29ed44415d8cd9331ef40a198d", "0d2a046858dd4c3e93115f396b3f1b79", "79675188267c4c79b7c61691f8ad5e16", "bc6d6676a80a4fd3b48850f172b69cea", "9fca380355764a0eb873b1a7cf96308e", "858d716681be4501b27b906ab13de4f6", "e205bcb3346745ab8b2ef4db297e4f33", "0b1a2d0c063945b68ef0be1de873d7e2", "bb2affb7b37d4888bfe2e6d8f19af2c1", "b861f2e0c4a74346879e8ee3ed20da3d", "7295fd10caae4291afbd38854880e6c2", "7d960aefb3dd44d88db6f1a705e6776c", "35a70deddc114184802a4727f06862d7", "63e724e6ff0a4955b4199a358b770c5d", "ab59c23868774090b714f5b01737dc84", "0c3ab3da9e634b7b9f8b1b6783bbf65e", "50d911d7831e4f0fb68d3ac111212f44", "27b1b6f0a11d455e805fff5abc66327a", "ba58f3fdb315401f8a05edfae007fb4c", "aa78d7b7f5504fdda7a2dae84ff83794", "a0ae466bf3db4358b4c1641d5e091cd8", "323b609e97bc4cbeb2ee849a9f398bc7", "181d38dfb8034370a4ba5d587de95481", "4a8ba935dc774124a683079faaacb12c", "387998e163d64971b40a0c00ed3ec4d8", "42bec54327c54ec2a6d1af6069a4ab87", "a9aefc47aa5f46a68d49cad95e6ccc49", "8b05976c5f7b49578d7715c22119ddd9", "ca546f198cc94dfdb70570ae258a664b", "c9b4cf04d3984102a585febef74ce376", "3c31cddd7c5b4d6bb80797974eaa73d6", "4c125daf21ac4e2dafd028b98139ab70", "8823206609c24c16a0d0bcc41c246066", "742f84db3a694efdae6f46a14dae04d3", "8f1d87809c5a466b994e120b958a34a3", "34a2a98f8b9a4ac2be5df3af3025ec6b", "862fc8d05339418692482aaa6eaac85e", "b531f1c8dc0e493ab5f180ea9c89996f", "f61e9939b068416a884512815865d14c", "5914c8e1c0d245a281ed783ddd076d7e", "0c65812b0abf4f6495cdbbafcf40f659", "8ace1cd6c759431b8b07c1c933f55068", "025572dfe10c43e68c840a3aef17ddf8", "16be4437154e428a8427f8466429daa2", "9de8e1be79534916a0e3fffc03cc6004", "d90245938ca5419b826f711e22770f72", "d9ce682c3cbc4db5a1f766422472304c", "526b8aac56f64a37bb9e8ef9af7ff3a6", "6cadd10394a84c3fbf6961c3bda9a3c1", "f8d7029443414964b9e5ef8e4a465e70", "f0a2d747ac83489797419906b955a74f", "89510c43a7d24af2b931caa995126ed7", "a52b9b5e548b4ad3913417823e464a26", "7bac3a3e677b49ae852e3ea637c32744", "e7a50d03fff946938fef9bca97ef5ad5", "941bce3fd3434fcc9b7c3d9e11e35ec7", "98f866508e7f4a8ba1efa8074a0e118d", "3e610103087644b7bfd72c727d987314", "56c2ea289f1e4e93aef059c839a0bb96", "d3861b5a5bf0411fa41c0fe43087965f", "c93f3ee56a974232bc2f1a18aced797f", "18a11285b3ce497b89b0d0897e54d477", "ef65908984c54df6b23b9ff4e8c4d8b2", "e7678beaaf00472892641b4bec8de005", "435b5799654948efae69bd864c0cf239", "a281ded82fce4af09f3e9920313ab300", "6b75188a8bd7435ba9227cc310986b52", "9950ade2674a430c8be861c936b82b8f", "003dcf6105db45479eddeedbef8ae1cc", "053d894192eb4ceb8bb64e4b2f3b4603", "dd8852e00c014754aa46e1a8e5b5bc50", "a9966a654ec14096bb328852e28905f0", "433cd1bd3b8b4948b730289e50c66327", "9051edfe6ba4430ba42179669874bc70", "8a91ef36e4c14ff382f1710b2f720b1b", "b46209bf486044fd9a23aa603be73f9e", "c4129ea2693842e8bd1ab1a2ec146f7f", "d2af52c7cfbd42bba4357e0389bcdfe1", "49363d597fd44f52994c09a3e65abb56", "3a083a6c1b804a10b7525c7ef974b33b", "1c1daeb044904342bc50eb9844752386", "5d6f45b281ff410facd9e5d5322bcaaa", "0093534224494dcb862280242fe01006", "721f3e82e1024417a7db8c143021f69a", "fb88305e953b48298da6345dec35a496", "2aef2eb1d3144fb1b86573b46a361f38", "aa716fb64486401093106d9b399bffa7", "996dd2656f394662b43053848618a276", "7b4e8d7a9ad64adbae7df9f5c4c97e4e", "ee16e418f6404866a9d2a78c879414ce", "8af11a43ac044b3da0602e5c514362ac", "30bc14bdbe1f4bd89b4dfd4a3b9d056c", "3f26e7e04f994a6480b55e6dc8129f20", "2031aab2b699488b9f32ee22c6b3928b", "baa4c8f02a4d47d79b414e07de6e5f4d", "ec6915159c874dd3884af37817ba064b", "802685683b8643e4a3c9ba05f060b417", "9155708b0cfe4dc3ae2ee3183480db5e", "058c14e867bd409d86b3eaa14c372931", "34986b21ec894ff3b928deed666fc618", "60e1bd7f031142d6bba05dd9478611a6", "be9f1f0f302a477d8af32348f67f671e", "cab756937cd249c78d8f729a6fe48656", "117783b53add473893c68ad4f8c9c0f1", "2c0c354cc27a488fb1042b3f8feb1efa", "7fd95967025941dda06465630df2c592", "b4dba085f83a4ae3a3a48d7d65084d57", "b5a0bbbd9cd74f808848636c7fb7d75d", "c524c0165be54f4693d3c4dba3e38887", "19be09e82f2e4532b1176845b5b9d5e2", "8a7ba12403214e758567c06c7142fe35", "752203384f884a3aac4a05a1793370ed", "e8b4186efafb496f8ac8746bf1ced5fc", "a1cdcdc757104010bdaa7a53075926d3", "4c950da3e78146d8a243749fa86c231e", "5dc13a3e869447a3a759619b28be08e0", "df52d339b8a648b5b9fc0abe5a8e97a5", "96c1b88b9c5447d9bb5e15959ede69bb", "ad13970ff6374795bdb795285271c0a6", "17ab7220260f40268be465ac121b23a8", "c76a35a22af640f2bee81e926d235503", "e7cd77877491479185fd025d67c112c5", "0e38497447784c5dabee06b520450db5", "f1e46fc16f894afda1c53da60b5390de", "33892727df6a48b4a600f19ef499dca5", "8c81c1cc43b8482288744dac73de2089", "897a00ba936d4435b62aa2628e592593", "64504f07d7b142a39419c6dc0bdbd3bf", "8f326370a2334f9b9f6283faa525c2cb", "86ca89b9470743f1b69d18e74eddd223", "975ede1fbfb644f19212845f19068075", "be8cc76538d84611bfd9715edd369e11", "eea6d8e5ea8249679fc81a2a455d6170", "ab7c655484344960a19c6bce6dc16da0", "9b01a5fc126941f3858d7b79cec428c1", "c5f1e29be67448ef9767dc5b227cca3c", "a8b89a806c94410092355162ffdb552f", "e08f6e571fa740df9b092ef53306d760", "5ab94a4a2daf4eac84127c7e965e2343", "4582a69a87044a68b61c44c37cd80dd2", "ebc7b8134a4c4de5a9219c08e884a93f", "ec05e6edfb1a43c08a83a5b43f24f8c1", "6febca8350cf4986b95eb7f3d4fad7d9", "fccb4beb50dd40709c29ae58d439ba02", "ea7b583eb6af44f8ad8429c73f37274f", "f5f6180a01b34792885fbd0bf4134dd1", "c1a7836e6e2e4abab880c983b8f050d7", "7fbf52cafa3f42eabad1c5b52f5c769c", "242e9c55e83f408dba6a086b4a80ce68", "3f5c7fdd8da54e7baa255a4aca05b123", "e018e7df8262488391c6d6afae4c8364", "634a715845a440b09364e9dd5f03a691", "af94e9e61e3141cc94c090107f4d5593", "64dc9bb53ff448efa40a18f7bb119b75", "7aaa05f8ef23470695a637e49e5707ec", "c1ee648700d3428fbe70b25492faadd3", "1cba9b043ecc43dab93fd96158a50c6c", "b3f4088d629149c2b2b0bb756998a354", "ce49d49dfa6c44679ce3fe93a01abf04", "8a63d4cdd3ed4e8e9c62edcb9f0c2e14", "d1b2c9a6ac514ce794e00a3896ca7289", "ede88a117a3a4f80b44ec8951944c2f2", "9b0a0f08f4e0484fbaf3200011dad04e", "0e40081e466743a7878f274e0b4a6a87", "ee9c7053fd87446aaac76edcb5cbf21f", "ea3e67b7ef7f4acfb1db0514ab0d2ef4", "2866763b8b784c919dcf0a9dcc5a1a69", "f80bc0a12b07450e91f07ae397b6b480", "00623d1912694cdab4f488574b571f59", "29ae68017cec476588b8e649c7387e0a", "33ff77a81ba84eaf9f7f1c4d9ecc7762", "3d263e003e304f26a70a1c332287d0af", "a02d5e11616b458880bc94a5aaa81247", "b1c23d22069b426197d6cb361f5f35d0", "7f76b3e693344f019973f82f3ac91c2a", "48d5386e12494a668511f826ba4a9175", "2964c2d4e7f54129a109afa7efa5347b", "09a0d80779104fe3a8d76c0315293722", "7b45a85c7bca41bf9050977eddb9b578", "22578c154a284e86a193541d48acd941", "bb21aab202244cef8a9943b4529157f6", "0f258459b25346f097a5d32bb7047f8d", "e09b18e0300749668bef181ab8a32f79", "e93c1f9f1e414d76852a004ce393a579", "a3ab1fa27c59416f939b28472239aa4e", "acc29ff68ac9402a9fe2b373e1049bf3", "897aa6fd68124d8eb8df905d26b87aad", "b3ae2476c30e4860837e5a08676ca17b", "a7fc0dfd60134254affb357a589f3cd5", "dc858c7968d24e898754a842f78b46b2", "a9867b5d6acf442f94f01261f13ee934", "bc021a1a803b4a539f57bee4e82d853f", "2d3c62f5ede64177b0c183b5dc1c8dd5", "bc658976f3bc447b8e699afb2e3b38c1", "c79e1b62f2024c23946df1e2f34fafc3", "f72f6decf184424ca921af9960490683", "99c349b61f6144b384852cf5ab51b1e9", "dd31e79826304ed1a46de11926bf3e6d", "0b61c93d0f1e4504a0731e426dab2315", "ab8076d82b4849219e855f49b93e4dde", "d3e84ba00d3c4f9d8a1650f9647e2292", "2f42f3192a594e68b4e82c6f6d1705b5", "22445d88462149f4bc15b719048e69c0", "2dcb7d47912b4cc0a9676700ada96ec5", "8c885c595b1e4957b3854b88bdeac0a0", "a9f033048a824555886545779686a123", "f48ee774c1c640909cb4020f4d588b4a", "30bdad0aceaf4ca88dfcc7a78064c2ba", "a78dcae5bd2e4f929472363ad1e8fc1c", "4df8ae8d7a5a465b89236facd964d3da", "c83e31b51d9a4cdfa227961ae912e5db", "20634465058c47019ff5345a7384032f", "fbcc992768344732b9293a2fb1b6d1f5", "2bf6d2a1259645b29d517f9c56d7891c", "a5081c4ae9014472840c0b6be22d2c40", "c59b0ec8aec3434e9af39ffbada8811a", "d773139ba9b6431bad7047a454c462c6", "2632aeb59c6d4239b6a654d1f04a8315", "eea748086ad841d4baf7b8b59fd6bdd6", "0c0fb2795c584451b23ad0e7bb9836eb", "a5dc60d91a6f48f585883cc12d96e288", "620358057d254432b8e4d847b7557098", "0f5559b107b84363a7a87f205bce9271", "ead5192328244bea97d74d6e854f6717", "f0564a210add4ee5a5aa49998e3c9517", "9ef16d4b049e4900a56b50420ddd6bf6", "b00e22811196493e84c5dfad047059da", "a1378414b6a5430ca07977014a946dac", "f13d9e9c69ba4850928081fef65a5fb4", "00642a06b7474113a0e6708ea79bfc80", "b63808c27b2845e5b28906f260da0d66", "5ac10cb0f4e9438eb80b046aafff4605", "5562fddc4b124152a29c674f9687d414", "30b16b9f9f2641f4864faf1cab2fc8cd", "f3d788d564eb40cb81f8829a72d88eb6", "c48bca6f09214de3a4d49277c55397e9", "e98fd1acb7fb45d99c576d56e60ac61b", "671e25934982476d968fb9cf0234c447", "c648389360094b409c02a0022c29d8b6", "ef58298c17cd456ab1c579df348ce6f6", "63f7502ccf974a06a848d34478b8dfa0", "af3c408b8fbb46c0be4a6ed1a6d278ef", "16154c018ee746e3a337252ba0b15cdd", "b1fc92e8eb48448ba4ad9dfde6443c32", "0813d3e2375140d5ae35e72cb03b0ae0", "7758a329408f443b8cf33e6be0144cb7", "6e240834fb664dc1a7eb17ecbf6e9cdc", "4fba14f2f01449f1813fecc11bda7ffb", "a653ec8ac8e84efb96da6af8d1c285f4", "dc7b422477cc482290399755e1ab5310", "198684e6b6674a8298586292843f2da0", "98bf2f38cdcb4994aee226029164ec00", "ac2b9a5221cc49a2b87f5225393aedcd", "60038da7d9f74d88a079b1ee74ff87ab", "b57e93440f9a4e339b8674d7e5690320", "f4680c7b5bae4e1ebebea874ab1bdbb5", "7c3793a01c884df8bc3b1ab88f825b5e", "abf7723f27904b2788455131acc0eaf0", "a5f1aba4745b4e9595369e5e29eabc26", "39728510e7d742c6ad7b6ff58ada4d2e", "fad57aa2764948fd9d725897e952a794", "88fd273815b546bea6c528fb8e425d19", "cf52b78bf1254d719ad41d14f26beea8", "c91ed5dd744249e7837c96a236f25464", "8c123c92daf64af58177278df0c86fce", "1f454892bfd0494996dd26579bd22279", "d6fcf6883c034745a27e4287e2706528", "9d9d293ed74d4b6abe52272d5b1f374b", "0f1e505ff2f64e8396eb86013cb89cea", "b9d83de05ecb466d9944ca6d681bd728", "95e27cc52e81430b9c22f6f7866f9850", "cc57e530a34e4e2ba8b87d755f2ecece", "8f309c930ccf4dd096ee59f3810afc45", "9b3db46406054317b6962c94de5d7862", "8b09bba898424d358f69bf408b10d14a", "ca18918fd6244e158c12b41210f22a60", "0d7d6318dcf3470683a08a1b7108f4cd", "36d472727dcb4184b3d4c9a542ba7618", "bef0040858b44baeb6cd2f02bd698912", "321a00d8fb654ba7b0c86dabe6726f8b", "3d04fd1bf36442d0b22b2d56c90d94bf", "f1fde636e64e464a96b3cb64ba0340a2", "2ababd644d3841e69c2e049d77f15464", "4e092cc7f4f04ead8d9354b99fffb740", "9c7eac996aef4987970f9dd024ce1079", "fb5dafa8fe934ab8b3d598837da4acea", "83ac8cef38b64833b2c4c8de400259ab", "44ae93b9fc0542a09d40f613e7cfc3e9", "791615a37ed9466caf93db4f25a15582", "dcc1e5db85174c41af265bd7841f8412", "1ece9f0c3c7c4e0d941361539dc58a53", "ad2ae1558d4540dca7bba1357463eaff", "9595b836f53d445ab1b0cf86a988d366", "f66e673ef76f4f28bdab408fd46976e9", "1c6e165a7f8c4230bffd448227fbda50", "8ef7d570c6ea4dd9896ba7ee5c99b3fa", "0443f12d41c84edfb6d860140f8222bf", "6ecf27f1c3244f2fa8fe38faee3849ad", "b67c0c8f4b704582a93b9f38aa685ff7", "0b59e0e2d46c48a3906234f56817a840", "a0655fcb74fb441aafbc5bd7b2901814", "815da882ae7c40d0bfdcbea88b8b6238", "de421f48870b4f7a9a9fedca90c665f4", "2166d90800544f3eacf7cf27583bf6f5", "79d7d049cd0c4844a95f58846037eef8", "a1b950b956ab4d4381540c4f94873ce1", "bc27636b81994a65b534df6141dbb21e", "b0dcf7d25b0d463d81c48c3de793bb99", "fb7e2d88cc274ba0af5955f2b34b664c", "c0cd0d24692c4f89ab9a2837b697626b", "264ac23136204d12a3edad31e425f84f", "4240a4413aa74617a339dd826cc4d72a", "17edd02002244014b258fb9138e8650f", "21eecd741a2e40b7b6dfb502cd5fea5e", "cc4599b1a3744e0b8909f0022fcb7ed3", "aac87e167647481bb8938f8ce1e7348c", "fed56053c7e34456980c67d7b2dcc9b9", "c9e055ccecde49d28e7079080587bf27", "9280ee5fe39541b7bfc29bf437e86af4", "a11e1b063ff146d4a1a134a4cd8678f5", "6a5ff3bae70249f58782fae1f8e056b4", "5c3bd723c6064144a923e8bed92bdfa1", "5f920eb0cb2646b990f5ad77aac84eea", "0a70fbcf00ea4fe387be568ed6217fb9", "cf554ca8867f4c029d827b7dff01c5cc", "1eb5bf3fd0344c0aaf43f08649e83bfd", "b5f2cbfff0f04a22b321598c91dbfdf7", "58eb44c6bbd6400284f8cedb38155cdc", "4b8ec325f9fd435b96333a6c89b0dab9", "22eb85e22e964be1acbe5de02f51f88e", "4924dc6f9f854a189dbe6671dbb871d3", "977af1cdc4654844b45532ea06fc3c6f", "01bb26c2b860437cbd301ef8cd5b5294", "56667e43acea4bb5b5e1b78ad74199d6", "e4cb9ed433fe4b0ca76c7c6fb9147336", "0b72374874dc42818cd70d099f64c21e", "bc3a5bb886684e5aada6ca85e4ce03a5", "ca0bf29729ac4620ba3608a2bdbf0c7a", "3015c6ca8a364905a36885d63055d128", "e0d9e4b1839641cd9b0ba190620ecf6a", "b95bc499c71b4929a0b32e78fa5e94e8", "3ad78c5f3d0a4349a72efad67df2aa45"]} id="710d4e3e" outputId="2ced0f67-3adf-492d-efac-a38ccdfd8332"
min_acc = 0
early_stop = 100
early_stop_cnt = 0
save_threshold = 0.002
path_nst = 'model_nst.pth'

# "cuda" only when GPUs are available.
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified.
model = Classifier().to(device)
PATH = 'model_nst.pth'
model.load_state_dict(torch.load(PATH))
model.device = device

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# The number of training epochs.
n_epochs = 200

# Whether to do semi-supervised learning.
do_semi = True

for epoch in range(n_epochs):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
    if do_semi and min_acc>0.7:
        # Obtain pseudo-labels for unlabeled data using trained model.
        pseudo_set = get_pseudo_labels(unlabeled_set, model)

        # Construct a new dataset and a data loader for training.
        # This is used in semi-supervised learning only.
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")
    
    if valid_acc - min_acc >= save_threshold:
        min_acc = valid_acc
        print('Saving model!')
        torch.save(model.state_dict(), path_nst)  # Save model to specified path
        early_stop_cnt = 0
    else:
        early_stop_cnt += 1
        print('Early stop count:', early_stop_cnt)

    if early_stop_cnt > early_stop:
        print('Early stop!')
        break

# %% [markdown] id="c6e73214"
# ## Testing
# For inference, we need to make sure the model is in eval mode, and the order of the dataset should not be shuffled ("shuffle=False" in test_loader).

# %% colab={"base_uri": "https://localhost:8080/", "height": 49, "referenced_widgets": ["9937821901e74608a9cf54b1ebbc4a5d", "df363d1c814f4ff1b81c6e65ea9d1fe7", "ed0fc3695dc7406ea67b0924acb0a6f7", "442249ac23164992b0e405721d3b21e1", "37dff1b0c27c4770b0b69224ced69423", "d26d1d9b8b8747668e502dda25e5bec3", "26e5ca65a6314e7eb90eff81d3138245", "87fe32fd14e74e7ca6880d0ed02f81f1", "1e0aeb0f78e744cf85c071b0c6764459", "c56a34703fae4b2b89af4287a2d61f13", "8e3dc307498c4116a5bd75df85d2cbc7"]} id="aa288a1b" outputId="2c15e577-524f-4a67-ccd7-7279bd2ee95e"
# Make sure the model is in eval mode.
# Some modules like Dropout or BatchNorm affect if the model is in training mode.
# model.load_state_dict(torch.load("model.ckpt", map_location=lambda storage, loc: storage))
# model.device = device

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = Classifier().to(device)
loadpath = 'model_nst.pth'
model.load_state_dict(torch.load(loadpath))

model.eval()

# Initialize a list to store the predictions.
predictions = []

# Iterate the testing set by batches.
for batch in tqdm(test_loader):
    # A batch consists of image data and corresponding labels.
    # But here the variable "labels" is useless since we do not have the ground-truth.
    # If printing out the labels, you will find that it is always 0.
    # This is because the wrapper (DatasetFolder) returns images and labels for each batch,
    # so we have to create fake labels to make it work normally.
    imgs, labels = batch

    # We don't need gradient in testing, and we don't even have labels to compute loss.
    # Using torch.no_grad() accelerates the forward process.
    with torch.no_grad():
        logits = model(imgs.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

# %% id="92607fcd"
# Save predictions into the file.
with open("predict.csv", "w") as f:

    # The first row must be "Id, Category"
    f.write("Id,Category\n")

    # For the rest of the rows, each image id corresponds to a predicted class.
    for i, pred in  enumerate(predictions):
         f.write(f"{i},{pred}\n")

# %% [markdown] id="c8b0acb3"
# ## Hints for better result
# * Design a better architecture
# * Adopt different data augmentations to improve the performance.
# * Utilize provided unlabeled data in training set

# %% id="6720af33"
