"""Autoencoder.
"""

import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url


from .BasicComponent import VGGConv, VGGMaxPool, Reshape, VGGMaxUnpool, \
    VGGConvTranspose


class Encoder(nn.Module):
    def __init__(self, embedding_size=3, num_classes=47):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            *VGGConv(3, 64),
            *VGGConv(64, 64),
            VGGMaxPool(),
            *VGGConv(64, 128),
            *VGGConv(128, 128),
            VGGMaxPool(),
            *VGGConv(128, 256),
            *VGGConv(256, 256),
            *VGGConv(256, 256),
            VGGMaxPool(),
            *VGGConv(256, 512),
            *VGGConv(512, 512),
            *VGGConv(512, 512),
            VGGMaxPool(),
            *VGGConv(512, 512),
            *VGGConv(512, 512),
            *VGGConv(512, 512),
            VGGMaxPool()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, embedding_size)
        )
        # self.load_state_dict(load_state_dict_from_url(
        #     'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
        #     progress=False))
        self.dfc = self.classifier
        del self.classifier

    def forward(self, inputs):
        output = self.features[0:6](inputs)
        output, ies_1 = self.features[6](output)

        output = self.features[7:13](output)
        output, ies_2 = self.features[13](output)

        output = self.features[14:23](output)
        output, ies_3 = self.features[23](output)

        output = self.features[24:33](output)
        output, ies_4 = self.features[33](output)

        output = self.features[34:43](output)
        output, ies_5 = self.features[43](output)

        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.dfc(output)
        return output, ies_1, ies_2, ies_3, ies_4, ies_5

    def get_embeddings(self, inputs):
        """Get embeddings form hidden space.
        """
        output = self.features[0:6](inputs)
        output, ies_1 = self.features[6](output)

        output = self.features[7:13](output)
        output, ies_2 = self.features[13](output)

        output = self.features[14:23](output)
        output, ies_3 = self.features[23](output)

        output = self.features[24:33](output)
        output, ies_4 = self.features[33](output)

        output = self.features[34:43](output)
        output, ies_5 = self.features[43](output)

        output = self.avgpool(output)
        output = output.view(output.size(0), -1)
        output = self.dfc(output)
        return output


class Decoder(nn.Module):
    def __init__(self, embedding_size=3):
        super(Decoder, self).__init__()
        self.dfc = nn.Sequential(
            nn.Linear(embedding_size, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 512 * 7 * 7),
            Reshape((512, 7, 7)),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # Layer 2
        self.decoder1 = VGGMaxUnpool()
        self.decoder2 = nn.Sequential(
            VGGConvTranspose(512, 512),
            VGGConvTranspose(512, 512),
            VGGConvTranspose(512, 512)
        )

        # Layer 3
        self.decoder3 = VGGMaxUnpool()
        self.decoder4 = nn.Sequential(
            VGGConvTranspose(512, 512),
            VGGConvTranspose(512, 512),
            VGGConvTranspose(512, 256)
        )

        # Layer 4
        self.decoder5 = VGGMaxUnpool()
        self.decoder6 = nn.Sequential(
            VGGConvTranspose(256, 256),
            VGGConvTranspose(256, 256),
            VGGConvTranspose(256, 128)
        )

        # Layer 5
        self.decoder7 = VGGMaxUnpool()
        self.decoder8 = nn.Sequential(
            VGGConvTranspose(128, 128),
            VGGConvTranspose(128, 128),
            VGGConvTranspose(128, 64)
        )

        # Layer 6
        self.decoder9 = VGGMaxUnpool()
        self.decoder10 = nn.Sequential(
            VGGConvTranspose(64, 64),
            VGGConvTranspose(64, 64),
            VGGConvTranspose(64, 3)
        )

    def forward(self, inputs, indicies):
        output = self.dfc(inputs)
        output = self.decoder1(output, indicies[4])
        output = self.decoder2(output)
        output = self.decoder3(output, indicies[3])
        output = self.decoder4(output)
        output = self.decoder5(output, indicies[2])
        output = self.decoder6(output)
        output = self.decoder7(output, indicies[1])
        output = self.decoder8(output)
        output = self.decoder9(output, indicies[0])
        output = self.decoder10(output)
        return output


class Autoencoder(nn.Module):
    """Autoencoder based on VGG16.
    """
    def __init__(self, embedding_size=3, num_classes=47,
                 freeze_encoder=False):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(embedding_size=embedding_size)
        self.decoder = Decoder(embedding_size=embedding_size)
        self.classifier = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(embedding_size, num_classes)
        )
        if freeze_encoder:
            self.freeze_encoder()

    def forward(self, inputs):
        """Forward.
        """
        # output, indicies_1, indicies_2, indicies_3, indicies_4, indicies_5 \
        #     = self.encoder(inputs, True)
        output, *indicies = self.encoder(inputs)
        preds = self.classifier(output)
        output = self.decoder(output, indicies)
        return output, preds

    def get_embeddings(self, inputs):
        """Get embeddings from hidden space in the encoder.
        """
        output = self.encoder.get_embeddings(inputs)
        return output

    def freeze_encoder(self):
        """Freeze parameters in encoder.
        """
        for param in self.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    import torch
    inputs = torch.rand(2, 3, 224, 224)
    net = Autoencoder()
    output, preds = net(inputs)
    print(output.size())
    print(preds.size())
