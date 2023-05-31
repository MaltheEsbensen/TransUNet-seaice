import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_size=768):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )

    def forward(self, x):
        x = self.projection(x)
        return x.transpose(1, 2)


class ViTBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super(ViTBlock, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

    def sinusoidal_position_embedding(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        b, seq_len, c = x.size()
        position_embedding = self.sinusoidal_position_embedding(seq_len, c).to(x.device)
        x = x + position_embedding
        x = self.norm1(x)
        x = x + self.dropout(self.attn(x, x, x)[0])
        x = self.norm2(x)
        x = x + self.dropout(self.ff(x))
        return x
    
class ViTBlock1(nn.Module):
    def __init__(self, d_model, nhead):
        super(ViTBlock1, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(0.1)

    def sinusoidal_position_embedding(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        b, seq_len, c = x.size()
        position_embedding = self.sinusoidal_position_embedding(seq_len, c).to(x.device)
        x = x + position_embedding
        x = self.norm1(x)
        x = x + self.dropout(self.attn(x, x, x)[0])
        x = self.norm2(x)
        x = x + self.dropout(self.ff(x))
        h = w = int(math.sqrt(seq_len))
        return x.reshape(b, c, h, w)


class UNetViT(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNetViT, self).__init__()

        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.encoder3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        self.encoder4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )

        # Vision Transformer
        self.patch_embedding = PatchEmbedding(128, 16, 128) #try 8 patches next
        self.vit = nn.Sequential(
            ViTBlock(128, 8),
            ViTBlock(128, 8),
            ViTBlock(128, 8),
            ViTBlock(128, 8),
            ViTBlock(128, 8),
            ViTBlock(128, 8),
            ViTBlock(128, 8),
            ViTBlock(128, 8),
            ViTBlock(128, 8),
            ViTBlock(128, 8),
            ViTBlock(128, 8),
            ViTBlock1(128, 8)
        )

        # Decoder

        self.decoder1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.decoder2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.decoder3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.decoder4 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, num_classes, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True)
        )


        self.sic_feature_map = FeatureMap(input_n=num_classes, output_n=12)
        self.sod_feature_map = FeatureMap(input_n=num_classes, output_n=11)
        self.floe_feature_map = FeatureMap(input_n=num_classes, output_n=7)

    def forward(self, x):
        if (x.size(2) or x.size(3)) < 513:
            b, c, h, w = x.size()
            
            x1 = self.encoder1(x)
            x2 = self.encoder2(x1)
            x3 = self.encoder3(x2)
            x4 = self.encoder4(x3)
                    
            if x4.size(2) < 33:
                while x4.size(2) < 33:
                    x4 = F.pad(x4, pad=(0, 0, 1, 0))
                    if x4.size(2) < 33:
                        x4 = F.pad(x4, pad=(0, 0, 0, 1))

            if x4.size(3) < 33:
                while x4.size(3) < 33:
                    x4 = F.pad(x4, pad=(1, 0, 0, 0))
                    if x4.size(3) < 33:
                        x4 = F.pad(x4, pad=(0, 1, 0, 0))
            

            x4 = self.patch_embedding(x4)

            x_vit = self.vit(x4)
            up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            x_vit = up1(x_vit)

            d1 = self.decoder1(x_vit)
            if d1.size(2) != x3.size(2) or d1.size(3) != x3.size(3):
                # Define the target size to which the input tensor will be upsampled or downsampled
                target_size = (x3.size(2), x3.size(3))
                # Use the interpolate function to resize the input tensor to the target size
                d1 = nn.functional.interpolate(d1, size=target_size, mode='nearest')

            d2 = self.decoder2(torch.cat((x3, d1), dim=1))

            if d2.size(2) != x2.size(2) or d2.size(3) != x2.size(3):
                # Define the target size to which the input tensor will be upsampled or downsampled
                target_size = (x2.size(2), x2.size(3))
                # Use the interpolate function to resize the input tensor to the target size
                d2 = nn.functional.interpolate(d2, size=target_size, mode='nearest')

            d3 = self.decoder3(torch.cat((x2, d2), dim=1))

            if d3.size(2) != x1.size(2) or d3.size(3) != x1.size(3):
                # Define the target size to which the input tensor will be upsampled or downsampled
                target_size = (x1.size(2), x1.size(3))
                # Use the interpolate function to resize the input tensor to the target size
                d3 = nn.functional.interpolate(d3, size=target_size, mode='nearest')

            d4 = self.decoder4(torch.cat((x1, d3), dim=1))

            if d4.size(2) != x.size(2) or d4.size(3) != x.size(3):
                # Define the target size to which the input tensor will be upsampled or downsampled
                target_size = (x.size(2), x.size(3))
                # Use the interpolate function to resize the input tensor to the target size
                x = nn.functional.interpolate(d4, size=target_size, mode='nearest')

            return {'SIC': self.sic_feature_map(x),
                    'SOD': self.sod_feature_map(x),
                    'FLOE': self.floe_feature_map(x)}
        else:
            # Split the input tensor into smaller chunks along the height and width dimensions
            tile_size = 512  # Adjust this value based on the available GPU memory
            h_splits = torch.split(x, tile_size, dim=2)
            tiles = [torch.split(h_split, tile_size, dim=3) for h_split in h_splits]

            outputs = []
            for h_row in tiles:
                output_row = []
                for tile in h_row:
                    b, c, h, w = tile.size()
                    
                    tilex1 = self.encoder1(tile)
                    tilex2 = self.encoder2(tilex1)
                    tilex3 = self.encoder3(tilex2)
                    tilex4 = self.encoder4(tilex3)
                    
                    if tilex4.size(2) < 33:
                        while tilex4.size(2) < 33:
                            tilex4 = F.pad(tilex4, pad=(0, 0, 1, 0))
                            if tilex4.size(2) < 33:
                                tilex4 = F.pad(tilex4, pad=(0, 0, 0, 1))

                    if tilex4.size(3) < 33:
                        while tilex4.size(3) < 33:
                            tilex4 = F.pad(tilex4, pad=(1, 0, 0, 0))
                            if tilex4.size(3) < 33:
                                tilex4 = F.pad(tilex4, pad=(0, 1, 0, 0))
                    
                    tilex4 = self.patch_embedding(tilex4)

                    tile_vit = self.vit(tilex4)
                    up1 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
                    tile_vit = up1(tile_vit)

                    tile_d1 = self.decoder1(tile_vit)

                    if tile_d1.size(2) != tilex3.size(2) or tile_d1.size(3) != tilex3.size(3):
                        # Define the target size to which the input tensor will be upsampled or downsampled
                        target_size = (tilex3.size(2), tilex3.size(3))
                        # Use the interpolate function to resize the input tensor to the target size
                        tile_d1 = nn.functional.interpolate(tile_d1, size=target_size, mode='nearest')

                    tile_d2 = self.decoder2(torch.cat((tilex3, tile_d1), dim=1))

                    if tile_d2.size(2) != tilex2.size(2) or tile_d2.size(3) != tilex2.size(3):
                        # Define the target size to which the input tensor will be upsampled or downsampled
                        target_size = (tilex2.size(2), tilex2.size(3))
                        # Use the interpolate function to resize the input tensor to the target size
                        tile_d2 = nn.functional.interpolate(tile_d2, size=target_size, mode='nearest')

                    tile_d3 = self.decoder3(torch.cat((tilex2, tile_d2), dim=1))

                    if tile_d3.size(2) != tilex1.size(2) or tile_d3.size(3) != tilex1.size(3):
                        # Define the target size to which the input tensor will be upsampled or downsampled
                        target_size = (tilex1.size(2), tilex1.size(3))
                        # Use the interpolate function to resize the input tensor to the target size
                        tile_d3 = nn.functional.interpolate(tile_d3, size=target_size, mode='nearest')

                    tile_d4 = self.decoder4(torch.cat((tilex1, tile_d3), dim=1))

                    if tile_d4.size(2) != tile.size(2) or tile_d4.size(3) != tile.size(3):
                        # Define the target size to which the input tensor will be upsampled or downsampled
                        target_size = (tile.size(2), tile.size(3))
                        # Use the interpolate function to resize the input tensor to the target size
                        tile_d4 = nn.functional.interpolate(tile_d4, size=target_size, mode='nearest')

                    tile_output = {'SIC': self.sic_feature_map(tile_d4),
                                   'SOD': self.sod_feature_map(tile_d4),
                                   'FLOE': self.floe_feature_map(tile_d4)}
                    output_row.append(tile_output)
                outputs.append(output_row)

            # Stitch the output tensors back together along the height and width dimensions
            final_output = {}
            for key in outputs[0][0].keys():
                stitched_rows = [torch.cat([tile[key] for tile in row], dim=3) for row in outputs]
                final_output[key] = torch.cat(stitched_rows, dim=2)

            return final_output

class FeatureMap(torch.nn.Module):
    """Class to perform final 1D convolution before calculating cross entropy or using softmax."""

    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()

        self.feature_out = torch.nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """Pass x through final layer."""
        return self.feature_out(x)