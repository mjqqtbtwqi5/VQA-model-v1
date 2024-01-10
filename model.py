import torch
import torch.nn as nn

class Model_V1(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:

        super().__init__()

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1036800,       #162000 648000 1036800
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))
    

class Model_V2(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:

        super().__init__()

        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.output_shape = output_shape

        # self.pretrain_x3d_xs = torch.hub.load('facebookresearch/pytorchvideo', 'efficient_x3d_xs', pretrained=True)

        self.conv_block_1 = nn.Sequential(
            nn.Conv3d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2)
        )

        self.flatten = nn.Flatten()
    
    def forward(self, x: torch.Tensor):
        print(f"x: {x.size()}")
        x = self.conv_block_1(x)
        print(f"x(Conv3d out): {x.size()}")
        # x = self.pretrain_x3d_xs(x)
        # print(f"x(pretrain_x3d_xs out): {x.size()}")
        x = self.flatten(x)
        print(f"x(Flatten out) x: {x.size()}")
        linear = nn.Linear(in_features=len(x[0]),
                           out_features=self.output_shape)
        return linear(x)