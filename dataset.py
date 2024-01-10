import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.io import read_video
from torch import DoubleTensor, FloatTensor

from typing import Tuple
from pandas import DataFrame

from transform import transform_ImgNet

class VideoDataset(Dataset):

    def __init__(self, video_dir: str, video_ext: str, targ_data_df: DataFrame) -> None:
        self.video_dir = video_dir
        self.video_format = video_ext
        self.targ_data_df = targ_data_df
        self.transform = None
        # self.transform = transform_ImgNet()

    def load_video_mos(self, index: int):

        row = self.targ_data_df.iloc[index]

        flickr_id = str(int(row['flickr_id']))
        mos = float(row['mos'])

        video_path = f"{self.video_dir}/{flickr_id}.{self.video_format}"
        video, _, _ = read_video(filename=video_path, 
                                  pts_unit="sec", 
                                  output_format="TCHW")

        if self.transform:
            for frame_idx in range(len(video)):
                frame = video[frame_idx]
                frame = self.transform(frame)
                video[frame_idx] = frame

        # video = torch.permute(video, (1, 0, 2, 3)).type(FloatTensor)
        video = video.type(FloatTensor)
        return video, mos
    
    def __len__(self) -> int:
        return len(self.targ_data_df)
    
    def __getitem__(self, index: int) -> Tuple[Tensor, float]:
        return self.load_video_mos(index)