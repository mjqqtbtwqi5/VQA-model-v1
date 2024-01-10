import zipfile
from pathlib import Path

import os
import torch
from torch import nn
from torch.utils.data import DataLoader

import pandas as pd


from timeit import default_timer as timer

from dataset import VideoDataset
from model import Model_V1, Model_V2
from engine import Engine


if __name__ == '__main__':
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # BATCH_SIZE = 32
    BATCH_SIZE = 1
    # NUM_WORKERS = os.cpu_count()
    NUM_WORKERS = 0
    NUM_EPOCHS = 1
    SEED = 22035001

    print(f"device: {DEVICE}, batch_size: {BATCH_SIZE}, num_worker: {NUM_WORKERS}, epochs: {NUM_EPOCHS}")

    SOURCE_PATH = Path("source/")
    DATA_PATH = Path("data/")

    # [KoNVid_1k] source files
    KONVID_1K_SOURCE_VIDEO_FILE = SOURCE_PATH / "KoNViD_1k" / "KoNViD_1k_videos.zip"
    KONVID_1K_SOURCE_SCORE_FILE = SOURCE_PATH / "KoNViD_1k" / "KoNViD_1k_metadata.zip"

    # [KoNVid_1k] data dir
    KONVID_1K_DATA_PATH = DATA_PATH / "KoNViD_1k"
    KONVID_1K_DATA_VIDEO_DIR = KONVID_1K_DATA_PATH / "KoNViD_1k_videos"
    KONVID_1K_DATA_SCORE_DIR = KONVID_1K_DATA_PATH / "KoNViD_1k_scores"

    # 1. Extract video data and score data
    # ==================================================
    # [KoNVid_1k] extract video data
    if KONVID_1K_DATA_VIDEO_DIR.is_dir():
        print(f"{KONVID_1K_DATA_VIDEO_DIR} directory exists.")
    else:
        with zipfile.ZipFile(KONVID_1K_SOURCE_VIDEO_FILE, "r") as zip_ref:
            print("Unzipping KoNVid_1k video data...")
            zip_ref.extractall(KONVID_1K_DATA_PATH)

    # [KoNVid_1k] extract score data
    if KONVID_1K_DATA_SCORE_DIR.is_dir():
        print(f"{KONVID_1K_DATA_SCORE_DIR} directory exists.")
    else:
        KONVID_1K_DATA_SCORE_DIR.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(KONVID_1K_SOURCE_SCORE_FILE, "r") as zip_ref:
            print("Unzipping KoNVid_1k score data...")
            zip_ref.extractall(KONVID_1K_DATA_SCORE_DIR)


    # 2. Prepare training data and testing data
    # ==================================================
    KONVID_1K_DATA_DF = pd.read_csv(KONVID_1K_DATA_SCORE_DIR / "KoNViD_1k_mos_1.csv")


    TRAIN_SPLIT = int(0.8 * len(KONVID_1K_DATA_DF))
    train_data_df = KONVID_1K_DATA_DF[:TRAIN_SPLIT]
    test_data_df = KONVID_1K_DATA_DF[TRAIN_SPLIT:]

    train_dataset = VideoDataset(video_dir=str(KONVID_1K_DATA_VIDEO_DIR),
                                video_ext="mp4",
                                targ_data_df=train_data_df)

    test_dataset = VideoDataset(video_dir=str(KONVID_1K_DATA_VIDEO_DIR),
                                video_ext="mp4",
                                targ_data_df=test_data_df)
    
    print(f"Number of training data: {len(train_dataset)} | Number of testing data: {len(test_dataset)}")

    train_dataloader = DataLoader(dataset=train_dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        shuffle=True)

    test_dataloader = DataLoader(dataset=test_dataset,
                                        batch_size=BATCH_SIZE,
                                        num_workers=NUM_WORKERS,
                                        shuffle=False)
    
    
    # 3. Train model
    # ==================================================

    torch.manual_seed(SEED) 
    torch.cuda.manual_seed(SEED)

    model = Model_V1(input_shape=3, # number of color channels (3 for RGB) 
                     hidden_units=32,
                    # output_shape=len(mos.shape).to(DEVICE))
                    output_shape=1).to(DEVICE)
    
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    engine = Engine(device=DEVICE,
                    epochs=NUM_EPOCHS)
    
    start_time = timer()
    model_results = engine.train(model=model,
                                 train_dataloader=train_dataloader,
                                 test_dataloader=test_dataloader,
                                 optimizer=optimizer,
                                 loss_fn=loss_fn)
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")

    

