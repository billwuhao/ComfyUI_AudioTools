import os
import json
import librosa
import numpy as np
from typing import Any, Tuple
import scipy
import soundfile as sf
import torch
import random
from collections import defaultdict
from pytorch_lightning import LightningDataModule
# from pytorch_lightning.core.mixins import HyperparametersMixin
import torchaudio
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from typing import Any, Dict, Optional, Tuple
from pytorch_lightning.utilities import rank_zero_only

@rank_zero_only
def print_(message: str):
    print(message)


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

def find_bottom_directories(root_dir):
    bottom_directories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # 如果一个目录下没有子目录，则认为它是最底层的
        if not dirnames:
            bottom_directories.append(dirpath)
    return bottom_directories

def compute_mch_rms_dB(mch_wav, fs=16000, energy_thresh=-50):
    """Return the wav RMS calculated only in the active portions"""
    mean_square = max(1e-20, torch.mean(mch_wav ** 2))
    return 10 * np.log10(mean_square)
    
class Libri2MixDataset(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
    ) -> None:
        super().__init__()
        self.EPS = 1e-8
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        if n_src not in [1, 2]:
            raise ValueError("{} is not in [1, 2]".format(n_src))
        self.json_dir = json_dir
        self.sample_rate = sample_rate
        self.normalize_audio = normalize_audio
        
        if segment is None:
            self.seg_len = None
            self.fps_len = None
        else:
            self.seg_len = int(segment * sample_rate)
            
        self.n_src = n_src
        self.test = self.seg_len is None
        mix_json = os.path.join(json_dir, "mix_both.json")
        sources_json = [
            os.path.join(json_dir, source + ".json") for source in ["s1", "s2"]
        ]

        with open(mix_json, "r") as f:
            mix_infos = json.load(f)
        sources_infos = []
        for src_json in sources_json:
            with open(src_json, "r") as f:
                sources_infos.append(json.load(f))

        self.mix = []
        self.sources = []
        if self.n_src == 1:
            orig_len = len(mix_infos) * 2
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]
                    else:
                        for src_inf in sources_infos:
                            self.mix.append(mix_infos[i])
                            self.sources.append(src_inf[i])
            else:
                for i in range(len(mix_infos)):
                    for src_inf in sources_infos:
                        self.mix.append(mix_infos[i])
                        self.sources.append(src_inf[i])

            print_(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 3600, orig_len, self.seg_len
                )
            )
            self.length = len(self.mix)

        elif self.n_src == 2:
            orig_len = len(mix_infos)
            drop_utt, drop_len = 0, 0
            if not self.test:
                for i in range(len(mix_infos) - 1, -1, -1):  # Go backward
                    if mix_infos[i][1] < self.seg_len:
                        drop_utt = drop_utt + 1
                        drop_len = drop_len + mix_infos[i][1]
                        del mix_infos[i]
                        for src_inf in sources_infos:
                            del src_inf[i]

            print_(
                "Drop {} utts({:.2f} h) from {} (shorter than {} samples)".format(
                    drop_utt, drop_len / sample_rate / 36000, orig_len, self.seg_len
                )
            )
            self.mix = mix_infos
            self.sources = sources_infos
            self.length = len(self.mix)

    def __len__(self):
        return self.length

    def preprocess_audio_only(self, idx: int):
        if self.n_src == 1:
            if self.mix[idx][1] == self.seg_len or self.test:
                rand_start = 0
            else:
                rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len
            # Load mixture
            x, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            # Load sources
            s, _ = sf.read(
                self.sources[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            # torch from numpy
            target = torch.from_numpy(s)
            mixture = torch.from_numpy(x)
            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                target = normalize_tensor_wav(target, eps=self.EPS, std=m_std)
            return mixture, target.unsqueeze(0), self.mix[idx][0].split("/")[-1]
        # import pdb; pdb.set_trace()
        if self.n_src == 2:
            if self.mix[idx][1] == self.seg_len or self.test:
                rand_start = 0
            else:
                rand_start = np.random.randint(0, self.mix[idx][1] - self.seg_len)
            if self.test:
                stop = None
            else:
                stop = rand_start + self.seg_len
            # Load mixture
            x, _ = sf.read(
                self.mix[idx][0], start=rand_start, stop=stop, dtype="float32"
            )
            # Load sources
            source_arrays = []
            for src in self.sources:
                s, _ = sf.read(
                    src[idx][0], start=rand_start, stop=stop, dtype="float32"
                )
                source_arrays.append(s)
            sources = torch.from_numpy(np.vstack(source_arrays))
            mixture = torch.sum(sources, dim=0)

            if self.normalize_audio:
                m_std = mixture.std(-1, keepdim=True)
                mixture = normalize_tensor_wav(mixture, eps=self.EPS, std=m_std)
                sources = normalize_tensor_wav(sources, eps=self.EPS, std=m_std)

            return mixture, sources, self.mix[idx][0].split("/")[-1]

    def __getitem__(self, index: int):
        return self.preprocess_audio_only(index)

class Libri2MixModuleRemix(LightningDataModule):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        n_src: int = 2,
        sample_rate: int = 8000,
        segment: float = 4.0,
        normalize_audio: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = Libri2MixDataset(
                json_dir=self.hparams.train_dir,
                n_src=self.hparams.n_src,
                sample_rate=self.hparams.sample_rate,
                segment=self.hparams.segment,
                normalize_audio=self.hparams.normalize_audio,
            )
            self.data_val = Libri2MixDataset(
                json_dir=self.hparams.valid_dir,
                n_src=self.hparams.n_src,
                sample_rate=self.hparams.sample_rate,
                segment=None,
                normalize_audio=self.hparams.normalize_audio,
            )
            self.data_test = Libri2MixDataset(
                json_dir=self.hparams.test_dir,
                n_src=self.hparams.n_src,
                sample_rate=self.hparams.sample_rate,
                segment=None,
                normalize_audio=self.hparams.normalize_audio,
            )
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        
    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()
        