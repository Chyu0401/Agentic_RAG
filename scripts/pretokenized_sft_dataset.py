"""
预分词 SFT Dataset：从 Parquet 读取已分词的 input_ids / attention_mask，供 VeRL SFT 使用。
与 jsonl_to_tokenized_parquet.py 产出的 train/val parquet 配套使用。
"""
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer



class PreTokenizedSFTDataset(Dataset):
    """
    从 Parquet 中读取已分词的 input_ids、attention_mask，直接用于 SFT。
    要求 Parquet 列：input_ids (list[int])、attention_mask (list[int])。
    loss_mask 默认与 attention_mask 一致（对所有非 pad 位置计算 loss）。
    """

    def __init__(
        self,
        parquet_files: Union[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor=None,
        max_samples: int = -1,
    ):
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ("right", "no_padding"), (
            f"Expect pad_mode 'right' or 'no_padding', got {self.pad_mode}"
        )
        self.max_length = config.get("max_length", 1024)
        self.max_samples = max_samples

        if not isinstance(parquet_files, (list, ListConfig)):
            parquet_files = [parquet_files]
        self.parquet_files = list(parquet_files)

        dfs = []
        for f in self.parquet_files:
            df = pd.read_parquet(f, dtype_backend="pyarrow")
            dfs.append(df)
        self.dataframe = pd.concat(dfs, ignore_index=True)

        if self.max_samples > 0 and len(self.dataframe) > self.max_samples:
            self.dataframe = self.dataframe.iloc[: self.max_samples].copy()
        assert "input_ids" in self.dataframe.columns and "attention_mask" in self.dataframe.columns, (
            "Parquet 需包含 input_ids 与 attention_mask 列"
        )
        print(f"PreTokenizedSFTDataset len={len(self.dataframe)}")

    def __len__(self) -> int:
        return len(self.dataframe)

    def _to_tensor(self, x):
        if isinstance(x, (list, tuple)):
            return torch.tensor(x, dtype=torch.long)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).long()
        if isinstance(x, torch.Tensor):
            return x.long()
        return torch.tensor(x, dtype=torch.long)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataframe.iloc[idx]
        input_ids = self._to_tensor(row["input_ids"])
        attention_mask = self._to_tensor(row["attention_mask"])
        seq_len = input_ids.shape[0]

        if self.pad_mode == "no_padding":
            if seq_len > self.max_length:
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                seq_len = self.max_length
            loss_mask = attention_mask.clone()
            position_ids = torch.arange(seq_len, dtype=torch.long)
            return {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
        else:
            # right padding：你方 parquet 已是定长，一般无需再 pad，但保持接口一致
            loss_mask = attention_mask.clone()
            position_ids = torch.arange(seq_len, dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
