import os
import pickle
from dataclasses import dataclass
from math import floor, ceil
import numpy as np
import torch as t

@dataclass
class DatasetAudioTimeseries:
    train_data: t.Tensor
    train_labels: t.Tensor
    validation_data: t.Tensor
    validation_labels: t.Tensor
    test_data: t.Tensor
    test_labels: t.Tensor

def preprocess_audio_timeseries(audio_timeseries: np.ndarray) -> np.ndarray:
    audio_timeseries = audio_timeseries[::2]  # 44100Hz -> 22500 Hz
    target_length = 22500 * 4  # 4s
    diff = len(audio_timeseries) - target_length
    if diff > 0:  # Crop 
        audio_timeseries = audio_timeseries[floor(diff / 2): len(audio_timeseries) - ceil(diff / 2)]
    elif diff < 0:  # 0 pad
        audio_timeseries = np.pad(audio_timeseries, (floor(abs(diff) / 2), ceil(abs(diff) / 2)), 
                                  mode='constant', constant_values=0)
    assert len(audio_timeseries) == target_length 
    audio_timeseries /= np.max(np.abs(audio_timeseries))  # Normalize

    return audio_timeseries

def compute_dataset_audio_timeseries(path: str, urbansound8k) -> DatasetAudioTimeseries:
    # Load
    if os.path.exists(path):
        with open(path, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    train_data, train_labels, validation_data, val_labels, test_data, test_labels  = [], [], [], [], [], []
    for i, clip_id in enumerate(urbansound8k.clip_ids):
        if i % 50 == 0: 
            print(i)
        clip = urbansound8k.clip(clip_id)
        audio_timeseries = preprocess_audio_timeseries(clip.audio[0])
        fold = clip.fold
        if 1 <= fold <= 7:
            train_data.append(audio_timeseries)
            train_labels.append(clip.class_id)
        elif fold == 8:
            validation_data.append(audio_timeseries)
            val_labels.append(clip.class_id)
        elif fold in [9, 10]:
            test_data.append(audio_timeseries)
            test_labels.append(clip.class_id)
        else:
            raise Exception()
  
    train_data = t.tensor(np.array(train_data), dtype=t.float32)  # [n_samples, length_samples]
    train_labels = t.tensor(train_labels)  # [n_samples]
    validation_data = t.tensor(np.array(validation_data), dtype=t.float32)
    validation_labels = t.tensor(val_labels)
    test_data = t.tensor(np.array(test_data), dtype=t.float32)
    test_labels = t.tensor(test_labels)

    dataset = DatasetAudioTimeseries(
        train_data=train_data,
        train_labels=train_labels,
        validation_data=validation_data,
        validation_labels=validation_labels,
        test_data=test_data,
        test_labels=test_labels,
    )

    # Save
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)

    return dataset
