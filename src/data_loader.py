"""Data loading utilities for ASR evaluation."""

import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional

from .text_normalizer import clean_reference_transcript


@dataclass
class AudioSample:
    """A single audio sample with metadata."""
    id: str
    audio_path: str
    reference_text: str  # Cleaned reference (no annotations)
    raw_reference: str   # Original reference with annotations
    emotion: Optional[str] = None


class EvalDataset:
    """Dataset loader for ASR evaluation.

    Loads audio samples and reference transcripts from CSV manifest.
    """

    def __init__(
        self,
        manifest_path: str,
        audio_dir: str,
        audio_extension: str = ".wav",
    ):
        """Initialize dataset.

        Args:
            manifest_path: Path to CSV manifest file
            audio_dir: Directory containing audio files
            audio_extension: Audio file extension (default: .wav)
        """
        self.manifest_path = Path(manifest_path)
        self.audio_dir = Path(audio_dir)
        self.audio_extension = audio_extension

        # Load manifest
        self.df = pd.read_csv(manifest_path)
        self._validate()

    def _validate(self):
        """Validate dataset structure."""
        required_cols = ['id', 'qa_edited_transcript']
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __iter__(self) -> Iterator[AudioSample]:
        """Iterate over all samples."""
        for idx in range(len(self)):
            yield self[idx]

    def __getitem__(self, idx: int) -> AudioSample:
        """Get a single sample by index."""
        row = self.df.iloc[idx]

        audio_path = self.audio_dir / f"{row['id']}{self.audio_extension}"

        return AudioSample(
            id=row['id'],
            audio_path=str(audio_path),
            reference_text=clean_reference_transcript(row['qa_edited_transcript']),
            raw_reference=row['qa_edited_transcript'],
            emotion=row.get('emotion'),
        )

    def get_by_id(self, sample_id: str) -> Optional[AudioSample]:
        """Get a sample by its ID."""
        matches = self.df[self.df['id'] == sample_id]
        if len(matches) == 0:
            return None

        idx = matches.index[0]
        return self[idx]

    def subset(self, n: int, random_seed: Optional[int] = None) -> 'EvalDataset':
        """Create a subset of the dataset.

        Args:
            n: Number of samples to include
            random_seed: Random seed for reproducibility

        Returns:
            New EvalDataset with subset of samples
        """
        if random_seed is not None:
            subset_df = self.df.sample(n=min(n, len(self.df)), random_state=random_seed)
        else:
            subset_df = self.df.head(n)

        # Create new instance with subset
        new_dataset = EvalDataset.__new__(EvalDataset)
        new_dataset.manifest_path = self.manifest_path
        new_dataset.audio_dir = self.audio_dir
        new_dataset.audio_extension = self.audio_extension
        new_dataset.df = subset_df.reset_index(drop=True)
        return new_dataset

    def get_ids(self) -> list[str]:
        """Get all sample IDs."""
        return self.df['id'].tolist()

    def summary(self) -> dict:
        """Get dataset summary statistics."""
        return {
            "num_samples": len(self),
            "manifest_path": str(self.manifest_path),
            "audio_dir": str(self.audio_dir),
            "emotions": self.df['emotion'].value_counts().to_dict() if 'emotion' in self.df.columns else {},
        }


def load_eval_dataset(
    data_dir: str = "data",
    dataset_name: str = "eval_10h",
) -> EvalDataset:
    """Convenience function to load the evaluation dataset.

    Args:
        data_dir: Base data directory
        dataset_name: Dataset name (e.g., "eval_10h", "eval")

    Returns:
        EvalDataset instance
    """
    data_dir = Path(data_dir)
    manifest_path = data_dir / f"{dataset_name}.csv"
    audio_dir = data_dir / dataset_name

    return EvalDataset(
        manifest_path=str(manifest_path),
        audio_dir=str(audio_dir),
    )
