#!/usr/bin/env python3
"""
Optimized DCSASS Dataset Data Loader
High-performance version with caching, parallel processing, and memory optimization
"""
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import logging
from concurrent.futures import ThreadPoolExecutor
import pickle
import time
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedDCSASSVideoDataset(Dataset):
    """
    Optimized dataset class for DCSASS video clips with caching and parallel processing
    """
    
    def __init__(
        self, 
        data_root: str = "data/DCSASS Dataset/",
        crime_types: Optional[List[str]] = None,
        max_videos_per_crime: int = 1,
        clip_duration: float = 2.0,
        target_size: Tuple[int, int] = (224, 224),
        transform=None,
        random_seed: int = 42,
        cache_dir: str = "cache",
        use_cache: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize Optimized DCSASS Dataset
        
        Args:
            data_root: Path to DCSASS Dataset directory
            crime_types: List of crime types to include (None = all)
            max_videos_per_crime: Always 1 - randomly select one video folder per crime type
            clip_duration: Duration of existing clips (2.0 seconds)
            target_size: Target frame size (height, width)
            transform: PyTorch transforms to apply
            random_seed: Random seed for reproducibility
            cache_dir: Directory to store cached frames
            use_cache: Whether to use frame caching
            max_workers: Number of worker threads for parallel processing
        """
        self.data_root = Path(data_root)
        self.clip_duration = clip_duration
        self.target_size = target_size
        self.transform = transform or self._get_default_transform()
        self.random_seed = random_seed
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.max_workers = max_workers

        if data_root is None:
            data_root = Path(__file__).parent.parent / "data" / "DCSASS Dataset"
        self.data_root = Path(data_root)
        
        # Create cache directory
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Get available crime types
        if crime_types is None:
            self.crime_types = [d.name for d in self.data_root.iterdir() 
                              if d.is_dir() and not d.name.startswith('.')]
        else:
            self.crime_types = crime_types
            
        logger.info(f"Found crime types: {self.crime_types}")
        
        # Build dataset with parallel processing
        self.samples = self._build_dataset_parallel(max_videos_per_crime)
        logger.info(f"Dataset built with {len(self.samples)} samples")
        
    def _get_default_transform(self):
        """Get optimized transforms for video frames"""
        return transforms.Compose([
            transforms.Resize(self.target_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _build_dataset_parallel(self, max_videos_per_crime: int) -> List[Dict]:
        """Build dataset with parallel processing and proper shuffling"""
        samples = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for crime_type in self.crime_types:
                future = executor.submit(self._process_crime_type, crime_type)
                futures.append(future)
            
            # Collect results
            for future in futures:
                crime_samples = future.result()
                samples.extend(crime_samples)
        
        # Shuffle the final samples to ensure proper randomization
        random.shuffle(samples)
        logger.info(f"Dataset shuffled with {len(samples)} samples")
                
        return samples
    
    def _process_crime_type(self, crime_type: str) -> List[Dict]:
        """Process a single crime type (runs in parallel)"""
        crime_dir = self.data_root / crime_type
        
        if not crime_dir.exists():
            logger.warning(f"Crime directory not found: {crime_dir}")
            return []
            
        # Get all video folders for this crime type
        video_folders = [d for d in crime_dir.iterdir() 
                       if d.is_dir() and d.name.endswith('.mp4')]
        
        if not video_folders:
            logger.warning(f"No video folders found for {crime_type}")
            return []
        
        # Randomly select ONE video folder per crime type
        selected_folder = random.choice(video_folders)
        
        logger.info(f"Selected video folder for {crime_type}: {selected_folder.name}")
        
        # Get ALL clips from the selected video folder
        clips = self._get_clips_from_folder_fast(selected_folder, crime_type)
        logger.info(f"  Loaded {len(clips)} clips from {selected_folder.name}")
        
        return clips
    
    def _get_clips_from_folder_fast(self, video_folder: Path, crime_type: str) -> List[Dict]:
        """Fast clip extraction with caching"""
        clips = []
        
        # Get all video files in the folder
        video_files = [f for f in video_folder.iterdir() 
                      if f.suffix == '.mp4']
        
        for video_file in video_files:
            clip_info = {
                'video_path': str(video_file),
                'crime_type': crime_type,
                'video_folder': video_folder.name,
                'clip_name': video_file.name,
                'label': self._get_crime_label(crime_type),
                'cache_key': f"{crime_type}_{video_folder.name}_{video_file.name}"
            }
            clips.append(clip_info)
            
        return clips
    
    def _get_crime_label(self, crime_type: str) -> int:
        """Convert crime type to numeric label"""
        crime_to_label = {
            'Abuse': 0,
            'Arrest': 1, 
            'Arson': 2,
            'Assault': 3,
            'Burglary': 4,
            'Explosion': 5,
            'Fighting': 6,
            'RoadAccidents': 7,
            'Robbery': 8,
            'Shooting': 9,
            'Shoplifting': 10,
            'Stealing': 11,
            'Vandalism': 12
        }
        return crime_to_label.get(crime_type, -1)
    
    @lru_cache(maxsize=1000)
    def _load_video_clip_cached(self, video_path: str, cache_key: str) -> np.ndarray:
        """Load video clip with LRU caching"""
        # Check if cached version exists
        if self.use_cache:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        return pickle.load(f)
                except:
                    pass  # If cache is corrupted, load normally
        
        # Load video clip
        frames = self._load_video_clip_fast(video_path)
        
        # Cache the result
        if self.use_cache and frames.size > 0:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(frames, f)
            except:
                pass  # If caching fails, continue without cache
        
        return frames
    
    def _load_video_clip_fast(self, video_path: str) -> np.ndarray:
        """Optimized video loading with efficient frame extraction"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        if not cap.isOpened():
            logger.warning(f"Could not open video: {video_path}")
            return np.array([])
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            return np.array([])
        
        # Calculate frame indices for 15 frames
        target_frames = 15
        if total_frames <= target_frames:
            # If video is shorter, read all frames
            frame_indices = list(range(total_frames))
        else:
            # Sample frames evenly
            frame_indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
        
        # Read frames efficiently
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if ret:
                # Convert BGR to RGB and resize in one step
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            logger.warning(f"No frames extracted from: {video_path}")
            return np.array([])
            
        return np.array(frames)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def remap_label(self, original_label: int) -> int:
        reverse_labels = {3:'Assault', 8:'Robbery', 10:'Shoplifting', 9:'Shooting', 11:'Stealing'} # else: 'Normal'
        if original_label in reverse_labels:
            return 1
        else:
            return 0
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset with optimized loading"""
        sample = self.samples[idx]
        
        # Load video clip with caching
        frames = self._load_video_clip_cached(
            sample['video_path'], 
            sample['cache_key']
        )
        
        if frames.size == 0:
            # Return dummy data if video couldn't be loaded
            dummy_frames = np.zeros((15, *self.target_size, 3), dtype=np.uint8)
            frames = dummy_frames
        
        # Convert to tensor and apply transforms (optimized)
        frame_tensors = []
        for frame in frames:
            # Skip PIL conversion for speed - use direct tensor conversion
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            frame_tensor = self.transform.transforms[0](frame_tensor)  # Resize
            frame_tensor = self.transform.transforms[2](frame_tensor)   # Normalize
            frame_tensors.append(frame_tensor)
        
        # Stack frames into video tensor
        video_tensor = torch.stack(frame_tensors)
        
        # Ensure exactly 15 frames
        if video_tensor.shape[0] < 15:
            # Pad with last frame
            padding = video_tensor[-1:].repeat(15 - video_tensor.shape[0], 1, 1, 1)
            video_tensor = torch.cat([video_tensor, padding], dim=0)
        elif video_tensor.shape[0] > 15:
            # Truncate to 15 frames
            video_tensor = video_tensor[:15]
        
        return video_tensor, self.remap_label(sample['label'])


class OptimizedDCSASSDataLoader:
    """
    Optimized data loader class for DCSASS Dataset with performance improvements
    """
    
    def __init__(
        self,
        data_root: str = "data/DCSASS Dataset/",
        batch_size: int = 1,
        num_workers: int = 4,
        train_split: float = 0.8,
        val_split: float = 0.1,
        test_split: float = 0.1,
        max_videos_per_crime: int = 1,
        clip_duration: float = 2.0,
        target_size: Tuple[int, int] = (224, 224),
        random_seed: int = 42,
        use_cache: bool = True,
        cache_dir: str = "cache",
        pin_memory: bool = True,
        persistent_workers: bool = True
    ):
        """
        Initialize Optimized DCSASS Data Loader
        
        Args:
            data_root: Path to DCSASS Dataset
            batch_size: Batch size for DataLoader
            num_workers: Number of worker processes (optimized)
            train_split: Fraction of data for training
            val_split: Fraction of data for validation
            test_split: Fraction of data for testing
            max_videos_per_crime: Max videos per crime type
            clip_duration: Duration of each clip in seconds
            target_size: Target frame size
            random_seed: Random seed for reproducibility
            use_cache: Whether to use frame caching
            cache_dir: Directory for caching
            pin_memory: Use pinned memory for faster GPU transfer
            persistent_workers: Keep workers alive between epochs
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        
        # Create optimized dataset
        self.dataset = OptimizedDCSASSVideoDataset(
            data_root=data_root,
            max_videos_per_crime=max_videos_per_crime,
            clip_duration=clip_duration,
            target_size=target_size,
            random_seed=random_seed,
            use_cache=use_cache,
            cache_dir=cache_dir,
            max_workers=num_workers
        )
        
        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = self._split_dataset(
            train_split, val_split, test_split
        )
        
        # Create optimized data loaders
        self.train_loader = self._create_optimized_data_loader(
            self.train_dataset, shuffle=True, pin_memory=pin_memory, persistent_workers=persistent_workers
        )
        self.val_loader = self._create_optimized_data_loader(
            self.val_dataset, shuffle=True, pin_memory=pin_memory, persistent_workers=persistent_workers
        )
        self.test_loader = self._create_optimized_data_loader(
            self.test_dataset, shuffle=True, pin_memory=pin_memory, persistent_workers=persistent_workers
        )
        
        logger.info(f"Optimized data loaders created:")
        logger.info(f"  Train: {len(self.train_dataset)} samples")
        logger.info(f"  Val: {len(self.val_dataset)} samples")
        logger.info(f"  Test: {len(self.test_dataset)} samples")
        logger.info(f"  Workers: {num_workers}")
        logger.info(f"  Cache: {'Enabled' if use_cache else 'Disabled'}")
    
    def _split_dataset(self, train_split: float, val_split: float, test_split: float):
        """Split dataset into train/val/test with proper shuffling"""
        total_samples = len(self.dataset)
        
        train_size = int(total_samples * train_split)
        val_size = int(total_samples * val_split)
        test_size = total_samples - train_size - val_size
        
        # Create indices for splitting with proper shuffling
        indices = list(range(total_samples))
        
        # Use different random state for shuffling to ensure proper randomization
        import random
        random_state = random.getstate()
        random.seed(self.random_seed)
        random.shuffle(indices)
        random.setstate(random_state)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
        val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
        test_dataset = torch.utils.data.Subset(self.dataset, test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_optimized_data_loader(self, dataset, shuffle: bool = True, pin_memory: bool = True, persistent_workers: bool = True) -> DataLoader:
        """Create optimized PyTorch DataLoader"""
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            persistent_workers=persistent_workers,
            prefetch_factor=2,  # Prefetch 2 batches per worker
            multiprocessing_context='spawn' if os.name == 'nt' else 'fork'
        )
    
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset"""
        crime_counts = {}
        for sample in self.dataset.samples:
            crime_type = sample['crime_type']
            crime_counts[crime_type] = crime_counts.get(crime_type, 0) + 1
        
        return {
            'total_samples': len(self.dataset),
            'crime_type_counts': crime_counts,
            'num_crime_types': len(crime_counts),
            'batch_size': self.batch_size,
            'clip_duration': self.dataset.clip_duration,
            'target_size': self.dataset.target_size,
            'cache_enabled': self.dataset.use_cache,
            'workers': self.num_workers
        }
    
    def clear_cache(self):
        """Clear the frame cache"""
        if self.dataset.use_cache and self.dataset.cache_dir.exists():
            import shutil
            shutil.rmtree(self.dataset.cache_dir)
            self.dataset.cache_dir.mkdir(exist_ok=True)
            logger.info("Cache cleared")


def test_optimized_data_loader():
    """Test the optimized data loader"""
    logger.info("🚀 Testing Optimized DCSASS Data Loader...")
    
    try:
        # Create optimized data loader
        data_loader = OptimizedDCSASSDataLoader(
            data_root="data/DCSASS Dataset/",
            batch_size=8,
            num_workers=4,
            max_videos_per_crime=1,
            clip_duration=2.0,
            target_size=(224, 224),
            use_cache=True,
            pin_memory=True,
            persistent_workers=True
        )
        
        # Get dataset info
        info = data_loader.get_dataset_info()
        logger.info(f"✅ Optimized dataset loaded successfully!")
        logger.info(f"   Total samples: {info['total_samples']}")
        logger.info(f"   Crime types: {info['num_crime_types']}")
        logger.info(f"   Batch size: {info['batch_size']}")
        logger.info(f"   Cache enabled: {info['cache_enabled']}")
        logger.info(f"   Workers: {info['workers']}")
        
        # Test loading speed
        logger.info("🔄 Testing batch loading speed...")
        start_time = time.time()
        
        for batch_idx, (videos, labels) in enumerate(data_loader.train_loader):
            logger.info(f"   Batch {batch_idx + 1}: {videos.shape}, {labels.shape}")
            if batch_idx >= 2:  # Test first 3 batches
                break
        
        end_time = time.time()
        logger.info(f"✅ Speed test completed in {end_time - start_time:.2f} seconds!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Optimized data loader test failed: {e}")
        return False


if __name__ == "__main__":
    # Test the optimized data loader
    test_optimized_data_loader()
