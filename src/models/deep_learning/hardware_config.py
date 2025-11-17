"""
Hardware Detection and Configuration for Multi-Platform Support
================================================================
Automatically detects and configures optimal hardware backend:
- NVIDIA CUDA GPUs
- Apple Silicon (M1/M2/M3/M4) with MPS (Metal Performance Shaders)
- CPU fallback

Author: ForeWatt Team
Date: November 2025
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import platform

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Install with: pip install torch")


class HardwareConfig:
    """
    Detects and configures optimal hardware backend for deep learning.
    """

    def __init__(self, force_device: Optional[str] = None):
        """
        Initialize hardware configuration.

        Args:
            force_device: Force specific device ('cuda', 'mps', 'cpu')
                         If None, auto-detects best available
        """
        self.force_device = force_device
        self.device_type = None
        self.device = None
        self.device_name = None
        self.memory_gb = None
        self.is_cuda = False
        self.is_mps = False
        self.is_cpu = False
        self.platform_info = {}

        self._detect_hardware()
        self._configure_device()
        self._get_memory_info()
        self._optimize_settings()

    def _detect_hardware(self):
        """Detect available hardware and platform information."""
        # Platform information
        self.platform_info = {
            'system': platform.system(),
            'processor': platform.processor(),
            'machine': platform.machine(),
            'python_version': platform.python_version(),
            'platform': platform.platform()
        }

        logger.info(f"\n{'='*80}")
        logger.info("HARDWARE DETECTION")
        logger.info(f"{'='*80}")
        logger.info(f"System: {self.platform_info['system']}")
        logger.info(f"Processor: {self.platform_info['processor']}")
        logger.info(f"Machine: {self.platform_info['machine']}")
        logger.info(f"Platform: {self.platform_info['platform']}")

        if not PYTORCH_AVAILABLE:
            logger.error("PyTorch not available. Cannot detect GPU.")
            self.device_type = 'cpu'
            return

        # Check CUDA (NVIDIA)
        if torch.cuda.is_available():
            self.is_cuda = True
            logger.info(f"\n✓ NVIDIA CUDA Available")
            logger.info(f"  CUDA Version: {torch.version.cuda}")
            logger.info(f"  Device Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name}")
                logger.info(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
                logger.info(f"    Compute Capability: {props.major}.{props.minor}")

        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.is_mps = True
            logger.info(f"\n✓ Apple Silicon MPS Available")
            logger.info(f"  MPS Backend: Enabled")

            # Detect M-series chip
            if 'arm64' in self.platform_info['machine'].lower():
                logger.info(f"  Detected: Apple Silicon (M-series)")
                # Try to detect specific M-chip version
                try:
                    import subprocess
                    result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        cpu_brand = result.stdout.strip()
                        logger.info(f"  CPU: {cpu_brand}")
                except Exception:
                    pass

        # CPU fallback
        self.is_cpu = not (self.is_cuda or self.is_mps)
        if self.is_cpu:
            logger.info(f"\n⚠ Using CPU (no GPU detected)")
            import multiprocessing
            logger.info(f"  CPU Cores: {multiprocessing.cpu_count()}")

    def _configure_device(self):
        """Configure PyTorch device based on available hardware."""
        if not PYTORCH_AVAILABLE:
            self.device_type = 'cpu'
            self.device = None
            self.device_name = 'CPU'
            return

        # Force specific device if requested
        if self.force_device:
            device_map = {
                'cuda': (torch.cuda.is_available(), 'cuda'),
                'mps': (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(), 'mps'),
                'cpu': (True, 'cpu')
            }

            available, device_str = device_map.get(self.force_device.lower(), (False, 'cpu'))
            if not available:
                logger.warning(f"Requested device '{self.force_device}' not available. Falling back to CPU.")
                self.device = torch.device('cpu')
                self.device_type = 'cpu'
                self.device_name = 'CPU'
            else:
                self.device = torch.device(device_str)
                self.device_type = device_str
                self.device_name = self.force_device.upper()
        else:
            # Auto-detect best device
            if self.is_cuda:
                self.device = torch.device('cuda')
                self.device_type = 'cuda'
                self.device_name = torch.cuda.get_device_name(0)
            elif self.is_mps:
                self.device = torch.device('mps')
                self.device_type = 'mps'
                self.device_name = 'Apple Silicon MPS'
            else:
                self.device = torch.device('cpu')
                self.device_type = 'cpu'
                self.device_name = 'CPU'

        logger.info(f"\n{'='*80}")
        logger.info(f"SELECTED DEVICE: {self.device_type.upper()}")
        logger.info(f"Device Name: {self.device_name}")
        logger.info(f"PyTorch Device: {self.device}")
        logger.info(f"{'='*80}\n")

    def _get_memory_info(self):
        """Get available memory information."""
        if not PYTORCH_AVAILABLE:
            return

        if self.is_cuda:
            # CUDA memory
            props = torch.cuda.get_device_properties(0)
            self.memory_gb = props.total_memory / 1024**3
            logger.info(f"GPU Memory: {self.memory_gb:.2f} GB")

        elif self.is_mps:
            # Apple Silicon shared memory
            try:
                import subprocess
                result = subprocess.run(['sysctl', 'hw.memsize'],
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    mem_bytes = int(result.stdout.split(':')[1].strip())
                    self.memory_gb = mem_bytes / 1024**3
                    logger.info(f"System Memory: {self.memory_gb:.2f} GB (shared with GPU)")
            except Exception:
                self.memory_gb = 16.0  # Default assumption
                logger.warning(f"Could not detect memory. Assuming {self.memory_gb:.2f} GB")
        else:
            # CPU memory
            try:
                import psutil
                self.memory_gb = psutil.virtual_memory().total / 1024**3
                logger.info(f"System Memory: {self.memory_gb:.2f} GB")
            except ImportError:
                self.memory_gb = 16.0
                logger.warning(f"psutil not installed. Assuming {self.memory_gb:.2f} GB")

    def _optimize_settings(self):
        """Set optimal PyTorch settings for detected hardware."""
        if not PYTORCH_AVAILABLE:
            return

        if self.is_cuda:
            # CUDA optimizations
            torch.backends.cudnn.benchmark = True  # Auto-tune kernels
            torch.backends.cudnn.enabled = True
            logger.info("CUDA Optimizations:")
            logger.info("  ✓ cuDNN benchmark enabled")
            logger.info("  ✓ cuDNN enabled")

        elif self.is_mps:
            # MPS optimizations
            # Note: MPS is relatively new, fewer optimization flags
            logger.info("MPS Optimizations:")
            logger.info("  ✓ MPS backend enabled")
            logger.info("  ✓ Using Metal Performance Shaders")

            # Set fallback for unsupported ops
            import os
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            logger.info("  ✓ CPU fallback enabled for unsupported ops")

        else:
            # CPU optimizations
            import os

            # Use all CPU cores
            num_threads = os.cpu_count()
            torch.set_num_threads(num_threads)
            logger.info(f"CPU Optimizations:")
            logger.info(f"  ✓ Using {num_threads} threads")

    def get_recommended_batch_size(self, model_type: str = 'medium') -> int:
        """
        Get recommended batch size based on available memory.

        Args:
            model_type: 'small', 'medium', or 'large'

        Returns:
            Recommended batch size
        """
        # Base batch sizes for different model types
        base_batch_sizes = {
            'small': 128,   # Small models (e.g., simple LSTM)
            'medium': 64,   # Medium models (e.g., N-HiTS, TFT)
            'large': 32     # Large models (e.g., PatchTST with many layers)
        }

        base_size = base_batch_sizes.get(model_type, 64)

        # Adjust based on available memory
        if self.memory_gb:
            if self.memory_gb < 8:
                multiplier = 0.5
            elif self.memory_gb < 16:
                multiplier = 1.0
            elif self.memory_gb < 32:
                multiplier = 1.5
            else:
                multiplier = 2.0

            # MPS typically uses shared memory, be more conservative
            if self.is_mps:
                multiplier *= 0.75

            recommended = int(base_size * multiplier)
            # Ensure power of 2
            recommended = 2 ** (recommended.bit_length() - 1)

            return max(16, min(recommended, 256))  # Clamp between 16 and 256

        return base_size

    def get_recommended_num_workers(self) -> int:
        """
        Get recommended number of data loader workers.

        Returns:
            Number of workers
        """
        import os
        cpu_count = os.cpu_count() or 4

        if self.is_cuda:
            # CUDA: Use more workers (data loading on CPU, training on GPU)
            return min(cpu_count, 8)
        elif self.is_mps:
            # MPS: Shared memory, use fewer workers
            return min(cpu_count // 2, 4)
        else:
            # CPU: Fewer workers to avoid overhead
            return min(cpu_count // 2, 2)

    def get_device_config(self) -> Dict:
        """
        Get complete device configuration.

        Returns:
            Dictionary with device configuration
        """
        return {
            'device': self.device,
            'device_type': self.device_type,
            'device_name': self.device_name,
            'is_cuda': self.is_cuda,
            'is_mps': self.is_mps,
            'is_cpu': self.is_cpu,
            'memory_gb': self.memory_gb,
            'recommended_batch_size_small': self.get_recommended_batch_size('small'),
            'recommended_batch_size_medium': self.get_recommended_batch_size('medium'),
            'recommended_batch_size_large': self.get_recommended_batch_size('large'),
            'recommended_num_workers': self.get_recommended_num_workers(),
            'platform_info': self.platform_info
        }

    def print_summary(self):
        """Print hardware configuration summary."""
        config = self.get_device_config()

        print(f"\n{'='*80}")
        print("HARDWARE CONFIGURATION SUMMARY")
        print(f"{'='*80}")
        print(f"Device: {config['device_type'].upper()} ({config['device_name']})")
        print(f"Memory: {config['memory_gb']:.2f} GB")
        print(f"\nRecommended Settings:")
        print(f"  Batch Size (small models):  {config['recommended_batch_size_small']}")
        print(f"  Batch Size (medium models): {config['recommended_batch_size_medium']}")
        print(f"  Batch Size (large models):  {config['recommended_batch_size_large']}")
        print(f"  Data Loader Workers: {config['recommended_num_workers']}")
        print(f"\nPlatform:")
        print(f"  System: {config['platform_info']['system']}")
        print(f"  Machine: {config['platform_info']['machine']}")
        print(f"{'='*80}\n")

    def enable_memory_efficient_mode(self):
        """Enable memory-efficient settings."""
        if not PYTORCH_AVAILABLE:
            return

        logger.info("Enabling memory-efficient mode...")

        if self.is_cuda:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            logger.info("  ✓ CUDA cache cleared")

        # Enable gradient checkpointing flag (models need to implement)
        self.use_gradient_checkpointing = True
        logger.info("  ✓ Gradient checkpointing recommended")

        # Use mixed precision training flag
        self.use_mixed_precision = self.is_cuda or self.is_mps
        if self.use_mixed_precision:
            logger.info("  ✓ Mixed precision training enabled")


# Global hardware config instance
_hardware_config = None


def get_hardware_config(force_device: Optional[str] = None) -> HardwareConfig:
    """
    Get or create global hardware configuration.

    Args:
        force_device: Force specific device ('cuda', 'mps', 'cpu')

    Returns:
        HardwareConfig instance
    """
    global _hardware_config

    if _hardware_config is None or force_device is not None:
        _hardware_config = HardwareConfig(force_device=force_device)

    return _hardware_config


def test_hardware():
    """
    Test hardware configuration and PyTorch installation.

    Returns:
        True if tests pass
    """
    logger.info(f"\n{'█'*80}")
    logger.info("HARDWARE TEST")
    logger.info(f"{'█'*80}\n")

    # Test hardware detection
    hw = get_hardware_config()
    hw.print_summary()

    if not PYTORCH_AVAILABLE:
        logger.error("✗ PyTorch not installed")
        return False

    # Test tensor creation
    logger.info("Testing tensor operations...")
    try:
        # CPU test
        x_cpu = torch.randn(100, 100)
        y_cpu = torch.matmul(x_cpu, x_cpu.T)
        logger.info("  ✓ CPU tensor operations work")

        # GPU test
        if hw.is_cuda:
            x_cuda = torch.randn(100, 100, device='cuda')
            y_cuda = torch.matmul(x_cuda, x_cuda.T)
            y_cuda_cpu = y_cuda.cpu()
            logger.info("  ✓ CUDA tensor operations work")

            # Memory test
            torch.cuda.empty_cache()
            logger.info("  ✓ CUDA memory management works")

        elif hw.is_mps:
            x_mps = torch.randn(100, 100, device='mps')
            y_mps = torch.matmul(x_mps, x_mps.T)
            y_mps_cpu = y_mps.cpu()
            logger.info("  ✓ MPS tensor operations work")

        logger.info("\n✓ All hardware tests passed!")
        return True

    except Exception as e:
        logger.error(f"\n✗ Hardware test failed: {e}")
        return False


if __name__ == '__main__':
    # Run hardware test
    success = test_hardware()
    sys.exit(0 if success else 1)
