"""
ChatCompressionHandler Service for handling response compression.

This service handles compression for chat completion responses,
separating compression concerns from the main chat handler.
"""

import logging
import gzip
import zlib
import json
import time
from typing import Dict, Any, Optional, List, Tuple, Union, BinaryIO
from dataclasses import dataclass
from enum import Enum
from io import BytesIO

logger = logging.getLogger(__name__)


class CompressionType(Enum):
    """Types of compression."""

    GZIP = "gzip"
    DEFLATE = "deflate"
    BROTLI = "brotli"
    NONE = "none"


@dataclass
class CompressionConfig:
    """Configuration for compression."""

    enabled: bool = True
    compression_type: CompressionType = CompressionType.GZIP
    min_size_bytes: int = 1024  # Only compress if response is larger than this
    compression_level: int = 6  # 1-9 for gzip/deflate, 0-11 for brotli
    max_compressed_size: int = 10 * 1024 * 1024  # 10 MB max compressed size
    enable_compression_stats: bool = True


@dataclass
class CompressionStats:
    """Statistics for compression."""

    total_compressed: int = 0
    total_uncompressed: int = 0
    compression_ratio: float = 0.0
    compression_time: float = 0.0
    decompression_time: float = 0.0
    compression_errors: int = 0


class ChatCompressionHandler:
    """Service for handling response compression."""

    def __init__(self, config: Optional[CompressionConfig] = None):
        """Initialize the ChatCompressionHandler."""
        self.config = config or CompressionConfig()
        self.stats = CompressionStats()

        # Import brotli if available
        try:
            import brotli

            self._brotli_available = True
        except ImportError:
            self._brotli_available = False
            if self.config.compression_type == CompressionType.BROTLI:
                logger.warning(
                    "Brotli compression requested but brotli library not available, falling back to gzip"
                )

    def compress_response(self, response: Union[Dict[str, Any], str, bytes]) -> bytes:
        """
        Compress a response.

        Args:
            response: Response to compress (dict, str, or bytes)

        Returns:
            Compressed response as bytes
        """
        try:
            start_time = time.time()

            # Convert response to bytes
            if isinstance(response, dict):
                response_bytes = json.dumps(response, default=str).encode("utf-8")
            elif isinstance(response, str):
                response_bytes = response.encode("utf-8")
            elif isinstance(response, bytes):
                response_bytes = response
            else:
                raise ValueError(f"Unsupported response type: {type(response)}")

            uncompressed_size = len(response_bytes)

            # Check if compression is enabled and response is large enough
            if (
                not self.config.enabled
                or uncompressed_size < self.config.min_size_bytes
                or self.config.compression_type == CompressionType.NONE
            ):
                self.stats.total_uncompressed += 1
                return response_bytes

            # Compress based on type
            compressed_bytes = self._compress_bytes(response_bytes)

            if compressed_bytes is None:
                # Compression failed, return original
                self.stats.compression_errors += 1
                return response_bytes

            # Check if compression actually reduced size
            if len(compressed_bytes) >= uncompressed_size:
                # Compression didn't help, return original
                self.stats.total_uncompressed += 1
                return response_bytes

            # Check max compressed size
            if len(compressed_bytes) > self.config.max_compressed_size:
                logger.warning(
                    f"Compressed response too large: {len(compressed_bytes)} bytes"
                )
                self.stats.compression_errors += 1
                return response_bytes

            # Update statistics
            if self.config.enable_compression_stats:
                self._update_compression_stats(
                    uncompressed_size, len(compressed_bytes), time.time() - start_time
                )

            return compressed_bytes

        except Exception as e:
            logger.error(f"Error compressing response: {e}")
            self.stats.compression_errors += 1
            return response_bytes if "response_bytes" in locals() else b""

    def decompress_response(
        self, compressed_data: bytes, compression_type: Optional[CompressionType] = None
    ) -> bytes:
        """
        Decompress a response.

        Args:
            compressed_data: Compressed data as bytes
            compression_type: Type of compression used (optional)

        Returns:
            Decompressed response as bytes
        """
        try:
            start_time = time.time()

            if compression_type is None:
                compression_type = self.config.compression_type

            # Check if data is actually compressed
            if not self._is_compressed_data(compressed_data, compression_type):
                # Data is not compressed, return as-is
                return compressed_data

            # Decompress based on type
            decompressed_bytes = self._decompress_bytes(
                compressed_data, compression_type
            )

            if decompressed_bytes is None:
                raise ValueError("Failed to decompress data")

            # Update statistics
            if self.config.enable_compression_stats:
                self._update_decompression_stats(
                    len(compressed_data),
                    len(decompressed_bytes),
                    time.time() - start_time,
                )

            return decompressed_bytes

        except Exception as e:
            logger.error(f"Error decompressing response: {e}")
            return compressed_data

    def get_compression_headers(self, response_size: int) -> Dict[str, str]:
        """
        Get HTTP headers for compressed response.

        Args:
            response_size: Size of the response

        Returns:
            Dictionary of HTTP headers
        """
        headers = {}

        if self.config.enabled and response_size >= self.config.min_size_bytes:
            if self.config.compression_type == CompressionType.GZIP:
                headers["Content-Encoding"] = "gzip"
            elif self.config.compression_type == CompressionType.DEFLATE:
                headers["Content-Encoding"] = "deflate"
            elif self.config.compression_type == CompressionType.BROTLI:
                headers["Content-Encoding"] = "br"

        return headers

    def should_compress(self, response_size: int) -> bool:
        """
        Check if a response should be compressed.

        Args:
            response_size: Size of the response in bytes

        Returns:
            True if response should be compressed
        """
        return (
            self.config.enabled
            and response_size >= self.config.min_size_bytes
            and self.config.compression_type != CompressionType.NONE
        )

    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        try:
            # Calculate compression ratio
            if self.stats.total_compressed > 0:
                compression_ratio = (
                    self.stats.total_uncompressed / self.stats.total_compressed
                )
            else:
                compression_ratio = 0.0

            return {
                "total_compressed": self.stats.total_compressed,
                "total_uncompressed": self.stats.total_uncompressed,
                "compression_ratio": compression_ratio,
                "compression_time": self.stats.compression_time,
                "decompression_time": self.stats.decompression_time,
                "compression_errors": self.stats.compression_errors,
                "config": {
                    "enabled": self.config.enabled,
                    "compression_type": self.config.compression_type.value,
                    "min_size_bytes": self.config.min_size_bytes,
                    "compression_level": self.config.compression_level,
                    "max_compressed_size": self.config.max_compressed_size,
                    "enable_compression_stats": self.config.enable_compression_stats,
                },
                "brotli_available": self._brotli_available,
            }

        except Exception as e:
            logger.error(f"Error getting compression stats: {e}")
            return {"error": str(e)}

    def update_compression_config(self, config: CompressionConfig) -> None:
        """Update compression configuration."""
        self.config = config
        logger.info("Updated compression configuration")

    def reset_stats(self) -> None:
        """Reset compression statistics."""
        try:
            self.stats = CompressionStats()
            logger.info("Reset compression statistics")

        except Exception as e:
            logger.error(f"Error resetting compression stats: {e}")

    def get_compression_health(self) -> Dict[str, Any]:
        """Get compression health information."""
        try:
            stats = self.get_compression_stats()
            current_time = time.time()

            # Check for potential issues
            issues = []

            # Check compression ratio
            if stats["compression_ratio"] < 0.5:  # Poor compression ratio
                issues.append(
                    f"Poor compression ratio: {stats['compression_ratio']:.2f}"
                )

            # Check error rate
            total_operations = (
                stats["total_compressed"]
                + stats["total_uncompressed"]
                + stats["compression_errors"]
            )
            if total_operations > 0:
                error_rate = stats["compression_errors"] / total_operations
                if error_rate > 0.05:  # More than 5% error rate
                    issues.append(f"High compression error rate: {error_rate:.2%}")

            # Check if brotli is requested but not available
            if (
                self.config.compression_type == CompressionType.BROTLI
                and not self._brotli_available
            ):
                issues.append("Brotli compression requested but library not available")

            return {
                "status": "healthy" if not issues else "warning",
                "issues": issues,
                "stats": stats,
                "timestamp": current_time,
            }

        except Exception as e:
            logger.error(f"Error getting compression health: {e}")
            return {"status": "error", "error": str(e)}

    def _compress_bytes(self, data: bytes) -> Optional[bytes]:
        """Compress bytes using the configured compression type."""
        try:
            if self.config.compression_type == CompressionType.GZIP:
                return gzip.compress(data, compresslevel=self.config.compression_level)

            elif self.config.compression_type == CompressionType.DEFLATE:
                return zlib.compress(data, level=self.config.compression_level)

            elif self.config.compression_type == CompressionType.BROTLI:
                if not self._brotli_available:
                    logger.warning("Brotli not available, falling back to gzip")
                    return gzip.compress(
                        data, compresslevel=self.config.compression_level
                    )

                import brotli

                return brotli.compress(data, quality=self.config.compression_level)

            else:
                return None

        except Exception as e:
            logger.error(f"Error compressing bytes: {e}")
            return None

    def _decompress_bytes(
        self, data: bytes, compression_type: CompressionType
    ) -> Optional[bytes]:
        """Decompress bytes using the specified compression type."""
        try:
            if compression_type == CompressionType.GZIP:
                return gzip.decompress(data)

            elif compression_type == CompressionType.DEFLATE:
                return zlib.decompress(data)

            elif compression_type == CompressionType.BROTLI:
                if not self._brotli_available:
                    logger.warning("Brotli not available, trying gzip decompression")
                    return gzip.decompress(data)

                import brotli

                return brotli.decompress(data)

            else:
                return data

        except Exception as e:
            logger.error(f"Error decompressing bytes: {e}")
            return None

    def _is_compressed_data(
        self, data: bytes, compression_type: CompressionType
    ) -> bool:
        """Check if data is compressed using the specified compression type."""
        try:
            if len(data) < 2:
                return False

            if compression_type == CompressionType.GZIP:
                # GZIP magic number
                return data[:2] == b"\x1f\x8b"

            elif compression_type == CompressionType.DEFLATE:
                # Check if it starts with valid zlib header
                return len(data) > 0

            elif compression_type == CompressionType.BROTLI:
                # Brotli magic number
                return len(data) >= 4 and data[:4] == b"\x42\x52\x4f\x54"

            return False

        except Exception as e:
            logger.error(f"Error checking if data is compressed: {e}")
            return False

    def _update_compression_stats(
        self, uncompressed_size: int, compressed_size: int, compression_time: float
    ) -> None:
        """Update compression statistics."""
        self.stats.total_compressed += 1
        self.stats.total_uncompressed += uncompressed_size
        self.stats.compression_time += compression_time

    def _update_decompression_stats(
        self, compressed_size: int, uncompressed_size: int, decompression_time: float
    ) -> None:
        """Update decompression statistics."""
        self.stats.decompression_time += decompression_time

    def estimate_compression_ratio(
        self, sample_data: Union[Dict[str, Any], str, bytes]
    ) -> float:
        """
        Estimate compression ratio for sample data.

        Args:
            sample_data: Sample data to estimate compression ratio for

        Returns:
            Estimated compression ratio
        """
        try:
            # Convert to bytes
            if isinstance(sample_data, dict):
                sample_bytes = json.dumps(sample_data, default=str).encode("utf-8")
            elif isinstance(sample_data, str):
                sample_bytes = sample_data.encode("utf-8")
            else:
                sample_bytes = sample_data

            # Compress sample
            compressed_bytes = self._compress_bytes(sample_bytes)

            if compressed_bytes is None:
                return 1.0  # No compression

            return len(compressed_bytes) / len(sample_bytes)

        except Exception as e:
            logger.error(f"Error estimating compression ratio: {e}")
            return 1.0

    def get_compression_recommendations(self) -> List[str]:
        """Get recommendations for compression configuration."""
        try:
            stats = self.get_compression_stats()
            recommendations = []

            # Poor compression ratio
            if stats["compression_ratio"] < 0.7:
                recommendations.append(
                    "Consider adjusting compression level or type for better ratios"
                )

            # High error rate
            total_operations = (
                stats["total_compressed"]
                + stats["total_uncompressed"]
                + stats["compression_errors"]
            )
            if total_operations > 0:
                error_rate = stats["compression_errors"] / total_operations
                if error_rate > 0.01:  # More than 1%
                    recommendations.append(
                        "High compression error rate, check compression settings"
                    )

            # Large responses not being compressed
            if stats["total_uncompressed"] > 100 and self.config.min_size_bytes > 512:
                recommendations.append(
                    "Consider lowering min_size_bytes to compress more responses"
                )

            # No recommendations
            if not recommendations:
                recommendations.append("Compression configuration appears optimal")

            return recommendations

        except Exception as e:
            logger.error(f"Error getting compression recommendations: {e}")
            return ["Error calculating recommendations"]

    def stream_compress(self, data_stream: BinaryIO) -> BinaryIO:
        """
        Create a compressed stream from an input stream.

        Args:
            data_stream: Input stream to compress

        Returns:
            Compressed stream
        """
        try:
            if (
                not self.config.enabled
                or self.config.compression_type == CompressionType.NONE
            ):
                return data_stream

            # Read all data from stream
            data = data_stream.read()

            # Compress data
            compressed_data = self.compress_response(data)

            # Create new stream with compressed data
            return BytesIO(compressed_data)

        except Exception as e:
            logger.error(f"Error creating compressed stream: {e}")
            return data_stream

    def stream_decompress(
        self,
        compressed_stream: BinaryIO,
        compression_type: Optional[CompressionType] = None,
    ) -> BinaryIO:
        """
        Create a decompressed stream from a compressed stream.

        Args:
            compressed_stream: Compressed stream to decompress
            compression_type: Type of compression used (optional)

        Returns:
            Decompressed stream
        """
        try:
            if compression_type is None:
                compression_type = self.config.compression_type

            # Read all data from stream
            compressed_data = compressed_stream.read()

            # Decompress data
            decompressed_data = self.decompress_response(
                compressed_data, compression_type
            )

            # Create new stream with decompressed data
            return BytesIO(decompressed_data)

        except Exception as e:
            logger.error(f"Error creating decompressed stream: {e}")
            return compressed_stream

    def get_stats(self) -> Dict[str, Any]:
        """
        Get compression handler statistics.

        Returns:
            Dict containing compression statistics
        """
        return {
            "total_compressed": getattr(self, "total_compressed", 0),
            "total_uncompressed": getattr(self, "total_uncompressed", 0),
            "compression_ratio": getattr(self, "compression_ratio", 0.0),
            "compression_time": getattr(self, "compression_time", 0.0),
            "compression_errors": getattr(self, "compression_errors", 0),
            "timestamp": time.time(),
        }
