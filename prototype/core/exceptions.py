"""
Spectra exception hierarchy.

All Spectra-specific errors derive from :class:`SpectraError` so callers
can catch a single base class while still distinguishing failure modes.
"""


class SpectraError(Exception):
    """Base class for all Spectra errors."""


class LoaderError(SpectraError):
    """Raised when a capture file cannot be loaded."""


class UnsupportedFormatError(LoaderError):
    """Raised for file formats or dtype combinations that are not supported."""


class SidecarError(LoaderError):
    """Raised when a metadata sidecar is missing, corrupt, or invalid."""


class PipelineError(SpectraError):
    """Raised when the end-to-end pipeline cannot complete."""


class DemodulationError(SpectraError):
    """Raised when demodulation cannot be performed for the given input."""


class ClassificationError(SpectraError):
    """Raised when classification cannot be performed."""


class FECError(SpectraError):
    """Raised for FEC encoding/decoding problems."""


class ExportError(SpectraError):
    """Raised when results cannot be serialized or exported."""


class ConfigurationError(SpectraError):
    """Raised when a configuration is invalid."""
