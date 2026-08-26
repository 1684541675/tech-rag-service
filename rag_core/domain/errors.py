"""Domain-specific errors."""


class InvalidRevisionTransition(ValueError):
    """Raised when a document revision attempts an unsupported state change."""
