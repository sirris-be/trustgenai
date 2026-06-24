"""Public package exports for the FAIR Assistant agent."""


# =================================== EXPORTS ===================================

__all__ = ["fair_agent"]


def __getattr__(name: str):
	"""Lazily expose the main agent without importing it on package import."""
	if name == "fair_agent":
		from .agent import fair_agent

		return fair_agent
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
