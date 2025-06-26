from typing import Dict, Type

from src import logger
from src.network_state.gsl_attachment.gsl_attachment_interface import GSLAttachmentStrategy

log = logger.get_logger(__name__)


class GSLAttachmentFactory:
    """Factory for creating GSL attachment strategy objects."""

    _strategies: Dict[str, Type[GSLAttachmentStrategy]] = {}

    @classmethod
    def register_strategy(cls, strategy_class: Type[GSLAttachmentStrategy]) -> None:
        """
        Register a new strategy class with the factory.

        Args:
            strategy_class: The strategy class to register
        """
        strategy_instance = strategy_class()
        cls._strategies[strategy_instance.name()] = strategy_class
        log.debug(f"Registered GSL attachment strategy: {strategy_instance.name()}")

    @classmethod
    def get_strategy(cls, strategy_name: str) -> GSLAttachmentStrategy:
        """
        Create and return a GSL attachment strategy by name.

        Args:
            strategy_name: Name of the strategy to create

        Returns:
            An instance of the requested strategy

        Raises:
            ValueError: If the strategy name is not registered
        """
        if strategy_name not in cls._strategies:
            available = ", ".join(cls._strategies.keys())
            raise ValueError(
                f"Unknown GSL attachment strategy '{strategy_name}'. "
                f"Available strategies: {available}"
            )

        strategy_class = cls._strategies[strategy_name]
        return strategy_class()

    @classmethod
    def list_strategies(cls) -> list[str]:
        """Return list of all registered strategy names."""
        return list(cls._strategies.keys())
