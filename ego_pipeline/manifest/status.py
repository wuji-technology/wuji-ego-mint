
from enum import Enum


class Status(str, Enum):
    """Internal helper."""

    PENDING = 'pending'
    RUNNING = 'running'
    DONE    = 'done'
    FAILED  = 'failed'
    SKIPPED = 'skipped'

    @classmethod
    def terminal(cls) -> set['Status']:
        """Internal helper."""
        return {cls.DONE, cls.SKIPPED}
