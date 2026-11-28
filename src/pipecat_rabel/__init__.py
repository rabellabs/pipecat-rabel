#
# Copyright (c) 2026, Rabel Labs Inc.
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Bell for Pipecat: start the reply on Bell's guess at the caller's words, before the caller finishes."""

__version__ = "0.4.0"

from pipecat_rabel.strategy import (  # noqa: E402
    RABEL_API_URL,
    BellStats,
    RabelBellUserTurnStopStrategy,
    RabelBellUserTurnStrategies,
)

__all__ = [
    "RABEL_API_URL",
    "BellStats",
    "RabelBellUserTurnStopStrategy",
    "RabelBellUserTurnStrategies",
    "__version__",
]
