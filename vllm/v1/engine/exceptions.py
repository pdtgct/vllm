# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus

from vllm.exceptions import GracefulHTTPError, VLLMServerError


class EngineGenerateError(VLLMServerError):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


class StreamAdmissionError(GracefulHTTPError):
    """The engine refused to admit a persistent-state stream, with `reason`
    `engine_paused` or `active_limit`. Returns HTTP 503 (Service Unavailable)
    so that clients retry, as for a full request queue."""

    def __init__(self, reason: str):
        super().__init__(
            f"Stream admission refused: {reason}", HTTPStatus.SERVICE_UNAVAILABLE
        )
        self.reason = reason


class EngineDeadError(VLLMServerError):
    """Raised when the EngineCore dies. Unrecoverable.

    `str(exc)` is the log-oriented message for server logs and offline
    tracebacks; `CLIENT_MESSAGE` is what API clients receive.
    """

    ENGINE_DEAD_MESSAGE = (
        "EngineCore encountered an issue. See stack trace (above) for the root cause."
    )
    CLIENT_MESSAGE = "The server had an error while processing your request."

    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        super().__init__(self.ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context
