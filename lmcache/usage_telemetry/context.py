# SPDX-License-Identifier: Apache-2.0
"""Context reporting: snapshot messages describing a deployment.

Context reports are sent once each (at startup, or at KV-cache registration
for the MP path in :mod:`.mp`), in contrast to the periodic interval
counters in :mod:`.continuous`.
"""

# Future
from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.usage_telemetry.env_probe import collect_env_message
from lmcache.usage_telemetry.guard import swallow_telemetry_errors
from lmcache.usage_telemetry.identity import (
    get_usage_identity,
    is_usage_tracking_enabled,
)
from lmcache.usage_telemetry.messages import (
    DeploymentMode,
    EngineMessage,
    MetadataMessage,
    UsageMessage,
)
from lmcache.usage_telemetry.transport import (
    DEFAULT_SENDER,
    UsageMessageSender,
    build_usage_payload,
    usage_server_url,
)
from lmcache.v1.lifecycle import RetainedCall
from lmcache.v1.mp_observability.errors import LMCacheTimeoutError

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata

logger = init_logger(__name__)


class UsageContextBase(ABC):
    """Shared plumbing for one-shot usage reporting.

    Subclasses define which messages the report contains via
    :meth:`_collect_messages`; this base owns the identity, transport, and
    optional local logging. Each message is POSTed to the endpoint it
    declares in the schema (:mod:`lmcache.usage_telemetry.messages`).
    """

    def __init__(
        self,
        mode: DeploymentMode,
        local_log: str | None,
        sender: UsageMessageSender | None,
    ) -> None:
        """Initialize shared reporting state.

        Args:
            mode: Deployment mode stamped on every payload this reporter
                sends.
            local_log: Path of a human-readable local log of every sent
                payload; ``None`` disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        self._mode = mode
        self._local_log = local_log
        self._sender = sender if sender is not None else DEFAULT_SENDER
        self._identity = get_usage_identity()
        self._start_time = datetime.now()

    @abstractmethod
    def _collect_messages(self) -> list[UsageMessage]:
        """Return the messages of the one-shot report."""
        raise NotImplementedError

    @swallow_telemetry_errors
    def report_once(self) -> None:
        """Collect and send every one-shot message on the calling thread.

        Failures are swallowed; never raises.
        """
        for message in self._collect_messages():
            payload = build_usage_payload(message, self._identity, self._mode)
            self._sender.send(usage_server_url(message.ENDPOINT), payload)
            self._write_local(payload)

    def _write_local(self, payload: dict[str, object]) -> None:
        """Append *payload* to the local log file, if one is configured."""
        if self._local_log is None:
            return
        text = "".join(f"{key}: {value}\n" for key, value in payload.items()) + "\n"
        try:
            with open(self._local_log, "a") as f:
                f.write(text)
        except OSError:
            logger.debug("Unable to write usage log to %s", self._local_log)


class UsageContext(UsageContextBase):
    """One-shot usage reporter for the single-process LMCacheEngine path.

    Sends an ``EnvMessage``, an ``EngineMessage``, and a
    ``MetadataMessage`` to the stats server.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        local_log: str | None = None,
        sender: UsageMessageSender | None = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            config: The engine configuration to snapshot.
            metadata: The engine metadata (model, world size, kv layout).
            local_log: Path of a local log of sent payloads; ``None``
                disables local logging.
            sender: Message transport; ``None`` selects the default HTTP
                sender.
        """
        super().__init__(DeploymentMode.SINGLE_PROCESS, local_log, sender)
        self._config = config
        self._metadata = metadata
        self._thread_state_lock = threading.RLock()
        self._closing = False
        self._closed = False
        # Publish the exact Thread before anyone can start it.  The engine can
        # therefore retain this context before ``Thread.start()`` enters its
        # ambiguous caller-interruption window.
        self._report_thread = self._make_report_thread()
        self._start_call = RetainedCall(
            self._report_thread.start,
            thread_name="lmcache-usage-report-start",
        )
        self._start_admitted = False

    REPORT_THREAD_JOIN_TIMEOUT_SECONDS = 10.0

    def _make_report_thread(self) -> threading.Thread:
        """Build, but do not start, the one-shot reporting thread."""
        return threading.Thread(
            target=self.report_once,
            daemon=True,
            name="lmcache-usage-report",
        )

    @property
    def report_thread(self) -> threading.Thread:
        """Return the exact retained reporting thread."""
        return self._report_thread

    @property
    def thread_alive(self) -> bool:
        """Return whether the retained reporting thread is still live."""
        return self._report_thread.is_alive()

    @property
    def closed(self) -> bool:
        """Return whether terminal shutdown proved the thread is absent."""
        with self._thread_state_lock:
            return self._closed

    def start_reporting(self) -> None:
        """Start the retained thread unless terminal shutdown won the race."""
        with self._thread_state_lock:
            if self._closing or self._closed:
                return
            # Admission publishes before the retained attempt can start.  A
            # concurrent close therefore waits for this exact attempt rather
            # than mistaking its temporarily unstarted Thread for no work.
            self._start_admitted = True
            start_call = self._start_call
        start_call.run()

    def close(self) -> None:
        """Fence startup and join the exact reporting thread.

        A timeout is reported truthfully and leaves this context retryable.
        Once close begins the startup fence is permanent, so a never-started
        reporter can be declared closed without creating new background work.
        """
        with self._thread_state_lock:
            if self._closed:
                return
            self._closing = True
            thread = self._report_thread
            start_admitted = self._start_admitted
            start_call = self._start_call
            if not start_admitted:
                self._closed = True
                return

        try:
            # If startup was admitted, first finish (or recover) its exact
            # retained attempt.  Only then is never-started a stable fact.
            start_call.run()
        except BaseException as error:
            if start_call.last_error is not error:
                raise
            # An exact Thread.start failure before its side effect is terminal
            # and owns no thread.  A failure raised after start retains a
            # populated Thread identity and must still be joined below.
            if thread.ident is None:
                with self._thread_state_lock:
                    self._closed = True
                return

        if thread is threading.current_thread():
            raise RuntimeError("usage reporting thread cannot join itself")
        thread.join(timeout=self.REPORT_THREAD_JOIN_TIMEOUT_SECONDS)
        if thread.is_alive():
            raise LMCacheTimeoutError("usage reporting thread did not terminate")

        with self._thread_state_lock:
            self._closed = True

    def _collect_messages(self) -> list[UsageMessage]:
        metadata_message = MetadataMessage(
            start_time=self._start_time.strftime("%Y-%m-%d %H:%M:%S"),
            duration=(datetime.now() - self._start_time).total_seconds(),
        )
        return [
            collect_env_message(),
            EngineMessage.from_config(self._config, self._metadata),
            metadata_message,
        ]


@swallow_telemetry_errors
def InitializeUsageContext(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    local_log: str | None = None,
    sender: UsageMessageSender | None = None,
    retain_context: Callable[[UsageContext], None] | None = None,
) -> UsageContext | None:
    """Start one-shot usage reporting for a single-process engine.

    Returns immediately; the report is sent in the background. Never
    blocks or raises.

    Args:
        config: The engine configuration to snapshot.
        metadata: The engine metadata (model, world size, kv layout).
        local_log: Path of a local log of sent payloads; ``None`` disables
            local logging.
        sender: Message transport; ``None`` selects the default HTTP sender.
        retain_context: Optional ownership callback invoked with the complete
            context (including its unstarted Thread) before startup begins.

    Returns:
        The usage context, or ``None`` when usage tracking is disabled or
        initialization failed.
    """
    if not is_usage_tracking_enabled():
        return None
    logger.info("Initializing usage context.")
    context = UsageContext(config, metadata, local_log, sender)
    if retain_context is not None:
        retain_context(context)
    context.start_reporting()
    return context
