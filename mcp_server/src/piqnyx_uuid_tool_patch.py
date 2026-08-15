"""Runtime MCP compatibility patch for caller-visible episode UUIDs.

The upstream MCP ``add_memory`` tool queues work asynchronously and returns before
``Graphiti.add_episode`` creates the EpisodicNode. This wrapper reserves an episode
UUID before queueing, forwards it through the existing ``uuid`` argument, and adds
that UUID to the successful structured response.
"""

import functools
import inspect
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP

_installed = False
_original_tool = FastMCP.tool


def wrap_add_memory_with_uuid(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap an ``add_memory`` coroutine while preserving its public signature."""
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        episode_uuid = bound.arguments.get('uuid') or str(uuid4())
        bound.arguments['uuid'] = episode_uuid

        result = await fn(*bound.args, **bound.kwargs)
        if isinstance(result, dict) and 'error' not in result:
            result = dict(result)
            result['uuid'] = episode_uuid
        return result

    # Be explicit for frameworks that inspect __signature__ directly rather than
    # following functools.wraps/__wrapped__.
    wrapped.__signature__ = signature  # type: ignore[attr-defined]
    return wrapped


def install_add_memory_uuid_response_patch() -> None:
    """Install the wrapper before ``graphiti_mcp_server`` registers its tools."""
    global _installed
    if _installed:
        return

    def patched_tool(self: FastMCP, *tool_args: Any, **tool_kwargs: Any) -> Any:
        # Preserve FastMCP's direct-decorator form unchanged. Graphiti currently
        # registers add_memory via @mcp.tool(), which is handled below.
        if tool_args and callable(tool_args[0]):
            return _original_tool(self, *tool_args, **tool_kwargs)

        decorator = _original_tool(self, *tool_args, **tool_kwargs)

        def decorate(fn: Callable[..., Any]) -> Any:
            if getattr(fn, '__name__', None) == 'add_memory':
                fn = wrap_add_memory_with_uuid(fn)
            return decorator(fn)

        return decorate

    FastMCP.tool = patched_tool  # type: ignore[method-assign]
    _installed = True
