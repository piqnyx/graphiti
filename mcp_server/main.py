#!/usr/bin/env python3
"""
Main entry point for Graphiti MCP Server

This is a backwards-compatible wrapper around the original graphiti_mcp_server.py
to maintain compatibility with existing deployment scripts and documentation.

Usage:
    python main.py [args...]

All arguments are passed through to the original server implementation.
"""

import sys
from pathlib import Path

# Add src directory to Python path for imports
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

# Import and run the original server
if __name__ == '__main__':
    # Install piqnyx's narrow compatibility layers before tool/service registration.
    from piqnyx_reliable_queue import install_reliable_queue_patch
    from piqnyx_uuid_tool_patch import install_add_memory_uuid_response_patch

    install_reliable_queue_patch()
    install_add_memory_uuid_response_patch()

    import graphiti_mcp_server as server
    from piqnyx_saga_state_tool import install_get_saga_tool

    # Register the fork-only read-only saga state tool after the upstream module
    # has created its FastMCP instance, but before the server starts accepting requests.
    install_get_saga_tool(server)

    # Pass all command line arguments to the original main function
    server.main()
