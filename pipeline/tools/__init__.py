"""
pipeline/tools/__init__.py  –  Public API for the tools subpackage.

Re-exports TOOLS (schema list) and dispatch_tool (router) so callers only
need to import from pipeline.tools.
"""
from pipeline.tools.schemas import TOOLS
from pipeline.tools.dispatcher import dispatch_tool

__all__ = ["TOOLS", "dispatch_tool"]
