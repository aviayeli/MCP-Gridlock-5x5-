"""Core OOP data model for the MCP-Gridlock-5x5 board game.

This package intentionally contains only the pure, config-agnostic data
model (Board, Player). Reading `config/game_config.json` and driving turns
belongs to `game_loop.py` (a later step) so this layer stays trivially
testable and reusable outside the MCP transport.
"""
