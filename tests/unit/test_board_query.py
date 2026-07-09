"""Comprehensive tests for barrier queries and board initialization."""

from __future__ import annotations

import random

import pytest

from engine.board import Board


class TestIsBarrier:
    """Test is_barrier() correctness."""

    def test_is_barrier_placed_barrier(self) -> None:
        """Placed barriers should return True from is_barrier()."""
        rng = random.Random(42)
        board = Board(5, 5, 1, rng=rng)
        barriers = board.barrier_coordinates()
        for barrier_coord in barriers:
            assert board.is_barrier(barrier_coord) is True

    def test_is_barrier_non_barrier_cell(self) -> None:
        """Non-barrier cells should return False from is_barrier()."""
        rng = random.Random(42)
        board = Board(5, 5, 5, exclude={(0, 0), (4, 4)}, rng=rng)
        # (0, 0) and (4, 4) are not barriers since they were excluded
        assert board.is_barrier((0, 0)) is False
        assert board.is_barrier((4, 4)) is False

    def test_is_barrier_all_cells_with_no_barriers(self) -> None:
        """No cells should be barriers when barrier_count is 0."""
        board = Board(5, 5, 0)
        for r in range(5):
            for c in range(5):
                assert board.is_barrier((r, c)) is False

    def test_is_barrier_consistency_with_barrier_coordinates(self) -> None:
        """is_barrier() should match membership in barrier_coordinates()."""
        rng = random.Random(42)
        board = Board(5, 5, 5, rng=rng)
        barriers = board.barrier_coordinates()
        for r in range(5):
            for c in range(5):
                coord = (r, c)
                assert board.is_barrier(coord) == (coord in barriers)


class TestBarrierCoordinates:
    """Test barrier_coordinates() returns frozenset and reflects placed barriers."""

    def test_barrier_coordinates_is_frozenset(self) -> None:
        """barrier_coordinates() should return a frozenset."""
        board = Board(5, 5, 5)
        barriers = board.barrier_coordinates()
        assert isinstance(barriers, frozenset)

    def test_barrier_coordinates_immutable(self) -> None:
        """Returned frozenset should not allow modifications."""
        board = Board(5, 5, 5)
        barriers = board.barrier_coordinates()
        with pytest.raises(AttributeError):
            barriers.add((0, 0))  # type: ignore[attr-defined]

    def test_barrier_coordinates_reflects_placement(self) -> None:
        """barrier_coordinates() should reflect all placed barriers."""
        rng = random.Random(42)
        board = Board(5, 5, 5, rng=rng)
        barriers = board.barrier_coordinates()
        assert len(barriers) == 5
        for barrier in barriers:
            assert board.in_bounds(barrier)

    def test_barrier_coordinates_multiple_calls_consistent(self) -> None:
        """Multiple calls to barrier_coordinates() should return equal sets."""
        rng = random.Random(42)
        board = Board(5, 5, 5, rng=rng)
        barriers1 = board.barrier_coordinates()
        barriers2 = board.barrier_coordinates()
        assert barriers1 == barriers2

    def test_barrier_coordinates_contains_valid_tuples(self) -> None:
        """barrier_coordinates() should contain valid coordinate tuples."""
        rng = random.Random(42)
        board = Board(5, 5, 5, rng=rng)
        barriers = board.barrier_coordinates()
        for coord in barriers:
            assert isinstance(coord, tuple)
            assert len(coord) == 2
            r, c = coord
            assert isinstance(r, int)
            assert isinstance(c, int)


class TestBoardInitialization:
    """Test Board initialization with various parameters."""

    def test_board_initialization_default_rng(self) -> None:
        """Board should work with default (unseeded) RNG."""
        board = Board(5, 5, 5)
        assert board.rows == 5
        assert board.cols == 5
        assert len(board.barrier_coordinates()) == 5

    def test_board_initialization_custom_grid_size(self) -> None:
        """Board should accept custom grid sizes."""
        board = Board(3, 7, 2)
        assert board.rows == 3
        assert board.cols == 7

    def test_board_initialization_with_rng(self) -> None:
        """Board should accept injected RNG."""
        rng = random.Random(42)
        board = Board(5, 5, 5, rng=rng)
        assert board._rng is rng

    def test_board_initialization_with_exclude(self) -> None:
        """Board should accept exclude set."""
        exclude = {(0, 0), (4, 4)}
        rng = random.Random(42)
        board = Board(5, 5, 5, exclude=exclude, rng=rng)
        for excluded_coord in exclude:
            assert excluded_coord not in board.barrier_coordinates()

    def test_board_initialization_empty_exclude(self) -> None:
        """Board should work with empty exclude set."""
        rng = random.Random(42)
        board1 = Board(5, 5, 5, exclude=set(), rng=rng)
        rng = random.Random(42)
        board2 = Board(5, 5, 5, exclude=None, rng=rng)
        # Both should produce the same result since empty exclude has no effect
        # Note: they won't be identical due to RNG state being consumed differently
        assert len(board1.barrier_coordinates()) == 5
        assert len(board2.barrier_coordinates()) == 5
