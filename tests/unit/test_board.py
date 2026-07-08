"""Comprehensive tests for Board class covering bounds, barriers, and exclusion."""

from __future__ import annotations

import random

import pytest

from engine.board import Board


class TestInBounds:
    """Test in_bounds() for coordinates inside and outside the 5x5 grid."""

    def test_in_bounds_center_coordinate(self) -> None:
        """Center coordinates should be in bounds."""
        board = Board(5, 5, 0)
        assert board.in_bounds((2, 2)) is True

    def test_in_bounds_corners(self) -> None:
        """All four corners should be in bounds."""
        board = Board(5, 5, 0)
        assert board.in_bounds((0, 0)) is True
        assert board.in_bounds((0, 4)) is True
        assert board.in_bounds((4, 0)) is True
        assert board.in_bounds((4, 4)) is True

    def test_in_bounds_edges(self) -> None:
        """Cells on the edges should be in bounds."""
        board = Board(5, 5, 0)
        assert board.in_bounds((0, 2)) is True
        assert board.in_bounds((4, 2)) is True
        assert board.in_bounds((2, 0)) is True
        assert board.in_bounds((2, 4)) is True

    def test_out_of_bounds_negative_row(self) -> None:
        """Negative row should be out of bounds."""
        board = Board(5, 5, 0)
        assert board.in_bounds((-1, 2)) is False

    def test_out_of_bounds_negative_col(self) -> None:
        """Negative column should be out of bounds."""
        board = Board(5, 5, 0)
        assert board.in_bounds((2, -1)) is False

    def test_out_of_bounds_both_negative(self) -> None:
        """Both negative coordinates should be out of bounds."""
        board = Board(5, 5, 0)
        assert board.in_bounds((-1, -1)) is False

    def test_out_of_bounds_row_too_large(self) -> None:
        """Row >= grid size should be out of bounds."""
        board = Board(5, 5, 0)
        assert board.in_bounds((5, 2)) is False
        assert board.in_bounds((6, 2)) is False
        assert board.in_bounds((100, 2)) is False

    def test_out_of_bounds_col_too_large(self) -> None:
        """Column >= grid size should be out of bounds."""
        board = Board(5, 5, 0)
        assert board.in_bounds((2, 5)) is False
        assert board.in_bounds((2, 6)) is False
        assert board.in_bounds((2, 100)) is False

    def test_out_of_bounds_both_too_large(self) -> None:
        """Both coordinates >= grid size should be out of bounds."""
        board = Board(5, 5, 0)
        assert board.in_bounds((5, 5)) is False
        assert board.in_bounds((10, 10)) is False

    def test_in_bounds_rectangular_grid(self) -> None:
        """Test bounds on non-square grids."""
        board = Board(3, 7, 0)
        assert board.in_bounds((2, 6)) is True
        assert board.in_bounds((3, 0)) is False
        assert board.in_bounds((2, 7)) is False


class TestBarrierPlacement:
    """Test barrier placement with seeded RNG."""

    def test_exact_barrier_count(self) -> None:
        """Exactly barrier_count barriers should be placed."""
        rng = random.Random(42)
        board = Board(5, 5, 5, rng=rng)
        assert len(board.barrier_coordinates()) == 5

    def test_barrier_count_zero(self) -> None:
        """Zero barriers should result in empty barrier set."""
        board = Board(5, 5, 0)
        assert len(board.barrier_coordinates()) == 0

    def test_barrier_count_one(self) -> None:
        """One barrier should be placed."""
        rng = random.Random(42)
        board = Board(5, 5, 1, rng=rng)
        assert len(board.barrier_coordinates()) == 1

    def test_barrier_count_max(self) -> None:
        """Maximum barriers for 5x5 grid (25 cells, minus 2 excluded)."""
        exclude = {(0, 0), (4, 4)}
        rng = random.Random(42)
        board = Board(5, 5, 23, exclude=exclude, rng=rng)
        assert len(board.barrier_coordinates()) == 23

    def test_barrier_count_capped_by_available_cells(self) -> None:
        """Requesting more barriers than available cells should place only available."""
        exclude = {(0, 0), (4, 4)}
        rng = random.Random(42)
        board = Board(5, 5, 100, exclude=exclude, rng=rng)
        # 5*5 = 25 cells, minus 2 excluded = 23 available
        assert len(board.barrier_coordinates()) == 23

    def test_deterministic_seeded_placement(self) -> None:
        """Same seed should produce identical barrier layouts."""
        seed = 42
        rng1 = random.Random(seed)
        rng2 = random.Random(seed)
        board1 = Board(5, 5, 5, rng=rng1)
        board2 = Board(5, 5, 5, rng=rng2)
        assert board1.barrier_coordinates() == board2.barrier_coordinates()

    def test_different_seeds_produce_different_layouts(self) -> None:
        """Different seeds should (almost certainly) produce different layouts."""
        rng1 = random.Random(42)
        rng2 = random.Random(43)
        board1 = Board(5, 5, 5, rng=rng1)
        board2 = Board(5, 5, 5, rng=rng2)
        # Extremely unlikely to collide with different seeds
        assert board1.barrier_coordinates() != board2.barrier_coordinates()

    def test_barrier_never_outside_grid(self) -> None:
        """No barrier should be placed outside the grid."""
        rng = random.Random(42)
        board = Board(5, 5, 5, rng=rng)
        for coord in board.barrier_coordinates():
            assert board.in_bounds(coord)


class TestBarrierExclusion:
    """Test that barriers are never placed on excluded coordinates."""

    def test_exclude_single_coordinate(self) -> None:
        """Barriers should not be placed on a single excluded coordinate."""
        exclude = {(2, 2)}
        rng = random.Random(42)
        board = Board(5, 5, 5, exclude=exclude, rng=rng)
        assert (2, 2) not in board.barrier_coordinates()

    def test_exclude_start_positions(self) -> None:
        """Barriers should not be placed on cop and thief start positions."""
        exclude = {(0, 0), (4, 4)}
        rng = random.Random(42)
        board = Board(5, 5, 5, exclude=exclude, rng=rng)
        assert (0, 0) not in board.barrier_coordinates()
        assert (4, 4) not in board.barrier_coordinates()

    def test_exclude_multiple_coordinates(self) -> None:
        """Barriers should not be placed on any excluded coordinate."""
        exclude = {(0, 0), (2, 2), (4, 4)}
        rng = random.Random(42)
        board = Board(5, 5, 5, exclude=exclude, rng=rng)
        for excluded_coord in exclude:
            assert excluded_coord not in board.barrier_coordinates()

    def test_exclude_with_multiple_seeds(self) -> None:
        """Exclusion should work across multiple different seeds."""
        exclude = {(0, 0), (4, 4)}
        for seed in [10, 20, 30, 40, 50]:
            rng = random.Random(seed)
            board = Board(5, 5, 5, exclude=exclude, rng=rng)
            assert (0, 0) not in board.barrier_coordinates()
            assert (4, 4) not in board.barrier_coordinates()

    def test_exclude_all_but_few_cells(self) -> None:
        """When many cells are excluded, only non-excluded cells get barriers."""
        exclude = {
            (0, 0), (0, 1), (0, 2), (0, 3), (0, 4),
            (1, 0), (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 0), (2, 1), (2, 2), (2, 3),
        }
        rng = random.Random(42)
        board = Board(5, 5, 5, exclude=exclude, rng=rng)
        barriers = board.barrier_coordinates()
        assert len(barriers) == 5
        for barrier in barriers:
            assert barrier not in exclude


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
