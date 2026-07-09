"""Comprehensive tests for barrier placement and exclusion."""

from __future__ import annotations

import random

from engine.board import Board


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
