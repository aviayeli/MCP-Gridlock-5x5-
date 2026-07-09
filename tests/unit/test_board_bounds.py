"""Comprehensive tests for Board.in_bounds()."""

from __future__ import annotations

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
