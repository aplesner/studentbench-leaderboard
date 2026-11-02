"""StudentBench lesson-plan leaderboard."""

from .comparison import ComparisonContext, compare_plans

__all__ = ["ComparisonContext", "compare_plans", "main"]


def main() -> None:
    from .cli import main as cli_main

    cli_main()
