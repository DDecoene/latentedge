"""The train command's progress screen."""

from pathlib import Path

from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Static


class TrainScreen(Screen):
    def __init__(self, swaps_path: Path) -> None:
        super().__init__()
        self.swaps_path = swaps_path

    def compose(self) -> ComposeResult:
        yield Static(f"Training on {self.swaps_path}...", id="train-placeholder")
