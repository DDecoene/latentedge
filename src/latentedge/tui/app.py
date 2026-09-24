"""The shared Textual application shell every dashboard command runs in."""

from textual.app import App
from textual.screen import Screen


class LatentEdgeApp(App[None]):
    """Hosts whichever screen a long-running command starts on."""

    def __init__(self, start_screen: Screen[None]) -> None:
        super().__init__()
        self._start_screen = start_screen

    def on_mount(self) -> None:
        self.push_screen(self._start_screen)
