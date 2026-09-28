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

    async def action_quit(self) -> None:
        # ctrl+q is bound to this by Textual itself (priority binding,
        # so a screen-level binding can't intercept it first). The
        # default just calls self.exit() unconditionally — fine for a
        # screen with no background work, but a screen with a live
        # worker thread (ingest, train) needs the chance to cancel that
        # thread and let it flush cleanly first; exiting out from under
        # it leaves it hung forever mid call_from_thread against an
        # event loop that no longer exists. A screen opts into that by
        # implementing request_stop() -> bool (True = "I'm handling
        # this, don't exit yet").
        request_stop = getattr(self.screen, "request_stop", None)
        if request_stop is not None and request_stop():
            return
        self.exit()
