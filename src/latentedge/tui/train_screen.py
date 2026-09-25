"""The train command's progress screen."""

import time
from collections.abc import Callable
from pathlib import Path

from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Static
from textual_plotext import PlotextPlot

from latentedge.features import save_feature_stats
from latentedge.metrics import build_training_metrics, save_training_metrics
from latentedge.model import NetReturnRegressor, save
from latentedge.model import train as default_train
from latentedge.training_data import AssembledTrainingData
from latentedge.tui.widgets import LogPanel, ProgressPanel

DEFAULT_MODEL_OUT_PATH = Path("data/model.safetensors")

TrainAssembleFn = Callable[[Path], AssembledTrainingData]


class TrainScreen(Screen[None]):
    BINDINGS = [
        Binding("q", "exit_now", "Exit", show=False),
    ]

    def __init__(
        self,
        swaps_path: Path,
        assemble_fn: TrainAssembleFn,
        out_path: Path = DEFAULT_MODEL_OUT_PATH,
        epochs: int = 100,
        learning_rate: float = 0.001,
        train_fn: Callable[..., list[float]] = default_train,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__()
        self.swaps_path = swaps_path
        self.out_path = out_path
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.assemble_fn = assemble_fn
        self.train_fn = train_fn
        self.time_fn = time_fn

        self.is_complete = False
        self.final_loss: float | None = None
        self.validate_correlation: float | None = None
        self.error: str | None = None
        self._losses: list[float] = []
        self._rate_start_time: float | None = None

    def compose(self) -> ComposeResult:
        yield ProgressPanel(id="train-progress")
        yield PlotextPlot(id="train-loss-plot")
        yield LogPanel(id="train-log")
        yield Static("", id="train-action-bar")

    def on_mount(self) -> None:
        self.query_one("#train-progress", ProgressPanel).update_progress(
            completed=0, total=self.epochs, unit_label="assembling training data...",
            rate_per_sec=0.0, rate_unit="epochs/sec",
        )
        self.run_worker(self._run_train, thread=True, exclusive=True)

    def _run_train(self) -> None:
        def on_epoch(epoch: int, total_epochs: int, loss: float) -> None:
            self.app.call_from_thread(self._handle_epoch, epoch, total_epochs, loss)

        try:
            assembled = self.assemble_fn(self.swaps_path)
            model = NetReturnRegressor(input_dim=assembled.input_dim)
            losses = self.train_fn(
                model, assembled.train.x, assembled.train.y, epochs=self.epochs,
                learning_rate=self.learning_rate, on_epoch=on_epoch,
            )
            self.out_path.parent.mkdir(parents=True, exist_ok=True)
            save(model, self.out_path)
            # Saved only after the model itself is safely on disk — an
            # earlier failure or interrupted run must never leave a stats
            # or metrics file that doesn't match the model SignalClient
            # will load alongside it.
            save_feature_stats(assembled.stats, Path(str(self.out_path) + ".stats.json"))
            metrics = build_training_metrics(model, assembled, losses)
            save_training_metrics(metrics, Path(str(self.out_path) + ".metrics.json"))
        except Exception as exc:
            self.app.call_from_thread(self._handle_error, str(exc))
            return
        self.app.call_from_thread(self._handle_complete, losses[-1], metrics["splits"]["validate"]["correlation"])

    def _handle_epoch(self, epoch: int, total_epochs: int, loss: float) -> None:
        self._losses.append(loss)
        if self._rate_start_time is None:
            self._rate_start_time = self.time_fn()
            rate = 0.0
        else:
            elapsed = self.time_fn() - self._rate_start_time
            rate = (epoch - 1) / elapsed if elapsed > 0 else 0.0

        self.query_one("#train-progress", ProgressPanel).update_progress(
            completed=epoch, total=total_epochs, unit_label=f"epoch {epoch}, loss {loss:.6f}",
            rate_per_sec=rate, rate_unit="epochs/sec",
        )
        self.query_one("#train-log", LogPanel).log_line(f"epoch {epoch}/{total_epochs}: loss {loss:.6f}")

        plot = self.query_one("#train-loss-plot", PlotextPlot)
        plot.plt.clear_data()
        plot.plt.plot(list(range(1, len(self._losses) + 1)), self._losses)
        plot.plt.title("Training loss")
        plot.refresh()

    def _handle_complete(self, final_loss: float, validate_correlation: float) -> None:
        self.is_complete = True
        self.final_loss = final_loss
        self.validate_correlation = validate_correlation
        self.query_one("#train-log", LogPanel).log_line(
            f"training complete — final loss {final_loss:.6f}, val corr {validate_correlation:.4f}, "
            f"saved to {self.out_path}"
        )
        self.query_one("#train-action-bar", Static).update(
            f"Training complete — final loss {final_loss:.6f}, val corr {validate_correlation:.4f}, "
            f"saved to {self.out_path}. Press [b]Q[/b] to exit."
        )

    def _handle_error(self, message: str) -> None:
        self.error = message
        self.query_one("#train-log", LogPanel).log_line(f"ERROR: {message}")
        self.query_one("#train-action-bar", Static).update(f"Training failed: {message}. Press [b]Q[/b] to exit.")

    def action_exit_now(self) -> None:
        if not (self.is_complete or self.error is not None):
            return
        self.app.exit()
