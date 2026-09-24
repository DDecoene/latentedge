# v1 Pipeline Implementation Plan

> **For implementers:** work through tasks in order; each ends with a
> commit. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end v1 loop — ingest real Uniswap v3 WETH/USDC
swap history, label it, train a regression model, and backtest it
honestly — as a single in-process Python pipeline.

**Architecture:** A linear pipeline of small, independently testable
modules (ingest → bars → labels → features/split → train → signal client
→ safety-guard → executor → backtest harness), wired together by a thin
CLI. No services, no network boundary between components — only the
ingestion step talks to the outside world (JSON-RPC).

**Tech Stack:** Python 3.12, `uv` for dependency management, `pandas` +
`pyarrow` for tabular data, `pydantic` for typed money-handling state,
`httpx` for RPC calls, `mlx` for the model, `click` for the CLI,
`pytest` for tests, `mypy --strict` for type checking.

**Spec:** `docs/specs/2026-09-24-v1-pipeline-design.md`

## Global Constraints

- Pool: WETH/USDC, Uniswap v3, 0.05% fee tier. Token order for this pool:
  **token0 = USDC (6 decimals), token1 = WETH (18 decimals)** — USDC's
  contract address sorts lower, which is what Uniswap v3 uses to assign
  token0/token1.
- Bar resolution: 1 minute, time-weighted, last price carried forward
  through gaps.
- Label horizon: 30 minutes, triple-barrier (take-profit / stop-loss /
  time-limit).
- Reference notional for labeling: **$1,000**, fixed, independent of
  actual backtest position sizing.
- Gas-used constant: **150,000 gas** per swap leg, multiplied by the
  block's base fee.
- Entry and exit prices for labeling must be **real swap prices only**,
  never a resampled bar's carried-forward value.
- Chronological train / validate / test split — never shuffled. Test set
  touched exactly once, at the end, for reported numbers and the backtest.
- No real money, no live execution, no service/API boundary — everything
  runs in one process.
- No AI-tooling attribution or internal process vocabulary in any
  committed file (commit messages, code comments, docs) — plain
  engineering prose only.

## Review Focus

- **Empty or near-empty ingested history** (a block range with almost no
  swaps): ingestion and the train/validate/test split must fail loudly
  with a clear error, not silently produce a near-empty split that trains
  a meaningless model.
- **Duplicate swap records** (an overlapping or retried RPC log query
  returning the same swap twice): storage must dedupe on
  `(tx_hash, log_index)`, or P&L/volume figures double-count real trades.
- **NaN/Inf model predictions** (numerical instability during training or
  a malformed feature row at inference time): the signal client must
  reject these explicitly, never let a NaN flow into position sizing.
- **Zero or negative position size from the safety-guard** (at a
  daily-loss-lockout boundary): the executor must treat this as "no
  trade," not attempt a zero-size fill or error out.
- **A bar whose forward 30-minute window runs past the end of ingested
  history** (the last ~30 minutes of any dataset): labeling must exclude
  these bars rather than silently label them off an incomplete window.

---

## Task 1: Project scaffolding and shared config

**Files:**
- Create: `pyproject.toml`
- Create: `src/latentedge/__init__.py`
- Create: `src/latentedge/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `latentedge.config` module with constants: `POOL_ADDRESS: str`,
  `TOKEN0_SYMBOL: str`, `TOKEN0_DECIMALS: int`, `TOKEN1_SYMBOL: str`,
  `TOKEN1_DECIMALS: int`, `FEE_TIER_BPS: int`, `BAR_INTERVAL_SECONDS: int`,
  `LABEL_HORIZON_SECONDS: int`, `REFERENCE_NOTIONAL_USD: float`,
  `GAS_USED_PER_SWAP: int`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from latentedge import config


def test_pool_and_token_constants():
    assert config.TOKEN0_SYMBOL == "USDC"
    assert config.TOKEN0_DECIMALS == 6
    assert config.TOKEN1_SYMBOL == "WETH"
    assert config.TOKEN1_DECIMALS == 18
    assert config.FEE_TIER_BPS == 5
    assert config.BAR_INTERVAL_SECONDS == 60
    assert config.LABEL_HORIZON_SECONDS == 1800
    assert config.REFERENCE_NOTIONAL_USD == 1000.0
    assert config.GAS_USED_PER_SWAP == 150_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge'`

- [ ] **Step 3: Create the project scaffold**

```toml
# pyproject.toml
[project]
name = "latentedge"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.2",
    "pyarrow>=17.0",
    "pydantic>=2.9",
    "httpx>=0.27",
    "mlx>=0.18",
    "click>=8.1",
    "plotext>=5.3",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.3",
    "mypy>=1.11",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/latentedge"]

[tool.mypy]
strict = true
```

```python
# src/latentedge/__init__.py
```

```python
# src/latentedge/config.py
"""Shared constants for the WETH/USDC v1 pipeline."""

POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # WETH/USDC 0.05%

TOKEN0_SYMBOL = "USDC"
TOKEN0_DECIMALS = 6
TOKEN1_SYMBOL = "WETH"
TOKEN1_DECIMALS = 18

FEE_TIER_BPS = 5  # 0.05%, i.e. 5 basis points

BAR_INTERVAL_SECONDS = 60
LABEL_HORIZON_SECONDS = 30 * 60

REFERENCE_NOTIONAL_USD = 1000.0
GAS_USED_PER_SWAP = 150_000
```

```python
# tests/__init__.py
```

- [ ] **Step 4: Install and run test to verify it passes**

Run: `uv sync && uv run pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/latentedge/__init__.py src/latentedge/config.py tests/__init__.py tests/test_config.py
git commit -m "Scaffold project and add shared pipeline config"
```

---

## Task 2: Uniswap v3 price math

**Files:**
- Create: `src/latentedge/uniswap_math.py`
- Create: `tests/test_uniswap_math.py`

**Interfaces:**
- Consumes: `config.TOKEN0_DECIMALS`, `config.TOKEN1_DECIMALS`.
- Produces: `sqrt_price_x96_to_price(sqrt_price_x96: int, decimals0: int,
  decimals1: int) -> float` — returns price of token0 in terms of token1
  (i.e. for this pool, USDC price in WETH terms would be the raw output;
  callers needing WETH-in-USDC invert it — see
  `sqrt_price_x96_to_weth_usdc_price`).
- Produces: `sqrt_price_x96_to_weth_usdc_price(sqrt_price_x96: int) ->
  float` — USDC price per 1 WETH, using this pool's fixed token order.
- Produces: `price_to_sqrt_price_x96(price_token1_per_token0: float,
  decimals0: int, decimals1: int) -> int` — inverse, used only by tests to
  construct known-good fixtures.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_uniswap_math.py
import pytest

from latentedge.uniswap_math import (
    price_to_sqrt_price_x96,
    sqrt_price_x96_to_price,
    sqrt_price_x96_to_weth_usdc_price,
)


def test_round_trip_price_conversion():
    # 1 raw token0 unit "buys" 0.0005 raw token1 units, i.e. a
    # constructed, self-verifying price — not a claim about any real
    # historical price.
    raw_price = 0.0005
    sqrt_price_x96 = price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18)
    recovered = sqrt_price_x96_to_price(sqrt_price_x96, decimals0=6, decimals1=18)
    assert recovered == pytest.approx(raw_price, rel=1e-9)


def test_weth_usdc_price_is_positive_and_plausible():
    # Construct a sqrtPriceX96 for a known, plausible WETH price of
    # $3,000 (in human terms) and confirm the pool-specific helper
    # recovers it, given token0=USDC(6dp)/token1=WETH(18dp).
    human_usdc_per_weth = 3000.0
    # token1-per-token0 raw price (WETH per USDC, decimal-adjusted) is the
    # inverse of the human USDC-per-WETH price, in raw-unit terms.
    raw_price = (1.0 / human_usdc_per_weth) * (10 ** (18 - 6)) ** -1
    sqrt_price_x96 = price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18)
    usdc_per_weth = sqrt_price_x96_to_weth_usdc_price(sqrt_price_x96)
    assert usdc_per_weth == pytest.approx(human_usdc_per_weth, rel=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_uniswap_math.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.uniswap_math'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/uniswap_math.py
"""Uniswap v3 sqrtPriceX96 <-> human price conversions.

Uniswap v3 stores price as sqrt(price) * 2**96, where "price" is the
amount of token1 you get per unit of token0, in raw (undecimaled) token
units. Converting to a human-readable price requires adjusting for each
token's decimals.
"""

Q96 = 2**96


def sqrt_price_x96_to_price(sqrt_price_x96: int, decimals0: int, decimals1: int) -> float:
    """Price of token0 in terms of token1, decimal-adjusted."""
    raw_price = (sqrt_price_x96 / Q96) ** 2
    return raw_price * (10 ** (decimals0 - decimals1))


def price_to_sqrt_price_x96(price_token1_per_token0: float, decimals0: int, decimals1: int) -> int:
    """Inverse of sqrt_price_x96_to_price. Test-fixture helper."""
    raw_price = price_token1_per_token0 / (10 ** (decimals0 - decimals1))
    return int((raw_price**0.5) * Q96)


def sqrt_price_x96_to_weth_usdc_price(sqrt_price_x96: int) -> float:
    """USDC price per 1 WETH for the fixed WETH/USDC pool (token0=USDC,
    token1=WETH). sqrt_price_x96_to_price gives WETH-per-USDC; invert."""
    weth_per_usdc = sqrt_price_x96_to_price(sqrt_price_x96, decimals0=6, decimals1=18)
    return 1.0 / weth_per_usdc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_uniswap_math.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/uniswap_math.py tests/test_uniswap_math.py
git commit -m "Add Uniswap v3 price conversion math"
```

---

## Task 3: Swap record schema and Parquet storage

**Files:**
- Create: `src/latentedge/schema.py`
- Create: `src/latentedge/store.py`
- Create: `tests/test_store.py`

**Interfaces:**
- Produces: `schema.SwapRecord` (pydantic model): `block_number: int`,
  `timestamp: int` (unix seconds), `tx_hash: str`, `log_index: int`,
  `sqrt_price_x96: int`, `tick: int`, `liquidity: int`, `amount0: float`,
  `amount1: float`, `base_fee_wei: int`.
- Produces: `store.write_swaps(records: list[SwapRecord], path: Path) ->
  None` — dedupes on `(tx_hash, log_index)` before writing, appends to an
  existing Parquet file if present.
- Produces: `store.read_swaps(path: Path) -> pd.DataFrame` — sorted by
  `timestamp` ascending.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
from pathlib import Path

import pandas as pd

from latentedge.schema import SwapRecord
from latentedge.store import read_swaps, write_swaps


def _record(tx_hash: str, log_index: int, timestamp: int) -> SwapRecord:
    return SwapRecord(
        block_number=100,
        timestamp=timestamp,
        tx_hash=tx_hash,
        log_index=log_index,
        sqrt_price_x96=1 << 96,
        tick=0,
        liquidity=10**18,
        amount0=1000.0,
        amount1=-0.3,
        base_fee_wei=20_000_000_000,
    )


def test_write_and_read_round_trip(tmp_path: Path):
    path = tmp_path / "swaps.parquet"
    write_swaps([_record("0xabc", 0, 100), _record("0xabc", 1, 101)], path)
    df = read_swaps(path)
    assert len(df) == 2
    assert list(df["timestamp"]) == [100, 101]


def test_write_dedupes_on_tx_hash_and_log_index(tmp_path: Path):
    path = tmp_path / "swaps.parquet"
    write_swaps([_record("0xabc", 0, 100)], path)
    write_swaps([_record("0xabc", 0, 100), _record("0xdef", 0, 102)], path)
    df = read_swaps(path)
    assert len(df) == 2
    assert set(df["tx_hash"]) == {"0xabc", "0xdef"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.schema'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/schema.py
from pydantic import BaseModel


class SwapRecord(BaseModel):
    block_number: int
    timestamp: int
    tx_hash: str
    log_index: int
    sqrt_price_x96: int
    tick: int
    liquidity: int
    amount0: float
    amount1: float
    base_fee_wei: int
```

```python
# src/latentedge/store.py
from pathlib import Path

import pandas as pd

from latentedge.schema import SwapRecord

DEDUPE_KEYS = ["tx_hash", "log_index"]


def write_swaps(records: list[SwapRecord], path: Path) -> None:
    new_df = pd.DataFrame([r.model_dump() for r in records])
    if path.exists():
        existing_df = pd.read_parquet(path)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=DEDUPE_KEYS, keep="first")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    combined.to_parquet(path, index=False)


def read_swaps(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df.sort_values("timestamp").reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_store.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/schema.py src/latentedge/store.py tests/test_store.py
git commit -m "Add swap record schema and deduping Parquet storage"
```

---

## Task 4: DROPPED — subgraph ingestion client

Dropped during implementation (see the ledger for the full ruling). The
Graph's decentralized-network gateway turned out to require a paid/metered
API key — not the anonymous public service the spec assumed. Since Task
6's direct `eth_getLogs` RPC path already decodes everything the subgraph
would have provided (price, tick, liquidity, all present in the raw `Swap`
event) and Task 5 already covers the one remaining gap (base fee) via a
plain RPC call, the subgraph bought nothing that RPC didn't already cover
— it was supposed to be the *simpler* option, and stopped being one.
Ingestion is RPC-only from here. Task 5 (pool-state backfill) now runs
**after** Task 6 (RPC ingestion) instead of before it, and its scope
narrows to base-fee-only backfill — see Task 5 below for the fields as
actually implemented, and Task 6 for the ingestion path that replaces
this one.

---

## Task 5: Pool state backfill (base fee only)

Runs **after** Task 6 now (RPC ingestion already populates `liquidity`
directly from the raw `Swap` event — see Task 4's drop note). This task's
scope narrows to the one field RPC logs don't carry: block base fee.

**Files:**
- Create: `src/latentedge/ingest/pool_state.py`
- Create: `tests/ingest/test_pool_state.py`

**Interfaces:**
- Consumes: `list[SwapRecord]` (with `base_fee_wei=0`, `liquidity` already
  populated, from Task 6).
- Produces: `pool_state.backfill_base_fee(records: list[SwapRecord],
  client: httpx.Client, rpc_url: str) -> list[SwapRecord]` — returns a new
  list with `base_fee_wei` populated per-record via `eth_getBlockByNumber`
  at each record's block. Leaves every other field, including `liquidity`,
  untouched.

- [ ] **Step 1: Write the failing test (real integration test)**

```python
# tests/ingest/test_pool_state.py
import httpx

from latentedge.ingest.pool_state import backfill_base_fee
from latentedge.schema import SwapRecord

RPC_URL = "https://eth.llamarpc.com"  # free public archive-capable endpoint; swap for a paid provider if this proves unreliable


def _bare_record(block_number: int, timestamp: int) -> SwapRecord:
    return SwapRecord(
        block_number=block_number,
        timestamp=timestamp,
        tx_hash="0x" + "0" * 64,
        log_index=0,
        sqrt_price_x96=1,
        tick=0,
        liquidity=123,  # already populated by Task 6; must survive untouched
        amount0=0.0,
        amount1=0.0,
        base_fee_wei=0,
    )


def test_backfill_populates_base_fee_and_preserves_liquidity():
    # A real, past mainnet block known to be well within archive range.
    record = _bare_record(block_number=17_500_000, timestamp=1_688_000_000)
    with httpx.Client(timeout=30.0) as client:
        result = backfill_base_fee([record], client, rpc_url=RPC_URL)

    assert len(result) == 1
    assert result[0].base_fee_wei > 0
    assert result[0].liquidity == 123
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingest/test_pool_state.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.ingest.pool_state'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/ingest/pool_state.py
"""Backfills per-block base fee via a JSON-RPC archive node.

Liquidity is populated directly from the raw Swap event during RPC
ingestion (see ingest.rpc_logs) and is never re-fetched here.
"""

import httpx

from latentedge.schema import SwapRecord


class PoolStateError(Exception):
    pass


def _rpc_call(client: httpx.Client, rpc_url: str, method: str, params: list) -> dict:
    response = client.post(rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    if response.status_code != 200:
        raise PoolStateError(f"RPC returned HTTP {response.status_code}: {response.text}")
    payload = response.json()
    if "error" in payload:
        raise PoolStateError(f"RPC error: {payload['error']}")
    return payload["result"]


def backfill_base_fee(records: list[SwapRecord], client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
    updated: list[SwapRecord] = []
    block_cache: dict[int, int] = {}

    for record in records:
        if record.block_number not in block_cache:
            block_hex = hex(record.block_number)
            block = _rpc_call(client, rpc_url, "eth_getBlockByNumber", [block_hex, False])
            block_cache[record.block_number] = int(block["baseFeePerGas"], 16)

        updated.append(record.model_copy(update={"base_fee_wei": block_cache[record.block_number]}))
    return updated
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingest/test_pool_state.py -v`
Expected: PASS. If the public RPC endpoint rejects the request (rate
limit, archive-data restriction), swap in a different provider and update
`RPC_URL` — this is exactly the kind of thing that needs proving against
the real service.

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/ingest/pool_state.py tests/ingest/test_pool_state.py
git commit -m "Backfill per-block base fee from an archive RPC"
```

---

## Task 6: Direct RPC ingestion (eth_getLogs)

First task to touch the `ingest` package now that Task 4 is dropped — its
`__init__.py` marker files are created here.

**Files:**
- Create: `src/latentedge/ingest/__init__.py`
- Create: `src/latentedge/ingest/rpc_logs.py`
- Create: `tests/ingest/__init__.py`
- Create: `tests/ingest/test_rpc_logs.py`

**Interfaces:**
- Consumes: `config.POOL_ADDRESS`.
- Produces: `rpc_logs.fetch_swaps(pool_address: str, from_block: int,
  to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]`
  — decodes raw `Swap` event logs directly, populating `liquidity` and
  `sqrt_price_x96`/`tick` from the event data itself (no separate
  backfill needed for records sourced this way — only base fee still
  requires the Task 5 helper).

- [ ] **Step 1: Write the failing test (real integration test)**

```python
# tests/ingest/test_rpc_logs.py
import httpx

from latentedge import config
from latentedge.ingest.rpc_logs import fetch_swaps

RPC_URL = "https://eth.llamarpc.com"


def test_fetch_swaps_decodes_real_logs():
    # A narrow, real, past block range with known WETH/USDC 0.05% activity.
    with httpx.Client(timeout=30.0) as client:
        records = fetch_swaps(config.POOL_ADDRESS, from_block=17_500_000, to_block=17_500_050, client=client, rpc_url=RPC_URL)

    assert len(records) > 0
    for r in records:
        assert r.liquidity > 0
        assert r.sqrt_price_x96 > 0
        assert r.tx_hash.startswith("0x")
        assert 17_500_000 <= r.block_number <= 17_500_050
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingest/test_rpc_logs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.ingest.rpc_logs'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/ingest/rpc_logs.py
"""Fallback ingestion: decode Swap events directly via eth_getLogs.

Swap(address indexed sender, address indexed recipient, int256 amount0,
     int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
"""

import httpx

from latentedge.schema import SwapRecord

# keccak256("Swap(address,address,int256,int256,uint160,uint128,int24)")
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"


class RpcLogsError(Exception):
    pass


def _rpc_call(client: httpx.Client, rpc_url: str, method: str, params: list) -> object:
    response = client.post(rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
    if response.status_code != 200:
        raise RpcLogsError(f"RPC returned HTTP {response.status_code}: {response.text}")
    payload = response.json()
    if "error" in payload:
        raise RpcLogsError(f"RPC error: {payload['error']}")
    return payload["result"]


def _decode_int(hex_str: str, bits: int, signed: bool) -> int:
    value = int(hex_str, 16)
    if signed and value >= (1 << (bits - 1)):
        value -= 1 << bits
    return value


def _decode_swap_data(data_hex: str) -> tuple[float, float, int, int, int]:
    data = data_hex[2:]  # strip 0x
    words = [data[i : i + 64] for i in range(0, len(data), 64)]
    amount0 = _decode_int(words[0], 256, signed=True)
    amount1 = _decode_int(words[1], 256, signed=True)
    sqrt_price_x96 = _decode_int(words[2], 256, signed=False)
    liquidity = _decode_int(words[3], 256, signed=False)
    tick = _decode_int(words[4], 256, signed=True)
    return float(amount0), float(amount1), sqrt_price_x96, liquidity, tick


def fetch_swaps(pool_address: str, from_block: int, to_block: int, client: httpx.Client, rpc_url: str) -> list[SwapRecord]:
    logs = _rpc_call(
        client,
        rpc_url,
        "eth_getLogs",
        [
            {
                "address": pool_address,
                "topics": [SWAP_TOPIC],
                "fromBlock": hex(from_block),
                "toBlock": hex(to_block),
            }
        ],
    )

    records: list[SwapRecord] = []
    block_timestamp_cache: dict[int, int] = {}

    for log in logs:
        block_number = int(log["blockNumber"], 16)
        if block_number not in block_timestamp_cache:
            block = _rpc_call(client, rpc_url, "eth_getBlockByNumber", [log["blockNumber"], False])
            block_timestamp_cache[block_number] = int(block["timestamp"], 16)

        amount0, amount1, sqrt_price_x96, liquidity, tick = _decode_swap_data(log["data"])

        records.append(
            SwapRecord(
                block_number=block_number,
                timestamp=block_timestamp_cache[block_number],
                tx_hash=log["transactionHash"],
                log_index=int(log["logIndex"], 16),
                sqrt_price_x96=sqrt_price_x96,
                tick=tick,
                liquidity=liquidity,
                amount0=amount0,
                amount1=amount1,
                base_fee_wei=0,  # populate via ingest.pool_state.backfill_base_fee
            )
        )

    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingest/test_rpc_logs.py -v`
Expected: PASS. If the decoded values look wrong (e.g. implausible price),
double-check the ABI word layout against the real log data returned —
this decoder is hand-written and the most likely place for an off-by-one
in word offsets.

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/ingest/__init__.py src/latentedge/ingest/rpc_logs.py tests/ingest/__init__.py tests/ingest/test_rpc_logs.py
git commit -m "Add direct eth_getLogs RPC ingestion"
```

---

## Task 7: Bar construction

**Files:**
- Create: `src/latentedge/bars.py`
- Create: `tests/test_bars.py`

**Interfaces:**
- Consumes: `pd.DataFrame` of swap records (as returned by
  `store.read_swaps`) with columns `timestamp`, `sqrt_price_x96`,
  `amount0`, `amount1`.
- Produces: `bars.build_bars(swaps: pd.DataFrame, interval_seconds: int) ->
  pd.DataFrame` with columns `bar_start` (int, unix seconds),
  `price_usdc_per_weth` (float, time-weighted, carried forward through
  gaps), `swap_count` (int), `volume_usdc` (float, sum of `abs(amount0)`),
  `has_gap` (bool, true if `swap_count == 0`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bars.py
import pandas as pd
import pytest

from latentedge.bars import build_bars
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _swap(timestamp: int, human_price: float, amount0: float) -> dict:
    raw_price = (1.0 / human_price) * (10 ** (18 - 6)) ** -1
    return {
        "timestamp": timestamp,
        "sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18),
        "amount0": amount0,
        "amount1": -amount0 / human_price,
    }


def test_bar_with_no_swaps_carries_price_forward_and_flags_gap():
    swaps = pd.DataFrame([_swap(0, 3000.0, 1000.0), _swap(150, 3010.0, 500.0)])
    bars = build_bars(swaps, interval_seconds=60)

    # bar[0] = [0,60): one swap at t=0
    # bar[1] = [60,120): no swaps -> carried forward from bar[0]
    # bar[2] = [120,180): one swap at t=150
    assert len(bars) == 3
    assert bars.iloc[1]["swap_count"] == 0
    assert bars.iloc[1]["has_gap"] is True
    assert bars.iloc[1]["price_usdc_per_weth"] == pytest.approx(bars.iloc[0]["price_usdc_per_weth"], rel=1e-6)
    assert bars.iloc[2]["has_gap"] is False


def test_bar_with_many_swaps_aggregates_volume():
    swaps = pd.DataFrame([_swap(0, 3000.0, 100.0), _swap(10, 3001.0, 200.0), _swap(20, 3002.0, 50.0)])
    bars = build_bars(swaps, interval_seconds=60)
    assert len(bars) == 1
    assert bars.iloc[0]["swap_count"] == 3
    assert bars.iloc[0]["volume_usdc"] == pytest.approx(350.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_bars.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.bars'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/bars.py
import pandas as pd

from latentedge.uniswap_math import sqrt_price_x96_to_weth_usdc_price


def build_bars(swaps: pd.DataFrame, interval_seconds: int) -> pd.DataFrame:
    if swaps.empty:
        return pd.DataFrame(columns=["bar_start", "price_usdc_per_weth", "swap_count", "volume_usdc", "has_gap"])

    df = swaps.copy()
    df["price_usdc_per_weth"] = df["sqrt_price_x96"].apply(sqrt_price_x96_to_weth_usdc_price)
    df["volume_usdc"] = df["amount0"].abs()
    df["bar_start"] = (df["timestamp"] // interval_seconds) * interval_seconds

    first_bar = int(df["bar_start"].min())
    last_bar = int(df["bar_start"].max())
    all_bar_starts = range(first_bar, last_bar + interval_seconds, interval_seconds)

    grouped = df.groupby("bar_start").agg(
        price_usdc_per_weth=("price_usdc_per_weth", "last"),
        swap_count=("price_usdc_per_weth", "count"),
        volume_usdc=("volume_usdc", "sum"),
    )

    bars = grouped.reindex(all_bar_starts)
    bars.index.name = "bar_start"
    bars["has_gap"] = bars["swap_count"].isna()
    bars["swap_count"] = bars["swap_count"].fillna(0).astype(int)
    bars["volume_usdc"] = bars["volume_usdc"].fillna(0.0)
    bars["price_usdc_per_weth"] = bars["price_usdc_per_weth"].ffill()

    return bars.reset_index()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_bars.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/bars.py tests/test_bars.py
git commit -m "Add 1-minute bar construction from raw swaps"
```

---

## Task 8: Slippage and gas cost estimation

**Files:**
- Create: `src/latentedge/costs.py`
- Create: `tests/test_costs.py`

**Interfaces:**
- Produces: `costs.estimate_slippage_fraction(notional_usd: float,
  liquidity: int, sqrt_price_x96: int) -> float` — a concentrated-liquidity
  approximation (spec 3.3/3.7): slippage scales with trade size relative
  to a liquidity-derived local depth estimate, and decreases as liquidity
  increases.
- Produces: `costs.estimate_gas_cost_usd(base_fee_wei: int,
  gas_used: int, weth_usdc_price: float) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_costs.py
import pytest

from latentedge.costs import estimate_gas_cost_usd, estimate_slippage_fraction
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _sqrt_price_for(human_usdc_per_weth: float) -> int:
    raw_price = (1.0 / human_usdc_per_weth) * (10 ** (18 - 6)) ** -1
    return price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18)


def test_slippage_decreases_with_more_liquidity():
    sqrt_price = _sqrt_price_for(3000.0)
    thin = estimate_slippage_fraction(notional_usd=1000.0, liquidity=10**12, sqrt_price_x96=sqrt_price)
    deep = estimate_slippage_fraction(notional_usd=1000.0, liquidity=10**18, sqrt_price_x96=sqrt_price)
    assert deep < thin
    assert thin > 0
    assert deep > 0


def test_slippage_increases_with_trade_size():
    sqrt_price = _sqrt_price_for(3000.0)
    small = estimate_slippage_fraction(notional_usd=100.0, liquidity=10**15, sqrt_price_x96=sqrt_price)
    large = estimate_slippage_fraction(notional_usd=10_000.0, liquidity=10**15, sqrt_price_x96=sqrt_price)
    assert large > small


def test_gas_cost_scales_with_base_fee():
    cheap = estimate_gas_cost_usd(base_fee_wei=10_000_000_000, gas_used=150_000, weth_usdc_price=3000.0)
    expensive = estimate_gas_cost_usd(base_fee_wei=100_000_000_000, gas_used=150_000, weth_usdc_price=3000.0)
    assert expensive == pytest.approx(cheap * 10, rel=1e-6)
    assert cheap > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_costs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.costs'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/costs.py
"""Slippage and gas-cost estimation.

Slippage: a concentrated-liquidity approximation, not exact tick-crossing
math (spec non-goal). Local liquidity L near the current price implies a
virtual USDC-denominated depth of roughly L * sqrt(price) (a standard
Uniswap v3 approximation for in-range depth in quote-token terms). Trade
impact is modeled the way constant-product slippage is commonly
approximated: impact_fraction ~= notional / (2 * virtual_depth_usd).
"""

from latentedge.uniswap_math import sqrt_price_x96_to_weth_usdc_price, Q96

WEI_PER_ETH = 10**18


def _virtual_depth_usd(liquidity: int, sqrt_price_x96: int) -> float:
    sqrt_price = sqrt_price_x96 / Q96
    # L * sqrtP, scaled from raw units to USDC (6 decimals) terms.
    virtual_reserve_token1_raw = liquidity * sqrt_price
    weth_price = sqrt_price_x96_to_weth_usdc_price(sqrt_price_x96)
    virtual_reserve_weth = virtual_reserve_token1_raw / (10**18)
    return virtual_reserve_weth * weth_price


def estimate_slippage_fraction(notional_usd: float, liquidity: int, sqrt_price_x96: int) -> float:
    depth_usd = _virtual_depth_usd(liquidity, sqrt_price_x96)
    if depth_usd <= 0:
        raise ValueError("non-positive virtual depth; check liquidity/sqrt_price_x96 inputs")
    return notional_usd / (2 * depth_usd)


def estimate_gas_cost_usd(base_fee_wei: int, gas_used: int, weth_usdc_price: float) -> float:
    cost_wei = base_fee_wei * gas_used
    cost_eth = cost_wei / WEI_PER_ETH
    return cost_eth * weth_usdc_price
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_costs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/costs.py tests/test_costs.py
git commit -m "Add slippage and gas cost estimation"
```

---

## Task 9: Triple-barrier labeling

**Files:**
- Create: `src/latentedge/labeling.py`
- Create: `tests/test_labeling.py`

**Interfaces:**
- Consumes: raw swaps `pd.DataFrame` (Task 6/7 shape: `timestamp`,
  `sqrt_price_x96`), `bars.build_bars` output, `costs.estimate_slippage_fraction`,
  `costs.estimate_gas_cost_usd`, `config.REFERENCE_NOTIONAL_USD`,
  `config.GAS_USED_PER_SWAP`, `config.LABEL_HORIZON_SECONDS`,
  `config.FEE_TIER_BPS`.
- Produces: `labeling.label_bars(bars: pd.DataFrame, swaps: pd.DataFrame,
  tp_sl_fraction: float) -> pd.DataFrame` — returns `bars` with an added
  `net_return` column (float, `NaN` for excluded bars) and an `excluded`
  column (bool) with a `reason` column (`"no_entry_fill"`,
  `"incomplete_horizon"`, or `None`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_labeling.py
import pandas as pd
import pytest

from latentedge.labeling import label_bars
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _swap(timestamp: int, human_price: float, liquidity: int = 10**18, base_fee_wei: int = 20_000_000_000) -> dict:
    raw_price = (1.0 / human_price) * (10 ** (18 - 6)) ** -1
    return {
        "timestamp": timestamp,
        "sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18),
        "liquidity": liquidity,
        "base_fee_wei": base_fee_wei,
    }


def test_bar_with_no_swap_in_window_is_excluded_for_no_entry_fill():
    # bar at t=0 has no swap at/after t=0 before the horizon ends at t=1800
    swaps = pd.DataFrame([_swap(2000, 3000.0)])  # only swap is after the horizon
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 0, "volume_usdc": 0.0, "has_gap": True}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)
    assert result.iloc[0]["excluded"] is True
    assert result.iloc[0]["reason"] == "no_entry_fill"
    assert pd.isna(result.iloc[0]["net_return"])


def test_take_profit_hit_produces_positive_net_return():
    swaps = pd.DataFrame(
        [
            _swap(0, 3000.0),  # entry fill
            _swap(60, 3100.0),  # +3.3%, above a 1% take-profit band -> exit here
        ]
    )
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 1, "volume_usdc": 1000.0, "has_gap": False}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)
    assert result.iloc[0]["excluded"] is False
    assert result.iloc[0]["net_return"] > 0
    # net return must be less than the raw 3.3% move once fee/slippage/gas are netted out
    assert result.iloc[0]["net_return"] < 0.033


def test_stop_loss_hit_produces_negative_net_return():
    swaps = pd.DataFrame([_swap(0, 3000.0), _swap(60, 2900.0)])  # -3.3% move
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 1, "volume_usdc": 1000.0, "has_gap": False}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)
    assert result.iloc[0]["net_return"] < 0


def test_bar_near_end_of_history_with_incomplete_horizon_is_excluded():
    swaps = pd.DataFrame([_swap(0, 3000.0), _swap(60, 3001.0)])  # history ends at t=60, horizon needs t=1800
    bars = pd.DataFrame([{"bar_start": 0, "price_usdc_per_weth": 3000.0, "swap_count": 1, "volume_usdc": 1000.0, "has_gap": False}])
    result = label_bars(bars, swaps, tp_sl_fraction=0.01)
    assert result.iloc[0]["excluded"] is True
    assert result.iloc[0]["reason"] == "incomplete_horizon"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_labeling.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.labeling'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/labeling.py
import numpy as np
import pandas as pd

from latentedge import config
from latentedge.costs import estimate_gas_cost_usd, estimate_slippage_fraction
from latentedge.uniswap_math import sqrt_price_x96_to_weth_usdc_price

FEE_FRACTION = config.FEE_TIER_BPS / 10_000


def _net_return(entry_price: float, exit_price: float, entry_swap: pd.Series, exit_swap: pd.Series) -> float:
    notional = config.REFERENCE_NOTIONAL_USD

    raw_return = (exit_price - entry_price) / entry_price
    gross_pnl = notional * raw_return

    fee_cost = 2 * notional * FEE_FRACTION

    entry_slippage = estimate_slippage_fraction(notional, entry_swap["liquidity"], entry_swap["sqrt_price_x96"])
    exit_slippage = estimate_slippage_fraction(notional, exit_swap["liquidity"], exit_swap["sqrt_price_x96"])
    slippage_cost = notional * (entry_slippage + exit_slippage)

    entry_gas = estimate_gas_cost_usd(entry_swap["base_fee_wei"], config.GAS_USED_PER_SWAP, entry_price)
    exit_gas = estimate_gas_cost_usd(exit_swap["base_fee_wei"], config.GAS_USED_PER_SWAP, exit_price)
    gas_cost = entry_gas + exit_gas

    net_pnl = gross_pnl - fee_cost - slippage_cost - gas_cost
    return net_pnl / notional


def label_bars(bars: pd.DataFrame, swaps: pd.DataFrame, tp_sl_fraction: float) -> pd.DataFrame:
    swaps = swaps.sort_values("timestamp").reset_index(drop=True)
    swaps["price_usdc_per_weth"] = swaps["sqrt_price_x96"].apply(sqrt_price_x96_to_weth_usdc_price)

    net_returns: list[float] = []
    excluded: list[bool] = []
    reasons: list[str | None] = []

    history_end = swaps["timestamp"].max() if not swaps.empty else -1

    for _, bar in bars.iterrows():
        t = bar["bar_start"]
        horizon_end = t + config.LABEL_HORIZON_SECONDS

        entry_candidates = swaps[swaps["timestamp"] >= t]
        if entry_candidates.empty or entry_candidates.iloc[0]["timestamp"] > horizon_end:
            net_returns.append(float("nan"))
            excluded.append(True)
            reasons.append("no_entry_fill")
            continue

        if history_end < horizon_end:
            net_returns.append(float("nan"))
            excluded.append(True)
            reasons.append("incomplete_horizon")
            continue

        entry_swap = entry_candidates.iloc[0]
        entry_price = entry_swap["price_usdc_per_weth"]

        forward = swaps[(swaps["timestamp"] > entry_swap["timestamp"]) & (swaps["timestamp"] <= horizon_end)]

        exit_swap = None
        for _, candidate in forward.iterrows():
            move = (candidate["price_usdc_per_weth"] - entry_price) / entry_price
            if abs(move) >= tp_sl_fraction:
                exit_swap = candidate
                break

        if exit_swap is None:
            exit_swap = forward.iloc[-1] if not forward.empty else entry_swap

        exit_price = exit_swap["price_usdc_per_weth"]
        net_returns.append(_net_return(entry_price, exit_price, entry_swap, exit_swap))
        excluded.append(False)
        reasons.append(None)

    result = bars.copy()
    result["net_return"] = net_returns
    result["excluded"] = excluded
    result["reason"] = reasons
    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_labeling.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/labeling.py tests/test_labeling.py
git commit -m "Add triple-barrier labeling with real-swap-price entry/exit"
```

---

## Task 10: Point-in-time feature engineering

**Files:**
- Create: `src/latentedge/features.py`
- Create: `tests/test_features.py`

**Interfaces:**
- Consumes: `bars.build_bars` output (`bar_start`, `price_usdc_per_weth`,
  `volume_usdc`).
- Produces: `features.compute_features(bars: pd.DataFrame,
  return_windows: list[int], volatility_window: int) -> pd.DataFrame` —
  returns `bars` with added columns: `return_<n>` for each `n` in
  `return_windows` (rolling return over the trailing `n` bars),
  `volatility` (rolling std-dev of 1-bar returns over
  `volatility_window`), `volume_usdc` (passthrough), `bars_since_swap`
  (int, count of consecutive prior bars with `swap_count == 0`, reset to 0
  on a bar with a swap).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_features.py
import pandas as pd
import pytest

from latentedge.features import compute_features


def test_features_use_only_past_data():
    bars = pd.DataFrame(
        {
            "bar_start": [0, 60, 120, 180, 240],
            "price_usdc_per_weth": [3000.0, 3010.0, 3005.0, 3020.0, 3015.0],
            "volume_usdc": [100.0, 200.0, 150.0, 300.0, 250.0],
            "swap_count": [1, 1, 0, 1, 1],
        }
    )
    result = compute_features(bars, return_windows=[2], volatility_window=3)

    # return_2 at index 2 uses prices at index 0 and 2 only, not index 3/4.
    expected_return_2_at_idx2 = (3005.0 - 3000.0) / 3000.0
    assert result.loc[2, "return_2"] == pytest.approx(expected_return_2_at_idx2)

    # Changing a *future* price must not change a past feature value.
    bars_altered = bars.copy()
    bars_altered.loc[4, "price_usdc_per_weth"] = 99999.0
    result_altered = compute_features(bars_altered, return_windows=[2], volatility_window=3)
    assert result_altered.loc[2, "return_2"] == result.loc[2, "return_2"]


def test_bars_since_swap_resets_on_activity():
    bars = pd.DataFrame(
        {
            "bar_start": [0, 60, 120, 180],
            "price_usdc_per_weth": [3000.0, 3000.0, 3000.0, 3000.0],
            "volume_usdc": [100.0, 0.0, 0.0, 50.0],
            "swap_count": [1, 0, 0, 1],
        }
    )
    result = compute_features(bars, return_windows=[1], volatility_window=2)
    assert list(result["bars_since_swap"]) == [0, 1, 2, 0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_features.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.features'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/features.py
import pandas as pd


def compute_features(bars: pd.DataFrame, return_windows: list[int], volatility_window: int) -> pd.DataFrame:
    result = bars.copy()
    price = result["price_usdc_per_weth"]

    for n in return_windows:
        result[f"return_{n}"] = price.pct_change(periods=n)

    one_bar_return = price.pct_change(periods=1)
    result["volatility"] = one_bar_return.rolling(window=volatility_window, min_periods=volatility_window).std()

    had_swap = result["swap_count"] > 0
    groups = had_swap.cumsum()
    result["bars_since_swap"] = result.groupby(groups).cumcount()
    result.loc[had_swap, "bars_since_swap"] = 0

    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/features.py tests/test_features.py
git commit -m "Add point-in-time rolling feature engineering"
```

---

## Task 11: Chronological train/validate/test split

**Files:**
- Create: `src/latentedge/split.py`
- Create: `tests/test_split.py`

**Interfaces:**
- Consumes: any `pd.DataFrame` sorted by `bar_start`.
- Produces: `split.chronological_split(df: pd.DataFrame,
  train_fraction: float, validate_fraction: float) -> tuple[pd.DataFrame,
  pd.DataFrame, pd.DataFrame]` — raises `split.InsufficientDataError` if
  any resulting split would have fewer than 100 rows (the "empty history"
  failure mode from Review Focus).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_split.py
import pandas as pd
import pytest

from latentedge.split import InsufficientDataError, chronological_split


def test_split_is_chronological_and_non_overlapping():
    df = pd.DataFrame({"bar_start": range(1000), "value": range(1000)})
    train, validate, test = chronological_split(df, train_fraction=0.7, validate_fraction=0.15)

    assert train["bar_start"].max() < validate["bar_start"].min()
    assert validate["bar_start"].max() < test["bar_start"].min()
    assert len(train) + len(validate) + len(test) == len(df)


def test_split_raises_on_insufficient_data():
    df = pd.DataFrame({"bar_start": range(50), "value": range(50)})
    with pytest.raises(InsufficientDataError):
        chronological_split(df, train_fraction=0.7, validate_fraction=0.15)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_split.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.split'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/split.py
import pandas as pd

MIN_ROWS_PER_SPLIT = 100


class InsufficientDataError(Exception):
    pass


def chronological_split(df: pd.DataFrame, train_fraction: float, validate_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("bar_start").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_fraction)
    validate_end = train_end + int(n * validate_fraction)

    train = df.iloc[:train_end]
    validate = df.iloc[train_end:validate_end]
    test = df.iloc[validate_end:]

    for name, split_df in [("train", train), ("validate", validate), ("test", test)]:
        if len(split_df) < MIN_ROWS_PER_SPLIT:
            raise InsufficientDataError(f"{name} split has only {len(split_df)} rows, need at least {MIN_ROWS_PER_SPLIT}")

    return train, validate, test
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_split.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/split.py tests/test_split.py
git commit -m "Add chronological train/validate/test split"
```

---

## Task 12: MLX regression model

**Files:**
- Create: `src/latentedge/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Produces: `model.NetReturnRegressor(input_dim: int)` (an `mlx.nn.Module`
  subclass) with `__call__(self, x: mx.array) -> mx.array`.
- Produces: `model.train(model: NetReturnRegressor, features: np.ndarray,
  labels: np.ndarray, epochs: int, learning_rate: float) -> list[float]`
  — returns the per-epoch training loss history.
- Produces: `model.save(model: NetReturnRegressor, path: Path) -> None`,
  `model.load(path: Path, input_dim: int) -> NetReturnRegressor`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
from pathlib import Path

import numpy as np

from latentedge.model import NetReturnRegressor, load, save, train


def test_training_reduces_loss_on_learnable_synthetic_data():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 4)).astype(np.float32)
    true_weights = np.array([0.5, -0.3, 0.1, 0.2], dtype=np.float32)
    y = (x @ true_weights).astype(np.float32)

    regressor = NetReturnRegressor(input_dim=4)
    losses = train(regressor, x, y, epochs=50, learning_rate=0.05)

    assert losses[-1] < losses[0]
    assert losses[-1] < 0.1  # should fit this simple linear signal closely


def test_save_and_load_round_trip_predicts_identically(tmp_path: Path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 3)).astype(np.float32)
    y = rng.normal(size=(50,)).astype(np.float32)

    regressor = NetReturnRegressor(input_dim=3)
    train(regressor, x, y, epochs=5, learning_rate=0.01)

    path = tmp_path / "model.safetensors"
    save(regressor, path)
    loaded = load(path, input_dim=3)

    import mlx.core as mx

    original_pred = regressor(mx.array(x))
    loaded_pred = loaded(mx.array(x))
    assert np.allclose(np.array(original_pred), np.array(loaded_pred), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.model'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/model.py
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class NetReturnRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 1)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.relu(self.layer1(x))
        return self.layer2(h).squeeze(-1)


def _loss_fn(model: NetReturnRegressor, x: mx.array, y: mx.array) -> mx.array:
    predictions = model(x)
    return mx.mean((predictions - y) ** 2)


def train(model: NetReturnRegressor, features: np.ndarray, labels: np.ndarray, epochs: int, learning_rate: float) -> list[float]:
    x = mx.array(features)
    y = mx.array(labels)
    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad = nn.value_and_grad(model, _loss_fn)

    losses: list[float] = []
    for _ in range(epochs):
        loss, grads = loss_and_grad(model, x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(float(loss))
    return losses


def save(model: NetReturnRegressor, path: Path) -> None:
    model.save_weights(str(path))


def load(path: Path, input_dim: int) -> NetReturnRegressor:
    model = NetReturnRegressor(input_dim=input_dim)
    model.load_weights(str(path))
    return model
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/model.py tests/test_model.py
git commit -m "Add MLX regression model with train/save/load"
```

---

## Task 13: Signal client (batched inference)

**Files:**
- Create: `src/latentedge/signal_client.py`
- Create: `tests/test_signal_client.py`

**Interfaces:**
- Consumes: `model.NetReturnRegressor`, `model.load`.
- Produces: `signal_client.SignalClient(model_path: Path, input_dim: int)`
  with method `predict_batch(self, features: np.ndarray) -> np.ndarray` —
  raises `signal_client.PredictionError` if any input row contains
  NaN/Inf, or if any output prediction is NaN/Inf.

- [ ] **Step 1: Write the failing test (real forward pass, not mocked)**

```python
# tests/test_signal_client.py
from pathlib import Path

import numpy as np
import pytest

from latentedge.model import NetReturnRegressor, save, train
from latentedge.signal_client import PredictionError, SignalClient


@pytest.fixture
def trained_model_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(100, 3)).astype(np.float32)
    y = rng.normal(size=(100,)).astype(np.float32)
    regressor = NetReturnRegressor(input_dim=3)
    train(regressor, x, y, epochs=5, learning_rate=0.01)
    path = tmp_path / "model.safetensors"
    save(regressor, path)
    return path


def test_predict_batch_runs_real_forward_pass(trained_model_path: Path):
    client = SignalClient(trained_model_path, input_dim=3)
    features = np.random.default_rng(3).normal(size=(10, 3)).astype(np.float32)
    predictions = client.predict_batch(features)
    assert predictions.shape == (10,)
    assert not np.isnan(predictions).any()


def test_predict_batch_rejects_nan_input(trained_model_path: Path):
    client = SignalClient(trained_model_path, input_dim=3)
    features = np.array([[1.0, float("nan"), 3.0]], dtype=np.float32)
    with pytest.raises(PredictionError):
        client.predict_batch(features)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_signal_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.signal_client'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/signal_client.py
from pathlib import Path

import numpy as np

from latentedge.model import load


class PredictionError(Exception):
    pass


class SignalClient:
    def __init__(self, model_path: Path, input_dim: int):
        self._model = load(model_path, input_dim=input_dim)

    def predict_batch(self, features: np.ndarray) -> np.ndarray:
        if not np.isfinite(features).all():
            raise PredictionError("input feature matrix contains NaN/Inf")

        import mlx.core as mx

        predictions = np.array(self._model(mx.array(features.astype(np.float32))))

        if not np.isfinite(predictions).all():
            raise PredictionError("model produced NaN/Inf predictions")

        return predictions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_signal_client.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/signal_client.py tests/test_signal_client.py
git commit -m "Add signal client with batched inference and NaN guards"
```

---

## Task 14: Safety-guard (position sizing + daily-loss lockout)

**Files:**
- Create: `src/latentedge/safety_guard.py`
- Create: `tests/test_safety_guard.py`

**Interfaces:**
- Produces: `safety_guard.GuardState` (pydantic model): `equity_usd:
  float`, `daily_loss_usd: float`, `current_day: int` (unix day number),
  `locked_out: bool`.
- Produces: `safety_guard.SafetyGuard(max_position_fraction: float,
  daily_loss_limit_fraction: float)` with method
  `size_position(self, state: GuardState, predicted_return: float,
  timestamp: int) -> tuple[float, GuardState]` — returns `(size_usd,
  updated_state)`. Returns `size_usd == 0.0` when locked out or when
  `predicted_return <= 0`. Resets `daily_loss_usd` and `locked_out` when
  `timestamp` crosses into a new day relative to `state.current_day`.
- Produces: `safety_guard.record_trade_result(state: GuardState,
  pnl_usd: float, daily_loss_limit_fraction: float) -> GuardState` — updates
  equity and daily loss, sets `locked_out = True` if the daily loss limit
  is breached.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_safety_guard.py
from latentedge.safety_guard import GuardState, SafetyGuard, record_trade_result

DAY_SECONDS = 86_400


def test_zero_or_negative_predicted_return_sizes_no_trade():
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05)
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=0.0, current_day=0, locked_out=False)

    size, _ = guard.size_position(state, predicted_return=0.0, timestamp=0)
    assert size == 0.0

    size, _ = guard.size_position(state, predicted_return=-0.01, timestamp=0)
    assert size == 0.0


def test_positive_predicted_return_sizes_a_fraction_of_equity():
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05)
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=0.0, current_day=0, locked_out=False)

    size, _ = guard.size_position(state, predicted_return=0.02, timestamp=0)
    assert 0 < size <= 1_000.0  # at most 10% of equity


def test_daily_loss_limit_locks_out_trading():
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=0.0, current_day=0, locked_out=False)
    state = record_trade_result(state, pnl_usd=-600.0, daily_loss_limit_fraction=0.05)  # -6% > 5% limit
    assert state.locked_out is True

    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05)
    size, _ = guard.size_position(state, predicted_return=0.05, timestamp=100)
    assert size == 0.0


def test_lockout_resets_on_new_day():
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=600.0, current_day=0, locked_out=True)
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05)

    next_day_timestamp = DAY_SECONDS + 10
    size, new_state = guard.size_position(state, predicted_return=0.02, timestamp=next_day_timestamp)

    assert new_state.locked_out is False
    assert new_state.daily_loss_usd == 0.0
    assert new_state.current_day == 1
    assert size > 0.0


def test_lockout_resets_across_multiple_day_boundaries_in_sequence():
    # Regression test: a previous version of this guard never reset and
    # silently froze a long-running backtest partway through. Simulate a
    # sequence of days, each starting locked out from the day before,
    # crossing several boundaries in a row.
    state = GuardState(equity_usd=10_000.0, daily_loss_usd=0.0, current_day=0, locked_out=False)
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.05)

    for day in range(1, 91):
        state = record_trade_result(state, pnl_usd=-600.0, daily_loss_limit_fraction=0.05)
        assert state.locked_out is True

        timestamp = day * DAY_SECONDS + 10
        size, state = guard.size_position(state, predicted_return=0.02, timestamp=timestamp)

        assert state.locked_out is False, f"guard stayed locked at day {day}"
        assert size > 0.0, f"guard produced no trade at day {day}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_safety_guard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.safety_guard'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/safety_guard.py
from pydantic import BaseModel

DAY_SECONDS = 86_400


class GuardState(BaseModel):
    equity_usd: float
    daily_loss_usd: float
    current_day: int
    locked_out: bool


def _day_for_timestamp(timestamp: int) -> int:
    return timestamp // DAY_SECONDS


def _roll_to_day_if_needed(state: GuardState, timestamp: int) -> GuardState:
    day = _day_for_timestamp(timestamp)
    if day == state.current_day:
        return state
    return state.model_copy(update={"current_day": day, "daily_loss_usd": 0.0, "locked_out": False})


class SafetyGuard(BaseModel):
    max_position_fraction: float
    daily_loss_limit_fraction: float

    def size_position(self, state: GuardState, predicted_return: float, timestamp: int) -> tuple[float, GuardState]:
        state = _roll_to_day_if_needed(state, timestamp)

        if state.locked_out or predicted_return <= 0:
            return 0.0, state

        size_usd = state.equity_usd * self.max_position_fraction
        return size_usd, state


def record_trade_result(state: GuardState, pnl_usd: float, daily_loss_limit_fraction: float) -> GuardState:
    new_equity = state.equity_usd + pnl_usd
    new_daily_loss = state.daily_loss_usd + max(0.0, -pnl_usd)
    limit_usd = state.equity_usd * daily_loss_limit_fraction
    locked_out = new_daily_loss >= limit_usd

    return state.model_copy(update={"equity_usd": new_equity, "daily_loss_usd": new_daily_loss, "locked_out": locked_out})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_safety_guard.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/safety_guard.py tests/test_safety_guard.py
git commit -m "Add safety-guard with position sizing and daily-loss lockout"
```

---

## Task 15: Executor (simulated fills)

**Files:**
- Create: `src/latentedge/executor.py`
- Create: `tests/test_executor.py`

**Interfaces:**
- Consumes: `costs.estimate_slippage_fraction`, `costs.estimate_gas_cost_usd`,
  `config.FEE_TIER_BPS`, `config.GAS_USED_PER_SWAP`.
- Produces: `executor.simulate_fill(size_usd: float, entry_price: float,
  exit_price: float, entry_swap: dict, exit_swap: dict) -> float` —
  returns net P&L in USD for a trade of `size_usd`, recomputing slippage
  and gas at that size (not the label's $1,000 reference). Returns `0.0`
  immediately if `size_usd <= 0` (no trade).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_executor.py
from latentedge.executor import simulate_fill
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _swap_dict(human_price: float, liquidity: int = 10**18, base_fee_wei: int = 20_000_000_000) -> dict:
    raw_price = (1.0 / human_price) * (10 ** (18 - 6)) ** -1
    return {"sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18), "liquidity": liquidity, "base_fee_wei": base_fee_wei}


def test_zero_size_produces_zero_pnl_with_no_computation():
    pnl = simulate_fill(0.0, entry_price=3000.0, exit_price=3100.0, entry_swap=_swap_dict(3000.0), exit_swap=_swap_dict(3100.0))
    assert pnl == 0.0


def test_profitable_move_produces_positive_pnl_net_of_costs():
    pnl = simulate_fill(1000.0, entry_price=3000.0, exit_price=3200.0, entry_swap=_swap_dict(3000.0), exit_swap=_swap_dict(3200.0))
    raw_pnl = 1000.0 * (3200.0 - 3000.0) / 3000.0
    assert 0 < pnl < raw_pnl  # positive but less than the uncosted move


def test_larger_size_incurs_proportionally_more_slippage_cost():
    small_pnl = simulate_fill(1000.0, entry_price=3000.0, exit_price=3010.0, entry_swap=_swap_dict(3000.0), exit_swap=_swap_dict(3010.0))
    large_pnl = simulate_fill(50_000.0, entry_price=3000.0, exit_price=3010.0, entry_swap=_swap_dict(3000.0), exit_swap=_swap_dict(3010.0))
    small_return_fraction = small_pnl / 1000.0
    large_return_fraction = large_pnl / 50_000.0
    assert large_return_fraction < small_return_fraction  # larger trade eats more slippage per dollar
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_executor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.executor'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/executor.py
from latentedge import config
from latentedge.costs import estimate_gas_cost_usd, estimate_slippage_fraction

FEE_FRACTION = config.FEE_TIER_BPS / 10_000


def simulate_fill(size_usd: float, entry_price: float, exit_price: float, entry_swap: dict, exit_swap: dict) -> float:
    if size_usd <= 0:
        return 0.0

    raw_return = (exit_price - entry_price) / entry_price
    gross_pnl = size_usd * raw_return

    fee_cost = 2 * size_usd * FEE_FRACTION

    entry_slippage = estimate_slippage_fraction(size_usd, entry_swap["liquidity"], entry_swap["sqrt_price_x96"])
    exit_slippage = estimate_slippage_fraction(size_usd, exit_swap["liquidity"], exit_swap["sqrt_price_x96"])
    slippage_cost = size_usd * (entry_slippage + exit_slippage)

    entry_gas = estimate_gas_cost_usd(entry_swap["base_fee_wei"], config.GAS_USED_PER_SWAP, entry_price)
    exit_gas = estimate_gas_cost_usd(exit_swap["base_fee_wei"], config.GAS_USED_PER_SWAP, exit_price)
    gas_cost = entry_gas + exit_gas

    return gross_pnl - fee_cost - slippage_cost - gas_cost
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_executor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/executor.py tests/test_executor.py
git commit -m "Add executor with size-dependent slippage/gas recomputation"
```

---

## Task 16: Backtest harness

**Files:**
- Create: `src/latentedge/backtest.py`
- Create: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `signal_client.SignalClient.predict_batch`,
  `safety_guard.SafetyGuard`, `safety_guard.GuardState`,
  `safety_guard.record_trade_result`, `executor.simulate_fill`.
- Produces: `backtest.BacktestResult` (pydantic model): `total_return_usd:
  float`, `max_drawdown_usd: float`, `win_rate: float`, `num_trades: int`,
  `equity_curve: list[float]`.
- Produces: `backtest.run_backtest(features: np.ndarray, entry_prices:
  np.ndarray, exit_prices: np.ndarray, entry_swaps: list[dict],
  exit_swaps: list[dict], signal_client: SignalClient, guard: SafetyGuard,
  initial_equity_usd: float, timestamps: np.ndarray) -> BacktestResult`.

- [ ] **Step 1: Write the failing test (hand-computable expected P&L)**

```python
# tests/test_backtest.py
from pathlib import Path

import numpy as np
import pytest

from latentedge.backtest import run_backtest
from latentedge.model import NetReturnRegressor, save, train
from latentedge.safety_guard import SafetyGuard
from latentedge.signal_client import SignalClient
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _swap_dict(human_price: float) -> dict:
    raw_price = (1.0 / human_price) * (10 ** (18 - 6)) ** -1
    return {"sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18), "liquidity": 10**18, "base_fee_wei": 20_000_000_000}


def test_backtest_produces_hand_computable_results(tmp_path: Path):
    # Train a trivial model that always predicts a positive number, by
    # fitting to an all-positive constant target — makes trade outcomes
    # (always taken) hand-computable from simulate_fill's own math.
    x = np.zeros((20, 1), dtype=np.float32)
    y = np.full(20, 0.02, dtype=np.float32)
    regressor = NetReturnRegressor(input_dim=1)
    train(regressor, x, y, epochs=200, learning_rate=0.05)
    model_path = tmp_path / "model.safetensors"
    save(regressor, model_path)

    client = SignalClient(model_path, input_dim=1)
    guard = SafetyGuard(max_position_fraction=0.1, daily_loss_limit_fraction=0.5)

    n_bars = 3
    features = np.zeros((n_bars, 1), dtype=np.float32)
    entry_prices = np.array([3000.0, 3050.0, 3100.0])
    exit_prices = np.array([3050.0, 3100.0, 3150.0])
    entry_swaps = [_swap_dict(p) for p in entry_prices]
    exit_swaps = [_swap_dict(p) for p in exit_prices]
    timestamps = np.array([0, 3600, 7200])

    result = run_backtest(
        features=features,
        entry_prices=entry_prices,
        exit_prices=exit_prices,
        entry_swaps=entry_swaps,
        exit_swaps=exit_swaps,
        signal_client=client,
        guard=guard,
        initial_equity_usd=10_000.0,
        timestamps=timestamps,
    )

    assert result.num_trades == 3
    assert result.total_return_usd > 0  # every simulated move here is profitable
    assert 0.0 <= result.win_rate <= 1.0
    assert len(result.equity_curve) == 3
    assert result.equity_curve[-1] == pytest.approx(10_000.0 + result.total_return_usd)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.backtest'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/backtest.py
import numpy as np
from pydantic import BaseModel

from latentedge.executor import simulate_fill
from latentedge.safety_guard import GuardState, SafetyGuard, record_trade_result
from latentedge.signal_client import SignalClient


class BacktestResult(BaseModel):
    total_return_usd: float
    max_drawdown_usd: float
    win_rate: float
    num_trades: int
    equity_curve: list[float]


def run_backtest(
    features: np.ndarray,
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    entry_swaps: list[dict],
    exit_swaps: list[dict],
    signal_client: SignalClient,
    guard: SafetyGuard,
    initial_equity_usd: float,
    timestamps: np.ndarray,
) -> BacktestResult:
    predictions = signal_client.predict_batch(features)

    state = GuardState(equity_usd=initial_equity_usd, daily_loss_usd=0.0, current_day=0, locked_out=False)

    equity_curve: list[float] = []
    wins = 0
    num_trades = 0
    peak_equity = initial_equity_usd
    max_drawdown = 0.0

    for i in range(len(features)):
        size_usd, state = guard.size_position(state, predicted_return=float(predictions[i]), timestamp=int(timestamps[i]))

        if size_usd > 0:
            pnl = simulate_fill(size_usd, float(entry_prices[i]), float(exit_prices[i]), entry_swaps[i], exit_swaps[i])
            num_trades += 1
            if pnl > 0:
                wins += 1
            state = record_trade_result(state, pnl_usd=pnl, daily_loss_limit_fraction=guard.daily_loss_limit_fraction)

        equity_curve.append(state.equity_usd)
        peak_equity = max(peak_equity, state.equity_usd)
        max_drawdown = max(max_drawdown, peak_equity - state.equity_usd)

    total_return_usd = state.equity_usd - initial_equity_usd
    win_rate = wins / num_trades if num_trades > 0 else 0.0

    return BacktestResult(
        total_return_usd=total_return_usd,
        max_drawdown_usd=max_drawdown,
        win_rate=win_rate,
        num_trades=num_trades,
        equity_curve=equity_curve,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/backtest.py tests/test_backtest.py
git commit -m "Add backtest harness with equity/drawdown/win-rate tracking"
```

---

## Task 17: CLI wiring

**Files:**
- Create: `src/latentedge/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: every module above.
- Produces: `latentedge` console script with three subcommands: `ingest`,
  `train`, `backtest`. Each is a thin wrapper — no new business logic — so
  the CLI is smoke-tested for wiring correctness, not re-testing
  already-covered logic.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from click.testing import CliRunner

from latentedge.cli import cli


def test_cli_exposes_expected_subcommands():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "train" in result.output
    assert "backtest" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'latentedge.cli'`

- [ ] **Step 3: Write the implementation**

```python
# src/latentedge/cli.py
from pathlib import Path

import click
import httpx
import pandas as pd

from latentedge import config
from latentedge.bars import build_bars
from latentedge.features import compute_features
from latentedge.ingest.pool_state import backfill_base_fee
from latentedge.ingest.rpc_logs import fetch_swaps
from latentedge.labeling import label_bars
from latentedge.model import NetReturnRegressor, save, train as train_model
from latentedge.split import chronological_split
from latentedge.store import read_swaps, write_swaps


@click.group()
def cli() -> None:
    """latentedge: ingest, train, and backtest the v1 WETH/USDC pipeline."""


DEFAULT_RPC_URL = "https://eth.llamarpc.com"


@cli.command()
@click.option("--from-block", type=int, required=True)
@click.option("--to-block", type=int, required=True)
@click.option("--rpc-url", type=str, default=DEFAULT_RPC_URL)
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
def ingest(from_block: int, to_block: int, rpc_url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=30.0) as client:
        records = fetch_swaps(config.POOL_ADDRESS, from_block, to_block, client, rpc_url)
        records = backfill_base_fee(records, client, rpc_url)
    write_swaps(records, out)
    click.echo(f"wrote {len(records)} swap records to {out}")


@cli.command()
@click.option("--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--out", type=click.Path(path_type=Path), default=Path("data/model.safetensors"))
@click.option("--epochs", type=int, default=100)
def train(swaps: Path, out: Path, epochs: int) -> None:
    swap_df = read_swaps(swaps)
    bar_df = build_bars(swap_df, config.BAR_INTERVAL_SECONDS)
    labeled = label_bars(bar_df, swap_df, tp_sl_fraction=0.01)
    labeled = labeled[~labeled["excluded"]].reset_index(drop=True)
    featured = compute_features(labeled, return_windows=[5, 15, 30], volatility_window=15)
    featured = featured.dropna().reset_index(drop=True)

    train_split, _validate_split, _test_split = chronological_split(featured, train_fraction=0.7, validate_fraction=0.15)

    feature_columns = ["return_5", "return_15", "return_30", "volatility", "volume_usdc", "bars_since_swap"]
    x = train_split[feature_columns].to_numpy(dtype="float32")
    y = train_split["net_return"].to_numpy(dtype="float32")

    regressor = NetReturnRegressor(input_dim=len(feature_columns))
    losses = train_model(regressor, x, y, epochs=epochs, learning_rate=0.001)

    out.parent.mkdir(parents=True, exist_ok=True)
    save(regressor, out)
    click.echo(f"trained {epochs} epochs, final loss {losses[-1]:.6f}, saved to {out}")


@cli.command()
@click.option("--swaps", type=click.Path(path_type=Path), default=Path("data/swaps.parquet"))
@click.option("--model", type=click.Path(path_type=Path), default=Path("data/model.safetensors"))
def backtest(swaps: Path, model: Path) -> None:
    click.echo("backtest command wiring — see backtest.run_backtest for the underlying logic")
```

Note for the implementer: the `backtest` subcommand above is intentionally
a thin stub — assembling the exact `entry_swaps`/`exit_swaps` arrays
`run_backtest` needs from the labeled+featured dataframe is a real piece
of glue code, not just wiring, and belongs in its own reviewed step once
Tasks 1–16 are all merged and the real column shapes are in hand. Flesh
it out then, following the same pattern as `train` above.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/latentedge/cli.py tests/test_cli.py
git commit -m "Add CLI wiring for ingest/train/backtest commands"
```

---

## Task 18: Full pipeline smoke test

**Files:**
- Create: `tests/test_pipeline_smoke.py`

**Interfaces:**
- Consumes: every module above.
- Produces: nothing new — this is a synthetic-data, no-network,
  end-to-end run confirming every stage's output shape feeds the next
  stage correctly, catching integration seams the per-module tests can't
  see.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline_smoke.py
from pathlib import Path

import numpy as np
import pandas as pd

from latentedge.bars import build_bars
from latentedge.features import compute_features
from latentedge.labeling import label_bars
from latentedge.model import NetReturnRegressor, save, train
from latentedge.signal_client import SignalClient
from latentedge.split import chronological_split
from latentedge.uniswap_math import price_to_sqrt_price_x96

FEATURE_COLUMNS = ["return_5", "return_15", "return_30", "volatility", "volume_usdc", "bars_since_swap"]


def _synthetic_swaps(n_minutes: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    price = 3000.0
    for minute in range(n_minutes):
        if rng.random() < 0.7:  # most minutes have at least one swap
            price *= 1 + rng.normal(0, 0.001)
            raw_price = (1.0 / price) * (10 ** (18 - 6)) ** -1
            rows.append(
                {
                    "timestamp": minute * 60,
                    "sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18),
                    "liquidity": 10**18,
                    "base_fee_wei": 20_000_000_000,
                    "amount0": abs(rng.normal(1000, 200)),
                }
            )
    return pd.DataFrame(rows)


def test_full_pipeline_runs_end_to_end_on_synthetic_data(tmp_path: Path):
    swaps = _synthetic_swaps(n_minutes=5000)

    bars = build_bars(swaps, interval_seconds=60)
    labeled = label_bars(bars, swaps, tp_sl_fraction=0.01)
    labeled = labeled[~labeled["excluded"]].reset_index(drop=True)
    assert len(labeled) > 0

    featured = compute_features(labeled, return_windows=[5, 15, 30], volatility_window=15)
    featured = featured.dropna().reset_index(drop=True)
    assert len(featured) > 100

    train_split, validate_split, test_split = chronological_split(featured, train_fraction=0.7, validate_fraction=0.15)

    x = train_split[FEATURE_COLUMNS].to_numpy(dtype="float32")
    y = train_split["net_return"].to_numpy(dtype="float32")
    regressor = NetReturnRegressor(input_dim=len(FEATURE_COLUMNS))
    train(regressor, x, y, epochs=10, learning_rate=0.001)

    model_path = tmp_path / "model.safetensors"
    save(regressor, model_path)

    client = SignalClient(model_path, input_dim=len(FEATURE_COLUMNS))
    test_features = test_split[FEATURE_COLUMNS].to_numpy(dtype="float32")
    predictions = client.predict_batch(test_features)

    assert predictions.shape == (len(test_split),)
    assert np.isfinite(predictions).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline_smoke.py -v`
Expected: FAIL initially if run before Tasks 1–14 exist; once they do,
this test should already pass on first run since it only composes
existing, already-tested functions — if it fails at that point, the
failure is a real integration bug between two modules whose unit tests
individually passed, and is exactly the kind of thing this task exists to
catch.

- [ ] **Step 3: (No new implementation — this task wires existing modules)**

If Step 2 fails due to a real integration mismatch (e.g. a column name
that Task 7 produces but Task 9 expects under a different name), fix the
mismatch in the relevant module from the earlier task, not by adding
translation logic here.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_pipeline_smoke.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_pipeline_smoke.py
git commit -m "Add full pipeline smoke test on synthetic data"
```

---

## Final step: run the whole suite

- [ ] Run: `uv run pytest -v` — all tests pass, including the real-service
  integration tests in Tasks 4–6 (these need network access; skip only if
  explicitly offline, and note in the final commit if any were skipped).
- [ ] Run: `uv run mypy src/` — no errors under strict mode.
