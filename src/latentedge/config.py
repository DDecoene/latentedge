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

# For turning a --days window into a block range when --from-block/
# --to-block are omitted. 12.0s is Ethereum's fixed post-merge slot
# time; a small buffer behind the fetched chain head avoids "block
# range extends beyond current head block" from a load-balanced RPC
# provider whose backends can lag each other by a few blocks (see
# tests/ingest/test_rpc_logs.py's real-network test for where this was
# first observed).
AVG_BLOCK_SECONDS = 12.0
DEFAULT_INGEST_DAYS = 365.0
HEAD_BLOCK_SAFETY_BUFFER = 5
