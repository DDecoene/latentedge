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

# Default --days window when --from-block/--to-block are omitted (the
# exact start block is found on-chain, not estimated — see
# get_block_at_or_after_timestamp). A small buffer behind the fetched
# chain head avoids "block range extends beyond current head block"
# from a load-balanced RPC provider whose backends can lag each other
# by a few blocks (see tests/ingest/test_rpc_logs.py's real-network
# test for where this was first observed).
DEFAULT_INGEST_DAYS = 365.0
HEAD_BLOCK_SAFETY_BUFFER = 5

# WETH/USDC 0.05% pool deployment block — no swap history is possible
# before it, so a --days window that turns out to already be fully
# ingested stops walking backward here rather than requesting
# pre-deployment blocks. Found by binary-searching eth_getCode against
# an archive RPC endpoint.
POOL_CREATION_BLOCK = 12_376_729
