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
