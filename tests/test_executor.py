from latentedge.executor import simulate_fill
from latentedge.uniswap_math import price_to_sqrt_price_x96


def _swap_dict(human_price: float, liquidity: int = 10**18, base_fee_wei: int = 20_000_000_000) -> dict:
    # See Task 2's ruling: price_to_sqrt_price_x96 already decimal-adjusts
    # internally, so pass 1.0 / human_price directly, no extra scaling.
    raw_price = 1.0 / human_price
    return {"sqrt_price_x96": price_to_sqrt_price_x96(raw_price, decimals0=6, decimals1=18), "liquidity": liquidity, "base_fee_wei": base_fee_wei}


def test_zero_size_produces_zero_pnl_with_no_computation():
    pnl = simulate_fill(0.0, entry_price=3000.0, exit_price=3100.0, entry_swap=_swap_dict(3000.0), exit_swap=_swap_dict(3100.0))
    assert pnl == 0.0


def test_profitable_move_produces_positive_pnl_net_of_costs():
    pnl = simulate_fill(1000.0, entry_price=3000.0, exit_price=3200.0, entry_swap=_swap_dict(3000.0), exit_swap=_swap_dict(3200.0))
    raw_pnl = 1000.0 * (3200.0 - 3000.0) / 3000.0
    assert 0 < pnl < raw_pnl  # positive but less than the uncosted move


def test_larger_size_incurs_proportionally_more_slippage_cost():
    # Gas is a fixed dollar cost per trade, so as a fraction of size it
    # *shrinks* for larger trades — the opposite direction from slippage,
    # which grows as a fraction of size. Isolating slippage's effect (the
    # thing this test is actually about) means removing gas from the
    # picture; a mixed-cost comparison would test whichever cost happens
    # to dominate at the chosen scale, not slippage scaling specifically.
    zero_gas_swap = lambda price: _swap_dict(price, base_fee_wei=0)
    small_pnl = simulate_fill(1000.0, entry_price=3000.0, exit_price=3010.0, entry_swap=zero_gas_swap(3000.0), exit_swap=zero_gas_swap(3010.0))
    large_pnl = simulate_fill(50_000.0, entry_price=3000.0, exit_price=3010.0, entry_swap=zero_gas_swap(3000.0), exit_swap=zero_gas_swap(3010.0))
    small_return_fraction = small_pnl / 1000.0
    large_return_fraction = large_pnl / 50_000.0
    assert large_return_fraction < small_return_fraction  # larger trade eats more slippage per dollar
