from datetime import datetime
from pathlib import Path
from src.strategy.intraday_data import IntradaySnapshot
from src.strategy.market_structure import MarketStructureAnalyzer
from src.strategy.trade_reflection import (
    TradeMistakeCategory,
    TradeReflectionEngine,
    TradeReflectionRecord,
)


def test_market_structure_analyzer():
    print("\n--- [1] Test MarketStructureAnalyzer ---")
    ms = MarketStructureAnalyzer()

    # 1. Normální odraz od VWAP (vstup blízko férové ceny)
    snap_good = IntradaySnapshot(
        ticker="NVDA",
        current_price=200.5,
        vwap=200.0,
        is_above_vwap=True,
        ema_9=200.4,
        ema_21=200.1,
        is_bullish_ema_cross=True,
        is_fresh_cross=False,
        rsi_5m=52.0,
        volume_surge_ratio=1.6,
        atr_5m=1.2,
        high_30m=201.0,
        low_30m=199.0,
        is_breaking_high=False,
        change_pct=0.8,
        timestamp=datetime.now(),
    )
    res_good = ms.analyze_snapshot(snap_good)
    assert not res_good.is_overextended_from_vwap
    assert not res_good.is_liquidity_sweep_suspect
    assert res_good.score_adjustment > 0
    print(f"  [OK] Zdravý odraz od VWAP: bonus {res_good.score_adjustment:+.1f} b")

    # 2. Přepálená vzdálenost od VWAP (> 1.4 %)
    snap_extended = IntradaySnapshot(
        ticker="NVDA",
        current_price=204.0,
        vwap=200.0,
        is_above_vwap=True,
        ema_9=203.5,
        ema_21=202.0,
        is_bullish_ema_cross=True,
        is_fresh_cross=False,
        rsi_5m=68.0,
        volume_surge_ratio=1.8,
        atr_5m=1.2,
        high_30m=204.0,
        low_30m=199.0,
        is_breaking_high=True,
        change_pct=2.5,
        timestamp=datetime.now(),
    )
    res_ext = ms.analyze_snapshot(snap_extended)
    assert res_ext.is_overextended_from_vwap
    assert res_ext.score_adjustment < 0
    print(f"  [OK] Přepálený VWAP ({res_ext.vwap_distance_pct:.2f} %): penalizace {res_ext.score_adjustment:+.1f} b")

    # 3. Liquidity Sweep / Fakeout (proražení High s extrémním RSI a objemem)
    snap_sweep = IntradaySnapshot(
        ticker="TQQQ",
        current_price=85.0,
        vwap=83.0,
        is_above_vwap=True,
        ema_9=84.5,
        ema_21=83.8,
        is_bullish_ema_cross=True,
        is_fresh_cross=False,
        rsi_5m=75.0,
        volume_surge_ratio=3.2,
        atr_5m=1.5,
        high_30m=85.0,
        low_30m=82.0,
        is_breaking_high=True,
        change_pct=3.8,
        timestamp=datetime.now(),
    )
    res_sweep = ms.analyze_snapshot(snap_sweep)
    assert res_sweep.is_liquidity_sweep_suspect
    assert res_sweep.score_adjustment <= -20.0
    print(f"  [OK] Detekován Liquidity Sweep (Turtle Soup): penalizace {res_sweep.score_adjustment:+.1f} b")

    # 4. Volatility Drag pro 3x páková ETF
    drag = ms.calculate_volatility_drag_pct(leverage=3.0, atr_5m=1.5, price=85.0)
    assert drag > 0
    print(f"  [OK] Odhadovaný pákový rozpad pro TQQQ: {drag:.1f} % p.a.")


def test_trade_reflection_engine():
    print("\n--- [2] Test TradeReflectionEngine ---")
    test_mem_file = Path("data") / "test_trade_memory.json"
    if test_mem_file.exists():
        test_mem_file.unlink()

    engine = TradeReflectionEngine(memory_file=test_mem_file)

    # 1. Reflexe ztrátového obchodu s překoupeným RSI
    dummy_snap_loss = IntradaySnapshot(
        ticker="SOXL",
        current_price=42.0,
        vwap=41.5,
        is_above_vwap=True,
        ema_9=41.8,
        ema_21=41.2,
        is_bullish_ema_cross=True,
        is_fresh_cross=False,
        rsi_5m=71.5,
        volume_surge_ratio=2.0,
        atr_5m=1.1,
        high_30m=42.0,
        low_30m=40.0,
        is_breaking_high=True,
        change_pct=3.0,
        timestamp=datetime.now(),
    )

    refl = engine.record_and_reflect(
        ticker="SOXL",
        entry_price=42.0,
        exit_price=41.5,
        pnl_pct=-1.19,
        allocated_eur=3300.0,
        exit_reason="STOP_LOSS",
        entry_snapshot=dummy_snap_loss,
        regime_name="NEUTRAL_CHOPPY",
    )

    assert refl.pnl_pct < 0
    assert refl.mistake_category == TradeMistakeCategory.OVERBOUGHT_RSI
    assert "max_entry_rsi" in refl.prevention_rule
    print(f"  [OK] AI Reflexe ztráty SOXL: Kategorie '{refl.mistake_category.value}' | Pravidlo: {refl.prevention_rule}")

    # 2. Prověření paměti: pokus znovu nakoupit SOXL při překoupeném RSI (> 66)
    check_veto = engine.check_candidate_against_memory(
        ticker="SOXL",
        current_price=42.2,
        rsi_5m=69.0,
        vwap=41.5,
        score=72.0,
    )
    assert not check_veto["approved"]
    print(f"  [OK] Paměť chyb udělila VETO: {check_veto['reason']}")

    # 3. Prověření paměti: nákup SOXL po zklidnění indikátoru (RSI 54) -> SCHVÁLENO
    check_ok = engine.check_candidate_against_memory(
        ticker="SOXL",
        current_price=41.6,
        rsi_5m=54.0,
        vwap=41.5,
        score=75.0,
    )
    assert check_ok["approved"]
    print(f"  [OK] Po zklidnění indikátoru paměť nákup schválila: {check_ok['reason']}")

    # Úklid testovacího souboru
    if test_mem_file.exists():
        test_mem_file.unlink()


if __name__ == "__main__":
    test_market_structure_analyzer()
    test_trade_reflection_engine()
    print("\n[OK] Vsechny testy trzni struktury a sebereflexe uspesne prosly!\n")
