"""
Production FastAPI Service for RL Portfolio Optimization.

Provides RESTful API for portfolio recommendations, risk monitoring,
and client reporting.

"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# code/ directory: config lives at code/config/config.yaml
_ROOT = Path(__file__).resolve().parent.parent

# Ensure code/ is on sys.path so sibling modules (agents, etc.) are importable
_CODE_DIR = str(_ROOT)
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

with open(_ROOT / "config" / "config.yaml") as _f:
    config: Dict[str, Any] = yaml.safe_load(_f)

# ---------------------------------------------------------------------------
# Model cache (populated at startup)
# ---------------------------------------------------------------------------

_model_cache: Dict[str, Any] = {}


def _load_all_models() -> None:
    """Load every model in models_dir into memory at startup."""
    from stable_baselines3 import DDPG, PPO, SAC

    _SB3_MAP = {"ppo": PPO, "ddpg": DDPG, "sac": SAC}
    models_dir = Path(config["output"]["models_dir"])

    if not models_dir.exists():
        logger.warning("Models directory not found: %s", models_dir)
        return

    for path in models_dir.glob("*.zip"):
        stem = path.stem  # e.g. "ppo_seed_0"
        prefix = stem.split("_seed_")[0]  # e.g. "ppo"
        if prefix in _SB3_MAP:
            try:
                _model_cache[stem] = _SB3_MAP[prefix].load(str(path.with_suffix("")))
                logger.info("Loaded model: %s", stem)
            except Exception as exc:
                logger.warning("Could not load %s: %s", stem, exc)

    # Custom QR-DDPG models (.pt)
    for path in models_dir.glob("*.pt"):
        try:
            _model_cache[path.stem] = {"checkpoint": path, "type": "qr_ddpg"}
            logger.info("Registered QR-DDPG checkpoint: %s", path.stem)
        except Exception as exc:
            logger.warning("Could not register %s: %s", path.stem, exc)


def _get_cached_model(model_name: str):
    """Return a cached model or raise 404."""
    # Prefer exact match; fall back to first seed
    if model_name in _model_cache:
        return _model_cache[model_name]
    # Try seed 0 fallback
    fallback = f"{model_name}_seed_0"
    if fallback in _model_cache:
        return _model_cache[fallback]
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting RL Portfolio Optimization API")
    _load_all_models()
    logger.info("Models loaded: %s", list(_model_cache.keys()))
    yield
    logger.info("Shutting down API")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

_allowed_origins: List[str] = (
    config.get("production", {})
    .get("api", {})
    .get(
        "allowed_origins",
        ["http://localhost", "http://localhost:3000"],  # safe default
    )
)

app = FastAPI(
    title="RL Portfolio Optimization API",
    description="Production API for AI-powered portfolio management",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,  # explicit allowlist, not "*"
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class PortfolioRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    risk_tolerance: str = Field(..., description="low | medium | high")
    investment_amount: float = Field(..., gt=0, description="Investment amount in USD")
    constraints: Optional[Dict] = Field(None, description="Custom constraints")


class PortfolioRecommendation(BaseModel):
    client_id: str
    timestamp: datetime
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    max_drawdown_estimate: float
    confidence_score: float


class RiskAlert(BaseModel):
    alert_id: str
    client_id: str
    timestamp: datetime
    alert_type: str
    severity: str
    message: str
    current_value: float
    threshold: float


class PerformanceMetrics(BaseModel):
    client_id: str
    period: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _get_all_tickers() -> List[str]:
    """Return the investable ticker list from config (excludes macro factors)."""
    assets = config["data"]["assets"]
    macro = set(config["data"].get("macro_factors", []))
    tickers: List[str] = []
    for cls_tickers in assets.values():
        tickers.extend([t for t in cls_tickers if t not in macro])
    return tickers


def _calculate_risk_metrics(weights: np.ndarray, returns_data: pd.DataFrame) -> Dict:
    mean_ret = returns_data.mean().values
    cov = returns_data.cov().values

    port_ret = float(np.dot(weights, mean_ret) * 252)
    port_vol = float(np.sqrt(weights @ cov @ weights) * np.sqrt(252))
    sharpe = (port_ret - 0.045) / port_vol if port_vol > 0 else 0.0

    return {
        "expected_return": port_ret * 100,
        "expected_volatility": port_vol * 100,
        "sharpe_ratio": sharpe,
    }


# Cache for the lazily-built inference dataset / environment so we do not
# re-run the data pipeline on every request.
_inference_cache: Dict[str, Any] = {}


def _get_inference_data() -> "pd.DataFrame":
    """
    Build (and cache) the most recent processed market data used for
    inference. Falls back to synthetic data when live data is unavailable
    (controlled by ``data.use_synthetic_data`` and automatic fetch fallback).
    """
    if "data" in _inference_cache:
        return _inference_cache["data"]

    sys.path.insert(0, str(_ROOT))
    from data import DataProcessor

    processor = DataProcessor(config)
    _, test_data = processor.process_all()
    _inference_cache["data"] = test_data
    return test_data


def _build_market_state(all_tickers: List[str]) -> np.ndarray:
    """
    Construct the current market state vector for model inference.

    Builds a live ``PortfolioEnv`` from the most recent processed market data
    and returns its terminal state (the latest available market snapshot). If
    the data pipeline is unavailable for any reason, falls back to a zero
    vector of the correct dimension so the API stays responsive.
    """
    n_assets = len(all_tickers)
    n_features = 6  # Close, MACD, RSI, CCI, DX, BollUB
    state_dim = 1 + n_assets + n_assets * n_features

    try:
        from environment import PortfolioEnv

        data = _get_inference_data()
        # Restrict to the investable universe, in the same order as all_tickers.
        data = data[data["tic"].isin(all_tickers)]
        if data.empty:
            raise RuntimeError("no processed rows for the configured tickers")

        env = PortfolioEnv(
            df=data,
            initial_amount=config["environment"]["initial_amount"],
            transaction_cost_pct=config["environment"]["transaction_cost_pct"],
            max_drawdown_penalty=config["risk"]["max_drawdown_penalty"],
            hmax=config["environment"].get("hmax", 0.30),
            print_verbosity=0,
        )
        # Advance to the final available snapshot to get the freshest state.
        env.reset()
        env.current_step = env.n_dates - 1
        state = env._get_state()
        if state.shape[0] != state_dim:
            # Universe mismatch (e.g. some tickers dropped); pad / truncate.
            fixed = np.zeros(state_dim, dtype=np.float32)
            fixed[: min(state_dim, state.shape[0])] = state[
                : min(state_dim, state.shape[0])
            ]
            return fixed
        return state.astype(np.float32)
    except Exception as exc:
        logger.warning(
            "Could not build live market state (%s); using zero vector.", exc
        )
        return np.zeros(state_dim, dtype=np.float32)


def _estimate_recommendation_risk(
    weights: np.ndarray, all_tickers: List[str]
) -> Dict[str, float]:
    """
    Estimate expected return / volatility / Sharpe and a drawdown estimate for
    a candidate weight vector from recent realised returns. Returns an empty
    dict if the underlying return data cannot be assembled.
    """
    try:
        data = _get_inference_data()
        pivot = (
            data[data["tic"].isin(all_tickers)]
            .pivot(index="Date", columns="tic", values="Close")
            .reindex(columns=all_tickers)
            .ffill()
            .bfill()
        )
        returns = pivot.pct_change().dropna()
        if returns.empty:
            return {}
        metrics = _calculate_risk_metrics(weights, returns)

        # Historical max drawdown of the weighted portfolio return series.
        port_returns = (returns.values * weights).sum(axis=1)
        equity = np.cumprod(1.0 + port_returns)
        peak = np.maximum.accumulate(equity)
        drawdown = (peak - equity) / peak
        metrics["max_drawdown_estimate"] = -float(drawdown.max()) * 100
        return metrics
    except Exception as exc:
        logger.warning("Risk-metric estimation failed: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {
        "message": "RL Portfolio Optimization API",
        "version": "1.0.0",
        "status": "operational",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(_model_cache),
    }


@app.post("/api/v1/portfolio/recommend", response_model=PortfolioRecommendation)
async def get_portfolio_recommendation(request: PortfolioRequest):
    """Return portfolio weight recommendations from the appropriate trained agent."""
    logger.info("Recommendation request for client %s", request.client_id)

    risk_to_model = {"low": "qr_ddpg", "medium": "ppo", "high": "sac"}
    model_name = risk_to_model.get(request.risk_tolerance, "ppo")
    model = _get_cached_model(model_name)

    all_tickers = _get_all_tickers()
    state = _build_market_state(all_tickers)

    try:
        if isinstance(model, dict) and model.get("type") == "qr_ddpg":
            # Lazy-load QR-DDPG on first use
            import torch
            from agents import QRDDPGAgent

            n = len(all_tickers)
            dim = 1 + n + n * 6
            agent = QRDDPGAgent(state_dim=dim, action_dim=n)
            ckpt = torch.load(
                model["checkpoint"], map_location="cpu", weights_only=True
            )
            agent.actor.load_state_dict(ckpt["actor_state_dict"])
            raw_weights = agent.select_action(state, noise=0.0)
        else:
            raw_weights, _ = model.predict(state[np.newaxis], deterministic=True)
            raw_weights = raw_weights[0]
    except Exception as exc:
        logger.error("Inference error: %s", exc)
        raise HTTPException(status_code=500, detail="Model inference failed") from exc

    weights = np.clip(raw_weights, 0, 1)
    _w_sum = weights.sum()
    # If all weights clip to zero, fall back to equal-weight allocation
    weights = (
        weights / _w_sum if _w_sum > 0 else np.ones(len(all_tickers)) / len(all_tickers)
    )

    weight_dict = {t: round(float(w), 6) for t, w in zip(all_tickers, weights)}

    # Estimate forward-looking risk metrics from recent realised returns.
    risk_metrics = _estimate_recommendation_risk(weights, all_tickers)

    # Confidence proxy: how concentrated the allocation is relative to an
    # equal-weight portfolio (a sharper, more decisive allocation ⇒ higher
    # confidence). Bounded to [0.5, 0.99].
    equal_weight = 1.0 / len(all_tickers)
    concentration = float(np.abs(weights - equal_weight).sum())
    confidence_score = float(np.clip(0.5 + 0.5 * concentration, 0.5, 0.99))

    return PortfolioRecommendation(
        client_id=request.client_id,
        timestamp=datetime.now(),
        weights=weight_dict,
        expected_return=risk_metrics.get("expected_return", 0.0),
        expected_volatility=risk_metrics.get("expected_volatility", 0.0),
        sharpe_ratio=risk_metrics.get("sharpe_ratio", 0.0),
        max_drawdown_estimate=risk_metrics.get("max_drawdown_estimate", 0.0),
        confidence_score=confidence_score,
    )


@app.get("/api/v1/portfolio/performance/{client_id}", response_model=PerformanceMetrics)
async def get_portfolio_performance(client_id: str, period: str = "1M"):
    """
    Return performance metrics for a client.

    NOTE: Currently returns stub values.
    Production: fetch real metrics from a time-series database keyed by client_id.
    """
    logger.info("Performance request: client=%s period=%s", client_id, period)
    # TODO: replace with real DB lookup
    return PerformanceMetrics(
        client_id=client_id,
        period=period,
        total_return=12.5,
        annualized_return=25.3,
        volatility=15.2,
        sharpe_ratio=1.85,
        max_drawdown=-8.5,
        win_rate=0.65,
    )


@app.get("/api/v1/risk/monitor/{client_id}", response_model=List[RiskAlert])
async def monitor_portfolio_risk(client_id: str):
    """Check portfolio for risk-threshold violations and return active alerts."""
    logger.info("Risk monitoring: client=%s", client_id)

    alerts: List[RiskAlert] = []
    current_dd = -12.5  # TODO: fetch live drawdown from portfolio store
    threshold = config["production"]["risk_monitoring"]["max_drawdown_alert"] * 100

    if abs(current_dd) > threshold:
        alerts.append(
            RiskAlert(
                alert_id=f"DD_{client_id}_{datetime.now().timestamp():.0f}",
                client_id=client_id,
                timestamp=datetime.now(),
                alert_type="MAX_DRAWDOWN",
                severity="WARNING",
                message=f"Drawdown {current_dd:.1f}% exceeds threshold {-threshold:.1f}%",
                current_value=current_dd,
                threshold=-threshold,
            )
        )
    return alerts


@app.post("/api/v1/rebalance/{client_id}")
async def trigger_rebalance(client_id: str, background_tasks: BackgroundTasks):
    """Schedule a portfolio rebalance to run in the background."""
    logger.info("Rebalance triggered for client %s", client_id)
    background_tasks.add_task(_execute_rebalance, client_id)
    return {
        "status": "pending",
        "client_id": client_id,
        "message": "Rebalancing scheduled",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v1/models/list")
async def list_available_models():
    """List all models currently loaded in the cache."""
    return {"models": list(_model_cache.keys()), "count": len(_model_cache)}


# ---------------------------------------------------------------------------
# Background tasks
# ---------------------------------------------------------------------------


async def _execute_rebalance(client_id: str) -> None:
    """Execute portfolio rebalancing (stub: wire up order management system)."""
    logger.info("Executing rebalance for %s", client_id)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    api_cfg = config["production"]["api"]
    uvicorn.run(
        "code.production.api:app",
        host=api_cfg["host"],
        port=api_cfg["port"],
        reload=False,  # never use reload=True in production
    )
