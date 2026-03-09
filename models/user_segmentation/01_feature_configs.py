# Databricks notebook source


# COMMAND ----------

FEATURE_DESCRIPTIONS = {
    "online_casino":{
        "R_days": "Number of days since the user's last betting activity",
        "days_since_last_bet": "Elapsed days since the most recent bet",
        "days_since_last_deposit": "Elapsed days since the most recent deposit",

        "R_min": "Minimum recency observed in the analysis window",
        "R_max": "Maximum recency observed in the analysis window",
        "R_norm": "Normalized recency score within the analysis window",

        "F_min": "Minimum betting frequency observed",
        "F_max": "Maximum betting frequency observed",
        "F_norm": "Normalized betting frequency score",

        "M_min": "Minimum monetary value observed",
        "M_max": "Maximum monetary value observed",
        "M_norm": "Normalized monetary value score",

        "rfm_score": "Composite score combining recency, frequency, and monetary behavior",

        "avg_bets_per_day": "Average number of bets placed per active day",
        "num_bets_total": "Total number of bets placed in the period",
        "num_active_days": "Number of distinct days with betting activity",
        "active_weeks_3m": "Number of weeks with activity in the last three months",
        "bet_days_90d": "Number of days with bets in the last 90 days",

        "interval_volatility": "Variability in time gaps between consecutive bets",
        "avg_time_between_bets": "Average time interval between consecutive bets",

        "stake_to_deposit_ratio": "Ratio between bet stakes and deposited amounts",
        "stake_stddev_ratio": "Relative variability of bet stake sizes",
        "prev_stake": "Stake amount of the previous bet",

        "net_profit": "Net monetary result over the period",
        "net_cash_flow": "Net cash movement considering deposits and withdrawals",
        "net_profit_ratio": "Net profit normalized by total wagered amount",

        "win_rate": "Proportion of bets resulting in wins",
        "loss_rate": "Proportion of bets resulting in losses",
        "is_loss": "Indicator that a bet resulted in a loss",
        "prev_result": "Outcome of the previous bet",

        "max_loss_streak": "Maximum number of consecutive losing bets",
        "chasing_ratio": "Tendency to increase betting after losses",
        "chasing_flag": "Indicator of loss-chasing behavior",

        "num_coupons": "Number of promotional coupons used",
        "deposit_days_count": "Number of days with deposit activity",

        "diff_hours": "Time difference in hours between consecutive events"
    },
    "online_live_casino_ap2":{
        "R_days": "Number of days since the user's last betting activity",
        "days_since_last_bet": "Elapsed days since the most recent bet",
        "days_since_last_deposit": "Elapsed days since the most recent deposit",

        "R_min": "Minimum recency observed in the analysis window",
        "R_max": "Maximum recency observed in the analysis window",
        "R_norm": "Normalized recency score within the analysis window",

        "F_min": "Minimum betting frequency observed",
        "F_max": "Maximum betting frequency observed",
        "F_norm": "Normalized betting frequency score",

        "M_min": "Minimum monetary value observed",
        "M_max": "Maximum monetary value observed",
        "M_norm": "Normalized monetary value score",

        "rfm_score": "Composite score combining recency, frequency, and monetary behavior",

        "avg_bets_per_day": "Average number of bets placed per active day",
        "num_bets_total": "Total number of bets placed in the period",
        "num_active_days": "Number of distinct days with betting activity",
        "active_weeks_3m": "Number of weeks with activity in the last three months",
        "bet_days_90d": "Number of days with bets in the last 90 days",

        "interval_volatility": "Variability in time gaps between consecutive bets",
        "avg_time_between_bets": "Average time interval between consecutive bets",

        "stake_to_deposit_ratio": "Ratio between bet stakes and deposited amounts",
        "stake_stddev_ratio": "Relative variability of bet stake sizes",
        "prev_stake": "Stake amount of the previous bet",

        "net_profit": "Net monetary result over the period",
        "net_cash_flow": "Net cash movement considering deposits and withdrawals",
        "net_profit_ratio": "Net profit normalized by total wagered amount",

        "win_rate": "Proportion of bets resulting in wins",
        "loss_rate": "Proportion of bets resulting in losses",
        "is_loss": "Indicator that a bet resulted in a loss",
        "prev_result": "Outcome of the previous bet",

        "max_loss_streak": "Maximum number of consecutive losing bets",
        "chasing_ratio": "Tendency to increase betting after losses",
        "chasing_flag": "Indicator of loss-chasing behavior",

        "num_coupons": "Number of promotional coupons used",
        "deposit_days_count": "Number of days with deposit activity",

        "diff_hours": "Time difference in hours between consecutive events"
    },
    "online_live_casino": {

        # ----------------------
        # RFM
        # ----------------------
        "R_days": "Recency in days since the last bet (last 90 days).",
        "R_days_live": "Recency in days since the last Live Casino bet (last 90 days).",
        "R_days_online": "Recency in days since the last Online Casino bet (last 90 days).",
        "R_norm": "Normalized recency (0–1), inverted so that higher values indicate more recent activity.",
        "F_norm": "Normalized betting frequency (0–1) based on bet_days_90d.",
        "M_norm": "Normalized monetary value (0–1) based on stake_sum_90d.",
        "rfm_score": "Weighted RFM score = 0.4*R_norm + 0.3*F_norm + 0.3*M_norm.",

        # ----------------------
        # Activity – Betting Volume
        # ----------------------
        "num_bets_total": "Total number of bets (Online + Live).",
        "num_bets_total_live": "Total number of Live Casino bets.",
        "num_bets_total_online": "Total number of Online Casino bets.",
        "bet_days_90d": "Number of distinct betting days in the last 90 days (overall).",
        "bet_days_90d_live": "Number of distinct Live Casino betting days in the last 90 days.",
        "bet_days_90d_online": "Number of distinct Online Casino betting days in the last 90 days.",
        "num_active_days": "Number of distinct active betting days (overall).",
        "num_active_days_live": "Number of distinct active Live Casino days.",
        "num_active_days_online": "Number of distinct active Online Casino days.",
        "active_weeks_3m": "Number of active weeks in the last 3 months (overall).",
        "active_weeks_3m_live": "Number of active Live Casino weeks in the last 3 months.",
        "active_weeks_3m_online": "Number of active Online Casino weeks in the last 3 months.",
        "avg_bets_per_day": "Average number of bets per active day (overall).",
        "avg_bets_per_day_live": "Average number of Live Casino bets per active day.",
        "avg_bets_per_day_online": "Average number of Online Casino bets per active day.",

        # ----------------------
        # Stake Metrics
        # ----------------------
        "total_stake_amount": "Total wagered amount (overall).",
        "total_stake_amount_live": "Total wagered amount in Live Casino.",
        "total_stake_amount_online": "Total wagered amount in Online Casino.",
        "stake_sum_90d": "Total wagered amount in the last 90 days (overall).",
        "stake_sum_90d_live": "Total Live Casino wagered amount in the last 90 days.",
        "stake_sum_90d_online": "Total Online Casino wagered amount in the last 90 days.",
        "avg_stake_amount": "Average stake per bet (overall).",
        "avg_stake_amount_live": "Average stake per bet in Live Casino.",
        "avg_stake_amount_online": "Average stake per bet in Online Casino.",
        "stake_avg": "Mean stake value used in risk block (overall).",
        "stake_avg_live": "Mean stake value in Live Casino.",
        "stake_avg_online": "Mean stake value in Online Casino.",
        "stake_stddev": "Standard deviation of stake values (overall).",
        "stake_stddev_live": "Standard deviation of stake values in Live Casino.",
        "stake_stddev_online": "Standard deviation of stake values in Online Casino.",
        "stake_stddev_ratio": "Stake volatility = std(stake) / mean(stake) (overall).",
        "stake_stddev_ratio_live": "Stake volatility in Live Casino.",
        "stake_stddev_ratio_online": "Stake volatility in Online Casino.",

        # ----------------------
        # Time Between Bets
        # ----------------------
        "avg_time_between_bets": "Average time between consecutive bets in hours (overall).",
        "avg_time_between_bets_live": "Average time between Live Casino bets in hours.",
        "avg_time_between_bets_online": "Average time between Online Casino bets in hours.",
        "interval_avg": "Mean interval between bets in hours (overall).",
        "interval_avg_live": "Mean interval between Live Casino bets.",
        "interval_avg_online": "Mean interval between Online Casino bets.",
        "interval_stddev": "Standard deviation of time between bets (overall).",
        "interval_stddev_live": "Standard deviation of time between Live Casino bets.",
        "interval_stddev_online": "Standard deviation of time between Online Casino bets.",
        "interval_volatility": "Timing volatility = std(interval) / mean(interval) (overall).",
        "interval_volatility_live": "Timing volatility in Live Casino.",
        "interval_volatility_online": "Timing volatility in Online Casino.",

        # ----------------------
        # Risk / Behavioral
        # ----------------------
        "chasing_ratio": "Proportion of bets classified as chasing behavior (increasing stake after loss) (overall).",
        "chasing_ratio_live": "Chasing behavior ratio in Live Casino.",
        "chasing_ratio_online": "Chasing behavior ratio in Online Casino.",
        "max_loss_streak": "Maximum consecutive loss streak (overall).",
        "max_loss_streak_live": "Maximum consecutive loss streak in Live Casino.",
        "max_loss_streak_online": "Maximum consecutive loss streak in Online Casino.",

        # ----------------------
        # Performance
        # ----------------------
        "win_rate": "Winning rate = wins / (wins + losses) (overall).",
        "win_rate_live": "Winning rate in Live Casino.",
        "win_rate_online": "Winning rate in Online Casino.",
        "loss_rate": "Loss rate = losses / (wins + losses) (overall).",
        "loss_rate_live": "Loss rate in Live Casino.",
        "loss_rate_online": "Loss rate in Online Casino.",
        "sum_wins": "Total winnings (overall).",
        "sum_wins_live": "Total winnings in Live Casino.",
        "sum_wins_online": "Total winnings in Online Casino.",
        "sum_losses": "Total losses (absolute value) (overall).",
        "sum_losses_live": "Total losses in Live Casino.",
        "sum_losses_online": "Total losses in Online Casino.",
        "net_profit": "Net profit = total wins minus total losses (overall).",
        "net_profit_live": "Net profit in Live Casino.",
        "net_profit_online": "Net profit in Online Casino.",
        "net_profit_ratio": "Net profit divided by total stake (overall).",
        "net_profit_ratio_live": "Net profit ratio in Live Casino.",
        "net_profit_ratio_online": "Net profit ratio in Online Casino.",

        # ----------------------
        # Vertical Comparison
        # ----------------------
        "live_share_of_total_stake": "Proportion of total stake allocated to Live Casino (0–1).",
        "live_share_of_stake_90d": "Proportion of stake in Live Casino during last 90 days.",
        "live_share_of_bet_days_90d": "Proportion of active betting days attributed to Live Casino (last 90 days).",
        "live_vs_online_total_stake_ratio": "Ratio between Live and Online total stake.",
        "live_vs_online_avg_stake_ratio": "Ratio between Live and Online average stake.",
        "live_vs_online_stake_volatility_ratio": "Ratio between Live and Online stake volatility.",
        "live_vs_online_chasing_ratio": "Ratio between Live and Online chasing behavior.",

        # ----------------------
        # Financial
        # ----------------------
        "total_deposit_amount": "Total deposited amount.",
        "total_withdraw_amount": "Total withdrawn amount.",
        "net_cash_flow": "Net cash flow = deposits minus withdrawals.",
        "deposit_days_count": "Number of distinct days with deposits.",
        "days_since_last_deposit": "Days since last deposit.",
        "stake_to_deposit_ratio": "Total stake divided by total deposits.",
        "stake_to_deposit_ratio_live": "Live stake divided by total deposits.",
        "stake_to_deposit_ratio_online": "Online stake divided by total deposits.",

        # ----------------------
        # Timestamp References
        # ----------------------
        "first_bet_time": "Timestamp of first recorded bet (overall).",
        "first_bet_time_live": "Timestamp of first Live Casino bet.",
        "first_bet_time_online": "Timestamp of first Online Casino bet.",
        "last_bet_time": "Timestamp of most recent bet (overall).",
        "last_bet_time_live": "Timestamp of most recent Live Casino bet.",
        "last_bet_time_online": "Timestamp of most recent Online Casino bet.",
        "last_bet_time_90d": "Timestamp of most recent bet in last 90 days.",
        "last_bet_time_90d_live": "Timestamp of most recent Live bet in last 90 days.",
        "last_bet_time_90d_online": "Timestamp of most recent Online bet in last 90 days.",
        "last_deposit_time": "Timestamp of most recent deposit.",
        "last_coupon_time": "Timestamp of most recent coupon expiration linked to the user.",
        "days_since_last_bet": "Days since most recent bet (overall).",
        "days_since_last_bet_live": "Days since most recent Live bet.",
        "days_since_last_bet_online": "Days since most recent Online bet.",

        # ----------------------
        # Coupon
        # ----------------------
        "num_coupons": "Number of distinct coupons used or linked to the user."
    }
}

# COMMAND ----------

from typing import Optional

def get_feature_descriptions(
    analysis_name: str,
    features: Optional[list[str]] = None,
    *,
    keep_missing: bool = False,
    missing_value: str | None = None,
) -> dict:
    """
    Returns a dict {feature_name: description} for the given analysis_name.

    - If features is None or empty list/tuple, returns ALL features for that analysis_name.
    - If features is provided, returns only those features (optionally keeping missing).
    """

    scope = FEATURE_DESCRIPTIONS.get(analysis_name) or {}

    # If features is None or empty -> return all
    if features is None or (isinstance(features, (list, tuple)) and len(features) == 0):
        return dict(scope)

    if not isinstance(features, (list, tuple)):
        raise TypeError("features must be a list/tuple of feature names (strings).")

    out = {}
    for f in features:
        if f in scope:
            out[f] = scope[f]
        elif keep_missing:
            out[f] = missing_value

    return out

# COMMAND ----------

# MAGIC %skip
# MAGIC get_feature_descriptions('online_casino')