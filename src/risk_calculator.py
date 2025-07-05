def calculate_risk_reward(entry_price, stop_loss_price, target_price, position_size):
    """
    Calculate risk/reward metrics for a trading decision.

    Parameters:
    - entry_price: The price at which the position is entered.
    - stop_loss_price: The price at which the position will be exited to prevent further loss.
    - target_price: The price at which the position will be exited to take profit.
    - position_size: The number of shares or contracts being traded.

    Returns:
    - risk: The amount of money risked on the trade.
    - reward: The potential profit from the trade.
    - risk_reward_ratio: The ratio of reward to risk.
    """
    risk = (entry_price - stop_loss_price) * position_size
    reward = (target_price - entry_price) * position_size
    risk_reward_ratio = reward / abs(risk) if risk != 0 else float('inf')

    return risk, reward, risk_reward_ratio


def assess_risk_tolerance(risk_score, risk_tolerance_level):
    """
    Assess if the risk score is within the acceptable range based on the user's risk tolerance.

    Parameters:
    - risk_score: A score representing the calculated risk of a trade (0-100).
    - risk_tolerance_level: A string representing the user's risk tolerance ('low', 'medium', 'high').

    Returns:
    - bool: True if the risk is acceptable, False otherwise.
    """
    tolerance_thresholds = {
        'low': 30,
        'medium': 70,
        'high': 100
    }

    return risk_score <= tolerance_thresholds.get(risk_tolerance_level, 100)