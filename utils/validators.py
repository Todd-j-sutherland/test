def validate_bank_symbol(symbol):
    """Validate the bank symbol format."""
    if not isinstance(symbol, str):
        return False
    return symbol.endswith('.AX')

def validate_alert_threshold(threshold):
    """Validate alert threshold value."""
    if not isinstance(threshold, (int, float)):
        return False
    return -100 <= threshold <= 100

def validate_email(email):
    """Validate email format."""
    import re
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email) is not None

def validate_cache_size(size):
    """Validate cache size in MB."""
    return isinstance(size, (int, float)) and size > 0

def validate_analysis_period(period):
    """Validate analysis period format."""
    valid_periods = ['1d', '1w', '1mo', '3mo', '6mo', '1y', '5y']
    return period in valid_periods