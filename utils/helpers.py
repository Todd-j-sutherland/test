# File: /asx-bank-analyzer/asx-bank-analyzer/utils/helpers.py

# This file includes various helper functions used throughout the project.

def format_currency(value):
    """Format a number as currency."""
    return f"${value:,.2f}"

def calculate_percentage(part, whole):
    """Calculate the percentage of a part relative to a whole."""
    if whole == 0:
        return 0
    return (part / whole) * 100

def is_valid_symbol(symbol):
    """Check if the provided symbol is valid (ends with .AX)."""
    return symbol.endswith('.AX')

def log_message(message):
    """Log a message to the console."""
    print(f"[LOG] {message}")