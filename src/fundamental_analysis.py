def calculate_pe_ratio(price, earnings_per_share):
    if earnings_per_share <= 0:
        return None
    return price / earnings_per_share

def calculate_dividend_yield(dividend, price):
    if price <= 0:
        return None
    return dividend / price

def calculate_book_value(total_assets, total_liabilities):
    return total_assets - total_liabilities

def calculate_return_on_equity(net_income, shareholder_equity):
    if shareholder_equity <= 0:
        return None
    return (net_income / shareholder_equity) * 100

def analyze_bank_fundamentals(bank_data):
    fundamentals = {}
    fundamentals['P/E Ratio'] = calculate_pe_ratio(bank_data['price'], bank_data['earnings_per_share'])
    fundamentals['Dividend Yield'] = calculate_dividend_yield(bank_data['dividend'], bank_data['price'])
    fundamentals['Book Value'] = calculate_book_value(bank_data['total_assets'], bank_data['total_liabilities'])
    fundamentals['ROE'] = calculate_return_on_equity(bank_data['net_income'], bank_data['shareholder_equity'])
    return fundamentals