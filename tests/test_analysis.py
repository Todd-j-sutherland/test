import pytest
from src.technical_analysis import TechnicalAnalysis
from src.fundamental_analysis import FundamentalAnalysis

def test_technical_analysis_rsi():
    ta = TechnicalAnalysis()
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    rsi_value = ta.calculate_rsi(data)
    assert isinstance(rsi_value, float), "RSI should be a float"
    assert 0 <= rsi_value <= 100, "RSI should be between 0 and 100"

def test_fundamental_analysis_pe_ratio():
    fa = FundamentalAnalysis()
    earnings = 10
    price = 100
    pe_ratio = fa.calculate_pe_ratio(earnings, price)
    assert isinstance(pe_ratio, float), "P/E Ratio should be a float"
    assert pe_ratio == 10.0, "P/E Ratio should be calculated correctly"

def test_technical_analysis_moving_average():
    ta = TechnicalAnalysis()
    data = [1, 2, 3, 4, 5]
    moving_average = ta.calculate_moving_average(data, period=3)
    assert isinstance(moving_average, float), "Moving Average should be a float"
    assert moving_average == 4.0, "Moving Average should be calculated correctly"