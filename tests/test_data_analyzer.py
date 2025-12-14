"""Tests for DataAnalyzer"""
import pytest
from src.core.data_analyzer import DataAnalyzer

def test_get_historical_data():
    analyzer = DataAnalyzer()
    data = analyzer.get_historical_data('AAPL', years=1)
    assert data is not None
    assert len(data) > 0

def test_calculate_indicators():
    analyzer = DataAnalyzer()
    data = analyzer.get_historical_data('AAPL', years=1)
    data = analyzer.calculate_technical_indicators(data)
    assert 'RSI' in data.columns
    assert 'MACD' in data.columns
