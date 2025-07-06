# src/technical_analysis.py
"""
Technical analysis module for calculating indicators and signals
Uses TA-Lib alternatives that are free and work with pandas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

# Import settings
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Settings

logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Performs technical analysis on price data"""
    
    def __init__(self):
        self.settings = Settings()
        self.indicators = self.settings.TECHNICAL_INDICATORS
    
    def analyze(self, symbol: str, data: pd.DataFrame) -> Dict:
        """Perform complete technical analysis"""
        if data.empty:
            return self._empty_analysis()
        
        try:
            # Calculate all indicators
            indicators = self._calculate_indicators(data)
            
            # Generate signals
            signals = self._generate_signals(indicators, data)
            
            # Calculate support and resistance
            support_resistance = self._calculate_support_resistance(data)
            
            # Determine trend
            trend = self._determine_trend(indicators, data)
            
            # Calculate overall signal
            overall_signal = self._calculate_overall_signal(signals)
            
            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'indicators': indicators,
                'signals': signals,
                'support_resistance': support_resistance,
                'trend': trend,
                'overall_signal': overall_signal,
                'signal_strength': self._calculate_signal_strength(signals),
                'key_levels': self._identify_key_levels(data)
            }
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {str(e)}")
            return self._empty_analysis()
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate all technical indicators"""
        indicators = {}
        
        # RSI
        indicators['rsi'] = self._calculate_rsi(data['Close'], self.indicators['RSI']['period'])
        
        # MACD
        macd, signal, histogram = self._calculate_macd(
            data['Close'],
            self.indicators['MACD']['fast'],
            self.indicators['MACD']['slow'],
            self.indicators['MACD']['signal']
        )
        indicators['macd'] = {'macd': macd, 'signal': signal, 'histogram': histogram}
        
        # Bollinger Bands
        upper, middle, lower = self._calculate_bollinger_bands(
            data['Close'],
            self.indicators['BB']['period'],
            self.indicators['BB']['std']
        )
        indicators['bollinger'] = {'upper': upper, 'middle': middle, 'lower': lower}
        
        # Moving Averages
        indicators['sma'] = {}
        for period in self.indicators['SMA']['periods']:
            indicators['sma'][f'sma_{period}'] = self._calculate_sma(data['Close'], period)
        
        indicators['ema'] = {}
        for period in self.indicators['EMA']['periods']:
            indicators['ema'][f'ema_{period}'] = self._calculate_ema(data['Close'], period)
        
        # ATR (Average True Range)
        indicators['atr'] = self._calculate_atr(
            data['High'], 
            data['Low'], 
            data['Close'], 
            self.indicators['ATR']['period']
        )
        
        # Volume indicators
        indicators['volume'] = {
            'current': data['Volume'].iloc[-1] if 'Volume' in data else 0,
            'average': data['Volume'].rolling(window=self.indicators['VOLUME']['ma_period']).mean().iloc[-1] if 'Volume' in data else 0,
            'ratio': data['Volume'].iloc[-1] / data['Volume'].rolling(window=20).mean().iloc[-1] if 'Volume' in data and data['Volume'].rolling(window=20).mean().iloc[-1] > 0 else 1
        }
        
        # Stochastic
        indicators['stochastic'] = self._calculate_stochastic(data['High'], data['Low'], data['Close'])
        
        # Additional indicators
        indicators['adx'] = self._calculate_adx(data['High'], data['Low'], data['Close'])
        indicators['cci'] = self._calculate_cci(data['High'], data['Low'], data['Close'])
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[float, float, float]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return (
            macd.iloc[-1] if not macd.empty else 0,
            macd_signal.iloc[-1] if not macd_signal.empty else 0,
            macd_histogram.iloc[-1] if not macd_histogram.empty else 0
        )
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std: int = 2) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands"""
        middle = prices.rolling(window=period).mean()
        std_dev = prices.rolling(window=period).std()
        
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return (
            upper.iloc[-1] if not upper.empty else 0,
            middle.iloc[-1] if not middle.empty else 0,
            lower.iloc[-1] if not lower.empty else 0
        )
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average"""
        sma = prices.rolling(window=period).mean()
        return sma.iloc[-1] if not sma.empty and not np.isnan(sma.iloc[-1]) else prices.iloc[-1]
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> float:
        """Calculate Exponential Moving Average"""
        ema = prices.ewm(span=period, adjust=False).mean()
        return ema.iloc[-1] if not ema.empty and not np.isnan(ema.iloc[-1]) else prices.iloc[-1]
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.iloc[-1] if not atr.empty else 0
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Dict:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=period).min()
        highest_high = high.rolling(window=period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()
        
        return {
            'k': k_percent.iloc[-1] if not k_percent.empty else 50,
            'd': d_percent.iloc[-1] if not d_percent.empty else 50
        }
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        try:
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate Directional Movement
            plus_dm = high.diff()
            minus_dm = low.diff()
            
            plus_dm[plus_dm < 0] = 0
            minus_dm[minus_dm > 0] = 0
            minus_dm = abs(minus_dm)
            
            # Calculate smoothed values
            tr_smooth = true_range.rolling(window=period).mean()
            plus_dm_smooth = plus_dm.rolling(window=period).mean()
            minus_dm_smooth = minus_dm.rolling(window=period).mean()
            
            # Calculate DI
            plus_di = 100 * (plus_dm_smooth / tr_smooth)
            minus_di = 100 * (minus_dm_smooth / tr_smooth)
            
            # Calculate DX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            
            # Calculate ADX
            adx = dx.rolling(window=period).mean()
            
            return adx.iloc[-1] if not adx.empty and not np.isnan(adx.iloc[-1]) else 25
            
        except Exception as e:
            logger.warning(f"Error calculating ADX: {str(e)}")
            return 25
    
    def _calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> float:
        """Calculate Commodity Channel Index"""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = (typical_price - sma).abs().rolling(window=period).mean()
        
        cci = (typical_price - sma) / (0.015 * mean_deviation)
        
        return cci.iloc[-1] if not cci.empty else 0
    
    def _generate_signals(self, indicators: Dict, data: pd.DataFrame) -> Dict:
        """Generate trading signals from indicators"""
        signals = {}
        current_price = data['Close'].iloc[-1]
        
        # RSI signals
        rsi = indicators['rsi']
        if rsi < self.indicators['RSI']['oversold']:
            signals['rsi'] = {'signal': 'buy', 'strength': (self.indicators['RSI']['oversold'] - rsi) / self.indicators['RSI']['oversold']}
        elif rsi > self.indicators['RSI']['overbought']:
            signals['rsi'] = {'signal': 'sell', 'strength': (rsi - self.indicators['RSI']['overbought']) / (100 - self.indicators['RSI']['overbought'])}
        else:
            signals['rsi'] = {'signal': 'neutral', 'strength': 0}
        
        # MACD signals
        macd = indicators['macd']
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            signals['macd'] = {'signal': 'buy', 'strength': min(abs(macd['histogram']) / 0.5, 1)}
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            signals['macd'] = {'signal': 'sell', 'strength': min(abs(macd['histogram']) / 0.5, 1)}
        else:
            signals['macd'] = {'signal': 'neutral', 'strength': 0}
        
        # Bollinger Bands signals
        bb = indicators['bollinger']
        if current_price < bb['lower']:
            signals['bollinger'] = {'signal': 'buy', 'strength': (bb['lower'] - current_price) / bb['lower']}
        elif current_price > bb['upper']:
            signals['bollinger'] = {'signal': 'sell', 'strength': (current_price - bb['upper']) / bb['upper']}
        else:
            signals['bollinger'] = {'signal': 'neutral', 'strength': 0}
        
        # Moving Average signals
        sma_50 = indicators['sma'].get('sma_50', current_price)
        sma_200 = indicators['sma'].get('sma_200', current_price)
        
        if current_price > sma_50 > sma_200:
            signals['ma_trend'] = {'signal': 'buy', 'strength': 0.8}
        elif current_price < sma_50 < sma_200:
            signals['ma_trend'] = {'signal': 'sell', 'strength': 0.8}
        else:
            signals['ma_trend'] = {'signal': 'neutral', 'strength': 0}
        
        # Volume signals
        vol = indicators['volume']
        if vol['ratio'] > 2 and data['Close'].iloc[-1] > data['Close'].iloc[-2]:
            signals['volume'] = {'signal': 'buy', 'strength': min(vol['ratio'] / 3, 1)}
        elif vol['ratio'] > 2 and data['Close'].iloc[-1] < data['Close'].iloc[-2]:
            signals['volume'] = {'signal': 'sell', 'strength': min(vol['ratio'] / 3, 1)}
        else:
            signals['volume'] = {'signal': 'neutral', 'strength': 0}
        
        # Stochastic signals
        stoch = indicators['stochastic']
        if stoch['k'] < 20 and stoch['d'] < 20:
            signals['stochastic'] = {'signal': 'buy', 'strength': (20 - stoch['k']) / 20}
        elif stoch['k'] > 80 and stoch['d'] > 80:
            signals['stochastic'] = {'signal': 'sell', 'strength': (stoch['k'] - 80) / 20}
        else:
            signals['stochastic'] = {'signal': 'neutral', 'strength': 0}
        
        return signals
    
    def _calculate_support_resistance(self, data: pd.DataFrame) -> Dict:
        """Calculate support and resistance levels"""
        if len(data) < 20:
            return {'support': [], 'resistance': []}
        
        # Method 1: Recent highs and lows
        recent_data = data.tail(20)
        
        highs = recent_data['High'].nlargest(3).tolist()
        lows = recent_data['Low'].nsmallest(3).tolist()
        
        # Method 2: Pivot points
        last_high = data['High'].iloc[-1]
        last_low = data['Low'].iloc[-1]
        last_close = data['Close'].iloc[-1]
        
        pivot = (last_high + last_low + last_close) / 3
        r1 = 2 * pivot - last_low
        r2 = pivot + (last_high - last_low)
        s1 = 2 * pivot - last_high
        s2 = pivot - (last_high - last_low)
        
        # Method 3: Volume-weighted levels
        volume_profile = self._calculate_volume_profile(data)
        
        # Combine all methods
        resistance_levels = sorted(list(set(highs + [r1, r2] + volume_profile['resistance'])))
        support_levels = sorted(list(set(lows + [s1, s2] + volume_profile['support'])))
        
        return {
            'support': support_levels[:3],  # Top 3 support levels
            'resistance': resistance_levels[-3:],  # Top 3 resistance levels
            'pivot': pivot,
            'current_price': last_close
        }
    
    def _calculate_volume_profile(self, data: pd.DataFrame) -> Dict:
        """Calculate volume-weighted price levels"""
        if 'Volume' not in data or len(data) < 50:
            return {'support': [], 'resistance': []}
        
        # Create price bins
        price_range = data['High'].max() - data['Low'].min()
        bin_size = price_range / 20
        
        # Calculate volume at each price level
        volume_profile = {}
        
        for _, row in data.iterrows():
            price_bin = round(row['Close'] / bin_size) * bin_size
            if price_bin not in volume_profile:
                volume_profile[price_bin] = 0
            volume_profile[price_bin] += row['Volume']
        
        # Find high volume nodes (potential support/resistance)
        sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        high_volume_levels = [level[0] for level in sorted_levels[:5]]
        
        current_price = data['Close'].iloc[-1]
        support = [level for level in high_volume_levels if level < current_price]
        resistance = [level for level in high_volume_levels if level > current_price]
        
        return {'support': support, 'resistance': resistance}
    
    def _determine_trend(self, indicators: Dict, data: pd.DataFrame) -> Dict:
        """Determine the current trend"""
        trends = []
        current_price = data['Close'].iloc[-1]
        
        # Price vs moving averages
        sma_20 = indicators['sma'].get('sma_20', current_price)
        sma_50 = indicators['sma'].get('sma_50', current_price)
        sma_200 = indicators['sma'].get('sma_200', current_price)
        
        if current_price > sma_20 > sma_50 > sma_200:
            trends.append(('strong_uptrend', 1.0))
        elif current_price > sma_50 > sma_200:
            trends.append(('uptrend', 0.7))
        elif current_price < sma_20 < sma_50 < sma_200:
            trends.append(('strong_downtrend', -1.0))
        elif current_price < sma_50 < sma_200:
            trends.append(('downtrend', -0.7))
        else:
            trends.append(('sideways', 0.0))
        
        # ADX trend strength
        adx = indicators['adx']
        if adx > 25:
            trend_strength = 'strong'
        elif adx > 20:
            trend_strength = 'moderate'
        else:
            trend_strength = 'weak'
        
        # Higher highs/lows analysis
        recent_data = data.tail(10)
        if len(recent_data) >= 4:
            highs = recent_data['High'].tolist()
            lows = recent_data['Low'].tolist()
            
            if highs[-1] > highs[-3] and lows[-1] > lows[-3]:
                structure = 'bullish'
            elif highs[-1] < highs[-3] and lows[-1] < lows[-3]:
                structure = 'bearish'
            else:
                structure = 'neutral'
        else:
            structure = 'unknown'
        
        # Determine primary trend
        trend_scores = [t[1] for t in trends]
        avg_trend_score = sum(trend_scores) / len(trend_scores) if trend_scores else 0
        
        if avg_trend_score > 0.5:
            primary_trend = 'bullish'
        elif avg_trend_score < -0.5:
            primary_trend = 'bearish'
        else:
            primary_trend = 'neutral'
        
        return {
            'primary': primary_trend,
            'strength': trend_strength,
            'structure': structure,
            'score': avg_trend_score * 100,  # Convert to -100 to 100 scale
            'details': trends
        }
    
    def _calculate_overall_signal(self, signals: Dict) -> float:
        """Calculate overall signal from all indicators (-100 to 100)"""
        signal_values = []
        weights = {
            'rsi': 0.2,
            'macd': 0.25,
            'bollinger': 0.15,
            'ma_trend': 0.2,
            'volume': 0.1,
            'stochastic': 0.1
        }
        
        for indicator, signal_data in signals.items():
            if indicator in weights:
                # Convert signal to numeric value
                if signal_data['signal'] == 'buy':
                    value = signal_data['strength'] * 100
                elif signal_data['signal'] == 'sell':
                    value = -signal_data['strength'] * 100
                else:
                    value = 0
                
                signal_values.append(value * weights[indicator])
        
        overall_signal = sum(signal_values)
        return max(-100, min(100, overall_signal))  # Clamp to -100 to 100
    
    def _calculate_signal_strength(self, signals: Dict) -> str:
        """Determine signal strength category"""
        # Count bullish and bearish signals
        bullish_count = sum(1 for s in signals.values() if s['signal'] == 'buy')
        bearish_count = sum(1 for s in signals.values() if s['signal'] == 'sell')
        
        # Calculate average strength
        strengths = [s['strength'] for s in signals.values() if s['signal'] != 'neutral']
        avg_strength = sum(strengths) / len(strengths) if strengths else 0
        
        if bullish_count >= 4 and avg_strength > 0.7:
            return 'very_strong_buy'
        elif bullish_count >= 3 and avg_strength > 0.5:
            return 'strong_buy'
        elif bullish_count > bearish_count:
            return 'buy'
        elif bearish_count >= 4 and avg_strength > 0.7:
            return 'very_strong_sell'
        elif bearish_count >= 3 and avg_strength > 0.5:
            return 'strong_sell'
        elif bearish_count > bullish_count:
            return 'sell'
        else:
            return 'neutral'
    
    def _identify_key_levels(self, data: pd.DataFrame) -> Dict:
        """Identify key price levels for trading"""
        if len(data) < 20:
            return {}
        
        current_price = data['Close'].iloc[-1]
        
        # Recent high/low
        recent_high = data['High'].tail(20).max()
        recent_low = data['Low'].tail(20).min()
        
        # 52-week high/low (if we have enough data)
        if len(data) >= 252:
            yearly_high = data['High'].tail(252).max()
            yearly_low = data['Low'].tail(252).min()
        else:
            yearly_high = data['High'].max()
            yearly_low = data['Low'].min()
        
        # Psychological levels (round numbers)
        round_levels = []
        for i in range(int(recent_low), int(recent_high) + 1):
            if i % 5 == 0:  # Multiples of 5
                round_levels.append(float(i))
        
        # Fibonacci retracements
        fib_high = data['High'].tail(50).max()
        fib_low = data['Low'].tail(50).min()
        fib_diff = fib_high - fib_low
        
        fib_levels = {
            'fib_0': fib_low,
            'fib_236': fib_low + fib_diff * 0.236,
            'fib_382': fib_low + fib_diff * 0.382,
            'fib_500': fib_low + fib_diff * 0.5,
            'fib_618': fib_low + fib_diff * 0.618,
            'fib_100': fib_high
        }
        
        return {
            'current_price': current_price,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'yearly_high': yearly_high,
            'yearly_low': yearly_low,
            'psychological_levels': round_levels,
            'fibonacci_levels': fib_levels,
            'distance_from_high': ((recent_high - current_price) / recent_high) * 100,
            'distance_from_low': ((current_price - recent_low) / recent_low) * 100
        }
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'symbol': '',
            'timestamp': datetime.now().isoformat(),
            'indicators': {},
            'signals': {},
            'support_resistance': {'support': [], 'resistance': []},
            'trend': {'primary': 'unknown', 'strength': 'unknown', 'score': 0},
            'overall_signal': 0,
            'signal_strength': 'neutral',
            'key_levels': {}
        }
    
    def get_entry_exit_points(self, analysis: Dict, risk_tolerance: str = 'medium') -> Dict:
        """Calculate recommended entry and exit points"""
        if not analysis or 'key_levels' not in analysis:
            return {}
        
        current_price = analysis['key_levels'].get('current_price', 0)
        atr = analysis['indicators'].get('atr', current_price * 0.02)  # Default 2% if no ATR
        
        risk_multipliers = {
            'low': {'stop': 1.5, 'target': 2.0},
            'medium': {'stop': 2.0, 'target': 3.0},
            'high': {'stop': 2.5, 'target': 4.0}
        }
        
        multipliers = risk_multipliers.get(risk_tolerance, risk_multipliers['medium'])
        
        if analysis['overall_signal'] > 20:  # Bullish
            entry = current_price
            stop_loss = current_price - (atr * multipliers['stop'])
            take_profit = current_price + (atr * multipliers['target'])
            
            # Adjust for support/resistance
            support = analysis['support_resistance'].get('support', [])
            if support:
                stop_loss = max(stop_loss, support[-1] - (atr * 0.5))
            
        elif analysis['overall_signal'] < -20:  # Bearish
            entry = current_price
            stop_loss = current_price + (atr * multipliers['stop'])
            take_profit = current_price - (atr * multipliers['target'])
            
            # Adjust for support/resistance
            resistance = analysis['support_resistance'].get('resistance', [])
            if resistance:
                stop_loss = min(stop_loss, resistance[0] + (atr * 0.5))
        
        else:  # Neutral
            return {
                'recommendation': 'wait',
                'reason': 'No clear signal'
            }
        
        risk_reward_ratio = abs(take_profit - entry) / abs(stop_loss - entry)
        
        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_amount': abs(stop_loss - entry),
            'reward_amount': abs(take_profit - entry),
            'risk_reward_ratio': risk_reward_ratio,
            'position_type': 'long' if analysis['overall_signal'] > 20 else 'short'
        }