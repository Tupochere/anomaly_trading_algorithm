"""
Advanced Adaptive Trading Algorithm
Combines multiple strategies with market regime detection
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class AdvancedTradingAlgorithm:
    def __init__(self, lookback_period: int = 252):
        self.lookback_period = lookback_period
        self.position = 0  # 0: neutral, 1: long, -1: short
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trades = []
        self.market_regime = "NEUTRAL"
        self.trailing_activated = False
        self.trailing_buffer = 1.5  # in ATR units
        self.trailing_trigger = 2.0  # price must move 2× ATR before we start trailing

    def update_trailing_stop(self, data: pd.DataFrame, idx: int):
        """Dynamically adjust stop-loss once price moves far enough in favor"""
        current = data.iloc[idx]
        atr = current['ATR']

        if self.position == 1:
            gain = current['close'] - self.entry_price
            if gain > atr * self.trailing_trigger:
                self.trailing_activated = True
                new_stop = current['close'] - atr * self.trailing_buffer
                self.stop_loss = max(self.stop_loss, new_stop)  # Don't lower stop
        elif self.position == -1:
           gain = self.entry_price - current['close']
           if gain > atr * self.trailing_trigger:
                self.trailing_activated = True
                new_stop = current['close'] + atr * self.trailing_buffer
                self.stop_loss = min(self.stop_loss, new_stop)  # Don't raise stop



    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        data = df.copy()
        
        # Moving Averages
        data['SMA_20'] = ta.trend.sma_indicator(data['close'], window=20)
        data['SMA_50'] = ta.trend.sma_indicator(data['close'], window=50)
        data['SMA_200'] = ta.trend.sma_indicator(data['close'], window=200)
        data['EMA_12'] = ta.trend.ema_indicator(data['close'], window=12)
        data['EMA_26'] = ta.trend.ema_indicator(data['close'], window=26)
        
        # Volatility Indicators
        data['ATR'] = ta.volatility.average_true_range(data['high'], data['low'], data['close'], window=14)
        bb_indicator = ta.volatility.BollingerBands(data['close'])
        data['BB_upper'] = bb_indicator.bollinger_hband()
        data['BB_middle'] = bb_indicator.bollinger_mavg() 
        data['BB_lower'] = bb_indicator.bollinger_lband()
        data['BB_width'] = (data['BB_upper'] - data['BB_lower']) / data['BB_middle']
        
        # Momentum Indicators
        data['RSI'] = ta.momentum.rsi(data['close'], window=14)
        macd_indicator = ta.trend.MACD(data['close'])
        data['MACD'] = macd_indicator.macd_diff()
        data['MACD_signal'] = macd_indicator.macd_signal()
        data['ADX'] = ta.trend.adx(data['high'], data['low'], data['close'], window=14)
        
        # Volume Indicators
        if 'volume' in data.columns:
            data['Volume_SMA'] = data['volume'].rolling(window=20).mean()
            data['OBV'] = ta.volume.on_balance_volume(data['close'], data['volume'])
        
        # Custom Indicators
        data['Price_STD'] = data['close'].rolling(window=20).std()
        data['Z_Score'] = (data['close'] - data['SMA_20']) / data['Price_STD']
        data['Trend_Strength'] = abs(data['SMA_20'] - data['SMA_50']) / data['ATR']
        
        return data
    
    def detect_market_regime(self, data: pd.DataFrame, idx: int) -> str:
        """Advanced market regime detection"""
        if idx < 50:  # Need enough data
            return "NEUTRAL"
        
        current_data = data.iloc[idx-20:idx+1]
        
        # Trend Analysis
        sma_slope = (current_data['SMA_20'].iloc[-1] - current_data['SMA_20'].iloc[-10]) / 10
        price_above_sma200 = current_data['close'].iloc[-1] > current_data['SMA_200'].iloc[-1]
        adx_strength = current_data['ADX'].iloc[-1]
        
        # Volatility Analysis
        bb_width_avg = current_data['BB_width'].mean()
        bb_width_current = current_data['BB_width'].iloc[-1]
        
        # Regime Logic
        if adx_strength > 25 and abs(sma_slope) > current_data['ATR'].iloc[-1] * 0.1:
            if sma_slope > 0 and price_above_sma200:
                return "STRONG_UPTREND"
            elif sma_slope < 0 and not price_above_sma200:
                return "STRONG_DOWNTREND"
            else:
                return "TRENDING"
        elif bb_width_current < bb_width_avg * 0.8:
            return "RANGING"
        elif bb_width_current > bb_width_avg * 1.2:
            return "HIGH_VOLATILITY"
        else:
            return "NEUTRAL"
    
    def mean_reversion_signal(self, data: pd.DataFrame, idx: int) -> Dict:
        """Enhanced mean reversion strategy"""
        if idx < 20:
            return {"signal": 0, "strength": 0, "reason": "insufficient_data"}
        
        current = data.iloc[idx]
        z_score = current['Z_Score']
        rsi = current['RSI']
        bb_position = (current['close'] - current['BB_lower']) / (current['BB_upper'] - current['BB_lower'])
        
        # Dynamic thresholds based on volatility
        volatility_factor = min(current['ATR'] / current['close'] * 100, 3)  # Cap at 3%
        entry_threshold = 1.5 + volatility_factor * 0.3
        
        signal = 0
        strength = 0
        reason = ""
        
        # Long signal (oversold)
        if z_score < -entry_threshold and rsi < 35 and bb_position < 0.2:
            signal = 1
            strength = min(abs(z_score) / 3, 1.0)
            reason = f"oversold: z={z_score:.2f}, rsi={rsi:.1f}"
        
        # Short signal (overbought) 
        elif z_score > entry_threshold and rsi > 65 and bb_position > 0.8:
            signal = -1
            strength = min(abs(z_score) / 3, 1.0)
            reason = f"overbought: z={z_score:.2f}, rsi={rsi:.1f}"
        
        return {"signal": signal, "strength": strength, "reason": reason}
    
    def momentum_signal(self, data: pd.DataFrame, idx: int) -> Dict:
        """Momentum/trend following strategy"""
        if idx < 26:
            return {"signal": 0, "strength": 0, "reason": "insufficient_data"}
        
        current = data.iloc[idx]
        prev = data.iloc[idx-1]
        
        # MACD crossover
        macd_bullish = current['MACD'] > current['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']
        macd_bearish = current['MACD'] < current['MACD_signal'] and prev['MACD'] <= prev['MACD_signal']
        
        # EMA crossover
        ema_bullish = current['EMA_12'] > current['EMA_26'] and prev['EMA_12'] <= prev['EMA_26']
        ema_bearish = current['EMA_12'] < current['EMA_26'] and prev['EMA_12'] >= prev['EMA_26']
        
        # Price momentum
        price_momentum = (current['close'] - data.iloc[idx-5]['close']) / data.iloc[idx-5]['close']
        
        signal = 0
        strength = 0
        reason = ""
        
        if (macd_bullish or ema_bullish) and price_momentum > 0.01:
            signal = 1
            strength = min(abs(price_momentum) * 10, 1.0)
            reason = "momentum_bullish"
        elif (macd_bearish or ema_bearish) and price_momentum < -0.01:
            signal = -1
            strength = min(abs(price_momentum) * 10, 1.0)
            reason = "momentum_bearish"
        
        return {"signal": signal, "strength": strength, "reason": reason}
    
    def calculate_position_size(self, data: pd.DataFrame, idx: int, signal_strength: float) -> float:
        """Dynamic position sizing based on volatility and confidence"""
        current = data.iloc[idx]
        
        # Base position size (1.0 = 100% of available capital)
        base_size = 0.1  # Conservative base
        
        # Adjust for signal strength
        strength_multiplier = 0.5 + (signal_strength * 1.5)  # 0.5x to 2x
        
        # Adjust for volatility (lower size in high vol)
        volatility_factor = current['ATR'] / current['close']
        vol_multiplier = max(0.3, 1 - (volatility_factor * 20))  # Cap between 0.3x and 1x
        
        position_size = base_size * strength_multiplier * vol_multiplier
        return min(position_size, 0.25)  # Never risk more than 25%
    
    def calculate_stops(self, data: pd.DataFrame, idx: int, signal: int, entry_price: float) -> Tuple[float, float]:
        """Dynamic stop loss and take profit calculation"""
        current = data.iloc[idx]
        atr = current['ATR']
        
        if signal == 1:  # Long position
            # Stop loss: 2x ATR below entry or recent swing low
            stop_loss = entry_price - (2 * atr)
            # Take profit: 3x ATR above entry or mean reversion target
            take_profit = max(entry_price + (3 * atr), current['SMA_20'])
        else:  # Short position
            # Stop loss: 2x ATR above entry or recent swing high
            stop_loss = entry_price + (2 * atr)
            # Take profit: 3x ATR below entry or mean reversion target
            take_profit = min(entry_price - (3 * atr), current['SMA_20'])
        
        return stop_loss, take_profit
    

    

    def execute_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main strategy execution"""
        results = []
        
        for i in range(len(data)):
            current = data.iloc[i]
            action = "WAIT"  # <-- Initialize action at the start
            
            # Detect market regime
            regime = self.detect_market_regime(data, i)
            
            # Get signals from different strategies
            if regime in ["RANGING", "NEUTRAL"]:
                primary_signal = self.mean_reversion_signal(data, i)
                secondary_signal = {"signal": 0, "strength": 0}
            else:
                primary_signal = self.momentum_signal(data, i)
                secondary_signal = self.mean_reversion_signal(data, i)
        
            # Combine signals
            combined_signal = primary_signal["signal"]
            if primary_signal["signal"] == 0 and abs(secondary_signal["signal"]) > 0:
                combined_signal = secondary_signal["signal"] * 0.5  # Reduced strength
            
            signal_strength = max(primary_signal["strength"], secondary_signal["strength"] * 0.5)

            # Position management
            if self.position == 0 and combined_signal != 0:
                # Enter position
                self.position = 1 if combined_signal > 0 else -1
                self.entry_price = current['close']
                position_size = self.calculate_position_size(data, i, signal_strength)
                self.stop_loss, self.take_profit = self.calculate_stops(data, i, self.position, self.entry_price)
                
                action = "BUY" if self.position == 1 else "SELL"
                
            elif self.position != 0:
                self.update_trailing_stop(data, i)
                # Check exit conditions
                action = "HOLD"
                
                # Stop loss hit
                if (self.position == 1 and current['close'] <= self.stop_loss) or \
                   (self.position == -1 and current['close'] >= self.stop_loss):
                    action = "EXIT_STOP"
                    self.position = 0
                
                # Take profit hit
                elif (self.position == 1 and current['close'] >= self.take_profit) or \
                     (self.position == -1 and current['close'] <= self.take_profit):
                    action = "EXIT_PROFIT"
                    self.position = 0
                
                # Regime change exit
                elif (self.position == 1 and combined_signal < -0.5) or \
                     (self.position == -1 and combined_signal > 0.5):
                    action = "EXIT_SIGNAL"
                    self.position = 0
            

        # Calculate P&L and record trade if exited
        if action.startswith("EXIT"):
            if self.position == 1:  # Long position
                pnl_pct = (current['close'] - self.entry_price) / self.entry_price
            elif self.position == -1:  # Short position
                pnl_pct = (self.entry_price - current['close']) / self.entry_price
            else:
                pnl_pct = 0  # Just in case

            self.trades.append({
                'entry_price': self.entry_price,
                'exit_price': current['close'],
                'pnl_pct': pnl_pct,
                'exit_reason': action
            })
        
        # Store results
        results.append({
            'timestamp': current.name if hasattr(current, 'name') else i,
            'close': current['close'],
            'regime': regime,
            'signal': combined_signal,
            'signal_strength': signal_strength,
            'position': self.position,
            'action': action,
            'entry_price': self.entry_price if self.position != 0 else None,
            'stop_loss': self.stop_loss if self.position != 0 else None,
            'take_profit': self.take_profit if self.position != 0 else None,
            'primary_reason': primary_signal.get("reason", ""),
            'secondary_reason': secondary_signal.get("reason", "")
        })
    
        return pd.DataFrame(results)
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {"error": "No trades executed"}
        
        trades_df = pd.DataFrame(self.trades)
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
        losing_trades = len(trades_df[trades_df['pnl_pct'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losing_trades > 0 else 0
        
        total_return = trades_df['pnl_pct'].sum()
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 else float('inf')
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return_pct': total_return * 100,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'profit_factor': profit_factor,
            'max_consecutive_wins': self._max_consecutive(trades_df['pnl_pct'] > 0),
            'max_consecutive_losses': self._max_consecutive(trades_df['pnl_pct'] < 0)
        }
    
    def _max_consecutive(self, series) -> int:
        """Helper function to calculate max consecutive occurrences"""
        if len(series) == 0:
            return 0
        
        max_consecutive = current_consecutive = 0
        prev_value = None
        
        for value in series:
            if value == prev_value:
                current_consecutive += 1
            else:
                max_consecutive = max(max_consecutive, current_consecutive)
                current_consecutive = 1
            prev_value = value
        
        return max(max_consecutive, current_consecutive)

# Example usage and backtesting function
def backtest_algorithm(df: pd.DataFrame, show_details: bool = True):
    """
    Backtest the advanced trading algorithm
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
    show_details: Whether to print detailed results
    """
    
    # Initialize algorithm
    algo = AdvancedTradingAlgorithm()
    
    # Calculate indicators
    print("Calculating technical indicators...")
    data_with_indicators = algo.calculate_indicators(df)
    
    # Execute strategy
    print("Executing trading strategy...")
    results = algo.execute_strategy(data_with_indicators)
    
    # Get performance metrics
    performance = algo.get_performance_metrics()
    
    if show_details:
        print("\n" + "="*50)
        print("ALGORITHM PERFORMANCE SUMMARY")
        print("="*50)
        
        if 'error' not in performance:
            print(f"Total Trades: {performance['total_trades']}")
            print(f"Win Rate: {performance['win_rate']:.2%}")
            print(f"Total Return: {performance['total_return_pct']:.2f}%")
            print(f"Average Win: {performance['avg_win_pct']:.2f}%")
            print(f"Average Loss: {performance['avg_loss_pct']:.2f}%")
            print(f"Profit Factor: {performance['profit_factor']:.2f}")
            print(f"Max Consecutive Wins: {performance['max_consecutive_wins']}")
            print(f"Max Consecutive Losses: {performance['max_consecutive_losses']}")
            
            # Show recent trades
            print(f"\nRecent Trades (Last 5):")
            for trade in algo.trades[-5:]:
                print(f"Entry: ${trade['entry_price']:.2f} → Exit: ${trade['exit_price']:.2f} | "
                      f"P&L: {trade['pnl_pct']:.2%} | Reason: {trade['exit_reason']}")
        else:
            print(performance['error'])
    
    return results, performance, algo

# Sample data generator for testing
def generate_sample_data(days: int = 252) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=days, freq='D')
    
    # Generate price data with some trends and noise
    base_price = 100
    prices = [base_price]
    
    for i in range(1, days):
        # Add trend component and noise
        trend = np.sin(i / 50) * 0.001  # Cyclical trend
        noise = np.random.normal(0, 0.02)  # Daily volatility
        change = trend + noise
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Prevent negative prices
    
    # Generate OHLC from close prices
    data = []
    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = prices[i-1] if i > 0 else close
        volume = np.random.randint(100000, 1000000)
        
        data.append({
            'date': dates[i],
            'open': open_price,
            'high': max(open_price, high, close),
            'low': min(open_price, low, close),
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    return df

# Example execution
if __name__ == "__main__":
    print("Advanced Trading Algorithm - Demo")
    print("Generating sample data...")
    
    # Generate sample data
    sample_data = generate_sample_data(500)  # 500 days of data
    
    # Run backtest
    results, performance, algorithm = backtest_algorithm(sample_data)
    
    print(f"\nAlgorithm completed analysis of {len(sample_data)} trading days")
    print("Results stored in 'results' DataFrame")
    print("Performance metrics in 'performance' dictionary")
    print("Algorithm object available as 'algorithm'")
