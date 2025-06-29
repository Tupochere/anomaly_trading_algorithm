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
    def __init__(self, lookback_period: int = 252, debug : bool = False, max_stop_pct: float = 0.05):
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
        self.debug = debug
        self.max_stop_pct = max_stop_pct  # Cap the stop-loss at this percentage of the entry price


    def log(self, message):
        if self.debug:
            print(message)

      
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
        entry_threshold = 1.5 + volatility_factor * 0.2  # Refined threshold
        
        signal = 0
        strength = 0
        reason = ""
        
        # Long signal (oversold)
        if z_score < -entry_threshold and rsi < 45 and bb_position < 0.2:
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
    
    def calculate_position_size(self, final_score, volatility, threshold=0.3, base_size=0.06, max_size=0.12, min_size=0.03):
        """
        Calculate position size based on signal strength and volatility.
        - final_score: final signal score (0 to 1)
        - volatility: normalized ATR as fraction of price (e.g., 0.02 means 2%)
        """
        if final_score <= threshold:
            scaled_size = base_size
        else:
            score_factor = min((final_score - threshold) / (1.0 - threshold), 1.0)
            scaled_size = base_size + score_factor * (max_size - base_size)

        # Volatility adjustment
        if volatility > 0.03:  # High volatility (>3%)
            vol_factor = 0.7  # reduce position
        elif volatility < 0.015:  # Low volatility (<1.5%)
            vol_factor = 1.2  # slightly increase
        else:
            vol_factor = 1.0

        final_size = scaled_size * vol_factor
        final_size = max(min(final_size, max_size), min_size)

        return round(final_size, 4)
    
    def calculate_stops(self, data: pd.DataFrame, idx: int, signal: int, entry_price: float) -> Tuple[float, float]:
        """Dynamic stop loss and take profit calculation with a max stop-loss cap"""
        current = data.iloc[idx]
        atr = current['ATR']

        # Default ATR-based stop (2x ATR)
        if signal == 1:  # Long position
            stop_loss = entry_price - (2 * atr)
            # Cap the stop-loss to a max percentage (e.g., 5% of entry price)
            stop_loss = max(stop_loss, entry_price * (1 - self.max_stop_pct))
            take_profit = max(entry_price + (3 * atr), current['SMA_20'])
        else:  # Short position
            stop_loss = entry_price + (2 * atr)
            # Cap the stop-loss to a max percentage (e.g., 5% of entry price)
            stop_loss = min(stop_loss, entry_price * (1 + self.max_stop_pct))
            take_profit = min(entry_price - (3 * atr), current['SMA_20'])

        return stop_loss, take_profit

    def execute_strategy(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main strategy execution"""
        results = []
        entry_index = None  # Track when we entered positions
        price_history = []  # Track price history for trailing stops
        
        for i in range(len(data)):
            current = data.iloc[i]
            action = "WAIT"  # Default action
            position_size = None  # Initialize position size
            
            # 1. Detect market regime
            regime = self.detect_market_regime(data, i)
            
            # 2. Generate smart combined signal with strength
            combined_signal = self.generate_signal(data, i)
            signal_direction, signal_strength_score = self.get_signal_with_strength(data, i)
            
            # Get individual signals for debugging
            primary_signal = self.mean_reversion_signal(data, i)
            secondary_signal = self.momentum_signal(data, i)
            
            # Calculate signal strength based on combined conditions
            if combined_signal != 0:
                # For the new scoring system, signal strength is inherent in the signal
                mean_rev = self.mean_reversion_signal(data, i)
                momentum = self.momentum_signal(data, i)
                signal_strength = min(
                    max(mean_rev["strength"], momentum["strength"]),
                    1.0
                )
            else:
                signal_strength = 0.0
            
            # Prepare debug reasons
            if combined_signal != 0:
                mean_rev = self.mean_reversion_signal(data, i)
                momentum = self.momentum_signal(data, i)
                primary_reason = f"Combined Score Signal: MR={mean_rev['signal']:.2f}*{mean_rev['strength']:.2f}, MOM={momentum['signal']:.2f}*{momentum['strength']:.2f}"
                secondary_reason = f"Regime: {regime}, Final Score: {abs(combined_signal):.2f} (threshold: 0.7)"
            else:
                mean_rev = self.mean_reversion_signal(data, i)
                momentum = self.momentum_signal(data, i)
                primary_reason = f"Signal too weak: MR={mean_rev['reason']}"
                secondary_reason = f"MOM={momentum['reason']}, Score below 0.7 threshold"
            exit_price = None  # Initialize exit price

            # 4. Position entry logic
            if self.position == 0 and combined_signal != 0:
                # Enter position
                self.position = 1 if combined_signal > 0 else -1
                self.entry_price = current['close']
                
                # Calculate normalized volatility for position sizing
                current_atr = current['ATR']
                current_price = current['close']
                normalized_volatility = current_atr / current_price
                
                # Use enhanced volatility-aware position sizing
                position_size = self.calculate_position_size(signal_strength_score, normalized_volatility, threshold=0.3)
                
                action = "BUY" if self.position == 1 else "SELL"
                entry_index = i  # Track when we entered
                price_history = [current['close']]  # Initialize price history
                
                # Debug logging for entries
                if self.debug:
                    self.log(f"\n=== ENTER {action} at {current['close']:.2f} (Bar {i}) ===")
                    self.log(f"Primary Signal: {primary_reason}")
                    self.log(f"Secondary Signal: {secondary_reason}")
                    self.log(f"Final Score: {signal_strength_score:.2f} | Volatility: {normalized_volatility:.3f} | Position Size: {position_size:.4f}")
                    self.log(f"Using improved dynamic exit logic")
            
            # 5. Position management
            elif self.position != 0:
                action = "HOLD"
                
                # Update price history for trailing stops
                price_history.append(current['close'])
                if len(price_history) > 20:  # Keep last 20 prices
                    price_history = price_history[-20:]
                
                # Calculate bars held
                bars_held = i - entry_index if entry_index is not None else 0
                
                # Prepare indicators dict for exit logic
                indicators = {
                    "RSI": current.get("RSI", 50),
                    "MACD": current.get("MACD", 0),
                    "MACD_signal": current.get("MACD_signal", 0)
                }
                
                # Determine position direction
                direction = "long" if self.position == 1 else "short"
                
                # Check improved dynamic exit conditions
                exit_now, exit_reason = self.should_exit(
                    self.position, 
                    current['close'], 
                    indicators, 
                    self.entry_price, 
                    current['ATR'], 
                    direction,
                    bars_held,
                    price_history
                )
                
                # Additional exit conditions
                exit_condition = None
                
                if exit_now:
                    action = "EXIT_DYNAMIC"
                    exit_price = current['close']
                    exit_condition = exit_reason
                
                # Signal-based exit (override dynamic if stronger signal)
                elif (self.position == 1 and combined_signal < -0.5) or \
                    (self.position == -1 and combined_signal > 0.5):
                    action = "EXIT_SIGNAL"
                    exit_price = current['close']
                    exit_condition = "Strong signal reversal"
                
                # Handle exits
                if exit_condition:
                    # Calculate P&L
                    if self.position == 1:  # Long position
                        pnl_pct = (exit_price - self.entry_price) / self.entry_price
                    else:  # Short position
                        pnl_pct = (self.entry_price - exit_price) / self.entry_price
                    
                    # Record trade
                    self.trades.append({
                        'entry_price': self.entry_price,
                        'exit_price': exit_price,
                        'pnl_pct': pnl_pct,
                        'exit_reason': action,
                        'entry_time': data.index[entry_index] if entry_index is not None else data.index[i],
                        'exit_time': data.index[i],
                        'duration': i - entry_index if entry_index is not None else 0
                    })
                    
                    # Debug logging for exits
                    if self.debug:
                        duration = i - entry_index if entry_index is not None else "N/A"
                        self.log(f"\n=== EXIT {action} at {exit_price:.2f} (Bar {i}) ===")
                        self.log(f"Reason: {exit_condition}")
                        self.log(f"Entry Price: {self.entry_price:.2f}")
                        self.log(f"Position Duration: {duration} bars")
                        self.log(f"P&L: {pnl_pct:.2%}")
                    
                    # Reset position
                    self.position = 0
                    entry_index = None
                    price_history = []
            
            # 6. Record results for this bar
            results.append({
                'timestamp': current.name if hasattr(current, 'name') else i,
                'close': current['close'],
                'regime': regime,
                'signal': combined_signal,
                'signal_strength': signal_strength,
                'position': self.position,
                'action': action,
                'entry_price': self.entry_price if self.position != 0 else None,
                'position_size': position_size,
                'primary_reason': primary_reason,
                'secondary_reason': secondary_reason
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

    def generate_signal(self, data: pd.DataFrame, idx: int) -> int:
        """
        Regime-adaptive signal generator with optimized thresholds.
        Returns: 1 for buy, -1 for sell, 0 for no trade.
        """
        if idx < 26:  # Need enough data for all indicators
            return 0
            
        row = data.iloc[idx]
        regime = self.detect_market_regime(data, idx)
        
        # Mean-reversion logic
        z_score = row['Z_Score']
        rsi = row['RSI']
        bollinger_position = (row['close'] - row['BB_lower']) / (row['BB_upper'] - row['BB_lower'])

        mean_rev_signal = 0
        if z_score < -1.5 and rsi < 45 and bollinger_position < 0.2:
            mean_rev_signal = 1  # Long candidate
        elif z_score > 1.5 and rsi > 55 and bollinger_position > 0.8:
            mean_rev_signal = -1  # Short candidate

        # Mean-reversion strength
        mean_rev_strength = min(abs(z_score) / 3, 1.0)  # Normalized cap

        # Momentum logic
        ema_diff = row['EMA_12'] - row['EMA_26']
        macd = row['MACD']
        macd_signal = row['MACD_signal']

        momentum_signal = 0
        if ema_diff > 0 and macd > macd_signal and rsi > 50:
            momentum_signal = 1
        elif ema_diff < 0 and macd < macd_signal and rsi < 50:
            momentum_signal = -1

        # Momentum strength
        momentum_strength = min(abs(ema_diff / row['close']), 1.0)

        # Combine
        combined_signal = 0.5 * (mean_rev_signal * mean_rev_strength) + 0.5 * (momentum_signal * momentum_strength)

        # Default threshold
        threshold = 0.3

        # Adjust threshold by regime
        if regime == "STRONG_UPTREND":
            threshold = 0.25  # Encourage longs
        elif regime == "STRONG_DOWNTREND":
            threshold = 0.25  # Encourage shorts
        else:
            threshold = 0.3

        # Final trade signal
        if combined_signal >= threshold:
            return 1  # Buy
        elif combined_signal <= -threshold:
            return -1  # Sell
        else:
            return 0  # Hold

    def get_signal_reason(self, data: pd.DataFrame, idx: int, signal: int) -> str:
        """Get detailed reason for signal generation"""
        if signal == 0:
            return "No signal - conditions not met"
        
        current = data.iloc[idx]
        regime = self.detect_market_regime(data, idx)
        z_score = current.get('Z_Score', 0)
        rsi = current.get('RSI', 50)
        
        if signal == 1:
            return f"BUY: {regime} regime, Z-Score={z_score:.2f}, RSI={rsi:.1f}"
        else:
            return f"SELL: {regime} regime, Z-Score={z_score:.2f}, RSI={rsi:.1f}"
    
    def should_exit(self, position, current_price, indicators, entry_price, atr, direction, bars_held=0, price_history=None):
        """
        Clean, strict exit logic with hard stops and clear profit targets.

        Returns:
            (exit_now: bool, reason: str)
        """
        # Calculate % P&L
        if direction == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        # ✅ Hard stop-loss: -5%
        if pnl_pct < -0.05:
            return True, 'EXIT_STOP_LOSS'

        # ✅ Take-profit: +10%
        if pnl_pct > 0.10:
            return True, 'EXIT_TAKE_PROFIT'

        # ✅ Trailing stop: trail by 2 × ATR
        if direction == "long" and price_history is not None and len(price_history) > 0:
            peak_price = max(price_history)
            if current_price < peak_price - 2 * atr:
                return True, 'EXIT_TRAILING_STOP'
        elif direction == "short" and price_history is not None and len(price_history) > 0:
            trough_price = min(price_history)
            if current_price > trough_price + 2 * atr:
                return True, 'EXIT_TRAILING_STOP'

        # ✅ Momentum exit
        rsi = indicators.get("RSI", 50)
        macd = indicators.get("MACD", 0)
        macd_signal = indicators.get("MACD_signal", 0)
        
        if direction == "long":
            if rsi < 50 and macd < macd_signal:
                return True, 'EXIT_MOMENTUM'
        else:
            if rsi > 50 and macd > macd_signal:
                return True, 'EXIT_MOMENTUM'

        # ✅ Max duration exit: 50 bars
        if bars_held > 50:
            return True, 'EXIT_MAX_DURATION'

        return False, "Hold"
    
    def get_signal_with_strength(self, data: pd.DataFrame, idx: int) -> Tuple[int, float]:
        """
        Get both signal direction and the actual combined signal strength.
        Returns: (direction: int, strength: float)
        """
        if idx < 26:  # Need enough data for all indicators
            return 0, 0.0
            
        row = data.iloc[idx]
        regime = self.detect_market_regime(data, idx)
        
        # Mean-reversion logic
        z_score = row['Z_Score']
        rsi = row['RSI']
        bollinger_position = (row['close'] - row['BB_lower']) / (row['BB_upper'] - row['BB_lower'])

        mean_rev_signal = 0
        if z_score < -1.5 and rsi < 45 and bollinger_position < 0.2:
            mean_rev_signal = 1  # Long candidate
        elif z_score > 1.5 and rsi > 55 and bollinger_position > 0.8:
            mean_rev_signal = -1  # Short candidate

        # Mean-reversion strength
        mean_rev_strength = min(abs(z_score) / 3, 1.0)  # Normalized cap

        # Momentum logic
        ema_diff = row['EMA_12'] - row['EMA_26']
        macd = row['MACD']
        macd_signal = row['MACD_signal']

        momentum_signal = 0
        if ema_diff > 0 and macd > macd_signal and rsi > 50:
            momentum_signal = 1
        elif ema_diff < 0 and macd < macd_signal and rsi < 50:
            momentum_signal = -1

        # Momentum strength
        momentum_strength = min(abs(ema_diff / row['close']), 1.0)

        # Combine
        combined_signal = 0.5 * (mean_rev_signal * mean_rev_strength) + 0.5 * (momentum_signal * momentum_strength)

        # Default threshold
        threshold = 0.3

        # Adjust threshold by regime
        if regime == "STRONG_UPTREND":
            threshold = 0.25  # Encourage longs
        elif regime == "STRONG_DOWNTREND":
            threshold = 0.25  # Encourage shorts
        else:
            threshold = 0.3

        # Return direction and actual strength
        if combined_signal >= threshold:
            return 1, abs(combined_signal)  # Buy with strength
        elif combined_signal <= -threshold:
            return -1, abs(combined_signal)  # Sell with strength
        else:
            return 0, 0.0  # Hold

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
