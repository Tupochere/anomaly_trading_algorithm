"""
Anomaly Trading Algorithm
Version: v2_prop_firm_optimized
Date: 2025-07-01
Notes:
- Prop firm optimized version
- Higher trade frequency, tighter risk, intraday confirmation
"""
from typing import Dict, Tuple, List
import pandas as pd
import numpy as np
import ta
import warnings
warnings.filterwarnings('ignore')

# Walk-Forward Testing Helper Functions
def calculate_max_drawdown(equity_curve):
    """Calculate maximum drawdown from equity curve."""
    roll_max = equity_curve.cummax()
    drawdown = (equity_curve - roll_max) / roll_max
    return drawdown.min()

def calculate_sharpe_ratio(pnl_series, risk_free_rate=0.0):
    """Calculate annualized Sharpe ratio."""
    mean_return = pnl_series.mean()
    std_return = pnl_series.std()
    if std_return == 0:
        return 0.0
    return (mean_return - risk_free_rate) / std_return * np.sqrt(252)

class AdvancedTradingAlgorithmPropV2:
    """
    Advanced Trading Algorithm - Prop Firm Optimized Version 2
    
    Features:
    - Higher trade frequency with tighter risk management
    - Intraday confirmation using 15-minute timeframe data
    - Dynamic position sizing based on signal strength and volatility
    - Regime-adaptive signal generation
    - Comprehensive risk management for prop firm compliance
    
    Intraday Confirmation:
    - Uses 15-minute data to confirm hourly signals
    - Long confirmation: RSI > 45 and EMA_12 > EMA_26
    - Short confirmation: RSI < 55 and EMA_12 < EMA_26
    - Helps reduce false signals and improve trade quality
    """
    
    def __init__(self, lookback_period=252, debug=False, max_stop_pct=0.03, account_phase="evaluation"):
        self.lookback_period = lookback_period
        self.debug = debug
        self.max_stop_pct = max_stop_pct
        self.account_phase = account_phase  # "evaluation" or "funded"
        self.daily_pnl = 0.0
        self.total_drawdown = 0.0
        self.high_watermark = 100_000  # Adjust to your initial balance
        self.active_days_traded = 0
        self.skip_today = False
        self.stop_trading = False
        
        # Existing attributes
        self.position = 0  # 0: neutral, 1: long, -1: short
        self.entry_price = 0
        self.stop_loss = 0
        self.take_profit = 0
        self.trades = []
        self.market_regime = "NEUTRAL"
        self.trailing_activated = False
        self.trailing_buffer = 1.5  # in ATR units
        self.trailing_trigger = 2.0  # price must move 2× ATR before we start trailing


    def log(self, msg):
        if self.debug:
            print(msg)

      
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
    
    def calculate_position_size(self, final_score, volatility, account_phase="evaluation", threshold=0.3, base_size=0.04, max_size=0.08, min_size=0.02):
        """
        Calculate position size based on signal strength and volatility - prop firm optimized.
        
        Args:
            final_score: final signal score (0 to 1)
            volatility: normalized ATR as fraction of price (e.g., 0.02 means 2%)
            account_phase: "evaluation" or "funded" - determines maximum position size cap
            threshold: minimum signal strength threshold
            base_size: base position size
            max_size: maximum position size (before account phase capping)
            min_size: minimum position size
            
        Returns:
            Position size as percentage of account (0.01 = 1%)
        """
        if final_score <= threshold:
            return min_size

        # Calculate raw position size based on signal strength
        raw_size = base_size + ((final_score - threshold) / (1 - threshold)) * (max_size - base_size)

        # Adjust for volatility
        if volatility > 0.03:
            raw_size *= 0.7  # Reduce size for high volatility
        elif volatility < 0.015:
            raw_size *= 1.2  # Increase size for low volatility

        # Apply account phase caps (prop firm risk management)
        if account_phase == "evaluation":
            # Conservative cap for evaluation phase
            phase_max_size = 0.03  # 3% maximum
        elif account_phase == "funded":
            # Slightly more aggressive for funded accounts
            phase_max_size = 0.05  # 5% maximum
        else:
            # Default to evaluation phase for safety
            phase_max_size = 0.03

        # Apply all constraints
        final_size = max(min_size, min(raw_size, max_size, phase_max_size))
        
        return final_size
    
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

    def execute_strategy(self, data: pd.DataFrame, data_15m: pd.DataFrame = None) -> pd.DataFrame:
        """Main strategy execution with optional 15-minute data for intraday confirmation"""
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
                secondary_reason = f"Regime: {regime}, Final Score: {signal_strength_score:.2f} (threshold: 0.7)"
            else:
                mean_rev = self.mean_reversion_signal(data, i)
                momentum = self.momentum_signal(data, i)
                primary_reason = f"Signal too weak: MR={mean_rev['reason']}"
                secondary_reason = f"MOM={momentum['reason']}, Score below 0.7 threshold"
            exit_price = None  # Initialize exit price

            # 4. Position entry logic
            if self.position == 0 and combined_signal != 0:
                # Check for intraday confirmation if 15-minute data is available
                intraday_confirmed = True  # Default to True if no 15m data
                
                if data_15m is not None:
                    # Map current hourly timestamp to 15-minute data
                    current_timestamp = current.name if hasattr(current, 'name') else data.index[i]
                    
                    # Find closest 15-minute timestamp
                    try:
                        # Use the exact timestamp or the closest available one
                        if current_timestamp in data_15m.index:
                            idx_15m = current_timestamp
                        else:
                            # Find the most recent 15-minute bar before current time
                            available_times = data_15m.index[data_15m.index <= current_timestamp]
                            if len(available_times) > 0:
                                idx_15m = available_times[-1]
                            else:
                                idx_15m = None
                        
                        if idx_15m is not None:
                            signal_direction = 1 if combined_signal > 0 else -1
                            intraday_confirmed = self.get_intraday_confirmation(data_15m, idx_15m, signal_direction)
                            
                            if self.debug and not intraday_confirmed:
                                self.log(f"Intraday confirmation failed for {signal_direction} signal at {current_timestamp}")
                        else:
                            intraday_confirmed = False
                            if self.debug:
                                self.log(f"No matching 15m data found for timestamp {current_timestamp}")
                                
                    except Exception as e:
                        if self.debug:
                            self.log(f"Error in intraday confirmation: {e}")
                        intraday_confirmed = False
                
                # Only enter position if confirmed (or no 15m data available)
                if intraday_confirmed:
                    # Enter position
                    self.position = 1 if combined_signal > 0 else -1
                    self.entry_price = current['close']
                    
                    # Calculate normalized volatility for position sizing
                    current_atr = current['ATR']
                    current_price = current['close']
                    normalized_volatility = current_atr / current_price
                    
                    # Use enhanced volatility-aware position sizing with account phase
                    position_size = self.calculate_position_size(
                        signal_strength_score, 
                        normalized_volatility, 
                        account_phase=self.account_phase,
                        threshold=0.3
                    )
                    
                    action = "BUY" if self.position == 1 else "SELL"
                    entry_index = i  # Track when we entered
                    price_history = [current['close']]  # Initialize price history
                    
                    # Debug logging for entries
                    if self.debug:
                        confirmation_status = "with 15m confirmation" if data_15m is not None else "no 15m data"
                        self.log(f"\n=== ENTER {action} at {current['close']:.2f} (Bar {i}) {confirmation_status} ===")
                        self.log(f"Primary Signal: {primary_reason}")
                        self.log(f"Secondary Signal: {secondary_reason}")
                        self.log(f"Final Score: {signal_strength_score:.2f} | Volatility: {normalized_volatility:.3f} | Position Size: {position_size:.4f}")
                        self.log(f"Using improved dynamic exit logic")
                else:
                    # Signal not confirmed by intraday analysis
                    action = "WAIT_CONFIRMATION"
                    if self.debug:
                        self.log(f"Signal rejected due to lack of intraday confirmation")
            
            
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
                
                # ✅ PRIORITY 1: Check for end-of-day exit first (prop firm compliance)
                current_timestamp = current.name if hasattr(current, 'name') else data.index[i]
                if self.should_exit_end_of_day(current_timestamp):
                    action = "EXIT_END_OF_DAY"
                    exit_price = current['close']
                    exit_condition = "End-of-day exit (prop firm compliance)"
                    
                    if self.debug:
                        self.log(f"Forced end-of-day exit at {current_timestamp}")
                
                # ✅ PRIORITY 2: Check improved dynamic exit conditions
                else:
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

    def evaluate_performance(self, data):
        """Evaluate performance on given data slice."""
        # Calculate portfolio value and PnL from trades
        initial_value = 100000  # Starting capital
        portfolio_value = [initial_value]
        pnl_series = []
        
        for i in range(1, len(data)):
            current_row = data.iloc[i]
            prev_row = data.iloc[i-1]
            
            # Simple PnL calculation based on position changes
            if 'position' in data.columns and current_row['position'] != 0:
                if prev_row['position'] == 0:  # New position
                    entry_price = current_row['close']
                    pnl = 0
                else:  # Existing position
                    price_change = (current_row['close'] - entry_price) / entry_price
                    if current_row['position'] == 1:  # Long
                        pnl = price_change
                    else:  # Short
                        pnl = -price_change
                    
                    if current_row['position'] == 0:  # Position closed
                        pnl_series.append(pnl)
            else:
                pnl = 0
            
            new_value = portfolio_value[-1] * (1 + pnl * 0.01)  # Assuming 1% risk per trade
            portfolio_value.append(new_value)
        
        # Convert to pandas Series for calculations
        portfolio_series = pd.Series(portfolio_value)
        pnl_series = pd.Series(pnl_series) if pnl_series else pd.Series([0])
        
        # Calculate metrics
        total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
        win_rate = (pnl_series > 0).mean() if len(pnl_series) > 0 else 0
        max_drawdown = calculate_max_drawdown(portfolio_series)
        sharpe = calculate_sharpe_ratio(pnl_series.dropna())
        
        return {
            "total_return_pct": round(total_return * 100, 2),
            "win_rate": round(win_rate * 100, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "sharpe_ratio": round(sharpe, 2),
            "num_trades": len(pnl_series)
        }

    def walk_forward_test(self, data, window_size_days=125, step_size_days=60, 
                         compliance_firm="FTMO", initial_balance=100000):
        """
        Enhanced walk-forward test using rolling windows with prop firm compliance validation.

        Args:
            data: pd.DataFrame with price and indicators.
            window_size_days: total length of each window (train + test), in trading days.
            step_size_days: how much to move the window forward each iteration.
            compliance_firm: "FTMO" or "FXIFY" for compliance testing.
            initial_balance: Starting balance for compliance calculations.

        Returns:
            pd.DataFrame summary of window-by-window performance metrics with compliance results.
        """
        results = []
        total_days = len(data)

        bars_per_day = 7  # approximate for 1-hour bars (7 trading hours per day)

        window_size = window_size_days * bars_per_day
        step_size = step_size_days * bars_per_day

        start = 0
        end = start + window_size

        print(f"Starting enhanced walk-forward test with {total_days} data points")
        print(f"Window size: {window_size} bars ({window_size_days} days)")
        print(f"Step size: {step_size} bars ({step_size_days} days)")
        print(f"Compliance testing: {compliance_firm} rules")
        print("=" * 60)

        window_count = 0
        compliance_passed_count = 0
        
        while end < total_days:
            window_count += 1
            
            in_sample_end = start + int(window_size * 0.6)  # ~60% in-sample
            out_sample_start = in_sample_end
            out_sample_end = end

            # Out-of-sample slice
            out_sample_data = data.iloc[out_sample_start:out_sample_end].copy()
            
            print(f"\nWindow {window_count}: Testing on {len(out_sample_data)} bars")
            print(f"  Period: {data.index[out_sample_start]} to {data.index[out_sample_end - 1]}")
            
            # Reset algorithm state for each window
            self.position = 0
            self.entry_price = 0
            self.trades = []
            
            # Calculate indicators for this slice
            out_sample_data = self.calculate_indicators(out_sample_data)
            
            # Run strategy on out-of-sample
            strategy_results = self.execute_strategy(out_sample_data)
            
            # Evaluate metrics
            try:
                metrics = self.evaluate_performance(strategy_results)
                metrics["window_start"] = data.index[out_sample_start]
                metrics["window_end"] = data.index[out_sample_end - 1]
                metrics["window_number"] = window_count
                
                # ===== PROP FIRM COMPLIANCE VALIDATION =====
                # Calculate P&L and equity progression for compliance testing
                window_pnl = 0.0
                min_daily_pnl = 0.0
                max_drawdown_experienced = 0.0
                equity_curve = [initial_balance]
                high_watermark = initial_balance
                
                # Calculate daily P&L and equity progression from trades
                if self.trades:
                    trades_df = pd.DataFrame(self.trades)
                    
                    # Calculate cumulative P&L
                    for _, trade in trades_df.iterrows():
                        trade_pnl = trade['pnl_pct'] * initial_balance * 0.01  # Assuming 1% risk per trade
                        window_pnl += trade_pnl
                        current_equity = equity_curve[-1] + trade_pnl
                        equity_curve.append(current_equity)
                        
                        # Track worst daily P&L (simplified: worst single trade)
                        if trade_pnl < min_daily_pnl:
                            min_daily_pnl = trade_pnl
                        
                        # Update high watermark and calculate drawdown
                        if current_equity > high_watermark:
                            high_watermark = current_equity
                        
                        drawdown = current_equity - high_watermark
                        if drawdown < max_drawdown_experienced:
                            max_drawdown_experienced = drawdown
                
                # Run compliance checks
                compliance_passed = True
                compliance_notes = []
                
                # Initialize compliance checker based on firm
                if compliance_firm.upper() == "FTMO":
                    compliance = FTMOCompliance(initial_balance=initial_balance, debug=False)
                elif compliance_firm.upper() == "FXIFY":
                    compliance = FXIFYCompliance(initial_balance=initial_balance, debug=False)
                else:
                    compliance = FTMOCompliance(initial_balance=initial_balance, debug=False)  # Default to FTMO
                
                # Check daily loss limits
                daily_compliant, daily_reason = compliance.check_daily_limits(min_daily_pnl)
                if not daily_compliant:
                    compliance_passed = False
                    compliance_notes.append(f"Daily loss: {daily_reason}")
                
                # Check total drawdown limits
                max_drawdown_limit = compliance.max_loss_limit if hasattr(compliance, 'max_loss_limit') else compliance.max_drawdown_limit
                drawdown_compliant, drawdown_reason = compliance.check_total_drawdown(
                    equity_curve[-1] if equity_curve else initial_balance, 
                    high_watermark, 
                    max_drawdown_limit
                )
                if not drawdown_compliant:
                    compliance_passed = False
                    compliance_notes.append(f"Total drawdown: {drawdown_reason}")
                
                # Check position sizing compliance (simplified check)
                if self.trades:
                    max_position_size = 0.0
                    for _, trade in trades_df.iterrows():
                        # Estimate position size from P&L (this is approximate)
                        estimated_size = abs(trade['pnl_pct']) / 2.0  # Rough estimate
                        if estimated_size > max_position_size:
                            max_position_size = estimated_size
                    
                    if max_position_size > (0.03 if self.account_phase == "evaluation" else 0.05):
                        compliance_passed = False
                        compliance_notes.append(f"Position size exceeded: {max_position_size:.2%} > {0.03 if self.account_phase == 'evaluation' else 0.05:.0%}")
                
                # Add compliance results to metrics
                metrics["compliance_passed"] = compliance_passed
                metrics["compliance_notes"] = "; ".join(compliance_notes) if compliance_notes else "All checks passed"
                metrics["compliance_firm"] = compliance_firm
                metrics["worst_daily_pnl"] = min_daily_pnl
                metrics["max_drawdown_amount"] = max_drawdown_experienced
                metrics["final_equity"] = equity_curve[-1] if equity_curve else initial_balance
                
                if compliance_passed:
                    compliance_passed_count += 1
                
                # Print window results with compliance status
                compliance_status = "✅ PASS" if compliance_passed else "❌ FAIL"
                print(f"  Return: {metrics['total_return_pct']:.2f}%, "
                      f"Win Rate: {metrics['win_rate']:.1f}%, "
                      f"Trades: {metrics['num_trades']}, "
                      f"Compliance: {compliance_status}")
                
                if not compliance_passed:
                    print(f"  Violations: {metrics['compliance_notes']}")
                
                results.append(metrics)
                
            except Exception as e:
                print(f"  ❌ Error in window {window_count}: {e}")
                # Add empty metrics to maintain sequence
                metrics = {
                    "total_return_pct": 0,
                    "win_rate": 0,
                    "max_drawdown_pct": 0,
                    "sharpe_ratio": 0,
                    "num_trades": 0,
                    "window_start": data.index[out_sample_start],
                    "window_end": data.index[out_sample_end - 1],
                    "window_number": window_count,
                    "compliance_passed": False,
                    "compliance_notes": f"Error during testing: {str(e)}",
                    "compliance_firm": compliance_firm,
                    "worst_daily_pnl": 0,
                    "max_drawdown_amount": 0,
                    "final_equity": initial_balance
                }
                results.append(metrics)

            # Move window forward
            start += step_size
            end = start + window_size

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            print(f"\n{'='*60}")
            print(f"🎯 ENHANCED WALK-FORWARD TEST SUMMARY")
            print(f"{'='*60}")
            print(f"Total Windows Tested: {len(results_df)}")
            print(f"Compliance Firm: {compliance_firm}")
            print(f"Account Phase: {self.account_phase}")
            print(f"Initial Balance: ${initial_balance:,.2f}")
            print()
            print(f"📊 PERFORMANCE METRICS:")
            print(f"Average Return: {results_df['total_return_pct'].mean():.2f}%")
            print(f"Average Win Rate: {results_df['win_rate'].mean():.1f}%")
            print(f"Average Sharpe: {results_df['sharpe_ratio'].mean():.2f}")
            print(f"Positive Windows: {(results_df['total_return_pct'] > 0).sum()}/{len(results_df)} ({(results_df['total_return_pct'] > 0).mean()*100:.1f}%)")
            print()
            print(f"🏛️ COMPLIANCE RESULTS:")
            print(f"Compliant Windows: {compliance_passed_count}/{len(results_df)} ({compliance_passed_count/len(results_df)*100:.1f}%)")
            print(f"Failed Windows: {len(results_df) - compliance_passed_count}/{len(results_df)} ({(len(results_df) - compliance_passed_count)/len(results_df)*100:.1f}%)")
            
            # Show compliance failure reasons
            failed_windows = results_df[~results_df['compliance_passed']]
            if not failed_windows.empty:
                print(f"\n❌ COMPLIANCE FAILURE ANALYSIS:")
                failure_reasons = {}
                for _, row in failed_windows.iterrows():
                    notes = row['compliance_notes']
                    if notes in failure_reasons:
                        failure_reasons[notes] += 1
                    else:
                        failure_reasons[notes] = 1
                
                for reason, count in failure_reasons.items():
                    print(f"  - {reason}: {count} window(s)")
            
            print(f"\n✅ DEPLOYMENT READINESS:")
            readiness_score = compliance_passed_count / len(results_df) * 100
            if readiness_score >= 80:
                print(f"🟢 EXCELLENT ({readiness_score:.1f}%): Ready for prop firm deployment")
            elif readiness_score >= 60:
                print(f"🟡 GOOD ({readiness_score:.1f}%): Suitable with risk monitoring")
            elif readiness_score >= 40:
                print(f"🟠 FAIR ({readiness_score:.1f}%): Requires optimization before deployment")
            else:
                print(f"🔴 POOR ({readiness_score:.1f}%): Major issues, not suitable for prop firm trading")
        
        return results_df
    
    def get_intraday_confirmation(self, data_15m, idx_15m, signal_direction):
        """
        Intraday confirmation logic for prop firm trading using 15-minute data.
        
        Args:
            data_15m: 15-minute timeframe data with calculated indicators
            idx_15m: index position in the 15-minute data
            signal_direction: 1 for long, -1 for short
            
        Returns:
            bool: True if signal is confirmed, False otherwise
        """
        try:
            # Handle different index types (integer vs timestamp)
            if isinstance(idx_15m, int):
                if idx_15m >= len(data_15m):
                    return False
                current_row = data_15m.iloc[idx_15m]
            else:
                # If idx_15m is a timestamp, use .loc
                if idx_15m not in data_15m.index:
                    return False
                current_row = data_15m.loc[idx_15m]
            
            # Extract indicators with fallback values
            rsi = current_row.get("RSI", 50)
            ema12 = current_row.get("EMA_12", 0)
            ema26 = current_row.get("EMA_26", 0)
            
            # Validate that we have meaningful indicator values
            if rsi == 0 or ema12 == 0 or ema26 == 0:
                return False
            
            # Long confirmation: RSI > 45 and EMA_12 > EMA_26
            if signal_direction == 1:
                if rsi > 45 and ema12 > ema26:
                    if self.debug:
                        self.log(f"Long confirmation: RSI={rsi:.1f}, EMA12={ema12:.4f}, EMA26={ema26:.4f}")
                    return True
            
            # Short confirmation: RSI < 55 and EMA_12 < EMA_26
            elif signal_direction == -1:
                if rsi < 55 and ema12 < ema26:
                    if self.debug:
                        self.log(f"Short confirmation: RSI={rsi:.1f}, EMA12={ema12:.4f}, EMA26={ema26:.4f}")
                    return True
            
            return False
            
        except (KeyError, IndexError, TypeError) as e:
            if self.debug:
                self.log(f"Intraday confirmation error: {e}")
            return False

    def should_exit_end_of_day(self, current_time):
        """
        Check if we should exit positions at end of day for prop firm compliance.
        Exits positions 30 minutes before market close to avoid overnight risk.
        
        Args:
            current_time: pandas Timestamp or datetime object
            
        Returns:
            bool: True if should exit, False otherwise
        """
        try:
            # Handle different time formats
            if hasattr(current_time, 'hour'):
                current_hour = current_time.hour
                current_minute = current_time.minute
            else:
                # If it's a string or other format, try to parse
                if isinstance(current_time, str):
                    current_time = pd.to_datetime(current_time)
                    current_hour = current_time.hour
                    current_minute = current_time.minute
                else:
                    return False
            
            # Market typically closes at 21:00 UTC (9 PM)
            # Exit 30 minutes before: 20:30 UTC (8:30 PM)
            market_close_hour = 21
            market_close_minute = 0
            exit_buffer_minutes = 30
            
            # Calculate exit time (30 minutes before market close)
            exit_hour = market_close_hour
            exit_minute = market_close_minute - exit_buffer_minutes
            
            # Handle minute underflow
            if exit_minute < 0:
                exit_minute += 60
                exit_hour -= 1
            
            # Check if current time is at or after the exit time
            if current_hour > exit_hour or (current_hour == exit_hour and current_minute >= exit_minute):
                if self.debug:
                    self.log(f"End-of-day exit triggered at {current_hour:02d}:{current_minute:02d}")
                return True
            
            return False
            
        except Exception as e:
            if self.debug:
                self.log(f"Error in should_exit_end_of_day: {e}")
            return False

    def update_risk_limits(self, equity, daily_pnl, max_daily_loss=-5000, max_total_drawdown=-10000):
        """
        Prop firm risk management - daily loss and total drawdown limits
        Enhanced with detailed debug logging for compliance monitoring
        """
        # Calculate current metrics for logging
        current_drawdown = equity - self.high_watermark
        daily_loss_pct = (daily_pnl / equity) * 100 if equity > 0 else 0
        total_drawdown_pct = (current_drawdown / self.high_watermark) * 100 if self.high_watermark > 0 else 0
        
        # Enhanced debug logging for risk monitoring
        if self.debug:
            self.log(f"Risk Limits Check:")
            self.log(f"  Current Equity: ${equity:,.2f}")
            self.log(f"  High Watermark: ${self.high_watermark:,.2f}")
            self.log(f"  Daily P&L: ${daily_pnl:,.2f} ({daily_loss_pct:+.2f}%)")
            self.log(f"  Current Drawdown: ${current_drawdown:,.2f} ({total_drawdown_pct:+.2f}%)")
            self.log(f"  Daily Loss Limit: ${max_daily_loss:,.2f}")
            self.log(f"  Total Drawdown Limit: ${max_total_drawdown:,.2f}")
        
        # Check daily loss limit
        if daily_pnl <= max_daily_loss:
            violation_msg = (f"DAILY LOSS LIMIT BREACHED! "
                           f"Daily P&L: ${daily_pnl:,.2f} ({daily_loss_pct:+.2f}%) "
                           f"exceeds limit: ${max_daily_loss:,.2f}. "
                           f"Current equity: ${equity:,.2f}. "
                           f"Blocking trades for today.")
            
            if self.debug:
                self.log(f"⚠️  {violation_msg}")
            else:
                self.log(violation_msg)
            
            self.skip_today = True
            return False

        # Check total drawdown limit
        if current_drawdown <= max_total_drawdown:
            violation_msg = (f"TOTAL DRAWDOWN LIMIT BREACHED! "
                           f"Current drawdown: ${current_drawdown:,.2f} ({total_drawdown_pct:+.2f}%) "
                           f"exceeds limit: ${max_total_drawdown:,.2f}. "
                           f"High watermark: ${self.high_watermark:,.2f}, "
                           f"Current equity: ${equity:,.2f}. "
                           f"Stopping trading permanently.")
            
            if self.debug:
                self.log(f"❌ {violation_msg}")
            else:
                self.log(violation_msg)
            
            self.stop_trading = True
            return False
        
        # Log successful risk check in debug mode
        if self.debug:
            self.log(f"✅ Risk limits check passed - trading continues")
            
        return True

    def execute_strategy_with_intraday_confirmation(self, data_1h: pd.DataFrame, data_15m: pd.DataFrame = None) -> pd.DataFrame:
        """
        Execute strategy with intraday confirmation using 15-minute data.
        
        Args:
            data_1h: Hourly timeframe data (main signals)
            data_15m: 15-minute timeframe data (confirmation)
            
        Returns:
            pd.DataFrame: Strategy execution results
        """
        if self.debug:
            self.log("Executing strategy with 15-minute intraday confirmation")
            self.log(f"Hourly data points: {len(data_1h)}")
            if data_15m is not None:
                self.log(f"15-minute data points: {len(data_15m)}")
            else:
                self.log("15-minute data: None (using hourly signals only)")
        
        # Ensure both datasets have required indicators
        data_1h_with_indicators = self.calculate_indicators(data_1h)
        data_15m_with_indicators = self.calculate_indicators(data_15m) if data_15m is not None else None
        
        # Execute strategy with intraday confirmation
        return self.execute_strategy(data_1h_with_indicators, data_15m_with_indicators)

# =============================================================================
# PROP FIRM COMPLIANCE CLASSES
# =============================================================================

class BasePropFirmCompliance:
    """Base class for prop firm compliance rules"""
    
    def __init__(self, debug=False):
        self.debug = debug
        self.firm_name = "Unknown"
        self.daily_loss_limit = 0.0
        self.max_loss_limit = 0.0
        self.restricted_strategies = []
        self.compliance_violations = []
    
    def log(self, msg):
        if self.debug:
            print(f"[{self.firm_name}] {msg}")
    
    def validate_trade(self, trade_params):
        """
        Validate if a trade is compliant with firm rules.
        
        Args:
            trade_params: Dict containing trade parameters
            
        Returns:
            tuple: (is_compliant: bool, violations: list)
        """
        raise NotImplementedError("Subclasses must implement validate_trade")
    
    def check_daily_limits(self, daily_pnl):
        """
        Check if daily PnL is within limits.
        
        Args:
            daily_pnl: Current daily profit/loss
            
        Returns:
            tuple: (is_compliant: bool, reason: str)
        """
        if daily_pnl <= self.daily_loss_limit:
            reason = f"Daily loss limit exceeded: {daily_pnl:.2f} <= {self.daily_loss_limit:.2f}"
            self.log(reason)
            return False, reason
        return True, "Daily limits OK"
    
    def check_total_drawdown(self, equity, high_watermark, max_drawdown):
        """
        Check if total drawdown is within limits.
        
        Args:
            equity: Current equity
            high_watermark: Peak equity value
            max_drawdown: Maximum allowed drawdown (negative value)
            
        Returns:
            tuple: (is_compliant: bool, reason: str)
        """
        current_drawdown = equity - high_watermark
        if current_drawdown < max_drawdown:
            reason = f"Max drawdown exceeded: {current_drawdown:.2f} < {max_drawdown:.2f}"
            self.log(reason)
            return False, reason
        return True, "Drawdown limits OK"


class FTMOCompliance(BasePropFirmCompliance):
    """
    FTMO prop firm compliance rules based on EA documentation.
    
    Key Rules:
    - Daily loss limit: 5% of initial balance
    - Max loss limit: 10% of initial balance
    - Prohibited: HFT, Arbitrage, Martingale, Grid Trading
    - Allowed: Standard trading EAs, scalping (non-HFT), custom strategies
    """
    
    def __init__(self, initial_balance=100000, debug=False):
        super().__init__(debug)
        self.firm_name = "FTMO"
        self.initial_balance = initial_balance
        
        # FTMO specific limits (as percentages, converted to absolute values)
        self.daily_loss_limit = -initial_balance * 0.05  # -5% daily loss
        self.max_loss_limit = -initial_balance * 0.10     # -10% max loss
        
        # Prohibited strategies
        self.restricted_strategies = [
            "high_frequency_trading",
            "arbitrage",
            "latency_arbitrage", 
            "martingale",
            "grid_trading"
        ]
        
        # Position sizing limits
        self.max_position_size = 0.02  # 2% risk per trade (conservative)
        self.max_leverage = 100  # 1:100 leverage
        
        if self.debug:
            self.log(f"Initialized with balance: ${initial_balance:,.2f}")
            self.log(f"Daily loss limit: ${self.daily_loss_limit:,.2f}")
            self.log(f"Max loss limit: ${self.max_loss_limit:,.2f}")
    
    def validate_trade(self, trade_params):
        """
        Validate trade against FTMO rules.
        
        Args:
            trade_params: Dict with keys:
                - strategy_type: str
                - position_size: float (as percentage)
                - leverage: float
                - trade_frequency: int (trades per hour)
                - risk_per_trade: float (as percentage)
                
        Returns:
            tuple: (is_compliant: bool, violations: list)
        """
        violations = []
        
        # Check strategy type
        strategy_type = trade_params.get("strategy_type", "").lower()
        if strategy_type in self.restricted_strategies:
            violations.append(f"Strategy '{strategy_type}' is prohibited by FTMO")
        
        # Check position size
        position_size = trade_params.get("position_size", 0)
        if position_size > self.max_position_size:
            violations.append(f"Position size {position_size:.2%} exceeds max {self.max_position_size:.2%}")
        
        # Check leverage
        leverage = trade_params.get("leverage", 1)
        if leverage > self.max_leverage:
            violations.append(f"Leverage {leverage}:1 exceeds max {self.max_leverage}:1")
        
        # Check for high frequency trading
        trade_frequency = trade_params.get("trade_frequency", 0)
        if trade_frequency > 10:  # More than 10 trades per hour = HFT
            violations.append(f"Trade frequency {trade_frequency}/hour suggests HFT (prohibited)")
        
        # Check risk per trade
        risk_per_trade = trade_params.get("risk_per_trade", 0)
        if risk_per_trade > 0.02:  # 2% max risk per trade
            violations.append(f"Risk per trade {risk_per_trade:.2%} exceeds recommended 2%")
        
        is_compliant = len(violations) == 0
        
        if not is_compliant:
            self.log(f"Trade validation failed: {violations}")
        
        return is_compliant, violations
    
    def check_profit_target(self, current_profit, phase="challenge"):
        """
        Check if profit target is met for specific phase.
        
        Args:
            current_profit: Current profit amount
            phase: "challenge" or "verification"
            
        Returns:
            tuple: (target_met: bool, required_profit: float, reason: str)
        """
        if phase == "challenge":
            target = self.initial_balance * 0.10  # 10% profit target
        elif phase == "verification":
            target = self.initial_balance * 0.05  # 5% profit target
        else:
            target = 0  # No target for funded account
        
        target_met = current_profit >= target
        reason = f"Profit target: {current_profit:.2f} / {target:.2f} ({phase})"
        
        return target_met, target, reason


class FXIFYCompliance(BasePropFirmCompliance):
    """
    FXIFY prop firm compliance rules based on EA documentation.
    
    Key Rules:
    - Daily loss limit: 2% (Starter) / 3% (Expert) 
    - Max drawdown: 4% (Starter) / 4-5% (Expert)
    - Consistency rule: 30% (Starter) / 40% (Expert)
    - Prohibited: Latency arbitrage, HFT, group hedging, reverse hedging
    - Note: EAs forbidden in Instant Funding accounts
    """
    
    def __init__(self, initial_balance=100000, account_type="starter", debug=False):
        super().__init__(debug)
        self.firm_name = "FXIFY"
        self.initial_balance = initial_balance
        self.account_type = account_type.lower()
        
        # FXIFY specific limits based on account type
        if self.account_type == "starter":
            self.daily_loss_limit = -initial_balance * 0.02  # -2% daily loss
            self.max_drawdown_limit = -initial_balance * 0.04  # -4% max drawdown
            self.consistency_limit = 0.30  # 30% consistency rule
        elif self.account_type == "expert":
            self.daily_loss_limit = -initial_balance * 0.03  # -3% daily loss
            self.max_drawdown_limit = -initial_balance * 0.05  # -5% max drawdown
            self.consistency_limit = 0.40  # 40% consistency rule
        else:
            # Default to starter limits
            self.daily_loss_limit = -initial_balance * 0.02
            self.max_drawdown_limit = -initial_balance * 0.04
            self.consistency_limit = 0.30
        
        # Prohibited strategies
        self.restricted_strategies = [
            "latency_arbitrage",
            "high_frequency_trading",
            "group_hedging",
            "reverse_hedging",
            "coordinated_trading"
        ]
        
        # Position sizing limits (more aggressive than FTMO)
        self.max_position_size = 0.05  # 5% risk per trade
        self.max_leverage = 500  # 1:500 leverage
        
        if self.debug:
            self.log(f"Initialized {account_type} account with balance: ${initial_balance:,.2f}")
            self.log(f"Daily loss limit: ${self.daily_loss_limit:,.2f}")
            self.log(f"Max drawdown limit: ${self.max_drawdown_limit:,.2f}")
            self.log(f"Consistency limit: {self.consistency_limit:.0%}")
    
    def validate_trade(self, trade_params):
        """
        Validate trade against FXIFY rules.
        
        Args:
            trade_params: Dict with keys:
                - strategy_type: str
                - position_size: float (as percentage)
                - leverage: float
                - trade_frequency: int (trades per hour)
                - is_instant_funding: bool
                - is_ea_trade: bool
                
        Returns:
            tuple: (is_compliant: bool, violations: list)
        """
        violations = []
        
        # Check if EA is allowed (forbidden in instant funding)
        is_instant_funding = trade_params.get("is_instant_funding", False)
        is_ea_trade = trade_params.get("is_ea_trade", False)
        
        if is_instant_funding and is_ea_trade:
            violations.append("EAs are strictly forbidden in FXIFY Instant Funding accounts")
        
        # Check strategy type
        strategy_type = trade_params.get("strategy_type", "").lower()
        if strategy_type in self.restricted_strategies:
            violations.append(f"Strategy '{strategy_type}' is prohibited by FXIFY")
        
        # Check position size
        position_size = trade_params.get("position_size", 0)
        if position_size > self.max_position_size:
            violations.append(f"Position size {position_size:.2%} exceeds max {self.max_position_size:.2%}")
        
        # Check leverage
        leverage = trade_params.get("leverage", 1)
        if leverage > self.max_leverage:
            violations.append(f"Leverage {leverage}:1 exceeds max {self.max_leverage}:1")
        
        # Check for high frequency trading
        trade_frequency = trade_params.get("trade_frequency", 0)
        if trade_frequency > 20:  # More than 20 trades per hour = HFT
            violations.append(f"Trade frequency {trade_frequency}/hour suggests HFT (prohibited)")
        
        is_compliant = len(violations) == 0
        
        if not is_compliant:
            self.log(f"Trade validation failed: {violations}")
        
        return is_compliant, violations
    
    def check_consistency_rule(self, largest_profit, total_profit):
        """
        Check FXIFY consistency rule.
        
        Args:
            largest_profit: Largest profitable trading day
            total_profit: Total profit
            
        Returns:
            tuple: (is_compliant: bool, ratio: float, reason: str)
        """
        if total_profit <= 0:
            return True, 0.0, "No profit to check consistency"
        
        consistency_ratio = largest_profit / total_profit
        is_compliant = consistency_ratio <= self.consistency_limit
        
        reason = f"Consistency: {consistency_ratio:.1%} (limit: {self.consistency_limit:.0%})"
        
        if not is_compliant:
            self.log(f"Consistency rule violated: {reason}")
        
        return is_compliant, consistency_ratio, reason


# =============================================================================
# UNIFIED COMPLIANCE CHECKER
# =============================================================================

def check_prop_firm_compliance(firm_name, trade_params, daily_pnl, equity, high_watermark, 
                             initial_balance=100000, account_type="starter", debug=False):
    """
    Unified compliance checker for multiple prop firms.
    
    Args:
        firm_name: "FTMO" or "FXIFY"
        trade_params: Dict with trade parameters
        daily_pnl: Current daily P&L
        equity: Current equity
        high_watermark: Peak equity
        initial_balance: Starting balance
        account_type: Account type (for FXIFY)
        debug: Enable debug logging
        
    Returns:
        dict: Compliance results with all checks
    """
    
    # Initialize appropriate compliance checker
    if firm_name.upper() == "FTMO":
        compliance = FTMOCompliance(initial_balance, debug)
    elif firm_name.upper() == "FXIFY":
        compliance = FXIFYCompliance(initial_balance, account_type, debug)
    else:
        return {"error": f"Unknown firm: {firm_name}"}
    
    # Run all compliance checks
    results = {
        "firm": firm_name,
        "account_type": account_type,
        "overall_compliant": True,
        "violations": []
    }
    
    # Check trade validation
    trade_compliant, trade_violations = compliance.validate_trade(trade_params)
    results["trade_compliant"] = trade_compliant
    results["trade_violations"] = trade_violations
    if not trade_compliant:
        results["overall_compliant"] = False
        results["violations"].extend(trade_violations)
    
    # Check daily limits
    daily_compliant, daily_reason = compliance.check_daily_limits(daily_pnl)
    results["daily_compliant"] = daily_compliant
    results["daily_reason"] = daily_reason
    if not daily_compliant:
        results["overall_compliant"] = False
        results["violations"].append(daily_reason)
    
    # Check total drawdown
    max_drawdown = compliance.max_loss_limit if hasattr(compliance, 'max_loss_limit') else compliance.max_drawdown_limit
    drawdown_compliant, drawdown_reason = compliance.check_total_drawdown(equity, high_watermark, max_drawdown)
    results["drawdown_compliant"] = drawdown_compliant
    results["drawdown_reason"] = drawdown_reason
    if not drawdown_compliant:
        results["overall_compliant"] = False
        results["violations"].append(drawdown_reason)
    
    # FXIFY specific: consistency check
    if firm_name.upper() == "FXIFY" and "largest_profit" in trade_params and "total_profit" in trade_params:
        consistency_compliant, consistency_ratio, consistency_reason = compliance.check_consistency_rule(
            trade_params["largest_profit"], trade_params["total_profit"]
        )
        results["consistency_compliant"] = consistency_compliant
        results["consistency_ratio"] = consistency_ratio
        results["consistency_reason"] = consistency_reason
        if not consistency_compliant:
            results["overall_compliant"] = False
            results["violations"].append(consistency_reason)
    
    return results

