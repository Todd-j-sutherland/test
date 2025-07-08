# src/trading_orchestrator.py
"""
Real-time trading orchestrator that integrates all analysis components
with automated position management and risk controls.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.async_main import ASXBankTradingSystemAsync
from src.advanced_risk_manager import AdvancedRiskManager
from src.alert_system import AlertSystem
from config.settings import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeAction(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"

class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    action: TradeAction
    quantity: float
    entry_price: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    status: PositionStatus
    current_price: float = 0.0
    current_pnl: float = 0.0
    risk_level: str = "medium"

@dataclass
class TradeSignal:
    """Represents a trade signal from analysis"""
    symbol: str
    action: TradeAction
    confidence: float
    price_target: float
    stop_loss: float
    take_profit: float
    risk_level: str
    reasoning: str
    timestamp: datetime

class TradingOrchestrator:
    """
    Real-time trading orchestrator that combines analysis, risk management,
    and position management with automated execution capabilities.
    """
    
    def __init__(self, paper_trading: bool = True):
        self.settings = Settings()
        self.analysis_system = None
        self.risk_manager = AdvancedRiskManager()
        self.alert_system = AlertSystem()
        self.paper_trading = paper_trading
        
        # Portfolio state
        self.positions: Dict[str, Position] = {}
        self.account_balance = 100000.0  # $100k starting balance
        self.cash_available = self.account_balance
        self.total_value = self.account_balance
        
        # Risk parameters
        self.max_position_size = 0.10  # 10% per position
        self.max_portfolio_risk = 0.20  # 20% total risk
        self.stop_loss_pct = 0.05  # 5% stop loss
        self.take_profit_pct = 0.10  # 10% take profit
        self.min_confidence = 0.3  # Minimum confidence for trades
        
        # Trading state
        self.is_running = False
        self.last_analysis_time = None
        self.analysis_interval = 300  # 5 minutes
        
        # Performance tracking
        self.trade_history = []
        self.daily_pnl = []
        
    async def initialize(self):
        """Initialize the trading system"""
        try:
            self.analysis_system = ASXBankTradingSystemAsync()
            await self.analysis_system.__aenter__()
            logger.info("🚀 Trading orchestrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize trading orchestrator: {e}")
            raise
    
    async def start_trading(self):
        """Start the real-time trading loop"""
        if not self.analysis_system:
            await self.initialize()
        
        self.is_running = True
        logger.info("🎯 Starting real-time trading orchestrator...")
        
        # Send startup notification
        try:
            self.alert_system.send_alert({
                'type': 'system_start',
                'message': f"Trading orchestrator started - Paper Trading: {self.paper_trading}",
                'account_balance': self.account_balance,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to send startup alert: {e}")
        
        try:
            while self.is_running:
                await self._trading_cycle()
                await asyncio.sleep(self.analysis_interval)
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
        finally:
            await self.stop_trading()
    
    async def stop_trading(self):
        """Stop the trading system gracefully"""
        self.is_running = False
        
        # Close all open positions
        for position in self.positions.values():
            if position.status == PositionStatus.OPEN:
                await self._close_position(position.symbol, "System shutdown")
        
        # Send shutdown notification
        try:
            self.alert_system.send_alert({
                'type': 'system_stop',
                'message': "Trading orchestrator stopped",
                'final_balance': self.account_balance,
                'total_pnl': self.account_balance - 100000.0,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to send shutdown alert: {e}")
        
        if self.analysis_system:
            await self.analysis_system.__aexit__(None, None, None)
        
        logger.info("🛑 Trading orchestrator stopped")
    
    async def _trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            # 1. Run comprehensive analysis
            analysis_results = await self.analysis_system.analyze_all_banks_async()
            
            # 2. Update current positions
            await self._update_positions(analysis_results)
            
            # 3. Generate new signals
            signals = await self._generate_signals(analysis_results)
            
            # 4. Execute trades based on signals
            for signal in signals:
                await self._execute_signal(signal)
            
            # 5. Update portfolio metrics
            await self._update_portfolio_metrics()
            
            # 6. Send periodic updates
            await self._send_periodic_update()
            
            self.last_analysis_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in trading cycle: {e}")
            try:
                self.alert_system.send_alert({
                    'type': 'error',
                    'message': f"Trading cycle error: {str(e)}",
                    'timestamp': datetime.now().isoformat()
                })
            except:
                pass  # Don't let alert failures break the system
    
    async def _update_positions(self, analysis_results: Dict):
        """Update current positions with latest prices and check exit conditions"""
        for symbol, position in self.positions.items():
            if position.status != PositionStatus.OPEN:
                continue
            
            # Get current price from analysis
            if symbol in analysis_results and 'current_price' in analysis_results[symbol]:
                current_price = analysis_results[symbol]['current_price']
                position.current_price = current_price
                
                # Calculate P&L
                if position.action == TradeAction.BUY:
                    position.current_pnl = (current_price - position.entry_price) * position.quantity
                else:  # SELL
                    position.current_pnl = (position.entry_price - current_price) * position.quantity
                
                # Check exit conditions
                await self._check_exit_conditions(position)
    
    async def _check_exit_conditions(self, position: Position):
        """Check if position should be closed based on stop loss, take profit, or signal change"""
        current_price = position.current_price
        
        # Stop loss check
        if position.action == TradeAction.BUY and current_price <= position.stop_loss:
            await self._close_position(position.symbol, f"Stop loss hit at ${current_price:.2f}")
            return
        elif position.action == TradeAction.SELL and current_price >= position.stop_loss:
            await self._close_position(position.symbol, f"Stop loss hit at ${current_price:.2f}")
            return
        
        # Take profit check
        if position.action == TradeAction.BUY and current_price >= position.take_profit:
            await self._close_position(position.symbol, f"Take profit hit at ${current_price:.2f}")
            return
        elif position.action == TradeAction.SELL and current_price <= position.take_profit:
            await self._close_position(position.symbol, f"Take profit hit at ${current_price:.2f}")
            return
        
        # Time-based exit (hold for max 5 days)
        if (datetime.now() - position.entry_time).days >= 5:
            await self._close_position(position.symbol, "Time-based exit (5 days)")
            return
    
    async def _generate_signals(self, analysis_results: Dict) -> List[TradeSignal]:
        """Generate trade signals from analysis results"""
        signals = []
        
        for symbol, analysis in analysis_results.items():
            if 'error' in analysis:
                continue
            
            # Extract prediction data
            prediction = analysis.get('prediction', {})
            confidence = prediction.get('confidence', 0)
            direction = prediction.get('direction', 'neutral')
            current_price = analysis.get('current_price', 0)
            
            # Skip if confidence too low
            if confidence < self.min_confidence:
                continue
            
            # Skip if we already have a position
            if symbol in self.positions and self.positions[symbol].status == PositionStatus.OPEN:
                continue
            
            # Generate signal based on direction
            if direction == 'bullish':
                signal = TradeSignal(
                    symbol=symbol,
                    action=TradeAction.BUY,
                    confidence=confidence,
                    price_target=current_price * (1 + self.take_profit_pct),
                    stop_loss=current_price * (1 - self.stop_loss_pct),
                    take_profit=current_price * (1 + self.take_profit_pct),
                    risk_level=self._determine_risk_level(analysis),
                    reasoning=f"Bullish signal with {confidence:.1%} confidence",
                    timestamp=datetime.now()
                )
                signals.append(signal)
            
            elif direction == 'bearish':
                signal = TradeSignal(
                    symbol=symbol,
                    action=TradeAction.SELL,
                    confidence=confidence,
                    price_target=current_price * (1 - self.take_profit_pct),
                    stop_loss=current_price * (1 + self.stop_loss_pct),
                    take_profit=current_price * (1 - self.take_profit_pct),
                    risk_level=self._determine_risk_level(analysis),
                    reasoning=f"Bearish signal with {confidence:.1%} confidence",
                    timestamp=datetime.now()
                )
                signals.append(signal)
        
        return signals
    
    def _determine_risk_level(self, analysis: Dict) -> str:
        """Determine risk level from analysis"""
        risk_analysis = analysis.get('risk_analysis', {})
        overall_risk = risk_analysis.get('overall_risk_score', 50)
        
        if overall_risk >= 70:
            return "high"
        elif overall_risk >= 40:
            return "medium"
        else:
            return "low"
    
    async def _execute_signal(self, signal: TradeSignal):
        """Execute a trade signal"""
        try:
            # Calculate position size based on risk
            position_size = self._calculate_position_size(signal)
            
            if position_size <= 0:
                logger.warning(f"Position size too small for {signal.symbol}")
                return
            
            # Simulate trade execution
            if self.paper_trading:
                await self._execute_paper_trade(signal, position_size)
            else:
                await self._execute_real_trade(signal, position_size)
            
        except Exception as e:
            logger.error(f"Error executing signal for {signal.symbol}: {e}")
            try:
                self.alert_system.send_alert({
                    'type': 'execution_error',
                    'symbol': signal.symbol,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
            except:
                pass  # Don't let alert failures break the system
    
    def _calculate_position_size(self, signal: TradeSignal) -> float:
        """Calculate appropriate position size based on risk"""
        # Base position size as percentage of account
        base_position_value = self.account_balance * self.max_position_size
        
        # Adjust based on confidence
        confidence_multiplier = signal.confidence
        
        # Adjust based on risk level
        risk_multipliers = {"low": 1.0, "medium": 0.8, "high": 0.6}
        risk_multiplier = risk_multipliers.get(signal.risk_level, 0.8)
        
        # Calculate final position value
        position_value = base_position_value * confidence_multiplier * risk_multiplier
        
        # Ensure we don't exceed available cash
        position_value = min(position_value, self.cash_available)
        
        # Calculate quantity (assuming we can buy fractional shares)
        quantity = position_value / signal.price_target
        
        return quantity
    
    async def _execute_paper_trade(self, signal: TradeSignal, position_size: float):
        """Execute a paper trade (simulation)"""
        # Create position
        position = Position(
            symbol=signal.symbol,
            action=signal.action,
            quantity=position_size,
            entry_price=signal.price_target,
            entry_time=datetime.now(),
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status=PositionStatus.OPEN,
            current_price=signal.price_target,
            risk_level=signal.risk_level
        )
        
        # Update portfolio
        position_value = position_size * signal.price_target
        self.cash_available -= position_value
        self.positions[signal.symbol] = position
        
        # Log trade
        trade_record = {
            'symbol': signal.symbol,
            'action': signal.action.value,
            'quantity': position_size,
            'price': signal.price_target,
            'timestamp': datetime.now().isoformat(),
            'confidence': signal.confidence,
            'reasoning': signal.reasoning,
            'type': 'entry'
        }
        self.trade_history.append(trade_record)
        
        # Send alert
        try:
            self.alert_system.send_alert({
                'type': 'trade_entry',
                'symbol': signal.symbol,
                'action': signal.action.value,
                'quantity': position_size,
                'price': signal.price_target,
                'confidence': signal.confidence,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'reasoning': signal.reasoning,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to send trade alert: {e}")
        
        logger.info(f"📈 Executed {signal.action.value} {position_size:.2f} {signal.symbol} @ ${signal.price_target:.2f}")
    
    async def _execute_real_trade(self, signal: TradeSignal, position_size: float):
        """Execute a real trade (would integrate with broker API)"""
        # This would integrate with a real broker API like Interactive Brokers, Alpaca, etc.
        # For now, we'll just log that this would be a real trade
        logger.info(f"🚨 REAL TRADE: {signal.action.value} {position_size:.2f} {signal.symbol}")
        
        # For demonstration, we'll still execute as paper trade
        await self._execute_paper_trade(signal, position_size)
    
    async def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        if position.status != PositionStatus.OPEN:
            return
        
        # Calculate final P&L
        final_pnl = position.current_pnl
        
        # Update portfolio
        position_value = position.quantity * position.current_price
        self.cash_available += position_value
        self.account_balance += final_pnl
        
        # Update position status
        position.status = PositionStatus.CLOSED
        
        # Log trade
        trade_record = {
            'symbol': symbol,
            'action': 'close',
            'quantity': position.quantity,
            'price': position.current_price,
            'pnl': final_pnl,
            'timestamp': datetime.now().isoformat(),
            'reason': reason,
            'type': 'exit'
        }
        self.trade_history.append(trade_record)
        
        # Send alert
        try:
            self.alert_system.send_alert({
                'type': 'trade_exit',
                'symbol': symbol,
                'action': 'close',
                'quantity': position.quantity,
                'price': position.current_price,
                'pnl': final_pnl,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.warning(f"Failed to send exit alert: {e}")
        
        logger.info(f"🔄 Closed position {symbol} @ ${position.current_price:.2f} | P&L: ${final_pnl:.2f}")
    
    async def _update_portfolio_metrics(self):
        """Update portfolio performance metrics"""
        # Calculate total portfolio value
        total_position_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values() 
            if pos.status == PositionStatus.OPEN
        )
        
        self.total_value = self.cash_available + total_position_value
        
        # Calculate unrealized P&L
        unrealized_pnl = sum(
            pos.current_pnl 
            for pos in self.positions.values() 
            if pos.status == PositionStatus.OPEN
        )
        
        # Update daily P&L tracking
        daily_pnl = {
            'date': datetime.now().date().isoformat(),
            'total_value': self.total_value,
            'unrealized_pnl': unrealized_pnl,
            'cash_available': self.cash_available,
            'num_positions': len([p for p in self.positions.values() if p.status == PositionStatus.OPEN])
        }
        
        # Only add if it's a new day
        if not self.daily_pnl or self.daily_pnl[-1]['date'] != daily_pnl['date']:
            self.daily_pnl.append(daily_pnl)
    
    async def _send_periodic_update(self):
        """Send periodic portfolio status update"""
        # Only send every hour
        if (datetime.now().minute != 0):
            return
        
        open_positions = [pos for pos in self.positions.values() if pos.status == PositionStatus.OPEN]
        
        update_message = {
            'type': 'portfolio_update',
            'account_balance': self.account_balance,
            'total_value': self.total_value,
            'cash_available': self.cash_available,
            'num_positions': len(open_positions),
            'unrealized_pnl': sum(pos.current_pnl for pos in open_positions),
            'positions': [
                {
                    'symbol': pos.symbol,
                    'action': pos.action.value,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.current_pnl,
                    'days_held': (datetime.now() - pos.entry_time).days
                }
                for pos in open_positions
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            self.alert_system.send_alert(update_message)
        except Exception as e:
            logger.warning(f"Failed to send portfolio update: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        open_positions = [pos for pos in self.positions.values() if pos.status == PositionStatus.OPEN]
        
        return {
            'account_balance': self.account_balance,
            'total_value': self.total_value,
            'cash_available': self.cash_available,
            'total_return': (self.total_value - 100000.0) / 100000.0,
            'num_positions': len(open_positions),
            'unrealized_pnl': sum(pos.current_pnl for pos in open_positions),
            'positions': [
                {
                    'symbol': pos.symbol,
                    'action': pos.action.value,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.current_pnl,
                    'pnl_pct': pos.current_pnl / (pos.quantity * pos.entry_price) * 100,
                    'days_held': (datetime.now() - pos.entry_time).days
                }
                for pos in open_positions
            ],
            'recent_trades': self.trade_history[-10:],  # Last 10 trades
            'daily_pnl': self.daily_pnl[-7:],  # Last 7 days
            'timestamp': datetime.now().isoformat()
        }

# Demo/Testing Functions
async def demo_trading_orchestrator():
    """Demo function to test the trading orchestrator"""
    orchestrator = TradingOrchestrator(paper_trading=True)
    
    try:
        # Initialize
        await orchestrator.initialize()
        
        # Run for a few cycles (demo mode)
        for i in range(3):
            print(f"\n--- Trading Cycle {i+1} ---")
            await orchestrator._trading_cycle()
            
            # Show portfolio summary
            summary = orchestrator.get_portfolio_summary()
            print(f"Portfolio Value: ${summary['total_value']:.2f}")
            print(f"Cash Available: ${summary['cash_available']:.2f}")
            print(f"Open Positions: {summary['num_positions']}")
            print(f"Unrealized P&L: ${summary['unrealized_pnl']:.2f}")
            
            # Wait a bit between cycles
            await asyncio.sleep(10)
        
    except Exception as e:
        print(f"Demo error: {e}")
    finally:
        await orchestrator.stop_trading()

if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_trading_orchestrator())
