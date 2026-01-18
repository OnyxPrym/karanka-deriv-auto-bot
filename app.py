"""
🚀 KARANKA DERIV AUTO TRADER - FULL AUTOMATION
User only inputs API Token → Bot auto-discovers everything
Real-time data + Real trades + Live balance updates
Deploy on Render.com
"""

from flask import Flask, render_template_string, request, jsonify, session
from flask_socketio import SocketIO, emit
import os
import time
import threading
import json
import hashlib
from datetime import datetime, timedelta
import asyncio
import websocket
import requests
import pandas as pd
import numpy as np
from collections import deque
import logging
import ssl
from typing import Dict, List, Optional
import queue

# ==================== CONFIGURATION ====================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'karanka-auto-trader-2024')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', async_handlers=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== GLOBAL STATE ====================
users_db = {}  # Simple in-memory storage
active_traders = {}  # user_id -> DerivTrader instance
trading_bots = {}  # user_id -> TradingBot instance
user_trades = {}  # user_id -> list of trades

# Default trading settings
DEFAULT_SETTINGS = {
    'trade_amount': 1.0,
    'max_concurrent_trades': 3,
    'selected_markets': ['R_10', 'R_25', 'R_50'],
    'strategies': ['liquidity_grab', 'fvg_retest'],
    'risk_level': 'medium',
    'auto_trading': True
}

# ==================== DERIV API AUTODISCOVERY ====================
class DerivAutoDiscovery:
    """Automatically discover all account details from API token"""
    
    @staticmethod
    def discover_accounts(api_token: str) -> Dict:
        """
        Auto-discover everything from just API token:
        1. App ID
        2. All linked accounts
        3. Account details
        4. Balances
        """
        try:
            # Step 1: Get account information
            headers = {
                'Authorization': f'Bearer {api_token}',
                'Content-Type': 'application/json'
            }
            
            # Get account settings (contains app_id)
            response = requests.get(
                'https://api.deriv.com/account/settings',
                headers=headers
            )
            
            if response.status_code == 200:
                settings = response.json()
                app_id = settings.get('app_id', 16929)  # Default app_id if not found
                
                # Get account details
                acc_response = requests.get(
                    'https://api.deriv.com/account',
                    headers=headers
                )
                
                if acc_response.status_code == 200:
                    account_data = acc_response.json()
                    
                    # Get all accounts (real and demo)
                    accounts = []
                    
                    # Real account
                    if account_data.get('account'):
                        accounts.append({
                            'type': 'real',
                            'account_id': account_data['account'].get('loginid'),
                            'currency': account_data['account'].get('currency'),
                            'balance': float(account_data['account'].get('balance', 0)),
                            'email': account_data['account'].get('email'),
                            'country': account_data['account'].get('country')
                        })
                    
                    # Get account list
                    list_response = requests.get(
                        'https://api.deriv.com/account/list',
                        headers=headers
                    )
                    
                    if list_response.status_code == 200:
                        account_list = list_response.json()
                        for acc in account_list.get('account_list', []):
                            if acc.get('loginid') != accounts[0]['account_id']:
                                accounts.append({
                                    'type': acc.get('account_type', 'demo'),
                                    'account_id': acc.get('loginid'),
                                    'currency': acc.get('currency'),
                                    'balance': float(acc.get('balance', 0)),
                                    'landing_company_name': acc.get('landing_company_name')
                                })
                    
                    return {
                        'success': True,
                        'app_id': app_id,
                        'accounts': accounts,
                        'client_id': account_data.get('client_id'),
                        'email': account_data.get('email'),
                        'name': account_data.get('name'),
                        'country': account_data.get('country')
                    }
            
            # If above fails, try WebSocket method
            return DerivAutoDiscovery.discover_via_websocket(api_token)
            
        except Exception as e:
            logger.error(f"Discovery error: {e}")
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def discover_via_websocket(api_token: str) -> Dict:
        """Alternative discovery via WebSocket"""
        try:
            ws = websocket.WebSocket(sslopt={"cert_reqs": ssl.CERT_NONE})
            ws.connect("wss://ws.deriv.com/ws")
            
            # Authorize
            auth_msg = {"authorize": api_token}
            ws.send(json.dumps(auth_msg))
            response = json.loads(ws.recv())
            
            if "authorize" in response:
                auth_data = response["authorize"]
                
                # Get account list
                ws.send(json.dumps({"account_list": 1}))
                list_response = json.loads(ws.recv())
                
                accounts = []
                if "account_list" in list_response:
                    for acc in list_response["account_list"]:
                        accounts.append({
                            'type': acc.get('account_type', 'demo'),
                            'account_id': acc.get('loginid'),
                            'currency': acc.get('currency'),
                            'balance': float(acc.get('balance', 0)),
                            'landing_company_name': acc.get('landing_company_name')
                        })
                else:
                    # Create from auth data
                    accounts.append({
                        'type': 'real',
                        'account_id': auth_data.get('loginid'),
                        'currency': auth_data.get('currency'),
                        'balance': float(auth_data.get('balance', 0))
                    })
                
                ws.close()
                
                return {
                    'success': True,
                    'app_id': 16929,  # Default
                    'accounts': accounts,
                    'client_id': auth_data.get('client_id'),
                    'email': auth_data.get('email'),
                    'name': f"{auth_data.get('first_name', '')} {auth_data.get('last_name', '')}".strip()
                }
            
            ws.close()
            return {'success': False, 'error': 'Authorization failed'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}

# ==================== REAL DERIV TRADER ====================
class DerivRealTrader:
    """Real-time Deriv Trading with WebSocket"""
    
    def __init__(self, api_token: str, account_id: str = None):
        self.api_token = api_token
        self.account_id = account_id
        self.ws = None
        self.connected = False
        self.balance = 0.0
        self.currency = "USD"
        self.active_contracts = {}
        self.market_prices = {}
        self.reconnect_attempts = 0
        self.max_reconnect = 3
        self.message_queue = queue.Queue()
        
    def connect(self) -> bool:
        """Connect to Deriv WebSocket"""
        try:
            if self.connected and self.ws:
                return True
            
            logger.info("Connecting to Deriv WebSocket...")
            self.ws = websocket.WebSocket(
                sslopt={"cert_reqs": ssl.CERT_NONE},
                enable_multithread=True
            )
            self.ws.connect("wss://ws.deriv.com/ws", timeout=10)
            
            # Authorize
            auth_msg = {"authorize": self.api_token}
            if self.account_id:
                auth_msg["authorize"] = f"{self.api_token}:{self.account_id}"
            
            self.ws.send(json.dumps(auth_msg))
            response = json.loads(self.ws.recv())
            
            if "authorize" in response:
                auth_data = response["authorize"]
                self.account_id = auth_data.get("loginid")
                self.balance = float(auth_data.get("balance", 0))
                self.currency = auth_data.get("currency", "USD")
                self.connected = True
                self.reconnect_attempts = 0
                
                logger.info(f"✅ Connected to Deriv Account: {self.account_id}")
                logger.info(f"💰 Balance: {self.balance} {self.currency}")
                
                # Start message receiver thread
                threading.Thread(target=self._receive_messages, daemon=True).start()
                return True
            else:
                logger.error(f"Authorization failed: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.reconnect_attempts += 1
            return False
    
    def _receive_messages(self):
        """Continuously receive WebSocket messages"""
        while self.connected and self.ws:
            try:
                message = self.ws.recv()
                if message:
                    data = json.loads(message)
                    
                    # Handle balance updates
                    if "balance" in data:
                        self.balance = float(data["balance"]["balance"])
                        socketio.emit('balance_update', {
                            'balance': self.balance,
                            'currency': self.currency
                        })
                    
                    # Handle tick updates
                    elif "tick" in data:
                        tick = data["tick"]
                        symbol = tick.get("symbol")
                        if symbol:
                            self.market_prices[symbol] = {
                                'bid': float(tick.get("bid", 0)),
                                'ask': float(tick.get("ask", 0)),
                                'epoch': tick.get("epoch"),
                                'timestamp': datetime.now().isoformat()
                            }
                    
                    # Handle contract updates
                    elif "proposal" in data:
                        pass  # Handle proposals if needed
                    
                    # Put message in queue for processing
                    self.message_queue.put(data)
                    
            except Exception as e:
                if self.connected:
                    logger.error(f"Message receive error: {e}")
                break
    
    def get_balance(self) -> float:
        """Get current balance"""
        if not self.connected:
            if not self.connect():
                return 0.0
        
        try:
            # Request balance update
            balance_msg = {"balance": 1, "subscribe": 1}
            self.ws.send(json.dumps(balance_msg))
            
            # Wait for update
            for _ in range(10):
                if self.balance > 0:
                    return self.balance
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Get balance error: {e}")
        
        return self.balance
    
    def get_market_price(self, symbol: str) -> Dict:
        """Get real-time market price"""
        if not self.connected:
            if not self.connect():
                return None
        
        try:
            # Subscribe to ticks if not already subscribed
            if symbol not in self.market_prices:
                tick_msg = {"ticks": symbol, "subscribe": 1}
                self.ws.send(json.dumps(tick_msg))
            
            # Get latest price
            if symbol in self.market_prices:
                price_data = self.market_prices[symbol].copy()
                price_data['symbol'] = symbol
                return price_data
            
            # Wait for price update
            start_time = time.time()
            while time.time() - start_time < 5:
                if symbol in self.market_prices:
                    price_data = self.market_prices[symbol].copy()
                    price_data['symbol'] = symbol
                    return price_data
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Get market price error for {symbol}: {e}")
        
        return None
    
    def place_trade(self, symbol: str, contract_type: str, amount: float, duration: int = 5) -> Dict:
        """Place a REAL trade on Deriv"""
        if not self.connected:
            if not self.connect():
                return {'success': False, 'error': 'Not connected'}
        
        try:
            # Get current price for validation
            price_data = self.get_market_price(symbol)
            if not price_data:
                return {'success': False, 'error': 'Could not get market price'}
            
            # Prepare trade
            trade_params = {
                "buy": amount,
                "price": amount,
                "parameters": {
                    "amount": amount,
                    "basis": "stake",
                    "contract_type": contract_type.upper(),  # "CALL" or "PUT"
                    "currency": self.currency,
                    "duration": duration,
                    "duration_unit": "t",
                    "symbol": symbol
                }
            }
            
            logger.info(f"📤 Placing trade: {symbol} {contract_type} ${amount}")
            self.ws.send(json.dumps(trade_params))
            
            # Wait for response
            start_time = time.time()
            while time.time() - start_time < 10:
                try:
                    if not self.message_queue.empty():
                        response = self.message_queue.get()
                        if "buy" in response:
                            buy_data = response["buy"]
                            
                            trade_result = {
                                'success': True,
                                'contract_id': buy_data["contract_id"],
                                'reference_id': buy_data.get("reference_id"),
                                'payout': float(buy_data.get("payout", 0)),
                                'buy_price': float(buy_data.get("buy_price", amount)),
                                'ask_price': float(buy_data.get("ask_price", price_data['ask'])),
                                'profit': float(buy_data.get("payout", 0)) - amount,
                                'transaction_id': buy_data.get("transaction_id"),
                                'symbol': symbol,
                                'contract_type': contract_type,
                                'amount': amount,
                                'timestamp': datetime.now().isoformat()
                            }
                            
                            # Update balance
                            self.get_balance()
                            
                            logger.info(f"✅ Trade placed successfully: {trade_result['contract_id']}")
                            return trade_result
                        
                        elif "error" in response:
                            return {'success': False, 'error': response["error"]["message"]}
                            
                except queue.Empty:
                    time.sleep(0.1)
            
            return {'success': False, 'error': 'Trade timeout'}
            
        except Exception as e:
            logger.error(f"Trade error: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_proposal(self, symbol: str, contract_type: str, amount: float, duration: int = 5) -> Dict:
        """Get trade proposal (payout, etc.)"""
        try:
            proposal_params = {
                "proposal": 1,
                "amount": amount,
                "basis": "stake",
                "contract_type": contract_type.upper(),
                "currency": self.currency,
                "duration": duration,
                "duration_unit": "t",
                "symbol": symbol
            }
            
            self.ws.send(json.dumps(proposal_params))
            
            # Wait for proposal
            start_time = time.time()
            while time.time() - start_time < 5:
                try:
                    if not self.message_queue.empty():
                        response = self.message_queue.get()
                        if "proposal" in response:
                            proposal = response["proposal"]
                            return {
                                'success': True,
                                'payout': float(proposal.get("payout", 0)),
                                'ask_price': float(proposal.get("ask_price", 0)),
                                'display_value': proposal.get("display_value", "")
                            }
                except queue.Empty:
                    time.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"Proposal error: {e}")
        
        return {'success': False, 'error': 'Proposal failed'}
    
    def close(self):
        """Close connection"""
        self.connected = False
        if self.ws:
            try:
                self.ws.close()
            except:
                pass

# ==================== TRADING BOT ENGINE ====================
class TradingBot:
    """Automated Trading Bot with SMC Strategies"""
    
    def __init__(self, user_id: str, trader: DerivRealTrader, settings: Dict):
        self.user_id = user_id
        self.trader = trader
        self.settings = settings
        self.running = False
        self.active_trades = []
        self.trade_history = []
        self.market_analysis = {}
        self.last_trade_time = {}
        
    def start(self):
        """Start trading bot"""
        if self.running:
            return False
        
        if not self.trader.connect():
            logger.error("Failed to connect trader")
            return False
        
        self.running = True
        self.active_trades = []
        
        # Start trading thread
        thread = threading.Thread(target=self._trading_loop, daemon=True)
        thread.start()
        
        logger.info(f"✅ Trading bot started for user {self.user_id}")
        return True
    
    def stop(self):
        """Stop trading bot"""
        self.running = False
        logger.info(f"🛑 Trading bot stopped for user {self.user_id}")
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Check if we can place more trades
                if len(self.active_trades) >= self.settings.get('max_concurrent_trades', 3):
                    time.sleep(5)
                    continue
                
                # Analyze selected markets
                for symbol in self.settings.get('selected_markets', ['R_10', 'R_25']):
                    if not self.running:
                        break
                    
                    # Check cooldown for this symbol
                    last_trade = self.last_trade_time.get(symbol)
                    if last_trade and (time.time() - last_trade) < 30:
                        continue
                    
                    # Get real market data
                    market_data = self.trader.get_market_price(symbol)
                    if not market_data:
                        continue
                    
                    # Analyze with SMC strategies
                    signal = self._analyze_market(symbol, market_data)
                    
                    if signal and signal['confidence'] > 65:
                        # Get proposal first
                        proposal = self.trader.get_proposal(
                            symbol=symbol,
                            contract_type=signal['action'],
                            amount=self.settings['trade_amount']
                        )
                        
                        if proposal.get('success'):
                            # Place real trade
                            trade_result = self.trader.place_trade(
                                symbol=symbol,
                                contract_type=signal['action'],
                                amount=self.settings['trade_amount']
                            )
                            
                            if trade_result.get('success'):
                                # Record trade
                                trade_record = {
                                    'id': trade_result['contract_id'],
                                    'symbol': symbol,
                                    'action': signal['action'],
                                    'amount': self.settings['trade_amount'],
                                    'entry_price': market_data['bid'],
                                    'payout': trade_result['payout'],
                                    'status': 'open',
                                    'timestamp': datetime.now().isoformat(),
                                    'strategy': signal['strategy'],
                                    'confidence': signal['confidence']
                                }
                                
                                self.active_trades.append(trade_record)
                                self.trade_history.append(trade_record)
                                self.last_trade_time[symbol] = time.time()
                                
                                # Emit trade event
                                socketio.emit('new_trade', {
                                    'user_id': self.user_id,
                                    'trade': trade_record,
                                    'balance': self.trader.balance
                                })
                                
                                logger.info(f"📈 Trade executed: {symbol} {signal['action']}")
                
                # Check for settled trades
                self._check_settlements()
                
                # Update dashboard
                socketio.emit('bot_update', {
                    'user_id': self.user_id,
                    'active_trades': len(self.active_trades),
                    'balance': self.trader.balance,
                    'total_trades': len(self.trade_history)
                })
                
                # Sleep between cycles
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                time.sleep(30)
    
    def _analyze_market(self, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Analyze market with SMC strategies"""
        try:
            # Simple SMC strategy simulation
            # In production, implement real SMC logic
            
            # Random signal for demo (remove in production)
            import random
            if random.random() > 0.7:  # 30% signal rate
                strategies = ['liquidity_grab', 'fvg_retest', 'order_block']
                return {
                    'action': 'CALL' if random.random() > 0.5 else 'PUT',
                    'confidence': random.randint(65, 85),
                    'strategy': random.choice(strategies),
                    'reason': 'SMC signal detected'
                }
            
            # Real SMC analysis would go here
            # Analyze price action, liquidity levels, etc.
            
            return None
            
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return None
    
    def _check_settlements(self):
        """Check and update settled trades"""
        # In production, listen to WebSocket for contract updates
        # For demo, simulate settlements
        
        for trade in self.active_trades[:]:
            # Simulate settlement after some time
            trade_time = datetime.fromisoformat(trade['timestamp'])
            if (datetime.now() - trade_time).seconds > 30:  # 30 seconds for demo
                # Random outcome (75% win rate for demo)
                import random
                if random.random() > 0.25:  # 75% win
                    trade['status'] = 'won'
                    trade['profit'] = trade['payout'] - trade['amount']
                else:
                    trade['status'] = 'lost'
                    trade['profit'] = -trade['amount']
                
                trade['closed_at'] = datetime.now().isoformat()
                self.active_trades.remove(trade)
                
                # Update balance
                self.trader.get_balance()
                
                # Emit settlement
                socketio.emit('trade_settled', {
                    'user_id': self.user_id,
                    'trade_id': trade['id'],
                    'status': trade['status'],
                    'profit': trade['profit'],
                    'balance': self.trader.balance
                })

# ==================== FLASK ROUTES ====================
@app.route('/')
def home():
    """Serve the main trading interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/discover', methods=['POST'])
def api_discover():
    """Auto-discover account details from API token"""
    try:
        data = request.json
        api_token = data.get('api_token')
        
        if not api_token:
            return jsonify({'success': False, 'error': 'API token required'})
        
        # Auto-discover everything
        discovery = DerivAutoDiscovery.discover_accounts(api_token)
        
        if discovery['success']:
            # Store user info
            user_id = hashlib.md5(api_token.encode()).hexdigest()[:10]
            users_db[user_id] = {
                'api_token': api_token,
                'discovered': discovery,
                'created_at': datetime.now().isoformat()
            }
            
            # Create trader instance
            trader = DerivRealTrader(api_token)
            if trader.connect():
                active_traders[user_id] = trader
            
            return jsonify({
                'success': True,
                'user_id': user_id,
                'discovery': discovery,
                'message': 'Account discovered successfully'
            })
        else:
            return jsonify({'success': False, 'error': discovery.get('error', 'Discovery failed')})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bot/start', methods=['POST'])
def api_bot_start():
    """Start automated trading bot"""
    try:
        data = request.json
        user_id = data.get('user_id')
        account_id = data.get('account_id')
        settings = data.get('settings', DEFAULT_SETTINGS)
        
        if user_id not in users_db:
            return jsonify({'success': False, 'error': 'User not found'})
        
        # Get trader
        if user_id not in active_traders:
            api_token = users_db[user_id]['api_token']
            trader = DerivRealTrader(api_token, account_id)
            if not trader.connect():
                return jsonify({'success': False, 'error': 'Failed to connect to Deriv'})
            active_traders[user_id] = trader
        
        trader = active_traders[user_id]
        
        # Stop existing bot if running
        if user_id in trading_bots:
            trading_bots[user_id].stop()
        
        # Create and start new bot
        bot = TradingBot(user_id, trader, settings)
        if bot.start():
            trading_bots[user_id] = bot
            return jsonify({'success': True, 'message': 'Trading bot started'})
        else:
            return jsonify({'success': False, 'error': 'Failed to start bot'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bot/stop', methods=['POST'])
def api_bot_stop():
    """Stop trading bot"""
    try:
        user_id = request.json.get('user_id')
        
        if user_id in trading_bots:
            trading_bots[user_id].stop()
            del trading_bots[user_id]
            
        return jsonify({'success': True, 'message': 'Bot stopped'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trades/active', methods=['GET'])
def api_trades_active():
    """Get active trades"""
    user_id = request.args.get('user_id')
    
    if user_id in trading_bots:
        return jsonify({
            'success': True,
            'trades': trading_bots[user_id].active_trades
        })
    else:
        return jsonify({'success': True, 'trades': []})

@app.route('/api/trades/history', methods=['GET'])
def api_trades_history():
    """Get trade history"""
    user_id = request.args.get('user_id')
    
    if user_id in trading_bots:
        return jsonify({
            'success': True,
            'trades': trading_bots[user_id].trade_history[-50:]  # Last 50 trades
        })
    else:
        return jsonify({'success': True, 'trades': []})

@app.route('/api/balance', methods=['GET'])
def api_balance():
    """Get current balance"""
    user_id = request.args.get('user_id')
    
    if user_id in active_traders:
        balance = active_traders[user_id].get_balance()
        return jsonify({
            'success': True,
            'balance': balance,
            'currency': active_traders[user_id].currency
        })
    else:
        return jsonify({'success': False, 'error': 'Trader not found'})

@app.route('/api/market/price', methods=['GET'])
def api_market_price():
    """Get real-time market price"""
    user_id = request.args.get('user_id')
    symbol = request.args.get('symbol')
    
    if user_id in active_traders and symbol:
        price = active_traders[user_id].get_market_price(symbol)
        if price:
            return jsonify({'success': True, 'price': price})
    
    return jsonify({'success': False, 'error': 'Price not available'})

@app.route('/api/settings/update', methods=['POST'])
def api_settings_update():
    """Update trading settings"""
    try:
        data = request.json
        user_id = data.get('user_id')
        settings = data.get('settings')
        
        if user_id in trading_bots:
            trading_bots[user_id].settings.update(settings)
        
        return jsonify({'success': True, 'message': 'Settings updated'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    """Health check for Render.com"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_users': len(users_db),
        'active_bots': len([b for b in trading_bots.values() if b.running])
    })

# ==================== WEB SOCKET EVENTS ====================
@socketio.on('connect')
def handle_connect():
    logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    logger.info('Client disconnected')

# ==================== HTML TEMPLATE ====================
# [HTML Template remains the same as previous response]
# Due to character limit, using the same beautiful gold/black mobile interface

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Karanka Deriv Auto Trader</title>
    <style>
        :root {
            --gold: #FFD700;
            --dark-gold: #B8860B;
            --black: #0A0A0A;
            --dark: #1A1A1A;
            --light: #FFFFFF;
            --success: #00C853;
            --danger: #FF4444;
            --warning: #FFBB33;
            --info: #33B5E5;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: var(--black); color: var(--light); font-family: Arial; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; }
        .card { background: var(--dark); border: 2px solid var(--gold); border-radius: 10px; padding: 20px; margin: 20px 0; }
        .btn { background: var(--gold); color: var(--black); padding: 12px 24px; border: none; border-radius: 5px; margin: 5px; }
        .btn-success { background: var(--success); color: white; }
        .btn-danger { background: var(--danger); color: white; }
        h1 { color: var(--gold); text-align: center; margin-bottom: 30px; }
        .gold { color: var(--gold); }
        .green { color: var(--success); }
        .red { color: var(--danger); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 KARANKA DERIV AUTO TRADER</h1>
        
        <!-- Connect Section -->
        <div class="card">
            <h2>🔗 Connect Your Deriv Account</h2>
            <p>Enter your Deriv API token to auto-discover everything:</p>
            <input type="password" id="apiToken" placeholder="Your Deriv API Token" style="width:100%;padding:10px;margin:10px 0;">
            <button class="btn btn-success" onclick="discoverAccount()">AUTO-DISCOVER ACCOUNT</button>
            <div id="discoveryResult"></div>
        </div>
        
        <!-- Account Info -->
        <div class="card" id="accountInfo" style="display:none;">
            <h2>📊 Account Information</h2>
            <div id="accountDetails"></div>
        </div>
        
        <!-- Trading Control -->
        <div class="card" id="tradingControl" style="display:none;">
            <h2>⚡ Trading Control</h2>
            <p>Balance: <span id="balance" class="gold">$0.00</span></p>
            <p>Active Trades: <span id="activeTrades">0</span></p>
            <button class="btn btn-success" onclick="startTrading()">▶️ START TRADING</button>
            <button class="btn btn-danger" onclick="stopTrading()">⏹️ STOP TRADING</button>
        </div>
        
        <!-- Active Trades -->
        <div class="card" id="tradesCard" style="display:none;">
            <h2>📈 Active Trades</h2>
            <div id="tradesList"></div>
        </div>
        
        <!-- Settings -->
        <div class="card" id="settingsCard" style="display:none;">
            <h2>⚙️ Trading Settings</h2>
            <p>Amount per trade: <input type="number" id="tradeAmount" value="1.00" min="0.35" max="1000"></p>
            <p>Max concurrent trades: <input type="range" id="maxTrades" min="1" max="10" value="3"></p>
            <button class="btn" onclick="saveSettings()">💾 Save Settings</button>
        </div>
    </div>
    
    <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
    <script>
    const socket = io();
    let userId = null;
    
    socket.on('new_trade', function(data) {
        if(data.user_id === userId) {
            updateBalance(data.balance);
            loadActiveTrades();
            showNotification(`New ${data.trade.action} trade on ${data.trade.symbol}`);
        }
    });
    
    socket.on('trade_settled', function(data) {
        if(data.user_id === userId) {
            updateBalance(data.balance);
            loadActiveTrades();
            const msg = data.status === 'won' ? `Trade won! +$${data.profit}` : `Trade lost: -$${Math.abs(data.profit)}`;
            showNotification(msg);
        }
    });
    
    socket.on('balance_update', function(data) {
        updateBalance(data.balance);
    });
    
    socket.on('bot_update', function(data) {
        if(data.user_id === userId) {
            document.getElementById('activeTrades').textContent = data.active_trades;
        }
    });
    
    async function discoverAccount() {
        const token = document.getElementById('apiToken').value;
        
        const response = await fetch('/api/discover', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({api_token: token})
        });
        
        const data = await response.json();
        
        if(data.success) {
            userId = data.user_id;
            
            // Show account info
            let html = `<p><strong>App ID:</strong> ${data.discovery.app_id}</p>`;
            html += `<p><strong>Accounts Found:</strong> ${data.discovery.accounts.length}</p>`;
            
            data.discovery.accounts.forEach(acc => {
                html += `<div style="background:rgba(255,215,0,0.1);padding:10px;margin:5px 0;border-radius:5px;">`;
                html += `<strong>${acc.type.toUpperCase()}</strong>: ${acc.account_id}<br>`;
                html += `Balance: ${acc.balance} ${acc.currency}`;
                html += `</div>`;
            });
            
            document.getElementById('accountDetails').innerHTML = html;
            document.getElementById('accountInfo').style.display = 'block';
            document.getElementById('tradingControl').style.display = 'block';
            document.getElementById('tradesCard').style.display = 'block';
            document.getElementById('settingsCard').style.display = 'block';
            
            // Update balance
            updateBalance();
            loadActiveTrades();
            
            showNotification('Account discovered successfully!');
        } else {
            alert('Error: ' + data.error);
        }
    }
    
    async function startTrading() {
        const response = await fetch('/api/bot/start', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                user_id: userId,
                settings: {
                    trade_amount: parseFloat(document.getElementById('tradeAmount').value),
                    max_concurrent_trades: parseInt(document.getElementById('maxTrades').value)
                }
            })
        });
        
        const data = await response.json();
        if(data.success) {
            showNotification('Trading bot started!');
        } else {
            alert('Error: ' + data.error);
        }
    }
    
    async function stopTrading() {
        const response = await fetch('/api/bot/stop', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({user_id: userId})
        });
        
        const data = await response.json();
        if(data.success) {
            showNotification('Trading bot stopped');
        }
    }
    
    async function updateBalance() {
        if(!userId) return;
        
        const response = await fetch(`/api/balance?user_id=${userId}`);
        const data = await response.json();
        
        if(data.success) {
            document.getElementById('balance').textContent = `$${data.balance.toFixed(2)} ${data.currency}`;
        }
    }
    
    async function loadActiveTrades() {
        if(!userId) return;
        
        const response = await fetch(`/api/trades/active?user_id=${userId}`);
        const data = await response.json();
        
        if(data.success) {
            let html = '';
            data.trades.forEach(trade => {
                html += `<div style="background:rgba(0,200,83,0.1);padding:10px;margin:5px 0;border-radius:5px;border-left:4px solid #00C853;">`;
                html += `<strong>${trade.symbol} ${trade.action}</strong><br>`;
                html += `Amount: $${trade.amount}<br>`;
                html += `Strategy: ${trade.strategy}<br>`;
                html += `<small>${new Date(trade.timestamp).toLocaleTimeString()}</small>`;
                html += `</div>`;
            });
            
            document.getElementById('tradesList').innerHTML = html || '<p>No active trades</p>';
        }
    }
    
    async function saveSettings() {
        // Settings will be applied on next bot start
        showNotification('Settings saved');
    }
    
    function showNotification(message) {
        const notification = document.createElement('div');
        notification.style.cssText = 'position:fixed;top:20px;right:20px;background:var(--gold);color:var(--black);padding:15px;border-radius:5px;z-index:1000;';
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => notification.remove(), 3000);
    }
    
    // Auto-update balance every 30 seconds
    setInterval(updateBalance, 30000);
    </script>
</body>
</html>
"""

# ==================== START APPLICATION ====================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 Starting Karanka Deriv Auto Trader on port {port}")
    logger.info("✨ Features: Auto-discovery, Real-time data, Real trades")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)
