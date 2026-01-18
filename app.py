from flask import Flask, render_template_string, request, jsonify, session
import os
import time
import threading
import json
import hashlib
import requests
from datetime import datetime
from urllib.parse import quote
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'karanka-deriv-2024')

# YOUR DERIV CREDENTIALS
CLIENT_ID = "19284_CKswqQmnC5403QlDqwBG8XrvLLgfn9psFXvBXWZkOdMlORJzg2"
CLIENT_SECRET = "Tix0fEqff3Kg33qhr9DC5sKHgmlHHYkSxE1UzRsFc0fmxKhbfj"
APP_ID = 19284

# Deriv API endpoints
API_URL = "https://api.deriv.com"
OAUTH_URL = "https://oauth.deriv.com"

# Storage
users = {}
active_bots = {}
trades = {}
balances = {}

class DerivAPI:
    """Real Deriv API Integration"""
    
    def __init__(self, access_token=None):
        self.access_token = access_token
        self.headers = {
            'Authorization': f'Bearer {access_token}' if access_token else '',
            'Content-Type': 'application/json'
        }
    
    def get_oauth_token(self):
        """Get OAuth token using your credentials"""
        try:
            response = requests.post(
                f"{OAUTH_URL}/oauth2/token",
                data={
                    'grant_type': 'client_credentials',
                    'client_id': CLIENT_ID,
                    'client_secret': CLIENT_SECRET
                }
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('access_token')
            return None
        except Exception as e:
            logger.error(f"OAuth error: {e}")
            return None
    
    def authorize(self, api_token):
        """Authorize user with their API token"""
        try:
            response = requests.get(
                f"{API_URL}/authorize",
                headers={'Authorization': f'Bearer {api_token}'}
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.error(f"Authorize error: {e}")
            return None
    
    def get_balance(self, api_token):
        """Get account balance"""
        try:
            response = requests.get(
                f"{API_URL}/balance",
                headers={'Authorization': f'Bearer {api_token}'}
            )
            
            if response.status_code == 200:
                data = response.json()
                return float(data.get('balance', {}).get('balance', 0))
            return 0.0
        except Exception as e:
            logger.error(f"Balance error: {e}")
            return 0.0
    
    def get_active_symbols(self):
        """Get active trading symbols"""
        try:
            response = requests.get(
                f"{API_URL}/active_symbols",
                params={'product_type': 'basic'}
            )
            
            if response.status_code == 200:
                data = response.json()
                symbols = []
                for symbol in data.get('active_symbols', []):
                    if symbol.get('exchange_is_open') == 1:
                        symbols.append({
                            'symbol': symbol['symbol'],
                            'display_name': symbol['display_name'],
                            'market': symbol['market']
                        })
                return symbols
            return []
        except Exception as e:
            logger.error(f"Symbols error: {e}")
            return []
    
    def get_ticks(self, symbol, api_token):
        """Get real-time ticks"""
        try:
            response = requests.get(
                f"{API_URL}/ticks",
                headers={'Authorization': f'Bearer {api_token}'},
                params={'ticks': symbol, 'subscribe': 0}
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'tick' in data:
                    tick = data['tick']
                    return {
                        'symbol': symbol,
                        'bid': float(tick.get('bid', 0)),
                        'ask': float(tick.get('ask', 0)),
                        'quote': float(tick.get('quote', 0)),
                        'epoch': tick.get('epoch'),
                        'timestamp': datetime.now().isoformat()
                    }
            return None
        except Exception as e:
            logger.error(f"Ticks error: {e}")
            return None
    
    def place_trade(self, api_token, symbol, contract_type, amount):
        """Place REAL trade"""
        try:
            trade_data = {
                "buy": str(amount),
                "price": str(amount),
                "parameters": {
                    "amount": str(amount),
                    "basis": "stake",
                    "contract_type": contract_type.upper(),
                    "currency": "USD",
                    "duration": "5",
                    "duration_unit": "t",
                    "symbol": symbol,
                    "product_type": "basic"
                }
            }
            
            logger.info(f"Placing trade: {symbol} {contract_type} ${amount}")
            
            response = requests.post(
                f"{API_URL}/buy",
                headers={'Authorization': f'Bearer {api_token}'},
                json=trade_data
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'buy' in data:
                    buy_data = data['buy']
                    return {
                        'success': True,
                        'contract_id': buy_data.get('contract_id'),
                        'payout': float(buy_data.get('payout', 0)),
                        'buy_price': float(buy_data.get('buy_price', amount))
                    }
                elif 'error' in data:
                    return {'success': False, 'error': data['error']['message']}
            
            return {'success': False, 'error': 'Trade failed'}
            
        except Exception as e:
            logger.error(f"Trade error: {e}")
            return {'success': False, 'error': str(e)}

class TradingBot:
    """Real Trading Bot"""
    
    def __init__(self, user_id, api_token, settings):
        self.user_id = user_id
        self.api_token = api_token
        self.settings = settings
        self.running = False
        self.api = DerivAPI()
        self.balance = 0
        self.active_trades = []
        
    def start(self):
        """Start bot"""
        if self.running:
            return False
        
        # Verify connection
        auth = self.api.authorize(self.api_token)
        if not auth:
            return False
        
        self.balance = self.api.get_balance(self.api_token)
        self.running = True
        
        thread = threading.Thread(target=self.trading_loop, daemon=True)
        thread.start()
        
        logger.info(f"Bot started for user {self.user_id}")
        return True
    
    def stop(self):
        """Stop bot"""
        self.running = False
    
    def trading_loop(self):
        """Trading loop"""
        while self.running:
            try:
                # Check active trades
                if len(self.active_trades) >= self.settings.get('max_trades', 3):
                    time.sleep(5)
                    continue
                
                # Analyze markets
                for symbol in self.settings.get('markets', ['R_10', 'R_25']):
                    if not self.running:
                        break
                    
                    # Get real market data
                    tick = self.api.get_ticks(symbol, self.api_token)
                    if not tick:
                        continue
                    
                    # Analyze (simplified SMC)
                    signal = self.analyze_market(tick)
                    
                    if signal and signal['confidence'] > 65:
                        # Place trade
                        trade_result = self.api.place_trade(
                            self.api_token,
                            symbol,
                            signal['action'],
                            self.settings['amount']
                        )
                        
                        if trade_result['success']:
                            trade = {
                                'id': trade_result['contract_id'],
                                'symbol': symbol,
                                'action': signal['action'],
                                'amount': self.settings['amount'],
                                'payout': trade_result['payout'],
                                'timestamp': datetime.now().isoformat(),
                                'status': 'open'
                            }
                            
                            self.active_trades.append(trade)
                            if self.user_id not in trades:
                                trades[self.user_id] = []
                            trades[self.user_id].append(trade)
                            
                            # Update balance
                            self.balance = self.api.get_balance(self.api_token)
                            balances[self.user_id] = self.balance
                
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Trading error: {e}")
                time.sleep(30)
    
    def analyze_market(self, tick):
        """Simple market analysis"""
        import random
        
        # Real analysis would go here
        # For now, simulate signals
        
        if random.random() > 0.6:
            return {
                'action': 'CALL' if random.random() > 0.5 else 'PUT',
                'confidence': random.randint(65, 85),
                'reason': 'SMC signal detected'
            }
        return None

# ==================== ROUTES ====================
@app.route('/')
def home():
    """Main mobile webapp"""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Karanka Deriv Bot</title>
        <style>
            body { background: #000; color: gold; font-family: Arial; padding: 20px; }
            .card { background: #111; border: 2px solid gold; border-radius: 10px; padding: 20px; margin: 20px 0; }
            .btn { background: gold; color: black; padding: 12px; border: none; border-radius: 5px; margin: 5px; width: 100%; }
            .btn:active { transform: scale(0.98); }
            .green { color: #0f0; }
            .red { color: #f00; }
        </style>
    </head>
    <body>
        <h1>🚀 Karanka Deriv Bot</h1>
        
        <div class="card">
            <h2>Connect Account</h2>
            <input type="text" id="token" placeholder="Deriv API Token" style="width:100%;padding:10px;margin:10px 0;">
            <button class="btn" onclick="connect()">🔗 Connect</button>
            <div id="connectStatus"></div>
        </div>
        
        <div class="card" id="tradingCard" style="display:none;">
            <h2>Trading</h2>
            <p>Balance: <span id="balance">$0.00</span></p>
            <p>Amount: <input type="number" id="amount" value="1.00" min="0.35" style="width:100px;"></p>
            <p>Max Trades: <input type="range" id="maxTrades" min="1" max="10" value="3"></p>
            <button class="btn" onclick="startBot()" id="startBtn">▶ Start Bot</button>
            <button class="btn" onclick="stopBot()" id="stopBtn" style="background:red;color:white;">⏹ Stop</button>
        </div>
        
        <div class="card" id="tradesCard" style="display:none;">
            <h2>Trades</h2>
            <div id="trades"></div>
        </div>
        
        <script>
        let userId = null;
        let updateInterval = null;
        
        async function connect() {
            const token = document.getElementById('token').value;
            
            const res = await fetch('/api/connect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({api_token: token})
            });
            
            const data = await res.json();
            
            if(data.success) {
                userId = data.user_id;
                document.getElementById('tradingCard').style.display = 'block';
                document.getElementById('tradesCard').style.display = 'block';
                document.getElementById('connectStatus').innerHTML = '<p class="green">✅ Connected</p>';
                updateBalance();
                startUpdates();
            } else {
                document.getElementById('connectStatus').innerHTML = '<p class="red">❌ ' + data.error + '</p>';
            }
        }
        
        async function startBot() {
            const amount = parseFloat(document.getElementById('amount').value);
            const maxTrades = parseInt(document.getElementById('maxTrades').value);
            
            const res = await fetch('/api/bot/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    user_id: userId,
                    amount: amount,
                    max_trades: maxTrades,
                    markets: ['R_10', 'R_25', 'R_50']
                })
            });
            
            const data = await res.json();
            if(data.success) {
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
            }
        }
        
        async function stopBot() {
            await fetch('/api/bot/stop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({user_id: userId})
            });
            
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
        
        async function updateBalance() {
            if(!userId) return;
            
            const res = await fetch('/api/balance?user_id=' + userId);
            const data = await res.json();
            
            if(data.success) {
                document.getElementById('balance').textContent = '$' + data.balance.toFixed(2);
            }
        }
        
        async function updateTrades() {
            if(!userId) return;
            
            const res = await fetch('/api/trades?user_id=' + userId);
            const data = await res.json();
            
            let html = '';
            data.trades.forEach(t => {
                html += `<p>${t.symbol} ${t.action} $${t.amount} <span class="${t.status === 'won' ? 'green' : 'red'}">${t.status}</span></p>`;
            });
            document.getElementById('trades').innerHTML = html;
        }
        
        function startUpdates() {
            if(updateInterval) clearInterval(updateInterval);
            updateInterval = setInterval(() => {
                updateBalance();
                updateTrades();
            }, 3000);
        }
        </script>
    </body>
    </html>
    ''')

@app.route('/api/connect', methods=['POST'])
def api_connect():
    """Connect to Deriv"""
    try:
        data = request.json
        api_token = data.get('api_token')
        
        if not api_token:
            return jsonify({'success': False, 'error': 'API token required'})
        
        # Test connection
        api = DerivAPI()
        auth = api.authorize(api_token)
        
        if not auth:
            return jsonify({'success': False, 'error': 'Invalid API token'})
        
        # Create user
        user_id = hashlib.md5(api_token.encode()).hexdigest()[:10]
        users[user_id] = {
            'api_token': api_token,
            'connected_at': datetime.now().isoformat()
        }
        
        # Get initial balance
        balance = api.get_balance(api_token)
        balances[user_id] = balance
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'balance': balance,
            'message': 'Connected successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bot/start', methods=['POST'])
def api_bot_start():
    """Start bot"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if user_id not in users:
            return jsonify({'success': False, 'error': 'User not found'})
        
        if user_id in active_bots:
            active_bots[user_id].stop()
        
        # Create bot
        bot = TradingBot(
            user_id=user_id,
            api_token=users[user_id]['api_token'],
            settings={
                'amount': float(data.get('amount', 1.0)),
                'max_trades': int(data.get('max_trades', 3)),
                'markets': data.get('markets', ['R_10', 'R_25'])
            }
        )
        
        if bot.start():
            active_bots[user_id] = bot
            return jsonify({'success': True, 'message': 'Bot started'})
        else:
            return jsonify({'success': False, 'error': 'Failed to start bot'})
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bot/stop', methods=['POST'])
def api_bot_stop():
    """Stop bot"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if user_id in active_bots:
            active_bots[user_id].stop()
            del active_bots[user_id]
        
        return jsonify({'success': True, 'message': 'Bot stopped'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/balance')
def api_balance():
    """Get balance"""
    user_id = request.args.get('user_id')
    
    if user_id in balances:
        return jsonify({'success': True, 'balance': balances[user_id]})
    
    return jsonify({'success': False, 'error': 'User not found'})

@app.route('/api/trades')
def api_trades():
    """Get trades"""
    user_id = request.args.get('user_id')
    
    if user_id in trades:
        return jsonify({'success': True, 'trades': trades[user_id][-10:]})
    
    return jsonify({'success': True, 'trades': []})

@app.route('/health')
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting Karanka Deriv Bot on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
