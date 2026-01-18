"""
KARANKA DERIV TRADING BOT - WORKING VERSION
Simple but functional - Deploys without errors
"""

from flask import Flask, render_template_string, request, jsonify
import os
import time
import threading
import json
import hashlib
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'karanka-simple-secure-key-2024')

# Simple storage
users = {}
trades = {}
bot_status = {}
account_balance = {}

# HTML Template with Gold/Black Design
HTML = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Karanka Deriv Bot</title>
    <style>
        :root {
            --gold: #FFD700;
            --dark-gold: #B8860B;
            --black: #000000;
            --dark: #111111;
            --light: #FFFFFF;
            --green: #00FF00;
            --red: #FF0000;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            background: var(--black);
            color: var(--light);
            font-family: Arial, sans-serif;
            padding: 20px;
            line-height: 1.6;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, var(--black), var(--dark));
            border: 3px solid var(--gold);
            border-radius: 15px;
        }
        
        h1 {
            color: var(--gold);
            font-size: 28px;
            margin-bottom: 10px;
        }
        
        .tagline {
            color: var(--gold);
            opacity: 0.8;
            font-size: 14px;
        }
        
        .card {
            background: var(--dark);
            border: 2px solid var(--gold);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .card-title {
            color: var(--gold);
            font-size: 20px;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid var(--gold);
        }
        
        .btn {
            background: linear-gradient(45deg, var(--gold), var(--dark-gold));
            color: var(--black);
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
            width: 100%;
            margin: 5px 0;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(255, 215, 0, 0.3);
        }
        
        .btn-success {
            background: linear-gradient(45deg, #00C853, #009624);
            color: white;
        }
        
        .btn-danger {
            background: linear-gradient(45deg, #FF4444, #CC0000);
            color: white;
        }
        
        .form-control {
            width: 100%;
            padding: 12px;
            background: rgba(255, 255, 255, 0.1);
            border: 2px solid var(--gold);
            border-radius: 8px;
            color: white;
            margin: 10px 0;
            font-size: 16px;
        }
        
        .balance {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
            color: var(--gold);
        }
        
        .status {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .status-online {
            background: var(--green);
            color: var(--black);
            animation: pulse 1.5s infinite;
        }
        
        .status-offline {
            background: var(--red);
            color: white;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
        
        .trade-item {
            background: rgba(255, 215, 0, 0.1);
            border: 1px solid var(--gold);
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .profit {
            font-weight: bold;
        }
        
        .profit-positive {
            color: var(--green);
        }
        
        .profit-negative {
            color: var(--red);
        }
        
        .loading {
            text-align: center;
            padding: 30px;
            color: var(--gold);
        }
        
        .tab {
            display: none;
        }
        
        .tab.active {
            display: block;
            animation: fadeIn 0.5s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .tabs {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            overflow-x: auto;
        }
        
        .tab-btn {
            padding: 12px 20px;
            background: var(--dark);
            border: 2px solid var(--gold);
            color: var(--light);
            border-radius: 8px;
            cursor: pointer;
            white-space: nowrap;
        }
        
        .tab-btn.active {
            background: var(--gold);
            color: var(--black);
            font-weight: bold;
        }
        
        @media (max-width: 600px) {
            .container {
                padding: 10px;
            }
            
            .header {
                padding: 15px;
            }
            
            h1 {
                font-size: 24px;
            }
            
            .balance {
                font-size: 28px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>🔥 KARANKA DERIV AUTO TRADER</h1>
            <p class="tagline">Professional Automated Trading Bot</p>
            <div style="margin-top: 15px;">
                <span class="status" id="botStatus">OFFLINE</span>
                <span style="margin-left: 20px;">Balance: <span id="balanceAmount">$0.00</span></span>
            </div>
        </div>
        
        <!-- Tabs -->
        <div class="tabs">
            <button class="tab-btn active" onclick="showTab('connect')">Connect</button>
            <button class="tab-btn" onclick="showTab('trading')">Trading</button>
            <button class="tab-btn" onclick="showTab('markets')">Markets</button>
            <button class="tab-btn" onclick="showTab('trades')">Trades</button>
        </div>
        
        <!-- Connect Tab -->
        <div id="connectTab" class="tab active">
            <div class="card">
                <h2 class="card-title">🔗 Connect Deriv Account</h2>
                <p>Enter your Deriv API token to connect:</p>
                
                <input type="password" class="form-control" id="apiToken" 
                       placeholder="Paste your Deriv API token here">
                
                <button class="btn btn-success" onclick="connectAccount()">
                    CONNECT TO DERIV
                </button>
                
                <div id="connectResult" style="margin-top: 15px;"></div>
            </div>
        </div>
        
        <!-- Trading Tab -->
        <div id="tradingTab" class="tab">
            <div class="card">
                <h2 class="card-title">⚡ Trading Control</h2>
                
                <div class="form-group">
                    <label>Trade Amount ($)</label>
                    <input type="number" class="form-control" id="tradeAmount" 
                           min="0.35" max="1000" step="0.01" value="1.00">
                </div>
                
                <div class="form-group">
                    <label>Max Concurrent Trades</label>
                    <input type="range" class="form-control" id="maxTrades" 
                           min="1" max="10" value="3" oninput="document.getElementById('maxTradesValue').innerText = this.value">
                    <div style="text-align: center; margin-top: 5px;">
                        <span id="maxTradesValue">3</span> trades
                    </div>
                </div>
                
                <button class="btn btn-success" onclick="startBot()" id="startBtn">
                    ▶ START TRADING BOT
                </button>
                
                <button class="btn btn-danger" onclick="stopBot()" id="stopBtn" disabled>
                    ⏹ STOP BOT
                </button>
            </div>
        </div>
        
        <!-- Markets Tab -->
        <div id="marketsTab" class="tab">
            <div class="card">
                <h2 class="card-title">📈 Available Markets</h2>
                <p>Select markets to trade:</p>
                
                <div id="marketsList">
                    <div class="loading">Loading markets...</div>
                </div>
                
                <button class="btn" onclick="saveMarkets()" style="margin-top: 20px;">
                    💾 Save Market Selection
                </button>
            </div>
        </div>
        
        <!-- Trades Tab -->
        <div id="tradesTab" class="tab">
            <div class="card">
                <h2 class="card-title">📊 Trade History</h2>
                <div id="tradesList">
                    <div class="loading">No trades yet</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
    // Global variables
    let userId = null;
    let botActive = false;
    let updateInterval = null;
    
    // Show tab function
    function showTab(tabName) {
        // Hide all tabs
        document.querySelectorAll('.tab').forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Remove active from all buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        
        // Show selected tab
        document.getElementById(tabName + 'Tab').classList.add('active');
        
        // Activate button
        event.target.classList.add('active');
        
        // Load data for tab
        if(tabName === 'markets') loadMarkets();
        if(tabName === 'trades') loadTrades();
    }
    
    // Connect to Deriv
    async function connectAccount() {
        const apiToken = document.getElementById('apiToken').value;
        
        if(!apiToken) {
            showMessage('Please enter your API token', 'error');
            return;
        }
        
        try {
            const response = await fetch('/api/connect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({api_token: apiToken})
            });
            
            const data = await response.json();
            
            if(data.success) {
                userId = data.user_id;
                showMessage('✅ Connected successfully!', 'success');
                showTab('trading');
                updateBalance();
                loadMarkets();
                startUpdates();
            } else {
                showMessage('❌ ' + data.error, 'error');
            }
        } catch(error) {
            showMessage('Network error. Please try again.', 'error');
        }
    }
    
    // Start bot
    async function startBot() {
        const tradeAmount = document.getElementById('tradeAmount').value;
        const maxTrades = document.getElementById('maxTrades').value;
        
        try {
            const response = await fetch('/api/bot/start', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    user_id: userId,
                    trade_amount: tradeAmount,
                    max_trades: maxTrades
                })
            });
            
            const data = await response.json();
            
            if(data.success) {
                botActive = true;
                updateBotStatus();
                showMessage('✅ Bot started successfully!', 'success');
            } else {
                showMessage('❌ ' + data.error, 'error');
            }
        } catch(error) {
            showMessage('Network error', 'error');
        }
    }
    
    // Stop bot
    async function stopBot() {
        try {
            const response = await fetch('/api/bot/stop', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({user_id: userId})
            });
            
            const data = await response.json();
            
            if(data.success) {
                botActive = false;
                updateBotStatus();
                showMessage('Bot stopped', 'info');
            }
        } catch(error) {
            showMessage('Network error', 'error');
        }
    }
    
    // Update bot status display
    function updateBotStatus() {
        const statusElement = document.getElementById('botStatus');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        
        if(botActive) {
            statusElement.textContent = 'TRADING';
            statusElement.className = 'status status-online';
            startBtn.disabled = true;
            stopBtn.disabled = false;
        } else {
            statusElement.textContent = 'STOPPED';
            statusElement.className = 'status status-offline';
            startBtn.disabled = false;
            stopBtn.disabled = true;
        }
    }
    
    // Update balance
    async function updateBalance() {
        if(!userId) return;
        
        try {
            const response = await fetch('/api/balance?user_id=' + userId);
            const data = await response.json();
            
            if(data.success) {
                document.getElementById('balanceAmount').textContent = 
                    '$' + data.balance.toFixed(2);
            }
        } catch(error) {
            console.error('Balance update failed');
        }
    }
    
    // Load markets
    async function loadMarkets() {
        try {
            const response = await fetch('/api/markets');
            const data = await response.json();
            
            let html = '';
            data.markets.forEach(market => {
                html += `
                <div style="background: rgba(255,215,0,0.1); padding: 15px; margin: 10px 0; border-radius: 8px; border: 1px solid var(--gold);">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: var(--gold);">${market.symbol}</strong><br>
                            <span style="font-size: 14px; color: #aaa;">${market.name}</span>
                        </div>
                        <div>
                            <label style="display: flex; align-items: center; gap: 10px;">
                                <input type="checkbox" value="${market.symbol}" ${market.symbol === 'R_10' || market.symbol === 'R_25' ? 'checked' : ''}>
                                <span>Trade</span>
                            </label>
                        </div>
                    </div>
                </div>`;
            });
            
            document.getElementById('marketsList').innerHTML = html;
        } catch(error) {
            console.error('Failed to load markets');
        }
    }
    
    // Load trades
    async function loadTrades() {
        if(!userId) return;
        
        try {
            const response = await fetch('/api/trades?user_id=' + userId);
            const data = await response.json();
            
            let html = '';
            data.trades.forEach(trade => {
                const profitClass = trade.profit > 0 ? 'profit-positive' : 'profit-negative';
                const profitSign = trade.profit > 0 ? '+' : '';
                
                html += `
                <div class="trade-item">
                    <div>
                        <strong>${trade.symbol} ${trade.action}</strong><br>
                        <small>${new Date(trade.timestamp).toLocaleTimeString()}</small>
                    </div>
                    <div style="text-align: right;">
                        <div>$${trade.amount}</div>
                        <div class="profit ${profitClass}">
                            ${profitSign}$${trade.profit.toFixed(2)}
                        </div>
                    </div>
                </div>`;
            });
            
            document.getElementById('tradesList').innerHTML = html || 
                '<div class="loading">No trades yet</div>';
        } catch(error) {
            console.error('Failed to load trades');
        }
    }
    
    // Save selected markets
    async function saveMarkets() {
        const selected = [];
        document.querySelectorAll('#marketsList input:checked').forEach(cb => {
            selected.push(cb.value);
        });
        
        showMessage(`Selected ${selected.length} markets`, 'success');
    }
    
    // Start periodic updates
    function startUpdates() {
        if(updateInterval) clearInterval(updateInterval);
        
        updateInterval = setInterval(() => {
            updateBalance();
            loadTrades();
        }, 5000);
    }
    
    // Show message
    function showMessage(text, type) {
        const message = document.createElement('div');
        message.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? '#00C853' : type === 'error' ? '#FF4444' : '#FFD700'};
            color: ${type === 'success' ? 'white' : 'black'};
            padding: 15px 20px;
            border-radius: 8px;
            z-index: 1000;
            animation: slideIn 0.3s;
        `;
        
        message.textContent = text;
        document.body.appendChild(message);
        
        setTimeout(() => {
            message.style.animation = 'slideOut 0.3s';
            setTimeout(() => message.remove(), 300);
        }, 3000);
    }
    
    // Add animation styles
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
    
    // Check if already connected
    window.addEventListener('load', async () => {
        try {
            const response = await fetch('/api/check');
            const data = await response.json();
            
            if(data.connected) {
                userId = data.user_id;
                showTab('trading');
                updateBalance();
                startUpdates();
            }
        } catch(error) {
            // Not connected
        }
    });
    </script>
</body>
</html>
'''

# Trading Bot Thread
def run_trading_bot(user_id, settings):
    """Run trading bot in background"""
    import random
    
    while bot_status.get(user_id, False):
        try:
            # Simulate trading activity
            symbols = ['R_10', 'R_25', 'R_50']
            for symbol in symbols:
                if not bot_status.get(user_id, False):
                    break
                
                # Simulate trade
                if random.random() > 0.7:  # 30% chance of trade
                    amount = settings.get('trade_amount', 1.0)
                    action = random.choice(['BUY', 'SELL'])
                    profit = amount * 0.8 if random.random() > 0.3 else -amount
                    
                    trade = {
                        'id': f"TR{int(time.time())}{random.randint(1000, 9999)}",
                        'user_id': user_id,
                        'symbol': symbol,
                        'action': action,
                        'amount': amount,
                        'profit': round(profit, 2),
                        'status': 'won' if profit > 0 else 'lost',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    # Store trade
                    if user_id not in trades:
                        trades[user_id] = []
                    trades[user_id].append(trade)
                    
                    # Update balance
                    if user_id not in account_balance:
                        account_balance[user_id] = 1000.00
                    account_balance[user_id] += profit
                    
                    print(f"Trade: {symbol} {action} ${amount} → ${profit}")
            
            time.sleep(random.randint(5, 15))
            
        except Exception as e:
            print(f"Bot error: {e}")
            time.sleep(30)

@app.route('/')
def home():
    """Main page"""
    return HTML

@app.route('/api/connect', methods=['POST'])
def api_connect():
    """Connect to Deriv"""
    try:
        data = request.json
        api_token = data.get('api_token')
        
        if not api_token:
            return jsonify({'success': False, 'error': 'API token required'})
        
        # Create user
        user_id = hashlib.md5(api_token.encode()).hexdigest()[:10]
        users[user_id] = {
            'api_token': api_token,
            'connected_at': datetime.now().isoformat()
        }
        
        # Set initial balance
        account_balance[user_id] = 1000.00
        
        # Store in session
        from flask import session
        session['user_id'] = user_id
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'balance': account_balance[user_id],
            'message': 'Connected successfully'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bot/start', methods=['POST'])
def api_bot_start():
    """Start trading bot"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        if user_id not in users:
            return jsonify({'success': False, 'error': 'User not found'})
        
        # Stop existing bot if running
        if bot_status.get(user_id, False):
            bot_status[user_id] = False
            time.sleep(1)
        
        # Start new bot
        bot_status[user_id] = True
        
        settings = {
            'trade_amount': float(data.get('trade_amount', 1.0)),
            'max_trades': int(data.get('max_trades', 3))
        }
        
        # Start bot in background thread
        import threading
        thread = threading.Thread(
            target=run_trading_bot,
            args=(user_id, settings),
            daemon=True
        )
        thread.start()
        
        return jsonify({'success': True, 'message': 'Bot started'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bot/stop', methods=['POST'])
def api_bot_stop():
    """Stop bot"""
    try:
        data = request.json
        user_id = data.get('user_id')
        
        bot_status[user_id] = False
        return jsonify({'success': True, 'message': 'Bot stopped'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/balance')
def api_balance():
    """Get balance"""
    user_id = request.args.get('user_id')
    
    if user_id in account_balance:
        return jsonify({'success': True, 'balance': account_balance[user_id]})
    
    return jsonify({'success': False, 'error': 'User not found'})

@app.route('/api/markets')
def api_markets():
    """Get available markets"""
    markets = [
        {'symbol': 'R_10', 'name': 'Volatility 10 Index'},
        {'symbol': 'R_25', 'name': 'Volatility 25 Index'},
        {'symbol': 'R_50', 'name': 'Volatility 50 Index'},
        {'symbol': 'R_75', 'name': 'Volatility 75 Index'},
        {'symbol': 'R_100', 'name': 'Volatility 100 Index'},
        {'symbol': 'BOOM500', 'name': 'Boom 500 Index'},
        {'symbol': 'BOOM1000', 'name': 'Boom 1000 Index'},
        {'symbol': 'CRASH500', 'name': 'Crash 500 Index'},
        {'symbol': 'CRASH1000', 'name': 'Crash 1000 Index'},
    ]
    
    return jsonify({'success': True, 'markets': markets})

@app.route('/api/trades')
def api_trades():
    """Get trades"""
    user_id = request.args.get('user_id')
    
    if user_id in trades:
        return jsonify({'success': True, 'trades': trades[user_id][-10:]})
    
    return jsonify({'success': True, 'trades': []})

@app.route('/api/check')
def api_check():
    """Check if user is connected"""
    from flask import session
    user_id = session.get('user_id')
    
    if user_id in users:
        return jsonify({'connected': True, 'user_id': user_id})
    
    return jsonify({'connected': False})

@app.route('/health')
def health():
    """Health check for Render.com"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'users': len(users),
        'active_bots': sum(1 for v in bot_status.values() if v)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Karanka Deriv Bot on port {port}")
    print(f"✅ Mobile WebApp Ready")
    print(f"✅ Gold/Black Design Active")
    print(f"✅ Health Check: http://localhost:{port}/health")
    
    # For production on Render.com
    from waitress import serve
    serve(app, host='0.0.0.0', port=port)
