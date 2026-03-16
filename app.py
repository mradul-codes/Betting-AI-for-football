from flask import Flask, render_template, request, jsonify
import joblib, numpy as np, pandas as pd, requests
from datetime import datetime

app = Flask(__name__)

# --- 1. Load Heavy Brain (13 Features) ---
try:
    MODEL = joblib.load('football_model.pkl')
    SCALER = joblib.load('scaler.pkl')
    LE = joblib.load('team_encoder.pkl')
    MASTER = pd.read_csv('master_data_elite.csv', low_memory=False)
    print("🚀 Heavy Brain Active: 13-Feature Logic Enabled!")
except Exception as e:
    print(f"❌ Initialization Error: {e}")

API_KEY = '78871a8e6ebf7fb7052329e852d9652f'

@app.route('/')
def index():
    now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    leagues = ['soccer_spain_la_liga', 'soccer_spain_copa_del_rey', 'soccer_epl', 'soccer_italy_serie_a', 'soccer_germany_bundesliga']
    matches = []
    for l in leagues:
        url = f'https://api.the-odds-api.com/v4/sports/{l}/odds/?apiKey={API_KEY}&regions=eu&markets=h2h&commenceTimeFrom={now}'
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for m in data:
                    if not m.get('bookmakers'): continue
                    outcomes = m['bookmakers'][0]['markets'][0]['outcomes']
                    matches.append({
                        'home_team': m['home_team'].replace("'", ""), 
                        'away_team': m['away_team'].replace("'", ""),
                        'h_odd': next(o['price'] for o in outcomes if o['name'] == m['home_team']),
                        'a_odd': next(o['price'] for o in outcomes if o['name'] == m['away_team']),
                        'd_odd': next(o['price'] for o in outcomes if o['name'].lower() == 'draw'),
                        'start_time': m['commence_time']
                    })
        except: continue
    return render_template('index.html', matches=sorted(matches, key=lambda x: x['start_time']))

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        oh, od, oa = float(data['oh']), float(data['od']), float(data['oa'])
        budget = float(data.get('budget', 1000))

        # --- 2. Team Matching Logic ---
        def fix_name(n):
            n = n.lower().strip()
            return next((t for t in LE.classes_ if n in t.lower() or t.lower() in n), None)

        h_m = fix_name(data['home'])
        a_m = fix_name(data['away'])
        
        h_s = MASTER[MASTER['HomeTeam'] == h_m].iloc[-1] if h_m and not MASTER[MASTER['HomeTeam'] == h_m].empty else None
        a_s = MASTER[MASTER['AwayTeam'] == a_m].iloc[-1] if a_m and not MASTER[MASTER['AwayTeam'] == a_m].empty else None

        # --- 3. Build 13 Features (Strictly Matching Training Code) ---
        p4 = h_s['P4'] if h_s is not None else 1.2
        p5 = a_s['P5'] if a_s is not None else 1.0
        p6 = p4 - p5
        p7 = oh / oa
        p8 = 1 if oh < 2.0 else 0
        p9 = h_s['P9'] if h_s is not None else 2
        p10 = a_s['P10'] if a_s is not None else 1
        
        # New Smart Features from training code
        p11 = oh * p4  # Home Odds * Home Power
        p12 = oa * p5  # Away Odds * Away Power
        p13 = oh / oa  # Odds Ratio (same as p7 but needed for index 13)

        # Order must be: P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13
        raw_input = np.array([oh, od, oa, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]).reshape(1, -1)

        # --- 4. Prediction ---
        scaled_input = SCALER.transform(raw_input)
        probs = MODEL.predict_proba(scaled_input)[0]
        # Mapping: [Away, Draw, Home]
        pa, pd, ph = probs[0]*100, probs[1]*100, probs[2]*100

        # --- 5. Profit & Investment Logic (Arbitrage Style) ---
        margin = (1/oh) + (1/oa)
        inv_h = (budget / oh) / margin
        inv_a = (budget / oa) / margin
        potential_profit = (inv_h * oh) - budget

        # --- 6. Verdict Based on Sniper Threshold ---
        max_conf = max(ph, pa, pd)
        verdict = "❌ NO ACTION"
        if max_conf >= 70:
            verdict = f"💎 ELITE SNIPER: {'HOME' if ph==max_conf else 'AWAY' if pa==max_conf else 'DRAW'}"
        elif max_conf >= 60:
            verdict = "✅ HIGH PROBABILITY"

        return jsonify({
            'p_home': "{:.2f}".format(ph),
            'p_draw': "{:.2f}".format(pd),
            'p_away': "{:.2f}".format(pa),
            'inv_h': "{:.2f}".format(inv_h),
            'inv_a': "{:.2f}".format(inv_a),
            'profit': "{:.2f}".format(potential_profit),
            'verdict': verdict,
            'confidence': "{:.2f}%".format(max_conf)
        })

    except Exception as e:
        print(f"🔥 Analysis Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)