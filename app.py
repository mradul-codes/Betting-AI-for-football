from flask import Flask, render_template, request, jsonify
import joblib, numpy as np, pandas as pd, requests, os
from datetime import datetime

app = Flask(__name__)

# --- 1. Load Heavy Brain ---
try:
    MODEL = joblib.load('football_model.pkl')
    SCALER = joblib.load('scaler.pkl')
    LE = joblib.load('team_encoder.pkl')
    MASTER = pd.read_csv('master_data_elite.csv', low_memory=False)
    print("🚀 Brain Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")

API_KEY = '78871a8e6ebf7fb7052329e852d9652f'

@app.route('/')
def index():
    now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    leagues = ['soccer_spain_la_liga', 'soccer_epl', 'soccer_italy_serie_a', 'soccer_germany_bundesliga']
    matches = []
    for l in leagues:
        url = f'https://api.the-odds-api.com/v4/sports/{l}/odds/?apiKey={API_KEY}&regions=eu&markets=h2h&commenceTimeFrom={now}'
        try:
            r = requests.get(url)
            if r.status_code == 200:
                for m in r.json():
                    if not m.get('bookmakers'): continue
                    outcomes = m['bookmakers'][0]['markets'][0]['outcomes']
                    matches.append({
                        'home_team': m['home_team'], 
                        'away_team': m['away_team'],
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
        oh, od, oa = float(data.get('oh', 2.0)), float(data.get('od', 3.0)), float(data.get('oa', 3.0))
        budget = float(data.get('budget', 1000))

        def fix_name(n):
            if not n: return None
            # Cleaning names for better matching (1. FC Leverkusen -> leverkusen)
            n_clean = n.lower().replace("1. ", "").replace("fc ", "").strip()
            for team in LE.classes_:
                if n_clean in team.lower() or team.lower() in n_clean:
                    return team
            return None

        h_m = fix_name(data['home'])
        a_m = fix_name(data['away'])
        
        h_s = MASTER[MASTER['HomeTeam'] == h_m].iloc[-1] if h_m and not MASTER[MASTER['HomeTeam'] == h_m].empty else None
        a_s = MASTER[MASTER['AwayTeam'] == a_m].iloc[-1] if a_m and not MASTER[MASTER['AwayTeam'] == a_m].empty else None

        # Fallback to defaults if data is missing to avoid "undefined"
        p4 = float(h_s['P4']) if h_s is not None else 1.2
        p5 = float(a_s['P5']) if a_s is not None else 1.0
        p9 = float(h_s['P9']) if h_s is not None else 2.0
        p10 = float(a_s['P10']) if a_s is not None else 1.0
        
        # Features 1-13
        p6, p7, p8 = p4 - p5, oh / oa, (1 if oh < 2.0 else 0)
        p11, p12, p13 = oh * p4, oa * p5, oh / oa

        raw_input = np.array([oh, od, oa, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13]).reshape(1, -1)
        scaled_input = SCALER.transform(raw_input)
        probs = MODEL.predict_proba(scaled_input)[0]
        
        pa, pd, ph = probs[0]*100, probs[1]*100, probs[2]*100

        # Investment Logic
        margin = (1/oh) + (1/oa)
        inv_h = (budget / oh) / margin
        inv_a = (budget / oa) / margin
        profit = (inv_h * oh) - budget

        max_conf = max(ph, pa, pd)
        verdict = "❌ NO ACTION"
        if max_conf >= 70: verdict = f"💎 ELITE SNIPER: {'HOME' if ph==max_conf else 'AWAY' if pa==max_conf else 'DRAW'}"
        elif max_conf >= 60: verdict = "✅ HIGH PROBABILITY"

        return jsonify({
            'p_home': round(ph, 2), 'p_draw': round(pd, 2), 'p_away': round(pa, 2),
            'inv_h': round(inv_h, 2), 'inv_a': round(inv_a, 2), 'profit': round(profit, 2),
            'verdict': verdict, 'confidence': f"{round(max_conf, 2)}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
