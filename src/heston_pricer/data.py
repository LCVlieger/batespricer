import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class MarketOption:
    strike: float
    maturity: float
    market_price: float
    option_type: str = "CALL" 
    bid: float = 0.0
    ask: float = 0.0

def fetch_options(ticker_symbol: str, target_size: int = 100) -> Tuple[List[MarketOption], float]:
    """
    SMART FETCHER (DISTRIBUTION SAMPLING):
    - Filters: Ghost Bids (2.65bps), Wide Spreads (>40%), Moneyness (0.75-1.25).
    - Selection: NON-UNIFORM. heavily weights ATM options (High Liquidity) 
      while sparsely sampling OTM wings (Smile Definition).
    """
    ticker = yf.Ticker(ticker_symbol)
    
    try:
        S0 = ticker.fast_info.get('last_price', None)
        if S0 is None:
            hist = ticker.history(period="1d")
            S0 = hist['Close'].iloc[-1]
    except:
        return [], 0.0

    print(f"--- Smart Calibration Set: {ticker_symbol} (Spot: {S0:.2f}) ---")
    
    expirations = ticker.options
    if not expirations: return [], 0.0
    today = datetime.now()
    
    all_candidates = []
    MIN_T, MAX_T = 0.46, 2.5 
    
    # Using the optimized 2.65bps (0.000265) we calculated for SPX/NVDA scaling
    PHI = 0.0005 #265

    print("Scanning option chains (filtering junk data)...")
    for exp_str in expirations:
        try:
            d = datetime.strptime(exp_str, "%Y-%m-%d")
            T = (d - today).days / 365.25
            if not (MIN_T <= T <= MAX_T): continue

            chain = ticker.option_chain(exp_str)
            
            puts = chain.puts
            puts['type'] = 'PUT'
            calls = chain.calls
            calls['type'] = 'CALL'
            combined = pd.concat([puts, calls])
            
            for _, row in combined.iterrows():
                K = row['strike']
                moneyness = K / S0
                
                # Standard Moneyness & OTM Filter
                if not (0.75 <= moneyness <= 1.25): continue 
                if row['type'] == 'PUT' and K >= S0: continue
                if row['type'] == 'CALL' and K < S0: continue
                
                bid = row.get('bid', 0.0)
                ask = row.get('ask', 0.0)
                
                # 1. Anti-Ghost Check
                dynamic_min_bid = S0 * PHI
                if bid < dynamic_min_bid: print(dynamic_min_bid)
                if bid < dynamic_min_bid: continue 

                # 2. Spread Check
                mid = (bid + ask) / 2.0 #ask 
                spread_ratio = (ask - bid) / mid
                #actual_mid = (bid + ask) / 2.0
                #spread_ratio = (ask - bid) / actual_mid if actual_mid > 0 else 1.0
                if spread_ratio > 0.40: continue 
                all_candidates.append({
                    'strike': K, 'maturity': T, 'market_price': mid,
                    'spread_ratio': spread_ratio, 'type': row['type'],
                    'bid': bid, 'ask': ask
                })
        except: continue

    if not all_candidates: return [], S0
        
    df = pd.DataFrame(all_candidates)
    
    # ---------------------------------------------------------
    # SELECTION STRATEGY: SKEWED DISTRIBUTION (ATM FOCUSED)
    # ---------------------------------------------------------
    
    unique_maturities = sorted(df['maturity'].unique())
    target_per_date = max(4, target_size // len(unique_maturities))
    selected_indices = set()
    
    # Skew Factor: 2.0 = Quadratic (Standard), 3.0 = Cubic (Very ATM heavy)
    SKEW_POWER = 0.75
    
    print(f"Stratifying {len(unique_maturities)} maturities with Quadratic ATM Skew...")
    
    for mat in unique_maturities:
        mat_slice = df[df['maturity'] == mat]
        
        for opt_type in ['PUT', 'CALL']:
            # Sort by strike (Low -> High)
            # Puts: Low Strike (Deep OTM) -> High Strike (ATM)
            # Calls: Low Strike (ATM) -> High Strike (Deep OTM)
            candidates = mat_slice[mat_slice['type'] == opt_type].sort_values('strike')
            
            count = len(candidates)
            if count == 0: continue

            # How many do we need?
            n_need = target_per_date // 2
            if opt_type == 'CALL': n_need += 1

            if count <= n_need:
                selected_indices.update(candidates.index)
            else:
                # 1. Create Linear Space 0 -> 1
                u = np.linspace(0, 1, n_need)
                
                # 2. Apply Quadratic Skew based on Option Type
                if opt_type == 'CALL':
                    # Calls start at ATM (Index 0). We want more points near 0.
                    # Mapping: 0->0, 0.5->0.25, 1->1
                    skewed_u = u ** SKEW_POWER
                else:
                    # Puts end at ATM (Index N). We want more points near 1.
                    # Mapping: 0->0, 0.5->0.75, 1->1
                    skewed_u = 1 - (1 - u) ** SKEW_POWER
                
                # 3. Convert back to integer indices
                idx_positions = (skewed_u * (count - 1)).astype(int)
                
                # Ensure uniqueness (rare edge case with small counts)
                idx_positions = np.unique(idx_positions)
                
                selected_indices.update(candidates.iloc[idx_positions].index)

    final_df = df.loc[list(selected_indices)].copy()
    
    # Cap excess (Prioritize tightest spreads if we overshoot target)
    if len(final_df) > target_size:
        final_df = final_df.sort_values('spread_ratio').head(target_size)
    
    market_options = [
        MarketOption(r['strike'], r['maturity'], r['market_price'], r['type'], r['bid'], r['ask'])
        for _, r in final_df.iterrows()
    ]
    market_options.sort(key=lambda x: (x.maturity, x.strike))
    
    return market_options, S0