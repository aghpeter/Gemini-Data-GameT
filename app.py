import streamlit as st
import yfinance as yf
import math
from datetime import datetime
import pandas as pd
from google import genai
from google.genai import types

# --- 1. KONFIGURACE STRÁNKY ---
st.set_page_config(page_title="Game-Theoretic Trading AI", page_icon="📈", layout="wide")
st.title("📈 Game-Theoretic Trading Agent")
st.markdown("Zadejte ticker a expiraci. AI stáhne reálná data a provede analýzu na základě teorie her.")

# --- 2. DEFINICE FUNKCÍ (NÁSTROJŮ) ---
def get_moving_averages(ticker: str) -> str:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        if hist.empty:
            return f"Data pro ticker {ticker} nebyla nalezena."
        
        current_price = hist['Close'].iloc[-1]
        ma9 = hist['Close'].rolling(window=9).mean().iloc[-1]
        ma20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ma50 = hist['Close'].rolling(window=50).mean().iloc[-1]
        
        return (f"Ticker: {ticker}\nAktuální cena: {current_price:.2f} USD\n"
                f"9denní MA: {ma9:.2f}\n20denní MA: {ma20:.2f}\n50denní MA: {ma50:.2f}")
    except Exception as e:
        return f"Chyba při stahování průměrů: {str(e)}"

def get_options_data(ticker: str, target_dates: list[str]) -> str:
    try:
        stock = yf.Ticker(ticker)
        current_price = stock.history(period="1d")['Close'].iloc[-1]
        available_expirations = stock.options
        
        result_str = f"Aktuální cena {ticker}: {current_price:.2f} USD\n\n"
        
        for target_date in target_dates:
            if target_date not in available_expirations:
                result_str += f"Expirace {target_date}: N/A (Nenalezena v datech Yahoo Finance)\n"
                continue
                
            opt = stock.option_chain(target_date)
            calls = opt.calls
            puts = opt.puts
            
            calls['distance'] = abs(calls['strike'] - current_price)
            atm_call = calls.loc[calls['distance'].idxmin()]
            atm_iv = atm_call['impliedVolatility']
            
            days_to_exp = (datetime.strptime(target_date, "%Y-%m-%d") - datetime.now()).days
            if days_to_exp <= 0: days_to_exp = 1
            
            one_sigma_move = current_price * atm_iv * math.sqrt(days_to_exp / 365.0)
            two_sigma_move = 2 * one_sigma_move
            upper_bound = current_price + two_sigma_move
            lower_bound = current_price - two_sigma_move
            
            calls['upper_dist'] = abs(calls['strike'] - upper_bound)
            puts['lower_dist'] = abs(puts['strike'] - lower_bound)
            call_2sigma = calls.loc[calls['upper_dist'].idxmin()]
            put_2sigma = puts.loc[puts['lower_dist'].idxmin()]
            
            result_str += f"--- Expirace: {target_date} ({days_to_exp} dní) ---\n"
            result_str += f"Implikovaná volatilita (ATM): {atm_iv*100:.2f}%\n"
            result_str += f"2-sigma pohyb: ±{two_sigma_move:.2f} USD (Rozsah: {lower_bound:.2f} až {upper_bound:.2f})\n"
            result_str += f"ATM Call (Strike {atm_call['strike']}): Cena = {atm_call['lastPrice']} USD\n"
            result_str += f"2-Sigma Upper Call (Strike {call_2sigma['strike']}): Cena = {call_2sigma['lastPrice']} USD\n"
            result_str += f"2-Sigma Lower Put (Strike {put_2sigma['strike']}): Cena = {put_2sigma['lastPrice']} USD\n\n"
            
        return result_str
    except Exception as e:
        return f"Chyba při stahování opcí: {str(e)}"

# --- 3. SYSTÉMOVÝ PROMPT ---
SYSTEM_PROMPT = """
ROLE
Jsi game-theoretický trading stratég specializovaný na: akcie, opce, komodity, futures.
Trh vnímáš jako opakovanou hru s neúplnými informacemi, kde různí hráči mají rozdílné cíle, omezení a časové horizonty.
Tvým cílem není predikovat cenu, ale identifikovat asymetrickou výhodu vznikající z nuceného chování ostatních hráčů.

ZDROJOVÁ HIERARCHIE (POVINNÁ)
1. Robert Gibbons – Game Theory for Applied Economists
2. Avinash Dixit & Barry Nalebuff – Thinking Strategically
3. Mark Minervini – Trade Like a Stock Market Wizard
4. Steven Tadelis – Game Theory
5. Fiona Carmichael – A Guide to Game Theory
6. Petrosjan & Zenkevich – Game Theory

POVINNÝ ANALYTICKÝ POSTUP (NIKDY NEVYNECHAT)
1. Identifikace hry (Xueqin styl): hráči, cíle, omezení.
2. Typ hry: simultánní/sekvenční, zero-sum/non-zero-sum.
3. Motivace a nucené chování: kdo musí jednat, tlak (margin, expiry).
4. Rovnováha: kde je stabilita, kde je křehká.
5. Second-level thinking: co očekává většina, co se stane, když se nenaplní.
6. Trading implikace: asymetrie payoff, katalyzátor, invalidace.

PRAVIDLA CHOVÁNÍ
❌ Nikdy: nepředpovídej cenu, nehovoř o „férové hodnotě“, nedávej doporučení bez reakce protistran.
✅ Vždy: pracuj s pravděpodobností, zohledňuj čas, zdůrazňuj risk a invalidaci.

STYL ODPOVĚDÍ: strukturovaný, stručný, poradenský.
DEFINIČNÍ VĚTA: Neobchoduji cenu. Obchoduji chování hráčů pod tlakem.
"""

# --- 4. INICIALIZACE AI ---
@st.cache_resource
def get_chat_session():
    # Bezpečné načtení klíče ze Streamlit Secrets
    api_key = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=api_key)
    
    config = types.GenerateContentConfig(
        tools=[get_moving_averages, get_options_data], 
        system_instruction=SYSTEM_PROMPT,
        temperature=0.2, 
    )
    return client.chats.create(model="gemini-2.5-flash", config=config)

# --- 5. UŽIVATELSKÉ ROZHRANÍ ---
user_input = st.text_area(
    "Zadejte svůj dotaz:", 
    height=100, 
    value="Analyzuj ticker NEM a navrhni opční strategii pro expiraci 15.5.2026 nebo 18.6.2026. Počítám s růstem ceny na 60 USD."
)

if st.button("Spustit analýzu", type="primary"):
    if not user_input:
        st.warning("Prosím, zadejte dotaz.")
    else:
        try:
            chat = get_chat_session()
            with st.spinner("⏳ Agenti pracují: Stahuji data z trhu a analyzuji herní prostředí (může to trvat 10-30 vteřin)..."):
                response = chat.send_message(user_input)
            
            st.success("Analýza dokončena!")
            st.markdown("### Výsledek analýzy:")
            st.markdown(response.text)
        except Exception as e:
            st.error(f"❌ Nastala chyba: {e}")
