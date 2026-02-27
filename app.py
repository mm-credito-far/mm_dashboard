import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# --- CONFIGURAÇÕES VISUAIS (Matplotlib) ---
COLOR_BG = "#1e1e1e"
COLOR_PANEL = "#2b2b2b"
COLOR_TEXT = "#e0e0e0"
COLOR_CURRENT_YEAR = "#2980b9"

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': COLOR_BG,
    'axes.facecolor': COLOR_BG,
    'grid.color': '#444444',
    'grid.linestyle': '--',
    'grid.alpha': 0.4,
    'text.color': COLOR_TEXT,
    'axes.labelcolor': '#aaaaaa',
    'xtick.color': '#aaaaaa',
    'ytick.color': '#aaaaaa',
    'font.size': 9,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# --- 1. CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Dashboard Sazonal", layout="wide")
st.title("📈 Dashboard Multiativos - Análise Sazonal")

# --- 2. PREPARAÇÃO DOS DADOS ---
@st.cache_data
def carregar_e_preparar_dados():
    # Lê o CSV (lembre-se de salvar seu Excel como dados.csv)
    try:
        df = pd.read_csv("dados.csv")
    except FileNotFoundError:
        st.error("Arquivo 'dados.csv' não encontrado. Coloque-o na mesma pasta do script.")
        return None

    # Encontra a coluna de data e converte
    col_data = [c for c in df.columns if c.lower() == 'data'][0]
    df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
    df = df.dropna(subset=[col_data]).sort_values(col_data)
    df.rename(columns={col_data: 'Data'}, inplace=True)
    
    return df

df_full = carregar_e_preparar_dados()

if df_full is not None:
    # --- 3. SELEÇÃO DE ATIVO (Substituindo as abas do Tkinter) ---
    ativos_disponiveis = [c for c in df_full.columns if c != 'Data']
    ativo_selecionado = st.sidebar.selectbox("SELECIONE O ATIVO", ativos_disponiveis)

    # Filtra e prepara os cálculos do ativo escolhido
    df_ativo = df_full[['Data', ativo_selecionado]].copy()
    df_ativo.rename(columns={ativo_selecionado: 'Close'}, inplace=True)
    df_ativo['Close'] = pd.to_numeric(df_ativo['Close'], errors='coerce')
    df_ativo = df_ativo.dropna()

    # Lógica original de cálculos
    df_ativo['Retorno'] = df_ativo['Close'].pct_change().fillna(0)
    df_ativo['Ano'] = df_ativo['Data'].dt.year
    df_ativo['Mes'] = df_ativo['Data'].dt.month
    df_ativo['Dia'] = df_ativo['Data'].dt.day
    df_ativo['TradingDay'] = df_ativo.groupby(['Ano', 'Mes'])['Data'].rank(method='first').astype(int)

    current_year = df_ativo['Ano'].max()
    anos_historicos = sorted(df_ativo[df_ativo['Ano'] < current_year]['Ano'].unique(), reverse=True)

    # --- 4. BARRA LATERAL: FILTRO DE ANOS ---
    st.sidebar.markdown("### FILTRO DE ANOS")
    todos_anos = st.sidebar.checkbox("Selecionar Todos", value=True)
    
    if todos_anos:
        anos_selecionados = anos_historicos
    else:
        anos_selecionados = st.sidebar.multiselect("Anos Históricos", anos_historicos, default=anos_historicos[:3])

    # O ano atual sempre entra na visualização
    mask = (df_ativo['Ano'].isin(anos_selecionados)) | (df_ativo['Ano'] == current_year)
    df_view = df_ativo[mask].copy()

    if df_view.empty:
        st.warning("Selecione pelo menos um ano para visualizar os dados.")
    else:
        # --- 5. GRÁFICO PRINCIPAL ---
        st.subheader(f"Evolução Histórica - {ativo_selecionado}")
        # Usando o gráfico nativo do Streamlit para ter interatividade (hover) grátis
        st.line_chart(data=df_view.set_index('Data')['Close'], color="#00b894")

        st.markdown("---")

        # --- 6. SESSÃO DE VOLATILIDADE ---
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Volatilidade Móvel")
            win = st.selectbox("Janela de Dias", [21, 42, 63, 252], index=0)
            vol = df_view['Retorno'].rolling(win).std() * np.sqrt(252) * 100
            df_vol = pd.DataFrame({'Data': df_view['Data'], 'Vol': vol}).dropna()
            st.line_chart(data=df_vol.set_index('Data')['Vol'], color="#ff7675")

        with col2:
            st.subheader("Volatilidade Média Anual (10y)")
            df_recent = df_view[df_view['Ano'] >= current_year - 10]
            if not df_recent.empty:
                vol_y = df_recent.groupby('Ano')['Retorno'].std() * np.sqrt(252) * 100
                st.bar_chart(vol_y)

        st.markdown("---")

        # --- 7. SESSÃO DE SAZONALIDADE (Usando seu Matplotlib) ---
        st.subheader("Análise de Sazonalidade")
        mes_escolhido = st.selectbox("Selecione o Período", 
                                     ["Ano Completo", "1 - Janeiro", "2 - Fevereiro", "3 - Março", "4 - Abril", 
                                      "5 - Maio", "6 - Junho", "7 - Julho", "8 - Agosto", "9 - Setembro", 
                                      "10 - Outubro", "11 - Novembro", "12 - Dezembro"])

        # Recria a figura do Matplotlib
        fig_sea, ax_sea = plt.subplots(figsize=(10, 4), layout='tight')
        ax_sea.axhline(0, color='gray', lw=1)

        if mes_escolhido == "Ano Completo":
            x_cur, ticks, lbls = 0, [], ['J','F','M','A','M','J','J','A','S','O','N','D']
            for m in range(1, 13):
                df_m = df_view[df_view['Mes'] == m]
                if df_m.empty: 
                    x_cur += 21
                    continue
                d_prof = df_m.groupby('TradingDay')['Retorno'].mean().cumsum() * 100
                days = len(d_prof)
                x_vals = np.arange(x_cur, x_cur + days)
                
                c = "#00e676" if d_prof.iloc[-1] >= 0 else "#ff1744"
                ax_sea.plot(x_vals, d_prof, color=c, lw=1.5, alpha=0.5)
                ax_sea.fill_between(x_vals, d_prof, 0, color=c, alpha=0.05)
                ax_sea.axvline(x_cur + days, color='#333', linestyle=':')
                ticks.append(x_cur + days/2)
                x_cur += days

            if ticks:
                ax_sea.set_xticks(ticks)
                ax_sea.set_xticklabels(lbls)
                
        else:
            # Pega apenas o número do mês da string (ex: "3 - Março" -> 3)
            month_idx = int(mes_escolhido.split(" - ")[0])
            df_m = df_view[df_view['Mes'] == month_idx].copy()
            
            if not df_m.empty:
                years = df_m['Ano'].unique()
                for yr in years:
                    df_yr = df_m[df_m['Ano'] == yr]
                    y_vals = df_yr['Retorno'].cumsum() * 100
                    x_vals = df_yr['TradingDay'].values 
                    # Linhas dos anos em cinza
                    if yr != current_year:
                        ax_sea.plot(x_vals, y_vals, color='gray', alpha=0.3, linewidth=1)

                # Média Histórica
                avg_prof = df_m.groupby('TradingDay')['Retorno'].mean().cumsum() * 100
                x_avg = avg_prof.index.values
                c = "#00e676" if avg_prof.iloc[-1] >= 0 else "#ff1744"
                ax_sea.plot(x_avg, avg_prof, color=c, lw=3, label="Média Histórica", zorder=10)

                # Ano Atual
                df_cur = df_full[(df_full['Ano'] == current_year) & (df_full['Mes'] == month_idx)].copy()
                if not df_cur.empty:
                    perf_cur = df_cur['Retorno'].cumsum() * 100
                    ax_sea.plot(df_cur['TradingDay'], perf_cur, color=COLOR_CURRENT_YEAR, lw=3, label=str(current_year), zorder=10)

                ax_sea.legend(loc='upper left')
                ax_sea.set_title(f"Sazonalidade: {mes_escolhido.split(' - ')[1]}")
                ax_sea.set_xlabel("Dia Útil do Mês")
                ax_sea.set_ylabel("Retorno Acumulado (%)")

        # Exibe o gráfico do Matplotlib no Streamlit
        st.pyplot(fig_sea)