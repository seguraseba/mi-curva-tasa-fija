import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import plotly.graph_objects as go
from pandas.tseries.offsets import CustomBusinessDay

from pathlib import Path
import pandas as pd
import streamlit as st

CER_LOCAL_PATH = Path(r"C:\Users\ssegura\OneDrive - BALANZ\Escritorio\CER.xlsx")
CER_REPO_PATH  = Path(__file__).parent / "CER.xlsx"

@st.cache_data
def cargar_cer(path: Path, file_version: float) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = df.columns.str.lower().str.strip()

    # requiere columnas: fecha, cer
    df["fecha"] = pd.to_datetime(df["fecha"], dayfirst=True).dt.normalize()
    df["cer"] = df["cer"].astype(str).str.replace(",", ".", regex=False)
    df["cer"] = pd.to_numeric(df["cer"], errors="coerce")

    df = (
        df.dropna(subset=["fecha", "cer"])
          .sort_values("fecha")
          .drop_duplicates("fecha", keep="last")
          .reset_index(drop=True)
    )
    return df

# 1) Local: tu PC
if CER_LOCAL_PATH.exists():
    cer_df = cargar_cer(CER_LOCAL_PATH, CER_LOCAL_PATH.stat().st_mtime)
    st.sidebar.success("✅ CER cargado desde tu Escritorio (modo local)")
# 2) Web / general: archivo en repo
elif CER_REPO_PATH.exists():
    cer_df = cargar_cer(CER_REPO_PATH, CER_REPO_PATH.stat().st_mtime)
    st.sidebar.success("✅ CER cargado desde el repo (modo web)")
# 3) Si falta todo: cortar con mensaje claro
else:
    st.sidebar.error("❌ No se encontró CER.xlsx ni en el Escritorio ni en el repo.")
    st.stop()

from pandas.tseries.offsets import BDay

def menos_10_habiles(d: date) -> pd.Timestamp:
    return (pd.Timestamp(d).normalize() - BDay(10))

def cer_en_o_antes(cer_df: pd.DataFrame, fecha: pd.Timestamp) -> float | None:
    s = cer_df.loc[cer_df["fecha"] <= fecha, "cer"]
    if s.empty:
        return None
    return float(s.iloc[-1])

def rendimiento_cer_bono(symbol: str, cer_df: pd.DataFrame, fecha_emision_map: dict) -> dict:
    """
    Devuelve un dict con:
      - fecha_liq
      - f_liq_m10 (liq - 10 hábiles)
      - f_emis_m10 (emisión - 10 hábiles)
      - cer_liq
      - cer_emis
      - factor_cer
      - rendimiento_cer_pct
    """
    if symbol not in fecha_emision_map:
        return {"error": "No hay fecha de emisión cargada"}

    fecha_liq = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1))  # hoy + 1
    f_liq_m10 = fecha_liq - BDay(10)

    fecha_emis = fecha_emision_map[symbol]
    f_emis_m10 = pd.Timestamp(fecha_emis).normalize() - BDay(10)

    cer_liq = cer_en_o_antes(cer_df, f_liq_m10)
    cer_emis = cer_en_o_antes(cer_df, f_emis_m10)

    if cer_liq is None or cer_emis is None or cer_emis == 0:
        return {"error": "No se encontró CER para alguna fecha (o cer_emis=0)"}

    factor = cer_liq / cer_emis
    rend_pct = (factor - 1) * 100

    return {
        "fecha_liq": fecha_liq.date(),
        "f_liq_m10": f_liq_m10.date(),
        "f_emis_m10": f_emis_m10.date(),
        "cer_liq": cer_liq,
        "cer_emis": cer_emis,
        "factor_cer": factor,
        "rendimiento_cer_pct": rend_pct,
    }

def rendimiento_real_por_precio(precio: float, factor_cer: float) -> float | None:
    if precio is None or factor_cer is None:
        return None
    if precio <= 0:
        return None
    return (precio / factor_cer  - 1) * 100



# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Curva tasa fija pesos",
    layout="wide",  # importante para ver tabla y gráfico lado a lado
)

# =========================
# MAPEO MES Y VENCIMIENTOS
# =========================

MONTH_CODE_MAP = {
    "E": 1,  # Enero
    "F": 2,  # Febrero
    "M": 3,  # Marzo
    "A": 4,  # Abril
    "Y": 5,  # Mayo
    "J": 6,  # Junio
    "L": 7,  # Julio
    "G": 8,  # Agosto
    "S": 9,  # Septiembre
    "O": 10, # Octubre
    "N": 11, # Noviembre
    "D": 12, # Diciembre
}

def vencimiento_desde_symbol(symbol: str, base_decade: int = 2020) -> date:
    if not symbol:
        raise ValueError("Símbolo vacío")

    s = symbol.strip().upper()
    if len(s) < 4:
        raise ValueError(f"Símbolo demasiado corto: {symbol!r}")

    year_digit = s[-1]
    month_code = s[-2]
    body = s[:-2]

    if not year_digit.isdigit():
        raise ValueError(f"El último carácter no es un dígito de año en {symbol!r}")
    if month_code not in MONTH_CODE_MAP:
        raise ValueError(f"Código de mes desconocido '{month_code}' en {symbol!r}")

    day_digits = "".join(ch for ch in body if ch.isdigit())
    if not day_digits:
        raise ValueError(f"No se encontraron dígitos de día en {symbol!r}")

    day = int(day_digits)
    year = base_decade + int(year_digit)
    month = MONTH_CODE_MAP[month_code]
    return date(year, month, day)

# =========================
# CONSTANTES DATA912
# =========================

URL_BONOS  = "https://data912.com/live/arg_bonds"
URL_LETRAS = "https://data912.com/live/arg_notes"

LETRAS_TARGET = [
    "S30N6", "S16E6", "S27F6", "S17A6", "S30A6", "S29Y6", "S31G6", "S30O6", "X29Y6", "X30N6",
]

BONOS_TARGET = [
    "T30E6",
    "T13F6",
    "T30J6",
    "T15E7",
    "T30A7",
    "T30J7",
]

BONOS_CER_TARGET = [
    "TZXM6",
    "TZX26",
    "TX26",
    "TZXO6",
    "TZXD6",
    "TZXM7",
    "TZX27",
    "TZXD7",
    "TZX28",
    "TX28",
    "TX31",
    "DICP",
    "PARP",
    "CUAP",
    "TZXA7",
    "TZXY7",
    "X29Y6",
    "X30N6",
]

FECHA_VENCIMIENTO = {
    "TZXM6": date(2026, 3, 31),
    "TZX26": date(2026, 6, 30),
    "TX26":  date(2026, 11, 9),
    "TZXO6": date(2026, 10, 30),
    "TZXD6": date(2026, 12, 15),
    "TZXM7": date(2027, 3, 31),
    "TZXA7": date(2027, 4, 30),
    "TZXY7": date(2027, 5, 30),
    "TZX27": date(2027, 6, 30),
    "TZXD7": date(2027, 12, 15),
    "TZX28": date(2028, 6, 30),
    "TX28":  date(2028, 11, 9),
    "TX31":  date(2031, 11, 30),
    "DICP":  date(2033, 12, 31),
    "PARP":  date(2038, 12, 31),
    "CUAP":  date(2045, 12, 31),
    "X29Y6": date(2026, 5, 29),
    "X30N6": date(2026, 11, 30),
}


# =========================
# SEPARAR TARGETS: TASA FIJA vs CER
# =========================

# Tasa fija: excluye letras que empiezan con X
LETRAS_TF_TARGET = [s for s in LETRAS_TARGET if not s.upper().startswith("X")]
BONOS_TF_TARGET = BONOS_TARGET[:]  # igual que BONOS_TARGET

# CER: letras que empiezan con X + bonos cer
LETRAS_CER_TARGET = [s for s in LETRAS_TARGET if s.upper().startswith("X")]
BONOS_CER_TARGET = BONOS_CER_TARGET[:]  # ya la tenés


PAGOS_FINALES = {
    "T30E6":142.22,
    "T13F6":144.97,
    "S27F6":125.84,
    "S30A6":127.49,
    "S29Y6":132.04,
    "T30J6":144.90,
    "S31G6":127.06,
    "S30O6":135.28,
    "T15E7":161.10,
    "T30A7":157.34,
    "S17A6":110.13,
    "S30N6":129.89,
    "T31Y7":151.56,
    "T30J7":156.04,
}

FECHA_EMISION = {
    "TZXM6": date(2024, 4, 30),
    "TZX26": date(2024, 2, 1),
    "TX26":  date(2020, 9, 4),
    "TZXO6": date(2024, 10, 31),
    "TZXD6": date(2024, 3, 15),
    "TZXM7": date(2024, 5, 20),
    "TZXA7": date(2025, 11, 28),
    "TZXY7": date(2025, 12, 15),
    "TZX27": date(2024, 2, 1),
    "TZXD7": date(2024, 3, 15),
    "TZX28": date(2024, 2, 1),
    "TX28":  date(2020, 9, 4),
    "TX31":  date(2022, 5, 31),
    "DICP":  date(2003, 12, 31),
    "PARP":  date(2003, 12, 31),
    "CUAP":  date(2003, 12, 31),
    "X29Y6": date(2025, 11, 28),
    "X30N6": date(2025, 12, 15),
}


# =========================
# HELPERS PARA API
# =========================

from pandas.tseries.offsets import BDay

def menos_10_habiles(d: date) -> pd.Timestamp:
    return (pd.Timestamp(d).normalize() - BDay(10))


def cer_en_o_antes(cer_df: pd.DataFrame, fecha: pd.Timestamp) -> float | None:
    s = cer_df.loc[cer_df["fecha"] <= fecha, "cer"]
    if s.empty:
        return None
    return float(s.iloc[-1])

def rendimiento_cer_bono(symbol: str, cer_df: pd.DataFrame, fecha_emision_map: dict) -> dict:
    if symbol not in fecha_emision_map:
        return {"error": "No hay fecha de emisión cargada"}

    # fecha de liquidación = hoy + 1
    fecha_liq = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1))
    f_liq_m10 = fecha_liq - BDay(10)

    # fecha de emisión
    fecha_emis = fecha_emision_map[symbol]
    f_emis_m10 = pd.Timestamp(fecha_emis).normalize() - BDay(10)

    cer_liq = cer_en_o_antes(cer_df, f_liq_m10)
    cer_emis = cer_en_o_antes(cer_df, f_emis_m10)

    if cer_liq is None or cer_emis is None or cer_emis == 0:
        return {"error": "No se encontró CER para alguna fecha (o cer_emis=0)"}

    factor = cer_liq / cer_emis
    rend_pct = (factor - 1) * 100

    return {
        "fecha_liq": fecha_liq.date(),
        "f_liq_m10": f_liq_m10.date(),
        "f_emis_m10": f_emis_m10.date(),
        "cer_liq": cer_liq,
        "cer_emis": cer_emis,
        "factor_cer": factor,
        "rendimiento_cer_pct": rend_pct,
    }


def _fetch_json(url: str):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

# =========================
# LISTAS DE LETRAS Y BONOS
# =========================

def letras_lista(con_vencimiento: bool = True) -> pd.DataFrame:
    """
    Devuelve solo las letras target en un DataFrame.
    Si con_vencimiento=True, agrega:
      - vencimiento
      - dias_a_vencimiento
    """
    datos = _fetch_json(URL_LETRAS)

    # filtrar universo target y sacar BNA6D por las dudas
  
    datos = [
        x for x in datos
        if x.get("symbol") in LETRAS_TF_TARGET and x.get("symbol") != "BNA6D"
    ]


    df = pd.DataFrame(datos)

    columnas = ['symbol', 'c', 'v', 'q_bid', 'px_bid', 'px_ask',
                'q_ask', 'q_op', 'pct_change']
    columnas = [c for c in columnas if c in df.columns]
    df = df[columnas].copy()

    if con_vencimiento:
        df["vencimiento"] = df["symbol"].apply(vencimiento_desde_symbol)
        df["vencimiento"] = pd.to_datetime(df["vencimiento"])
        hoy = pd.Timestamp.today().normalize()
        df["dias_a_vencimiento"] = (df["vencimiento"] - hoy).dt.days
        df = df.sort_values("dias_a_vencimiento", ascending=True)

    return df


def bonos_lista(con_vencimiento: bool = True) -> pd.DataFrame:
    """
    Devuelve solo los bonos target en un DataFrame.
    Si con_vencimiento=True, agrega columnas:
      - vencimiento (fecha)
      - dias_a_vencimiento (int)
    """
    datos = _fetch_json(URL_BONOS)

    # filtrar solo universo target
    datos = [x for x in datos if x.get("symbol") in BONOS_TARGET]

    df = pd.DataFrame(datos)

    columnas = ['symbol', 'c', 'v', 'q_bid', 'px_bid', 'px_ask', 'q_ask', 'q_op', 'pct_change']
    columnas = [c for c in columnas if c in df.columns]
    df = df[columnas].copy()

    if con_vencimiento:
        df["vencimiento"] = df["symbol"].apply(vencimiento_desde_symbol)
        df["vencimiento"] = pd.to_datetime(df["vencimiento"])
        hoy = pd.Timestamp.today().normalize()
        df["dias_a_vencimiento"] = (df["vencimiento"] - hoy).dt.days
        df = df.sort_values("dias_a_vencimiento", ascending=True)

    return df

# =========================
# LISTAS CER (LETRAS X... y BONOS CER)
# =========================

def letras_cer_lista(con_vencimiento: bool = True) -> pd.DataFrame:
    """
    Devuelve las 'letras CER' (tickers que empiezan con X) desde la API de letras (arg_notes).
    """
    datos = _fetch_json(URL_LETRAS)

    # Solo X... (CER) dentro de tu universo
    datos = [x for x in datos if x.get("symbol") in LETRAS_CER_TARGET]

    df = pd.DataFrame(datos)

    columnas = ['symbol', 'c', 'v', 'q_bid', 'px_bid', 'px_ask',
                'q_ask', 'q_op', 'pct_change']
    columnas = [c for c in columnas if c in df.columns]
    df = df[columnas].copy()

    if con_vencimiento and not df.empty:
        df["vencimiento"] = df["symbol"].apply(vencimiento_desde_symbol)
        df["vencimiento"] = pd.to_datetime(df["vencimiento"])
        hoy = pd.Timestamp.today().normalize()
        df["dias_a_vencimiento"] = (df["vencimiento"] - hoy).dt.days
        df = df.sort_values("dias_a_vencimiento", ascending=True)

    return df


def bonos_cer_lista() -> pd.DataFrame:
    """
    Devuelve bonos CER desde la API de bonos (arg_bonds).
    NOTA: muchos CER no siguen el patrón de vencimiento_desde_symbol (TX26, DICP, etc.)
    por eso por ahora NO calculamos vencimiento aquí para no romper.
    """
    datos = _fetch_json(URL_BONOS)

    # Excluimos X... acá porque esos los traemos desde notas (letras_cer_lista)
    datos = [
        x for x in datos
        if x.get("symbol") in BONOS_CER_TARGET and not str(x.get("symbol", "")).upper().startswith("X")
    ]

    df = pd.DataFrame(datos)

    columnas = ['symbol', 'c', 'v', 'q_bid', 'px_bid', 'px_ask',
                'q_ask', 'q_op', 'pct_change']
    columnas = [c for c in columnas if c in df.columns]
    df = df[columnas].copy()


    if not df.empty:
    # vencimiento desde diccionario
        df["vencimiento"] = df["symbol"].apply(lambda s: FECHA_VENCIMIENTO.get(str(s).strip().upper()))
        df["vencimiento"] = pd.to_datetime(df["vencimiento"])

        hoy = pd.Timestamp.today().normalize()
        df["dias_a_vencimiento"] = (df["vencimiento"] - hoy).dt.days


    return df

# =========================
# ARMAR TABLAS: TASA FIJA y CER
# =========================

def instrumentos_tasa_fija():
    df_letras = letras_lista(con_vencimiento=True).copy()
    df_letras["tipo"] = "LETRA"

    df_bonos = bonos_lista(con_vencimiento=True).copy()
    df_bonos["tipo"] = "BONO"

    df = pd.concat([df_letras, df_bonos], ignore_index=True, sort=True)
    df = df.sort_values(["vencimiento", "tipo", "symbol"]).reset_index(drop=True)
    return df



def instrumentos_cer():
    df_letras_cer = letras_cer_lista(con_vencimiento=True).copy()
    df_letras_cer["tipo"] = "LETRA CER"

    df_bonos_cer = bonos_cer_lista().copy()
    df_bonos_cer["tipo"] = "BONO CER"

    df = pd.concat([df_letras_cer, df_bonos_cer], ignore_index=True, sort=True)

    # Orden simple (ya debería existir vencimiento en ambos)
    if "vencimiento" in df.columns:
        df = df.sort_values(["vencimiento", "tipo", "symbol"], ascending=[True, True, True])

    return df.reset_index(drop=True)


# =========================
# FUNCIONES DE TASAS
# =========================

def calcular_tna(row, pagos_finales: dict, base_dias=365):
    """
    Calcula TNA simple para una fila del df_all.
    Requiere:
      - precio 'c'
      - dias_a_vencimiento
      - pago_final cargado manualmente por símbolo
    """
    symbol = row["symbol"]

    if symbol not in pagos_finales:
        return None

    pago_final = pagos_finales[symbol]
    precio = row["c"]
    dias = row["dias_a_vencimiento"]

    if precio is None or precio <= 0 or dias <= 0:
        return None

    return ((pago_final / precio - 1) / (dias - 1) * base_dias ) * 100


def calcular_tir(row, pagos_finales: dict, base_dias=365):
    """
    Calcula la TIR efectiva anual para una fila del df_all.
    """
    symbol = row["symbol"]

    if symbol not in pagos_finales:
        return None

    pago_final = pagos_finales[symbol]
    precio = row["c"]
    dias = row["dias_a_vencimiento"]

    if precio is None or precio <= 0 or dias is None or dias <= 0:
        return None

    return ((pago_final / precio) ** (base_dias / (dias-1)) - 1) * 100


def calcular_tem_desde_tir(row):
    tir_pct = row["TIR (%)"]

    if tir_pct is None or pd.isna(tir_pct):
        return None

    tir = tir_pct / 100   # decimal

    return ((1 + tir) ** (1/12) - 1) * 100

def tir_real_cer(precio: float, factor_cer: float, dias: int, base_dias=365):
    if precio is None or factor_cer is None or dias is None:
        return None
    if pd.isna(precio) or pd.isna(factor_cer) or pd.isna(dias):
        return None

    try:
        precio = float(precio)
        factor_cer = float(factor_cer)
        dias = int(dias)
    except Exception:
        return None

    if precio <= 0 or factor_cer <= 0 or dias <= 0:
        return None

    vf = 100 * factor_cer  # 100 VN
    tir = (vf / precio) ** (base_dias / dias) - 1
    return tir * 100


# =========================
# MAIN APP
# =========================

st.title("Curva de instrumentos en pesos 💸")

try:
    df_tf = instrumentos_tasa_fija()
    df_cer = instrumentos_cer()
except Exception as e:
    st.error(f"Error al cargar datos de instrumentos: {e}")
    df_tf = None
    df_cer = None

# =========================
# CALCULO RENDIMIENTO CER
# =========================

if df_cer is not None and not df_cer.empty and cer_df is not None:

    def _calc(row):
        out = rendimiento_cer_bono(row["symbol"], cer_df, FECHA_EMISION)
        return pd.Series({
            "CER factor": out.get("factor_cer"),
            "CER rend (%)": out.get("rendimiento_cer_pct"),
            "Liq-10": out.get("f_liq_m10"),
            "Emis-10": out.get("f_emis_m10"),
            "err": out.get("error"),
        })

    df_cer = df_cer.copy()
    df_cer[["CER factor","CER rend (%)","Liq-10","Emis-10","err"]] = df_cer.apply(_calc, axis=1)

# =========================
# RENDIMIENTO REAL POR PRECIO
# =========================

if df_cer is not None and not df_cer.empty:
    df_cer["TIR real CER (%)"] = df_cer.apply(
        lambda row: tir_real_cer(
            row.get("c"),
            row.get("CER factor"),
            row.get("dias_a_vencimiento")
        ),
        axis=1
    )



def tir_real_cer(precio: float, factor_cer: float, dias: int, base_dias=365):
    if precio is None or factor_cer is None or dias is None:
        return None
    if pd.isna(precio) or pd.isna(factor_cer) or pd.isna(dias):
        return None

    try:
        precio = float(precio)
        factor_cer = float(factor_cer)
        dias = int(dias)
    except Exception:
        return None

    if precio <= 0 or factor_cer <= 0 or dias <= 0:
        return None

    # ojo unidades: precio está por 100 VN, entonces VF debe ser por 100 VN
    vf = 100 * factor_cer

    tir = (vf / precio) ** (base_dias / dias) - 1
    return tir * 100




if df_tf is not None:


    # Calcular tasas
    df_tf["TNA (%)"] = df_tf.apply(
        lambda row: calcular_tna(row, PAGOS_FINALES),
        axis=1
    )
    df_tf["TIR (%)"] = df_tf.apply(
        lambda row: calcular_tir(row, PAGOS_FINALES),
        axis=1
    )
    df_tf["TEM (%)"] = df_tf.apply(calcular_tem_desde_tir, axis=1)

    # =========================
    # LAYOUT: TABLA (IZQ) Y GRÁFICO (DER)
    # =========================
    col_tabla, col_grafico = st.columns([1.2, 1])

    with col_tabla:
        st.subheader("Tabla de instrumentos TASA FIJA")

        columnas_mostrar = [
            "tipo", "symbol", "c",
            "dias_a_vencimiento",
            "TNA (%)", "TIR (%)", "TEM (%)"
        ]

        # Copia para mostrar
        df_display = df_tf[columnas_mostrar].copy()

        # Aseguramos tipos numéricos y redondeamos
        for col in ["c", "TNA (%)", "TIR (%)", "TEM (%)"]:
            df_display[col] = pd.to_numeric(df_display[col], errors="coerce").round(2)

        # dias_a_vencimiento como entero
        df_display["dias_a_vencimiento"] = pd.to_numeric(
            df_display["dias_a_vencimiento"], errors="coerce"
        ).astype("Int64")

        # Renombrar columnas para mostrar
        df_display = df_display.rename(columns={
            "tipo": "Tipo",
            "symbol": "Ticker",
            "c": "Precio",
            "dias_a_vencimiento": "Días a vencimiento",
            "TNA (%)": "TNA (%)",
            "TIR (%)": "TIR (%)",
            "TEM (%)": "TEM (%)"
        })

        # altura dinámica tabla tasa fija
        row_height = 35
        max_height = 900
        height_tf = min(max_height, 40 + len(df_display) * row_height)

        st.dataframe(
            df_display,
            use_container_width=True,
            height=height_tf
        )


        st.subheader("Tabla instrumentos CER")

        if df_cer is None or df_cer.empty:
            st.info("No se encontraron instrumentos CER.")
        else:
            
            cols_cer = ["tipo","symbol","c","v","pct_change", "dias_a_vencimiento","TIR real CER (%)"]
            cols_cer = [c for c in cols_cer if c in df_cer.columns]

            df_cer_display = df_cer[cols_cer].copy()
            if "c" in df_cer_display.columns:
                df_cer_display["c"] = pd.to_numeric(df_cer_display["c"], errors="coerce").round(2)

            df_cer_display = df_cer_display.rename(columns={
                "tipo": "Tipo",
                "symbol": "Ticker",
                "c": "Precio",
                "v": "Volumen",
                "pct_change": "% Var"
            })

            row_height = 35
            max_height = 900
            height = min(max_height, 40 + len(df_cer_display) * row_height)

            st.dataframe(
                df_cer_display,
                use_container_width=True,
                height=height
            )



    with col_grafico:
        st.subheader("")

        # Selector de tasa
        tasa_elegida = st.selectbox(
            "Tasa a graficar:",
            ["TIR (%)", "TNA (%)", "TEM (%)"],
            index=0
        )

        # Filtrar datos válidos (evitar NaN y días <= 0)
        df_plot = df_tf.dropna(subset=["dias_a_vencimiento", tasa_elegida]).copy()
        df_plot = df_plot[df_plot["dias_a_vencimiento"] > 0]

        x = df_plot["dias_a_vencimiento"].values
        y = df_plot[tasa_elegida].values

        # Ajuste logarítmico
        a, b = np.polyfit(np.log(x), y, 1)
        x_line = np.linspace(x.min(), x.max(), 300)
        y_line = a * np.log(x_line) + b

        # Figura
        fig = go.Figure()

        # Puntos (scatter), separados por tipo
        tipos = df_plot["tipo"].unique()
        colores = {"LETRA": "blue", "BONO": "red"}

        for tipo in tipos:
            sub = df_plot[df_plot["tipo"] == tipo]

            fig.add_trace(go.Scatter(
                x=sub["dias_a_vencimiento"],
                y=sub[tasa_elegida],
                mode="markers",
                name=tipo,
                marker=dict(size=10, opacity=0.8, color=colores.get(tipo, "gray")),
                text=sub["symbol"],  # aparece en hover
                hovertemplate=(
                    "<b>%{text}</b><br><br>"
                    "Días: %{x}<br>"
                    f"{tasa_elegida}: %{{y:.2f}}%<br>"
                    "Precio: %{customdata[0]:.2f}<br>"
                    "Vencimiento: %{customdata[1]}<extra></extra>"
                ),
                customdata=np.stack([
                    sub["c"].round(2),
                    sub["vencimiento"].dt.strftime("%Y-%m-%d")
                ], axis=-1)
            ))

        # Línea de regresión logarítmica
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Regresión logarítmica",
            line=dict(color="purple", width=3, dash="dash")
        ))

        # Layout
        fig.update_layout(
            title=f"Curva {tasa_elegida} ",
            xaxis_title="Días a vencimiento",
            yaxis_title=tasa_elegida,
            hovermode="closest",
            template="plotly_white",
            legend=dict(title="Tipo de instrumento")
        )

        st.plotly_chart(fig, use_container_width=True)


#py -m streamlit run curva.py
#cd "C:\Users\ssegura\OneDrive - BALANZ\Escritorio\curvas"

"""
git add curva.py
git commit -m "Arreglo hovertemplate plotly"
git push

"""

#para modificar excel de CER
"""
git add CER.xlsx
git commit -m "Update CER file"
git push

"""