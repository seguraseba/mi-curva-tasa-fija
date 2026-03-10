import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, date
import plotly.graph_objects as go
from pandas.tseries.offsets import CustomBusinessDay
from pathlib import Path

# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Curva tasa fija pesos",
    layout="wide",  # importante para ver tabla y gráfico lado a lado
)


CER_LOCAL_PATH = Path(r"C:\Users\ssegura\OneDrive - BALANZ\Escritorio\CER.xlsx")
CER_REPO_PATH  = Path(__file__).parent / "CER.xlsx"

CORPOS_LOCAL_PATH = Path(r"C:\Users\ssegura\OneDrive - BALANZ\Escritorio\corpos.xlsx")
CORPOS_REPO_PATH  = Path(__file__).parent / "corpos.xlsx"

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


@st.cache_data
def cargar_corpos(path: Path, file_version: float) -> pd.DataFrame:
    # header=1 porque:
    # fila 1 = título
    # fila 2 = encabezados reales
    df = pd.read_excel(path, sheet_name="Hoja1", header=1, engine="openpyxl")

    # eliminar filas completamente vacías
    df = df.dropna(how="all").reset_index(drop=True)

    # limpiar nombres de columnas
    df.columns = [str(c).strip() for c in df.columns]

    # convertir columnas de fecha si existen
    columnas_fecha_posibles = ["Vencimiento", "Próx. Cupón", "Prox. Cupón", "Próximo Cupón", "Fecha"]

    for col in columnas_fecha_posibles:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

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

# =========================
# CARGA EXCEL CORPORATIVOS
# =========================
if CORPOS_LOCAL_PATH.exists():
    corpos_df = cargar_corpos(CORPOS_LOCAL_PATH, CORPOS_LOCAL_PATH.stat().st_mtime)
    st.sidebar.success("✅ Corporativos cargados desde tu Escritorio")
elif CORPOS_REPO_PATH.exists():
    corpos_df = cargar_corpos(CORPOS_REPO_PATH, CORPOS_REPO_PATH.stat().st_mtime)
    st.sidebar.success("✅ Corporativos cargados desde el repo")
else:
    corpos_df = pd.DataFrame()
    st.sidebar.warning("⚠️ No se encontró corpos.xlsx")

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

def cer_coef_desde_emision(symbol: str, cer_df: pd.DataFrame, fecha_emision_map: dict) -> dict:
    symbol = str(symbol).strip().upper()

    if symbol not in fecha_emision_map:
        return {"error": "No hay fecha de emisión cargada"}

    # liquidación = hoy + 1
    fecha_liq = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1))
    f_liq_m10 = fecha_liq - BDay(10)

    fecha_emis = fecha_emision_map[symbol]
    f_emis_m10 = pd.Timestamp(fecha_emis).normalize() - BDay(10)

    cer_liq = cer_en_o_antes(cer_df, f_liq_m10)
    cer_emis = cer_en_o_antes(cer_df, f_emis_m10)

    if cer_liq is None or cer_emis is None or cer_emis == 0:
        return {"error": "No se encontró CER para alguna fecha (o cer_emis=0)"}

    coef = float(cer_liq) / float(cer_emis)
    vf = 100.0 * coef  # único flujo cupón cero

    return {
        "liq_m10": f_liq_m10.date(),
        "emis_m10": f_emis_m10.date(),
        "cer_liq": float(cer_liq),
        "cer_emis": float(cer_emis),
        "cer_coef": coef,
        "vf_cupon_cero": vf,
        "error": None,
    }


def tir_cer_cupon_cero(precio: float, vf: float, dias: int, base_dias=365) -> float | None:
    if precio is None or vf is None or dias is None:
        return None
    if pd.isna(precio) or pd.isna(vf) or pd.isna(dias):
        return None
    try:
        precio = float(precio)
        vf = float(vf)
        dias = int(dias)
    except Exception:
        return None
    if precio <= 0 or vf <= 0 or dias <= 0:
        return None
    return (((vf / precio) ** (base_dias / dias)) - 1) * 100


def tna_cer_cupon_cero(precio: float, vf: float, dias: int, base_dias=365) -> float | None:
    if precio is None or vf is None or dias is None:
        return None
    if pd.isna(precio) or pd.isna(vf) or pd.isna(dias):
        return None
    try:
        precio = float(precio)
        vf = float(vf)
        dias = int(dias)
    except Exception:
        return None
    if precio <= 0 or vf <= 0 or dias <= 0:
        return None
    return ((vf / precio - 1) / dias) * base_dias * 100

def rendimiento_real_por_precio(precio: float, factor_cer: float) -> float | None:
    if precio is None or factor_cer is None:
        return None
    if precio <= 0:
        return None
    return (precio / factor_cer  - 1) * 100




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
URL_ONS  = "https://data912.com/live/arg_corp"


# =========================
# PARES LEGISLACIÓN
# =========================
PARES_LEGISLACION = [
    {"par": "AL29 / GD29", "ticker_al": "AL29D", "ticker_gd": "GD29D"},
    {"par": "AL30 / GD30", "ticker_al": "AL30D", "ticker_gd": "GD30D"},
    {"par": "AL35 / GD35", "ticker_al": "AL35D", "ticker_gd": "GD35D"},
    {"par": "AE38 / GD38", "ticker_al": "AE38D", "ticker_gd": "GD38D"},
    {"par": "AL41 / GD41", "ticker_al": "AL41D", "ticker_gd": "GD41D"},
]

LETRAS_TARGET = [
    "S30N6", "S16E6", "S27F6","S16M6", "S17A6", "S30A6", "S29Y6", "S31L6", "S31G6", "S30O6", "X29Y6", "X30N6", "X31L6"
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
    "X31L6",
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
    "X31L6": date(2026, 7, 31),
}

# =========================
# REGLAS BONOS CER (MODELO REAL)
# =========================

BOND_RULES = {
    "TX26": {
        "coupon_real": 0.02,
        "freq": 2,              # semestral
        "amort_last_n": 5,
        "daycount": "ACT/ACT",
    },
    "TX28": {
        "coupon_real": 0.0225,
        "freq": 2,
        "amort_last_n": 10,
        "daycount": "ACT/ACT",
    },
    "TX31": {
        "coupon_real": 0.025,
        "freq": 2,
        "amort_last_n": 10,
        "daycount": "ACT/ACT",
    },
    "DICP": {
        "coupon_real": 0.0583,
        "freq": 2,
        "amort_last_n": 20,
        "daycount": "ACT/ACT",
    },
    "CUAP": {
        "coupon_real": 0.0331,
        "freq": 2,
        "amort_last_n": 20,
        "daycount": "ACT/ACT",
    },
}

# =========================
# PARP - CUPONES POR TRAMOS
# =========================

PARP_RULE = {
    "freq": 2,
    "amort_last_n": 20,
    "daycount": "ACT/ACT",
    "coupon_tramos": [
        {"desde": date(2003,12,31), "hasta": date(2009,3,31),  "coupon": 0.0063},
        {"desde": date(2009,3,31),  "hasta": date(2019,3,31),  "coupon": 0.0118},
        {"desde": date(2019,3,31),  "hasta": date(2029,3,31),  "coupon": 0.0177},
        {"desde": date(2029,3,31),  "hasta": date(2038,12,31), "coupon": 0.0248},
    ]
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
    "S16M6":104.62,
    "S30A6":127.49,
    "S29Y6":132.04,
    "S31L6":117.68,
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
    "X31L6": date(2026, 1, 30),
}

CER_ESPECIALES_CON_FLUJOS = {"DICP", "PARP", "CUAP", "TX26", "TX28", "TX31"}

# =========================
# HELPERS PARA API
# =========================

from pandas.tseries.offsets import BDay

def _fetch_json(url: str):
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

from dateutil.relativedelta import relativedelta

def generar_fechas_pago(fecha_emision: date, fecha_venc: date, freq: int):
    meses = int(12 / freq)
    fechas = []
    f = pd.Timestamp(fecha_emision)

    while f < pd.Timestamp(fecha_venc):
        f = f + relativedelta(months=meses)
        if f <= pd.Timestamp(fecha_venc):
            fechas.append(f.normalize())

    return fechas

def generar_flujos_reales(symbol: str, vn=100):
    symbol = symbol.strip().upper()

    if symbol not in FECHA_EMISION or symbol not in FECHA_VENCIMIENTO:
        return None

    fecha_emis = FECHA_EMISION[symbol]
    fecha_venc = FECHA_VENCIMIENTO[symbol]

    # PARP separado
    if symbol == "PARP":
        rule = PARP_RULE
        fechas = generar_fechas_pago(fecha_emis, fecha_venc, rule["freq"])

        n_total = len(fechas)
        n_amort = rule["amort_last_n"]
        amort_por_periodo = vn / n_amort

        flujos = []
        saldo = vn

        for i, f in enumerate(fechas, start=1):

            # cupón por tramo
            coupon = 0.0
            for tramo in rule["coupon_tramos"]:
                if tramo["desde"] <= f.date() < tramo["hasta"]:
                    coupon = tramo["coupon"]
                    break

            interes = saldo * coupon / rule["freq"]

            amort = 0
            if i > n_total - n_amort:
                amort = amort_por_periodo
                saldo -= amort

            flujo = interes + amort

            flujos.append({
                "fecha": f.date(),
                "interes_real": interes,
                "amort_real": amort,
                "flujo_real": flujo,
                "saldo": saldo
            })

        return pd.DataFrame(flujos)

    # Bonos normales
    rule = BOND_RULES.get(symbol)
    if rule is None:
        return None

    fechas = generar_fechas_pago(fecha_emis, fecha_venc, rule["freq"])

    n_total = len(fechas)
    n_amort = rule["amort_last_n"]
    amort_por_periodo = vn / n_amort

    flujos = []
    saldo = vn

    for i, f in enumerate(fechas, start=1):

        interes = saldo * rule["coupon_real"] / rule["freq"]

        amort = 0
        if i > n_total - n_amort:
            amort = amort_por_periodo
            saldo -= amort

        flujo = interes + amort

        flujos.append({
            "fecha": f.date(),
            "interes_real": interes,
            "amort_real": amort,
            "flujo_real": flujo,
            "saldo": saldo
        })

    return pd.DataFrame(flujos)

def xirr_dates(dates, cashflows, guess=0.05):
    dates = pd.to_datetime(pd.Series(dates))
    cfs = pd.Series(cashflows, dtype="float64")

    t0 = dates.iloc[0]
    years = (dates - t0).dt.days / 365.0

    def npv(r):
        return (cfs / (1 + r) ** years).sum()

    def dnpv(r):
        return (-(years) * cfs / (1 + r) ** (years + 1)).sum()

    r = float(guess)
    for _ in range(100):
        f = npv(r)
        df = dnpv(r)
        if abs(df) < 1e-12:
            break
        r_new = r - f / df
        if abs(r_new - r) < 1e-12:
            r = r_new
            break
        r = r_new

    return r


def tir_real_por_flujos(symbol: str, precio_nominal: float, cer_liq: float | None, vn=100):
    symbol = str(symbol).strip().upper()

    if precio_nominal is None or pd.isna(precio_nominal) or float(precio_nominal) <= 0:
        return None

    # Si no tenés cer_liq disponible todavía, podés devolver None.
    # (o setear cer_liq=1 para debug)
    if cer_liq is None or pd.isna(cer_liq) or float(cer_liq) <= 0:
        return None

    cf_df = generar_flujos_reales(symbol, vn=vn)
    if cf_df is None or cf_df.empty:
        return None

    # Asegurar tipos
    cf_df = cf_df.copy()
    cf_df["fecha"] = pd.to_datetime(cf_df["fecha"], errors="coerce")
    cf_df["flujo_real"] = pd.to_numeric(cf_df["flujo_real"], errors="coerce")

    cf_df = cf_df.dropna(subset=["fecha", "flujo_real"])
    if cf_df.empty:
        return None

    # Convertir precio a "real" para consistencia con flujos reales
    precio_real = float(precio_nominal) / float(cer_liq)

    fechas = [pd.Timestamp.today().normalize()] + cf_df["fecha"].tolist()
    flujos = [-precio_real] + cf_df["flujo_real"].astype(float).tolist()

    try:
        tir = xirr_dates(fechas, flujos)
        return float(tir) * 100
    except Exception:
        return None



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

def precios_vivos_bonos_config(df_config: pd.DataFrame) -> pd.DataFrame:
    """
    Toma la configuración cargada en st.session_state['bonos_spread']
    y la cruza con la API de bonos en vivo.
    """

    if df_config is None or df_config.empty:
        return pd.DataFrame()

    # API bonos en vivo
    datos_bonos = _fetch_json(URL_BONOS)
    df_api = pd.DataFrame(datos_bonos)

    if df_api.empty:
        return pd.DataFrame()

    # Normalizar ticker
    df_api["ticker"] = df_api["symbol"].astype(str).str.strip().str.upper()

    # Campos que nos interesa conservar desde la API
    cols_api = [
        "ticker", "symbol", "c", "v", "q_bid", "px_bid",
        "px_ask", "q_ask", "q_op", "pct_change"
    ]
    cols_api = [c for c in cols_api if c in df_api.columns]
    df_api = df_api[cols_api].copy()

    # Normalizar config
    df_cfg = df_config.copy()
    df_cfg["ticker"] = df_cfg["ticker"].astype(str).str.strip().str.upper()

    # Merge
    df_merge = df_cfg.merge(df_api, on="ticker", how="left")

    # Precio seleccionado según tipo_precio
    def _precio_seleccionado(row):
        campo = str(row.get("tipo_precio", "")).strip()
        if campo in ["c", "px_bid", "px_ask"]:
            return row.get(campo)
        return row.get("c")

    df_merge["precio_seleccionado"] = df_merge.apply(_precio_seleccionado, axis=1)

    # Orden sugerido
    cols_finales = [
        "ticker",
        "legislacion",
        "par",
        "tipo_precio",
        "precio_seleccionado",
        "c",
        "px_bid",
        "px_ask",
        "pct_change",
        "v",
        "q_bid",
        "q_ask",
        "q_op",
        "comentario"
    ]
    cols_finales = [c for c in cols_finales if c in df_merge.columns]

    return df_merge[cols_finales].copy()

def tabla_spread_legislacion(precio_col="c") -> pd.DataFrame:
    """
    Arma una tabla de spreads AL/GD usando precios en vivo desde URL_BONOS.

    precio_col puede ser:
    - 'c'
    - 'px_bid'
    - 'px_ask'
    """
    datos = _fetch_json(URL_BONOS)
    df_api = pd.DataFrame(datos)

    if df_api.empty:
        return pd.DataFrame()

    df_api["symbol"] = df_api["symbol"].astype(str).str.strip().str.upper()

    cols_needed = ["symbol", "c", "px_bid", "px_ask", "pct_change", "v"]
    cols_needed = [c for c in cols_needed if c in df_api.columns]
    df_api = df_api[cols_needed].copy()

    df_pairs = pd.DataFrame(PARES_LEGISLACION)

    # merge AL
    df_out = df_pairs.merge(
        df_api,
        left_on="ticker_al",
        right_on="symbol",
        how="left"
    ).rename(columns={
        "c": "c_al",
        "px_bid": "px_bid_al",
        "px_ask": "px_ask_al",
        "pct_change": "pct_change_al",
        "v": "v_al"
    }).drop(columns=["symbol"], errors="ignore")

    # merge GD
    df_out = df_out.merge(
        df_api,
        left_on="ticker_gd",
        right_on="symbol",
        how="left"
    ).rename(columns={
        "c": "c_gd",
        "px_bid": "px_bid_gd",
        "px_ask": "px_ask_gd",
        "pct_change": "pct_change_gd",
        "v": "v_gd"
    }).drop(columns=["symbol"], errors="ignore")

    # elegir columna de precio
    mapa_al = {
        "c": "c_al",
        "px_bid": "px_bid_al",
        "px_ask": "px_ask_al",
    }
    mapa_gd = {
        "c": "c_gd",
        "px_bid": "px_bid_gd",
        "px_ask": "px_ask_gd",
    }

    col_al = mapa_al.get(precio_col, "c_al")
    col_gd = mapa_gd.get(precio_col, "c_gd")

    df_out["precio_al"] = pd.to_numeric(df_out[col_al], errors="coerce")
    df_out["precio_gd"] = pd.to_numeric(df_out[col_gd], errors="coerce")

    df_out["spread_gd_al"] = df_out["precio_gd"] / df_out["precio_al"]
    df_out["prima_pct"] = (df_out["spread_gd_al"] - 1) * 100

    return df_out


def ticker_mep_desde_excel(symbol: str) -> str:
    """
    Convierte el ticker del Excel al ticker MEP para buscar en la API.
    Regla:
    - si termina en O -> reemplaza por D
    - si ya termina en D -> lo deja igual
    - en otros casos, devuelve el mismo ticker
    """
    if symbol is None:
        return ""

    s = str(symbol).strip().upper()

    if not s:
        return ""

    if s.endswith("O"):
        return s[:-1] + "D"

    return s

def completar_precio_dirty_desde_api(df_corpos: pd.DataFrame) -> pd.DataFrame:
    """
    Completa/actualiza 'Precio Dirty (MEP)' usando el precio 'c'
    de la API de obligaciones negociables.
    Regla MEP:
    - si el ticker termina en O, busca el mismo ticker terminado en D
    """

    if df_corpos is None or df_corpos.empty:
        return df_corpos

    df = df_corpos.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # detectar columna ticker
    col_ticker = None
    for c in df.columns:
        if str(c).strip().lower() in ["ticker", "symbol", "especie"]:
            col_ticker = c
            break

    if col_ticker is None:
        return df

    # traer API de ONs
    datos_api = _fetch_json(URL_ONS)
    df_api = pd.DataFrame(datos_api)

    if df_api.empty or "symbol" not in df_api.columns:
        return df

    df_api["symbol"] = df_api["symbol"].astype(str).str.strip().str.upper()

    if "c" in df_api.columns:
        df_api["c"] = pd.to_numeric(df_api["c"], errors="coerce")

    cols_api = [c for c in ["symbol", "c", "pct_change", "v"] if c in df_api.columns]
    df_api = df_api[cols_api].copy()

    # ticker que se va a buscar en la API
    df["ticker_api_mep"] = (
        df[col_ticker]
        .astype(str)
        .str.strip()
        .str.upper()
        .apply(ticker_mep_desde_excel)
    )

    # debug visual
    df["Ticker API MEP"] = df["ticker_api_mep"]

    # merge contra ONs
    df = df.merge(
        df_api,
        left_on="ticker_api_mep",
        right_on="symbol",
        how="left"
    )

    # actualizar Precio Dirty (MEP) con c
    col_dirty = None
    for c in df.columns:
        if str(c).strip().lower() == "precio dirty (mep)":
            col_dirty = c
            break

    if col_dirty is not None and "c" in df.columns:
        df[col_dirty] = df["c"]

    # opcional: guardar también variación/volumen si después los querés mostrar
    if "pct_change" in df.columns and "% Var API" not in df.columns:
        df["% Var API"] = df["pct_change"]

    if "v" in df.columns and "Vol API" not in df.columns:
        df["Vol API"] = df["v"]

    # limpiar auxiliares técnicas, pero dejamos Ticker API MEP para validar
    df = df.drop(
        columns=[c for c in ["ticker_api_mep", "symbol", "c", "pct_change", "v"] if c in df.columns],
        errors="ignore"
    )

    return df

# =========================
# MAIN APP (CON PESTAÑAS)
# =========================

st.title("Curva de instrumentos en pesos")

# --- Cargar universos (una sola vez) ---
try:
    df_tf = instrumentos_tasa_fija()
    df_cer = instrumentos_cer()
except Exception as e:
    st.error(f"Error al cargar datos de instrumentos: {e}")
    df_tf = None
    df_cer = None

# =========================
# CER cupón cero (para BONOS CER y LETRAS CER que NO son especiales)
# =========================
if df_cer is not None and not df_cer.empty and cer_df is not None:

    df_cer = df_cer.copy()
    df_cer["sym_u"] = df_cer["symbol"].astype(str).str.upper()

    # Bonos CER que son "cupón cero": excluimos especiales
    mask_cupon_cero = (
        (df_cer["tipo"].isin(["BONO CER", "LETRA CER"]))
        & (~df_cer["sym_u"].isin(CER_ESPECIALES_CON_FLUJOS))
    )

    def _calc_cupon_cero(row):
        out = cer_coef_desde_emision(row["sym_u"], cer_df, FECHA_EMISION)

        if out.get("error"):
            return pd.Series({
                "CER coef": None,
                "VF CER (cupón cero)": None,
                "CER liq-10": None,
                "CER emis-10": None,
                "CER liq": None,
                "CER emis": None,
                "TIR CER cupón cero (%)": None,
                "TNA CER cupón cero (%)": None,
                "err_cer": out.get("error"),
            })

        precio = row.get("c")
        dias = row.get("dias_a_vencimiento")
        vf = out["vf_cupon_cero"]

        return pd.Series({
            "CER coef": out["cer_coef"],
            "VF CER (cupón cero)": vf,
            "CER liq-10": out["liq_m10"],
            "CER emis-10": out["emis_m10"],
            "CER liq": out["cer_liq"],
            "CER emis": out["cer_emis"],
            "TIR CER cupón cero (%)": tir_cer_cupon_cero(precio, vf, dias),
            "TNA CER cupón cero (%)": tna_cer_cupon_cero(precio, vf, dias),
            "err_cer": None,
        })

    # Calculamos SOLO donde corresponde
    df_cer.loc[mask_cupon_cero, [
        "CER coef", "VF CER (cupón cero)",
        "CER liq-10", "CER emis-10",
        "CER liq", "CER emis",
        "TIR CER cupón cero (%)", "TNA CER cupón cero (%)",
        "err_cer"
    ]] = df_cer.loc[mask_cupon_cero].apply(_calc_cupon_cero, axis=1)

    df_cer = df_cer.drop(columns=["sym_u"])

# =========================
# TIR REAL CER POR FLUJOS (safe, no rompe la app)
# =========================
def _safe_tir(row):
    try:
        return tir_real_por_flujos(
            row.get("symbol"),
            row.get("c"),
            row.get("cer_liq"),
            vn=100
        )
    except Exception:
        return None

# =========================
# TASA FIJA: calcular tasas
# =========================
if df_tf is not None and not df_tf.empty:

    df_tf = df_tf.copy()

    df_tf["TNA (%)"] = df_tf.apply(lambda row: calcular_tna(row, PAGOS_FINALES), axis=1)
    df_tf["TIR (%)"] = df_tf.apply(lambda row: calcular_tir(row, PAGOS_FINALES), axis=1)
    df_tf["TEM (%)"] = df_tf.apply(calcular_tem_desde_tir, axis=1)

# =========================
# HELPERS CARRY TRADE
# =========================
def _get_pago_final(symbol: str) -> float | None:
    s = str(symbol).strip().upper()
    v = PAGOS_FINALES.get(s)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None

def _simular_carry_trade(
    monto_usd: float,
    tc_inicial: float,
    precio: float,
    pago_final: float,
    comision_pct: float,
    tcs_finales: list[float],
):
    """
    Asume que 'precio' y 'pago_final' están en la MISMA unidad (por 100 VN, típico en pesos).
    Compra: ARS = USD * TC0, descuenta comisión, compra instrumento al 'precio',
    cobra 'pago_final' al vencimiento, y convierte a USD con TC final.
    """
    if monto_usd <= 0 or tc_inicial <= 0 or precio <= 0 or pago_final <= 0:
        return None, None

    ars_inicial = monto_usd * tc_inicial
    ars_neto = ars_inicial * (1 - comision_pct / 100.0)

    # Retorno en pesos del instrumento (manteniendo unidad precio/pago_final)
    factor_ars = pago_final / precio
    ars_final = ars_neto * factor_ars

    # break-even TC final para quedar igual en USD
    tc_breakeven = ars_final / monto_usd  # USD_final = ars_final / tc_final = monto_usd => tc_final = ars_final/monto_usd

    rows = []
    for tc_f in tcs_finales:
        if tc_f <= 0:
            continue
        usd_final = ars_final / tc_f
        retorno_usd_pct = (usd_final / monto_usd - 1) * 100.0
        rows.append({
            "TC final": tc_f,
            "ARS inicial": ars_inicial,
            "ARS neto (post comisión)": ars_neto,
            "ARS final (cobro)": ars_final,
            "USD final": usd_final,
            "Retorno USD (%)": retorno_usd_pct,
        })

    out = pd.DataFrame(rows)
    return out, tc_breakeven

# =========================
# SESSION STATE - BONOS SPREAD
# =========================
if "bonos_spread" not in st.session_state:
    st.session_state["bonos_spread"] = pd.DataFrame(columns=[
        "ticker",
        "legislacion",
        "par",
        "tipo_precio",
        "comentario"
    ])


# =========================
# PESTAÑAS
# =========================

tab_curvas, tab_carry, tab_spreads, tab_leg, tab_corpos = st.tabs(
    ["Curvas", "Carry Trade", "Bonos / Spreads", "Spread Legislación", "Corporativos"]
)

# =========================
# TAB 1: CURVAS (TU APP ACTUAL)
# =========================
with tab_curvas:

# --- Layout NUEVO: 2 filas, cada una con tabla (izq) + gráfico (der) ---

# =========================
# FILA 1: TASA FIJA
# =========================
    st.markdown("## Tasa fija")

if df_tf is None or df_tf.empty:
    st.warning("No se encontraron instrumentos tasa fija.")
else:
    col_tf_tabla, col_tf_graf = st.columns([1.2, 1])

    # --- Tabla TF (izquierda) ---
    with col_tf_tabla:
        st.subheader("Tabla de instrumentos TASA FIJA")

        columnas_mostrar = [
            "tipo", "symbol", "c",
            "dias_a_vencimiento",
            "TNA (%)", "TIR (%)", "TEM (%)"
        ]

        df_display = df_tf[columnas_mostrar].copy()

        for col in ["c", "TNA (%)", "TIR (%)", "TEM (%)"]:
            df_display[col] = pd.to_numeric(df_display[col], errors="coerce").round(2)

        df_display["dias_a_vencimiento"] = pd.to_numeric(
            df_display["dias_a_vencimiento"], errors="coerce"
        ).astype("Int64")

        df_display = df_display.rename(columns={
            "tipo": "Tipo",
            "symbol": "Ticker",
            "c": "Precio",
            "dias_a_vencimiento": "Días a vencimiento",
            "TNA (%)": "TNA (%)",
            "TIR (%)": "TIR (%)",
            "TEM (%)": "TEM (%)"
        })

        row_height = 35
        max_height = 650
        height_tf = min(max_height, 40 + len(df_display) * row_height)

        st.dataframe(df_display, use_container_width=True, height=height_tf)

    # --- Gráfico TF (derecha) ---
    with col_tf_graf:
        tasa_elegida = st.selectbox("Tasa a graficar (TF):", ["TIR (%)", "TNA (%)", "TEM (%)"], index=0)

        df_plot = df_tf.dropna(subset=["dias_a_vencimiento", tasa_elegida]).copy()
        df_plot = df_plot[df_plot["dias_a_vencimiento"] > 0]

        if df_plot.empty:
            st.info("No hay puntos suficientes para graficar tasa fija.")
        else:
            x = df_plot["dias_a_vencimiento"].values
            y = df_plot[tasa_elegida].values

            a, b = np.polyfit(np.log(x), y, 1)
            x_line = np.linspace(x.min(), x.max(), 300)
            y_line = a * np.log(x_line) + b

            fig = go.Figure()

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
                    text=sub["symbol"],
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

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Regresión logarítmica",
                line=dict(color="purple", width=3, dash="dash")
            ))

            fig.update_layout(
                title=f"Curva {tasa_elegida} (TF)",
                xaxis_title="Días a vencimiento",
                yaxis_title=tasa_elegida,
                hovermode="closest",
                template="plotly_white",
                legend=dict(title="Tipo de instrumento")
            )

            st.plotly_chart(fig, use_container_width=True)

# =========================
# FILA 2: CER (cupón cero)
# =========================
st.markdown("## Bonos CER")

if df_cer is None or df_cer.empty:
    st.info("No se encontraron instrumentos CER.")
else:
    col_cer_tabla, col_cer_graf = st.columns([1.2, 1])

    # --- Tabla CER (izquierda) ---
    with col_cer_tabla:
        st.subheader("Tabla instrumentos CER")

        cols_cer = [
            "tipo", "symbol", "c", "v", "pct_change", "dias_a_vencimiento",
            "TIR CER cupón cero (%)"
        ]
        cols_cer = [c for c in cols_cer if c in df_cer.columns]

        df_cer_display = df_cer[cols_cer].copy()

        if "c" in df_cer_display.columns:
            df_cer_display["c"] = pd.to_numeric(df_cer_display["c"], errors="coerce").round(2)

        if "TIR CER cupón cero (%)" in df_cer_display.columns:
            df_cer_display["TIR CER cupón cero (%)"] = pd.to_numeric(
                df_cer_display["TIR CER cupón cero (%)"], errors="coerce"
            ).round(4)

        df_cer_display = df_cer_display.rename(columns={
            "tipo": "Tipo",
            "symbol": "Ticker",
            "c": "Precio",
            "v": "Volumen",
            "pct_change": "% Var",
            "dias_a_vencimiento": "Días a vencimiento",
            "TIR CER cupón cero (%)": "TIR CER (%)",
        })

        row_height = 35
        max_height = 650
        height_cer = min(max_height, 40 + len(df_cer_display) * row_height)

        st.dataframe(df_cer_display, use_container_width=True, height=height_cer)

    # --- Gráfico CER (derecha) ---
    with col_cer_graf:
        tir_col = "TIR CER cupón cero (%)"

        df_plot = df_cer.dropna(subset=["dias_a_vencimiento", tir_col]).copy()
        df_plot = df_plot[df_plot["dias_a_vencimiento"] > 0]

        if df_plot.empty:
            st.info("No hay puntos CER con TIR y días a vencimiento.")
        else:
            x = df_plot["dias_a_vencimiento"].astype(float).values
            y = pd.to_numeric(df_plot[tir_col], errors="coerce").astype(float).values

            a, b = np.polyfit(np.log(x), y, 1)
            x_line = np.linspace(x.min(), x.max(), 300)
            y_line = a * np.log(x_line) + b

            fig = go.Figure()

            colores = {"LETRA CER": "blue", "BONO CER": "red"}
            for tipo in df_plot["tipo"].unique():
                sub = df_plot[df_plot["tipo"] == tipo]
                fig.add_trace(go.Scatter(
                    x=sub["dias_a_vencimiento"],
                    y=sub[tir_col],
                    mode="markers",
                    name=tipo,
                    marker=dict(size=10, opacity=0.8, color=colores.get(tipo, "gray")),
                    text=sub["symbol"],
                    hovertemplate=(
                        "<b>%{text}</b><br><br>"
                        "Días: %{x}<br>"
                        "TIR CER: %{y:.2f}%<br>"
                        "Precio: %{customdata[0]:.2f}<br>"
                        "Vencimiento: %{customdata[1]}<extra></extra>"
                    ),
                    customdata=np.stack([
                        pd.to_numeric(sub["c"], errors="coerce").round(2),
                        pd.to_datetime(sub["vencimiento"]).dt.strftime("%Y-%m-%d")
                    ], axis=-1)
                ))

            fig.add_trace(go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                name="Regresión logarítmica",
                line=dict(color="purple", width=3, dash="dash")
            ))

            fig.update_layout(
                title="Curva TIR CER",
                xaxis_title="Días a vencimiento",
                yaxis_title="TIR CER (%)",
                hovermode="closest",
                template="plotly_white",
                legend=dict(title="Tipo de instrumento")
            )

            st.plotly_chart(fig, use_container_width=True)

# =========================
# TAB 2: CARRY TRADE
# =========================
with tab_carry:
    st.subheader("Emulador de Carry Trade (Tasa fija en pesos → retorno en USD)")

    if df_tf is None or df_tf.empty:
        st.warning("No hay universo tasa fija cargado.")
        st.stop()

    # Universo para elegir: solo símbolos que tengan pago_final cargado
    df_univ = df_tf.copy()
    df_univ["pago_final"] = df_univ["symbol"].apply(_get_pago_final)
    df_univ = df_univ.dropna(subset=["pago_final"])

    if df_univ.empty:
        st.warning("No hay instrumentos tasa fija con pago final cargado en PAGOS_FINALES.")
        st.stop()

    colA, colB = st.columns([1, 1])

    with colA:
        ticker = st.selectbox(
            "Instrumento (tasa fija)",
            options=df_univ["symbol"].tolist(),
            index=0
        )

        row = df_univ[df_univ["symbol"] == ticker].iloc[0]

        precio_sugerido = float(row["c"]) if row.get("c") is not None and not pd.isna(row.get("c")) else 0.0
        pago_final_sugerido = float(row["pago_final"])

        precio = st.number_input("Precio (cada 100vn, precio en vivo por default)", value=float(round(precio_sugerido, 4)))
        pago_final = st.number_input("Pago final (cada 100vn, no modificar)", value=float(round(pago_final_sugerido, 4)))

        comision_pct = st.number_input("Comisión (%)", value=0.50, step=0.05)

    with colB:
        monto_usd = st.number_input("Monto a invertir (USD)", value=10000.0, step=500.0)
        tc_inicial = st.number_input("Tipo de cambio inicial (ARS/USD)", value=1100.0, step=10.0)

        st.caption("Escenarios de TC final")
        modo = st.radio("Modo de escenarios", ["Rango", "Manual"], horizontal=True)

        if modo == "Rango":
            tc_min = st.number_input("TC final mínimo", value=float(tc_inicial), step=10.0)
            tc_max = st.number_input("TC final máximo", value=float(tc_inicial * 1.5), step=10.0)
            n_pts = st.slider("Cantidad de puntos (gráfico)", 10, 200, 50)
            tcs_finales = list(np.linspace(tc_min, tc_max, n_pts))
        else:
            txt = st.text_area(
                "Pegá TC finales (uno por línea)",
                value=f"{tc_inicial}\n{tc_inicial*1.10:.2f}\n{tc_inicial*1.20:.2f}"
            )
            tcs_finales = []
            for line in txt.splitlines():
                line = line.strip().replace(",", ".")
                if not line:
                    continue
                try:
                    tcs_finales.append(float(line))
                except Exception:
                    pass

    df_res, tc_be = _simular_carry_trade(
        monto_usd=monto_usd,
        tc_inicial=tc_inicial,
        precio=precio,
        pago_final=pago_final,
        comision_pct=comision_pct,
        tcs_finales=tcs_finales,
    )

    if df_res is None or df_res.empty:
        st.error("Revisá inputs: monto, TC, precio y pago final deben ser > 0, y escenarios válidos.")
        st.stop()

    # KPIs
    st.markdown("### Resumen")
    k1, k2, k3 = st.columns(3)
    k1.metric("TC break-even (ARS/USD)", f"{tc_be:,.2f}")
    k2.metric("Factor ARS (pago_final/precio)", f"{(pago_final/precio):.6f}")
    k3.metric("ARS final / USD inicial", f"{( (monto_usd*tc_inicial)*(1-comision_pct/100)*(pago_final/precio) / monto_usd ):,.2f}")

    st.markdown("### Resultados por escenario")
    df_show = df_res.copy()
    for c in ["TC final", "ARS inicial", "ARS neto (post comisión)", "ARS final (cobro)", "USD final", "Retorno USD (%)"]:
        df_show[c] = pd.to_numeric(df_show[c], errors="coerce")

    st.dataframe(
        df_show[["TC final", "USD final", "Retorno USD (%)", "ARS neto (post comisión)", "ARS final (cobro)"]]
            .round({"TC final": 2, "USD final": 2, "Retorno USD (%)": 4, "ARS neto (post comisión)": 2, "ARS final (cobro)": 2}),
        use_container_width=True,
        height=min(900, 40 + 35 * len(df_show))
    )

    st.markdown("### Sensibilidad: retorno USD vs TC final")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df_res["TC final"],
        y=df_res["Retorno USD (%)"],
        mode="lines+markers",
        name="Retorno USD (%)",
        hovertemplate="TC final: %{x:.2f}<br>Retorno USD: %{y:.4f}%<extra></extra>"
    ))
    fig2.add_vline(x=tc_be, line_dash="dash", annotation_text="Break-even", annotation_position="top right")
    fig2.update_layout(
        xaxis_title="Tipo de cambio final (ARS/USD)",
        yaxis_title="Retorno en USD (%)",
        template="plotly_white",
        hovermode="closest"
    )
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Notas de supuestos (importante)"):
        st.write(
            "- Este módulo asume que 'precio' y 'pago_final' están en la MISMA unidad (típicamente por 100 VN).\n"
            "- La comisión se descuenta solo al inicio (compra).\n"
        )

        
# =========================
# TAB 3: BONOS / SPREADS
# =========================
with tab_spreads:
    st.subheader("Configuración de bonos para análisis de spreads")

    st.markdown(
        "En esta pestaña podés armar el universo de bonos que después "
        "vamos a usar para cruzar con tu Excel histórico y calcular spreads / percentiles."
    )

    col_form, col_tabla = st.columns([1, 1.4])

    # -------------------------------------------------
    # FORMULARIO DE ALTA
    # -------------------------------------------------
    with col_form:
        st.markdown("### Agregar bono")

        with st.form("form_agregar_bono", clear_on_submit=True):
            ticker = st.text_input("Ticker", value="").strip().upper()

            legislacion = st.selectbox(
                "Legislación",
                options=["Ley local", "Ley NY", "Otra"]
            )

            tipo_precio = st.selectbox(
                "Campo de precio a usar",
                options=["c", "px_bid", "px_ask"],
                index=0,
                help="c = último precio, px_bid = bid, px_ask = ask"
            )

            par = st.text_input(
                "Par o grupo",
                value="",
                help="Ej: GD30-AL30, GD35-AL35, Globales 2030, etc."
            ).strip().upper()

            comentario = st.text_input(
                "Comentario",
                value="",
                help="Campo opcional"
            ).strip()

            agregar = st.form_submit_button("Agregar bono")

            if agregar:
                if not ticker:
                    st.warning("Ingresá un ticker antes de agregar.")
                else:
                    df_actual = st.session_state["bonos_spread"].copy()

                    nuevo = pd.DataFrame([{
                        "ticker": ticker,
                        "legislacion": legislacion,
                        "par": par,
                        "tipo_precio": tipo_precio,
                        "comentario": comentario
                    }])

                    # evitar duplicados exactos por ticker
                    if not df_actual.empty:
                        ya_existe = df_actual["ticker"].astype(str).str.upper().eq(ticker).any()
                    else:
                        ya_existe = False

                    if ya_existe:
                        st.warning(f"El ticker {ticker} ya fue cargado.")
                    else:
                        st.session_state["bonos_spread"] = pd.concat(
                            [df_actual, nuevo],
                            ignore_index=True
                        )
                        st.success(f"{ticker} agregado correctamente.")

        st.markdown("### Acciones")

        if st.button("Limpiar toda la lista"):
            st.session_state["bonos_spread"] = pd.DataFrame(columns=[
                "ticker",
                "legislacion",
                "par",
                "tipo_precio",
                "comentario"
            ])
            st.success("Se limpió la lista de bonos.")

        df_export = st.session_state["bonos_spread"].copy()
        if not df_export.empty:
            csv_data = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Descargar lista en CSV",
                data=csv_data,
                file_name="bonos_spreads_config.csv",
                mime="text/csv"
            )

    # -------------------------------------------------
    # TABLA DE BONOS CARGADOS
    # -------------------------------------------------
    with col_tabla:
        st.markdown("### Bonos cargados")

        df_bonos_spread = st.session_state["bonos_spread"].copy()

        if df_bonos_spread.empty:
            st.info("Todavía no cargaste bonos.")
        else:
            df_bonos_spread.index = range(1, len(df_bonos_spread) + 1)
            st.dataframe(df_bonos_spread, use_container_width=True, height=min(600, 40 + 35 * len(df_bonos_spread)))

            st.markdown("### Eliminar bono cargado")

            opciones_delete = [
                f"{row.ticker} | {row.legislacion} | {row.par}"
                for _, row in df_bonos_spread.reset_index(drop=True).iterrows()
            ]

            seleccion = st.selectbox(
                "Seleccioná una fila para eliminar",
                options=opciones_delete
            )

            if st.button("Eliminar seleccionado"):
                idx = opciones_delete.index(seleccion)
                df_tmp = st.session_state["bonos_spread"].copy().reset_index(drop=True)
                eliminado = df_tmp.loc[idx, "ticker"]
                df_tmp = df_tmp.drop(index=idx).reset_index(drop=True)
                st.session_state["bonos_spread"] = df_tmp
                st.success(f"Se eliminó {eliminado}.")
                st.rerun()

    # -------------------------------------------------
    # RESUMEN
    # -------------------------------------------------
    st.markdown("### Resumen")

    df_resumen = st.session_state["bonos_spread"].copy()

    if not df_resumen.empty:
        c1, c2, c3 = st.columns(3)

        c1.metric("Cantidad de bonos", len(df_resumen))
        c2.metric("Ley local", int((df_resumen["legislacion"] == "Ley local").sum()))
        c3.metric("Ley NY", int((df_resumen["legislacion"] == "Ley NY").sum()))

        if df_resumen["par"].astype(str).str.strip().ne("").any():
            resumen_par = (
                df_resumen[df_resumen["par"].astype(str).str.strip() != ""]
                .groupby("par", as_index=False)
                .size()
                .rename(columns={"size": "cantidad"})
                .sort_values("cantidad", ascending=False)
            )
            st.markdown("### Agrupación por par")
            st.dataframe(resumen_par, use_container_width=True, hide_index=True)

    # -------------------------------------------------
    # PRECIOS EN VIVO
    # -------------------------------------------------
    st.markdown("### Precios en vivo")

    df_live = precios_vivos_bonos_config(st.session_state["bonos_spread"])

    if df_live.empty:
        st.info("No hay bonos cargados para consultar precios en vivo.")
    else:
        df_live_show = df_live.copy()

        # Redondeos
        for col in ["precio_seleccionado", "c", "px_bid", "px_ask", "pct_change"]:
            if col in df_live_show.columns:
                df_live_show[col] = pd.to_numeric(df_live_show[col], errors="coerce").round(4)

        for col in ["v", "q_bid", "q_ask", "q_op"]:
            if col in df_live_show.columns:
                df_live_show[col] = pd.to_numeric(df_live_show[col], errors="coerce")

        df_live_show = df_live_show.rename(columns={
            "ticker": "Ticker",
            "legislacion": "Legislación",
            "par": "Par",
            "tipo_precio": "Campo precio",
            "precio_seleccionado": "Precio usado",
            "c": "Último",
            "px_bid": "Bid",
            "px_ask": "Ask",
            "pct_change": "% Var",
            "v": "Volumen",
            "q_bid": "Cant bid",
            "q_ask": "Cant ask",
            "q_op": "Operaciones",
            "comentario": "Comentario"
        })

        st.dataframe(
            df_live_show,
            use_container_width=True,
            height=min(700, 40 + 35 * len(df_live_show))
        )

# =========================
# TAB 4: SPREAD LEGISLACIÓN
# =========================
with tab_leg:
    st.subheader("Spread legislación en vivo")

    precio_base = st.selectbox(
        "Campo de precio",
        options=["c", "px_bid", "px_ask"],
        index=0,
        help="c = último precio, px_bid = bid, px_ask = ask"
    )

    try:
        df_leg = tabla_spread_legislacion(precio_col=precio_base)
    except Exception as e:
        st.error(f"Error al cargar spread de legislación: {e}")
        df_leg = pd.DataFrame()

    if df_leg.empty:
        st.info("No se pudieron cargar datos para los pares de legislación.")
    else:
        df_show = df_leg[[
            "par",
            "ticker_al",
            "precio_al",
            "ticker_gd",
            "precio_gd",
            "prima_pct"
        ]].copy()

        for col in ["precio_al", "precio_gd", "prima_pct"]:
            df_show[col] = pd.to_numeric(df_show[col], errors="coerce")

        df_show["precio_al"] = df_show["precio_al"].round(2)
        df_show["precio_gd"] = df_show["precio_gd"].round(2)
        df_show["prima_pct"] = df_show["prima_pct"].round(2)

        df_show = df_show.rename(columns={
            "par": "Par",
            "ticker_al": "Ticker AL",
            "precio_al": "Precio AL",
            "ticker_gd": "Ticker GD",
            "precio_gd": "Precio GD",
            "prima_pct": "Spread %"
        })

        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True,
            height=min(500, 40 + 35 * len(df_show))
        )

# =========================
# TAB 5: CORPORATIVOS
# =========================
# =========================
# TAB 5: CORPORATIVOS
# =========================
with tab_corpos:
    st.subheader("Bonos corporativos en dólares")

    if corpos_df is None or corpos_df.empty:
        st.info("No se pudo cargar la tabla de corporativos.")
    else:
        df_corpos_show = completar_precio_dirty_desde_api(corpos_df)

        # eliminar columnas que no queremos mostrar
        cols_eliminar = [
            "Precio Clean (MEP)",
            "TIR Efectiva",
            "TNA",
            "CY",
            "MD",
            "YTW (TNA)"
        ]

        df_corpos_show = df_corpos_show.drop(
            columns=[c for c in cols_eliminar if c in df_corpos_show.columns],
            errors="ignore"
        )

        # Limpiar nombres de columnas por si vienen con espacios
        df_corpos_show.columns = [str(c).strip() for c in df_corpos_show.columns]

        # -------------------------
        # FILTROS
        # -------------------------
        col_f1, col_f2 = st.columns(2)

        # Detectar nombres de columnas esperados
        col_moneda = None
        col_ley = None

        for c in df_corpos_show.columns:
            c_norm = str(c).strip().lower()
            if c_norm in ["moneda pago", "moneda", "currency"]:
                col_moneda = c
            if c_norm in ["ley", "law"]:
                col_ley = c

        with col_f1:
            if col_moneda is not None:
                monedas = sorted(
                    [x for x in df_corpos_show[col_moneda].dropna().astype(str).str.strip().unique() if x != ""]
                )
                opciones_moneda = ["Todas"] + monedas
                moneda_sel = st.selectbox("Filtrar por moneda", opciones_moneda, index=0)
            else:
                moneda_sel = "Todas"
                st.caption("No se encontró columna de moneda.")

        with col_f2:
            if col_ley is not None:
                leyes = sorted(
                    [x for x in df_corpos_show[col_ley].dropna().astype(str).str.strip().unique() if x != ""]
                )
                opciones_ley = ["Todas"] + leyes
                ley_sel = st.selectbox("Filtrar por ley", opciones_ley, index=0)
            else:
                ley_sel = "Todas"
                st.caption("No se encontró columna de ley.")

        # -------------------------
        # APLICAR FILTROS
        # -------------------------
        if col_moneda is not None and moneda_sel != "Todas":
            df_corpos_show = df_corpos_show[
                df_corpos_show[col_moneda].astype(str).str.strip() == moneda_sel
            ]

        if col_ley is not None and ley_sel != "Todas":
            df_corpos_show = df_corpos_show[
                df_corpos_show[col_ley].astype(str).str.strip() == ley_sel
            ]
            

        # -------------------------
        # TABLA
        # -------------------------
        st.markdown(f"### Resultados: {len(df_corpos_show)} bono(s)")

        # asegurar que las columnas de fecha sean datetime
        for col in ["Vencimiento", "Próx. Cupón", "Prox. Cupón", "Próximo Cupón", "Fecha"]:
            if col in df_corpos_show.columns:
                try:
                    df_corpos_show[col] = pd.to_datetime(
                        df_corpos_show[col],
                        errors="coerce",
                        dayfirst=True
                    ).dt.date
                except Exception:
                    pass

        st.dataframe(
            df_corpos_show,
            use_container_width=True,
            hide_index=True,
            height=min(900, 40 + 35 * len(df_corpos_show))
        )


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

