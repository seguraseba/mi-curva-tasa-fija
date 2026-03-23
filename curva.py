import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, date, timedelta
import plotly.graph_objects as go
from pandas.tseries.offsets import CustomBusinessDay
from pathlib import Path
import calendar


# =========================
# CONFIG STREAMLIT
# =========================
st.set_page_config(
    page_title="Curva tasa fija pesos",
    layout="wide",  # importante para ver tabla y gráfico lado a lado
)

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# BUSQUEDA DE ARCHIVOS (MULTI PC)
# =========================

CER_PATHS = [
    Path.home() / "OneDrive - BALANZ" / "Escritorio" / "CER.xlsx",
    Path(__file__).parent / "CER.xlsx",
]

CORPOS_PATHS = [
    Path.home() / "OneDrive - BALANZ" / "Escritorio" / "corpos.xlsx",
    Path(__file__).parent / "corpos.xlsx",
]


def buscar_archivo(paths: list[Path]) -> Path | None:
    """
    Busca un archivo en una lista de rutas posibles
    y devuelve la primera que exista.
    """
    for path in paths:
        if path.exists():
            return path
    return None

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

# =========================
# CARGA CER
# =========================

cer_file = buscar_archivo(CER_PATHS)

if cer_file is not None:
    cer_df = cargar_cer(cer_file, cer_file.stat().st_mtime)
else:
    st.error("❌ No se encontró CER.xlsx en ninguna ruta configurada.")
    st.stop()

# =========================
# CARGA EXCEL CORPORATIVOS
# =========================
# =========================
# CARGA CORPORATIVOS
# =========================

corpos_file = buscar_archivo(CORPOS_PATHS)

if corpos_file is not None:
    corpos_df = cargar_corpos(corpos_file, corpos_file.stat().st_mtime)
else:
    corpos_df = pd.DataFrame()

from pandas.tseries.offsets import BDay

def menos_10_habiles(d: date) -> pd.Timestamp:
    return (pd.Timestamp(d).normalize() - 10 * ARG_BDAY)

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
    f_liq_m10 = fecha_liq - 10 * ARG_BDAY

    fecha_emis = fecha_emision_map[symbol]
    f_emis_m10 = pd.Timestamp(fecha_emis).normalize() - 10 * ARG_BDAY

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
    f_liq_m10 = fecha_liq - 10 * ARG_BDAY

    fecha_emis = fecha_emision_map[symbol]
    f_emis_m10 = pd.Timestamp(fecha_emis).normalize() - 10 * ARG_BDAY

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


from pandas.tseries.offsets import CustomBusinessDay

feriados_arg = [
    "2026-01-01",
    "2026-02-16",
    "2026-02-17",
    "2026-03-23",
    "2026-03-24",
    "2026-04-02",
    "2026-04-03",
    "2026-05-01",
    "2026-05-25",
    "2026-06-15",
    "2026-06-20",
    "2026-07-09",
    "2026-07-10",
    "2026-08-17",
    "2026-10-12",
    "2026-11-23",
    "2026-12-07",
    "2026-12-08",
    "2026-12-24",
    "2026-12-25",
    "2026-12-31"
]

feriados_arg = pd.to_datetime(feriados_arg)

ARG_BDAY = CustomBusinessDay(holidays=feriados_arg)


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

# =========================
# PARES INFLACION IMPLICITA
# =========================
PARES_INFLACION_IMPLICITA = [
    {"ticker_fija": "S29Y6", "ticker_cer": "X29Y6", "par": "S29Y6 / X29Y6"},
    {"ticker_fija": "T30J6", "ticker_cer": "TZX26", "par": "T30J6 / TZX26"},
    {"ticker_fija": "S31L6", "ticker_cer": "X31L6", "par": "S31L6 / X31L6"},
    {"ticker_fija": "S30N6", "ticker_cer": "X30N6", "par": "S30N6 / X30N6"},
    {"ticker_fija": "T30A7", "ticker_cer": "TZXA7", "par": "T30A7 / TZXA7"},
]

LETRAS_TARGET = [
    "S30N6", "S16E6", "S27F6","S16M6", "S17A6", "S30A6", "S29Y6", "S31L6", "S31G6", "S30O6", "X29Y6","X15Y6", "X30N6", "X31L6"
]

BONOS_TARGET = [
    "T13F6",
    "T30J6",
    "T15E7",
    "T30A7",
    "T31Y7",
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
    "X15Y6",
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
    "X15Y6": date(2026, 5, 15),
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
    "X15Y6": date(2026, 2, 27),
    "X29Y6": date(2025, 11, 28),
    "X30N6": date(2025, 12, 15),
    "X31L6": date(2026, 1, 30),
}

CER_ESPECIALES_CON_FLUJOS = {"DICP", "PARP", "CUAP", "TX26", "TX28", "TX31"}

# =========================
# HELPERS PARA API
# =========================

from datetime import date

BOND_RULES = {

    # ── A. AL30 – Bono USD Step Up 2030 Ley Argentina ──────────────────────
    "AL30": {
        "full_name": "Bonos de la República Argentina en USD Step Up 2030 – Ley Argentina",
        "currency": "USD",
        "issue_date": date(2020, 9, 4),
        "maturity": date(2030, 7, 9),
        "frequency": 2,                  # semestral
        "day_count": "30/360",
        "min_denomination": 1.0,
        "governing_law": "Argentina",
        "tipo": "step_up",
        # Tasas step-up: (fecha_desde_inclusive, fecha_hasta_exclusive, tasa_anual)
        "coupon_schedule": [
            (date(2020, 9, 4),  date(2021, 7, 9),  0.00125),
            (date(2021, 7, 9),  date(2023, 7, 9),  0.00500),
            (date(2023, 7, 9),  date(2027, 7, 9),  0.00750),
            (date(2027, 7, 9),  date(2030, 7, 9),  0.01750),
        ],
        "first_coupon_date": date(2021, 7, 9),
        "coupon_dates": ("01-09", "07-09"),   # día-mes de cada pago (ene y jul)
        # Amortización: 13 cuotas semestrales
        # 1ra cuota (4%) el 09/07/2024, luego 12 cuotas de 8% hasta 09/07/2030
        "amortization_schedule": (
            [(date(2024, 7, 9), 0.04)] +
            [(date(2025, 1, 9), 0.08), (date(2025, 7, 9), 0.08),
             (date(2026, 1, 9), 0.08), (date(2026, 7, 9), 0.08),
             (date(2027, 1, 9), 0.08), (date(2027, 7, 9), 0.08),
             (date(2028, 1, 9), 0.08), (date(2028, 7, 9), 0.08),
             (date(2029, 1, 9), 0.08), (date(2029, 7, 9), 0.08),
             (date(2030, 1, 9), 0.08), (date(2030, 7, 9), 0.08)]
        ),
    },

    # ── B. AL35 – Bono USD Step Up 2035 Ley Argentina ──────────────────────
    "AL35": {
        "full_name": "Bonos de la República Argentina en USD Step Up 2035 – Ley Argentina",
        "currency": "USD",
        "issue_date": date(2020, 9, 4),
        "maturity": date(2035, 7, 9),
        "frequency": 2,
        "day_count": "30/360",
        "min_denomination": 1.0,
        "governing_law": "Argentina",
        "tipo": "step_up",
        "coupon_schedule": [
            (date(2020, 9, 4),  date(2021, 7, 9),  0.00125),
            (date(2021, 7, 9),  date(2022, 7, 9),  0.01125),
            (date(2022, 7, 9),  date(2023, 7, 9),  0.01500),
            (date(2023, 7, 9),  date(2024, 7, 9),  0.03625),
            (date(2024, 7, 9),  date(2027, 7, 9),  0.04125),
            (date(2027, 7, 9),  date(2028, 7, 9),  0.04750),
            (date(2028, 7, 9),  date(2035, 7, 9),  0.05000),
        ],
        "first_coupon_date": date(2021, 7, 9),
        "coupon_dates": ("01-09", "07-09"),
        # 10 cuotas semestrales iguales (10%) desde 09/01/2031 hasta 09/07/2035
        "amortization_schedule": [
            (date(2031, 1, 9), 0.10), (date(2031, 7, 9), 0.10),
            (date(2032, 1, 9), 0.10), (date(2032, 7, 9), 0.10),
            (date(2033, 1, 9), 0.10), (date(2033, 7, 9), 0.10),
            (date(2034, 1, 9), 0.10), (date(2034, 7, 9), 0.10),
            (date(2035, 1, 9), 0.10), (date(2035, 7, 9), 0.10),
        ],
    },

    # ── C. AE38 – Bono USD Step Up 2038 Ley Argentina ──────────────────────
    "AE38": {
        "full_name": "Bonos de la República Argentina en USD Step Up 2038 – Ley Argentina",
        "currency": "USD",
        "issue_date": date(2020, 9, 4),
        "maturity": date(2038, 1, 9),
        "frequency": 2,
        "day_count": "30/360",
        "min_denomination": 1.0,
        "governing_law": "Argentina",
        "tipo": "step_up",
        "coupon_schedule": [
            (date(2020, 9, 4),  date(2021, 7, 9),  0.00125),
            (date(2021, 7, 9),  date(2022, 7, 9),  0.02000),
            (date(2022, 7, 9),  date(2023, 7, 9),  0.03875),
            (date(2023, 7, 9),  date(2024, 7, 9),  0.04250),
            (date(2024, 7, 9),  date(2038, 1, 9),  0.05000),
        ],
        "first_coupon_date": date(2021, 7, 9),
        "coupon_dates": ("01-09", "07-09"),
        # 22 cuotas semestrales iguales (~4.545%) desde 09/07/2027 hasta 09/01/2038
        "amortization_schedule": [
            (date(2027, 7, 9),  round(1/22, 6)),
            (date(2028, 1, 9),  round(1/22, 6)),
            (date(2028, 7, 9),  round(1/22, 6)),
            (date(2029, 1, 9),  round(1/22, 6)),
            (date(2029, 7, 9),  round(1/22, 6)),
            (date(2030, 1, 9),  round(1/22, 6)),
            (date(2030, 7, 9),  round(1/22, 6)),
            (date(2031, 1, 9),  round(1/22, 6)),
            (date(2031, 7, 9),  round(1/22, 6)),
            (date(2032, 1, 9),  round(1/22, 6)),
            (date(2032, 7, 9),  round(1/22, 6)),
            (date(2033, 1, 9),  round(1/22, 6)),
            (date(2033, 7, 9),  round(1/22, 6)),
            (date(2034, 1, 9),  round(1/22, 6)),
            (date(2034, 7, 9),  round(1/22, 6)),
            (date(2035, 1, 9),  round(1/22, 6)),
            (date(2035, 7, 9),  round(1/22, 6)),
            (date(2036, 1, 9),  round(1/22, 6)),
            (date(2036, 7, 9),  round(1/22, 6)),
            (date(2037, 1, 9),  round(1/22, 6)),
            (date(2037, 7, 9),  round(1/22, 6)),
            (date(2038, 1, 9),  round(1/22, 6)),
        ],
    },

    # ── D. AL41 – Bono USD Step Up 2041 Ley Argentina ──────────────────────
    "AL41": {
        "full_name": "Bonos de la República Argentina en USD Step Up 2041 – Ley Argentina",
        "currency": "USD",
        "issue_date": date(2020, 9, 4),
        "maturity": date(2041, 7, 9),
        "frequency": 2,
        "day_count": "30/360",
        "min_denomination": 1.0,
        "governing_law": "Argentina",
        "tipo": "step_up",
        "coupon_schedule": [
            (date(2020, 9, 4),  date(2021, 7, 9),  0.00125),
            (date(2021, 7, 9),  date(2022, 7, 9),  0.02500),
            (date(2022, 7, 9),  date(2029, 7, 9),  0.03500),
            (date(2029, 7, 9),  date(2041, 7, 9),  0.04875),
        ],
        "first_coupon_date": date(2021, 7, 9),
        "coupon_dates": ("01-09", "07-09"),
        # 28 cuotas semestrales iguales (~3.571%) desde 09/01/2028 hasta 09/07/2041
        "amortization_schedule": [
            (date(2028, 1, 9),  round(1/28, 6)),
            (date(2028, 7, 9),  round(1/28, 6)),
            (date(2029, 1, 9),  round(1/28, 6)),
            (date(2029, 7, 9),  round(1/28, 6)),
            (date(2030, 1, 9),  round(1/28, 6)),
            (date(2030, 7, 9),  round(1/28, 6)),
            (date(2031, 1, 9),  round(1/28, 6)),
            (date(2031, 7, 9),  round(1/28, 6)),
            (date(2032, 1, 9),  round(1/28, 6)),
            (date(2032, 7, 9),  round(1/28, 6)),
            (date(2033, 1, 9),  round(1/28, 6)),
            (date(2033, 7, 9),  round(1/28, 6)),
            (date(2034, 1, 9),  round(1/28, 6)),
            (date(2034, 7, 9),  round(1/28, 6)),
            (date(2035, 1, 9),  round(1/28, 6)),
            (date(2035, 7, 9),  round(1/28, 6)),
            (date(2036, 1, 9),  round(1/28, 6)),
            (date(2036, 7, 9),  round(1/28, 6)),
            (date(2037, 1, 9),  round(1/28, 6)),
            (date(2037, 7, 9),  round(1/28, 6)),
            (date(2038, 1, 9),  round(1/28, 6)),
            (date(2038, 7, 9),  round(1/28, 6)),
            (date(2039, 1, 9),  round(1/28, 6)),
            (date(2039, 7, 9),  round(1/28, 6)),
            (date(2040, 1, 9),  round(1/28, 6)),
            (date(2040, 7, 9),  round(1/28, 6)),
            (date(2041, 1, 9),  round(1/28, 6)),
            (date(2041, 7, 9),  round(1/28, 6)),
        ],
    },

    # ── E. TX26 – BONCER 2026 2.00% ────────────────────────────────────────
    "TX26": {
        "full_name": "Bonos del Tesoro Nacional en Pesos con Ajuste por CER 2,00% Vto. 2026",
        "currency": "ARS",
        "issue_date": date(2020, 9, 4),
        "maturity": date(2026, 11, 9),
        "frequency": 2,
        "day_count": "30/360",
        "min_denomination": 1.0,
        "governing_law": "Argentina",
        "tipo": "CER",
        "coupon": 0.02,                  # tasa fija real sobre capital ajustado CER
        "cer_adjusted": True,
        # CER: t-10 hábiles desde emisión hasta t-10 hábiles antes de cada vencimiento
        "cer_lag_business_days": 10,
        "first_coupon_date": date(2021, 5, 9),
        "coupon_dates": ("05-09", "11-09"),   # mayo y noviembre
        # 5 cuotas semestrales iguales (20%) desde 09/11/2024 hasta 09/11/2026
        "amortization_schedule": [
            (date(2024, 11, 9), 0.20),
            (date(2025, 5,  9), 0.20),
            (date(2025, 11, 9), 0.20),
            (date(2026, 5,  9), 0.20),
            (date(2026, 11, 9), 0.20),
        ],
    },

    # ── F. TX28 – BONCER 2028 2.25% ────────────────────────────────────────
    "TX28": {
        "full_name": "Bonos del Tesoro Nacional en Pesos con Ajuste por CER 2,25% Vto. 2028",
        "currency": "ARS",
        "issue_date": date(2020, 9, 4),
        "maturity": date(2028, 11, 9),
        "frequency": 2,
        "day_count": "30/360",
        "min_denomination": 1.0,
        "governing_law": "Argentina",
        "tipo": "CER",
        "coupon": 0.0225,
        "cer_adjusted": True,
        "cer_lag_business_days": 10,
        "first_coupon_date": date(2021, 5, 9),
        "coupon_dates": ("05-09", "11-09"),
        # 10 cuotas semestrales iguales (10%) desde 09/05/2024 hasta 09/11/2028
        "amortization_schedule": [
            (date(2024, 5,  9), 0.10),
            (date(2024, 11, 9), 0.10),
            (date(2025, 5,  9), 0.10),
            (date(2025, 11, 9), 0.10),
            (date(2026, 5,  9), 0.10),
            (date(2026, 11, 9), 0.10),
            (date(2027, 5,  9), 0.10),
            (date(2027, 11, 9), 0.10),
            (date(2028, 5,  9), 0.10),
            (date(2028, 11, 9), 0.10),
        ],
    },

    # ── G. AL29 – Bono USD 1% 2029 Ley Argentina ───────────────────────────
    "AL29": {
        "full_name": "Bonos de la República Argentina en USD al 1% 2029 – Ley Argentina",
        "currency": "USD",
        "issue_date": date(2020, 9, 4),
        "maturity": date(2029, 7, 9),
        "frequency": 2,
        "day_count": "30/360",
        "min_denomination": 1.0,
        "governing_law": "Argentina",
        "tipo": "tasa_fija",
        "coupon": 0.01,                  # tasa flat, sin step-up
        "coupon_schedule": [
            (date(2020, 9, 4), date(2029, 7, 9), 0.01),
        ],
        "first_coupon_date": date(2021, 7, 9),
        "coupon_dates": ("01-09", "07-09"),
        # 10 cuotas semestrales iguales (10%) desde 09/01/2025 hasta 09/07/2029
        "amortization_schedule": [
            (date(2025, 1, 9), 0.10),
            (date(2025, 7, 9), 0.10),
            (date(2026, 1, 9), 0.10),
            (date(2026, 7, 9), 0.10),
            (date(2027, 1, 9), 0.10),
            (date(2027, 7, 9), 0.10),
            (date(2028, 1, 9), 0.10),
            (date(2028, 7, 9), 0.10),
            (date(2029, 1, 9), 0.10),
            (date(2029, 7, 9), 0.10),
        ],
    },
}

SOBERANOS_API_MAP = {
    "AL29D": "AL29",
    "AL30D": "AL30",
    "AL35D": "AL35",
    "AE38D": "AE38",
    "AL41D": "AL41",
}

def accrued_interest(symbol, rules, fecha_val):
    freq = rules["frequency"]

    # convertir fecha_val a date
    if isinstance(fecha_val, pd.Timestamp):
        fecha_val = fecha_val.date()

    fechas = sorted([f for f, _ in rules["amortization_schedule"]])

    prev_coupon = max([f for f in fechas if f <= fecha_val], default=None)
    next_coupon = min([f for f in fechas if f > fecha_val], default=None)

    if not prev_coupon or not next_coupon:
        return 0.0

    tasa = tasa_cupon_en_fecha(rules, next_coupon)

    dias_total = (next_coupon - prev_coupon).days
    dias_corridos = (fecha_val - prev_coupon).days

    if dias_total <= 0:
        return 0.0

    frac = dias_corridos / dias_total

    return (tasa / freq) * frac * 100.0


def tasa_cupon_en_fecha(rules: dict, fecha):
    if rules.get("tipo") == "step_up":
        for f_ini, f_fin, tasa in rules.get("coupon_schedule", []):
            if f_ini <= fecha < f_fin:
                return tasa
        return 0.0

    return float(rules.get("coupon", 0.0))

@st.cache_data(ttl=30)
def soberanos_usd_lista():
    df = pd.read_json(URL_BONOS)

    if df.empty:
        return pd.DataFrame()

    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"].isin(SOBERANOS_API_MAP.keys())].copy()

    if df.empty:
        return pd.DataFrame()

    df["bono"] = df["symbol"].map(SOBERANOS_API_MAP)
    df["vencimiento"] = df["bono"].map(lambda b: BOND_RULES[b]["maturity"])
    df["vencimiento"] = pd.to_datetime(df["vencimiento"])

    hoy = pd.Timestamp.today().normalize()
    df["años_al_vto"] = (df["vencimiento"] - hoy).dt.days / 365.0
    df["precio"] = pd.to_numeric(df["c"], errors="coerce")

    df["tir"] = df.apply(
        lambda row: calcular_tir_soberano(row["bono"], row["precio"]),
        axis=1
    )

    df = df[["bono", "symbol", "precio", "vencimiento", "años_al_vto", "tir"]]
    df = df.sort_values("vencimiento").reset_index(drop=True)

    return df

def tasa_cupon_en_fecha(rules: dict, fecha):
    if rules.get("tipo") == "step_up":
        for f_ini, f_fin, tasa in rules.get("coupon_schedule", []):
            if f_ini <= fecha < f_fin:
                return tasa
        return 0.0

    return float(rules.get("coupon", 0.0))


def generar_flujos_soberano(symbol: str, rules: dict, vn: float = 100.0):
    amort_sched = rules.get("amortization_schedule", [])
    amort_dict = {fecha: pct for fecha, pct in amort_sched}

    fechas_pago = sorted(amort_dict.keys())
    outstanding = vn
    flows = []

    for fecha in fechas_pago:
        tasa_anual = tasa_cupon_en_fecha(rules, fecha)
        interes = outstanding * tasa_anual / rules["frequency"]

        amort_pct = amort_dict.get(fecha, 0.0)
        amort = amort_pct * vn

        flujo = interes + amort

        flows.append({
            "symbol": symbol,
            "fecha": pd.Timestamp(fecha),
            "outstanding_previo": outstanding,
            "tasa_anual": tasa_anual,
            "interes": interes,
            "amort_pct": amort_pct,
            "amort": amort,
            "flujo": flujo
        })

        outstanding -= amort

    df = pd.DataFrame(flows)

    hoy = pd.Timestamp.today().normalize()
    df_futuros = df[df["fecha"] > hoy].copy().reset_index(drop=True)

    return df_futuros

def tabla_flujos_bono(symbol: str, vn: float = 100.0):
    if symbol not in BOND_RULES:
        return pd.DataFrame()

    rules = BOND_RULES[symbol]
    df = generar_flujos_soberano(symbol, rules, vn=vn)

    if df.empty:
        return df

    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df

def xnpv(rate, cashflows):
    total = 0.0
    t0 = cashflows[0][0]

    for fecha, flujo in cashflows:
        dias = (fecha - t0).days
        total += flujo / ((1 + rate) ** (dias / 365.0))

    return total


def xirr(cashflows, guess=0.10):
    low, high = -0.99, 5.0

    f_low = xnpv(low, cashflows)
    f_high = xnpv(high, cashflows)

    intentos = 0
    while f_low * f_high > 0 and intentos < 10:
        high *= 2
        f_high = xnpv(high, cashflows)
        intentos += 1

    if f_low * f_high > 0:
        return None

    for _ in range(200):
        mid = (low + high) / 2
        f_mid = xnpv(mid, cashflows)

        if abs(f_mid) < 1e-8:
            return mid

        if f_low * f_mid < 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return mid


def calcular_tir_soberano(symbol: str, precio_limpio: float, vn: float = 100.0):
    if symbol not in BOND_RULES:
        return None

    rules = BOND_RULES[symbol]
    df_flujos = generar_flujos_soberano(symbol, rules, vn=vn)

    if df_flujos.empty:
        return None

    hoy = pd.Timestamp.today().normalize()
    hoy_date = hoy.date()
    
    accrued = accrued_interest(symbol, rules, hoy_date)
    dirty_price = precio_limpio + accrued

    cashflows = [(hoy, -dirty_price)]

    cashflows += list(zip(df_flujos["fecha"], df_flujos["flujo"]))

    try:
        tir = xirr(cashflows)
        return tir
    except Exception:
        return None

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

    # limpiar columnas auxiliares y debug
    df = df.drop(
        columns=[
            c for c in [
                "ticker_api_mep",
                "Ticker API MEP",
                "% Var API",
                "Vol API",
                "symbol",
                "c",
                 "pct_change",
                "v"
            ] if c in df.columns
        ],
        errors="ignore"
    )

    return df


def color_spread(val):
    """
    Colorea la columna de spread:
    - verde si > 0
    - rojo si < 0
    - neutro si = 0 o NaN
    """
    try:
        if pd.isna(val):
            return ""
        v = float(val)
        if v > 0:
            return "color: #00c853; font-weight: 700;"
        elif v < 0:
            return "color: #ff5252; font-weight: 700;"
        else:
            return "color: #e0e0e0; font-weight: 700;"
    except Exception:
        return ""

def color_precio_dirty(val):
    """
    Resalta Precio Dirty (MEP).
    """
    try:
        if pd.isna(val):
            return ""
        return "color: #40c4ff; font-weight: 700;"
    except Exception:
        return ""


def color_vencimiento(val):
    """
    Resalta la fecha de vencimiento.
    """
    try:
        if pd.isna(val):
            return ""
        return "color: #ffd54f; font-weight: 700;"
    except Exception:
        return ""

def color_tir(val):
    """
    Resalta TIR en verde.
    """
    try:
        if pd.isna(val):
            return ""
        return "color: #00c853; font-weight: 700;"
    except Exception:
        return ""


def color_precio(val):
    """
    Resalta precios en celeste.
    """
    try:
        if pd.isna(val):
            return ""
        return "color: #40c4ff; font-weight: 700;"
    except Exception:
        return ""


# =========================
# HELPERS PROYECCION CER
# =========================
def fecha_15_mes_siguiente(fecha_base: date) -> date:
    """
    Devuelve el día 15 del mes siguiente a la fecha_base.
    """
    f = pd.Timestamp(fecha_base)
    primer_dia_mes = pd.Timestamp(year=f.year, month=f.month, day=1)
    f_sig = primer_dia_mes + relativedelta(months=1)
    return date(f_sig.year, f_sig.month, 15)


def proyectar_cer_tramo(
    fecha_cer_conocido: date,
    cer_conocido: float,
    ipc_estimado_pct: float,
    fecha_objetivo: date,
) -> dict:
    """
    Proyecta el CER diariamente entre la fecha conocida y el 15 del mes siguiente,
    distribuyendo el IPC mensual estimado con capitalización diaria.
    """
    if cer_conocido is None or ipc_estimado_pct is None:
        return {"error": "Faltan datos de entrada."}

    try:
        cer_conocido = float(cer_conocido)
        ipc_estimado_pct = float(ipc_estimado_pct)
    except Exception:
        return {"error": "CER conocido o IPC estimado inválido."}

    if cer_conocido <= 0:
        return {"error": "El CER conocido debe ser mayor a 0."}

    fecha_inicio = pd.Timestamp(fecha_cer_conocido).date()
    fecha_fin = fecha_15_mes_siguiente(fecha_inicio)
    fecha_obj = pd.Timestamp(fecha_objetivo).date()

    dias_tramo = (fecha_fin - fecha_inicio).days
    if dias_tramo <= 0:
        return {"error": "El tramo calculado no es válido."}

    if fecha_obj < fecha_inicio or fecha_obj > fecha_fin:
        return {
            "error": f"La fecha objetivo debe estar entre {fecha_inicio.strftime('%d/%m/%Y')} y {fecha_fin.strftime('%d/%m/%Y')}."
        }

    dias_hasta_obj = (fecha_obj - fecha_inicio).days
    factor_total = 1 + ipc_estimado_pct / 100.0
    factor_diario = factor_total ** (1 / dias_tramo)

    cer_proyectado_obj = cer_conocido * (factor_diario ** dias_hasta_obj)
    cer_proyectado_fin = cer_conocido * (factor_diario ** dias_tramo)

    return {
        "fecha_inicio": fecha_inicio,
        "fecha_fin": fecha_fin,
        "fecha_objetivo": fecha_obj,
        "dias_tramo": dias_tramo,
        "dias_hasta_obj": dias_hasta_obj,
        "factor_total": factor_total,
        "factor_diario": factor_diario,
        "cer_proyectado_obj": cer_proyectado_obj,
        "cer_proyectado_fin": cer_proyectado_fin,
        "error": None,
    }

def rendimiento_esperado_cer_cupon_cero(
    symbol: str,
    precio_actual: float,
    cer_proyectado_final: float,
    fecha_emision_map: dict,
    fecha_vencimiento_map: dict,
    cer_df: pd.DataFrame,
    base_dias: int = 365,
) -> dict:
    """
    Calcula rendimiento esperado para un bono CER cupón cero,
    usando un CER final proyectado en vencimiento - 10 hábiles.
    """

    symbol = str(symbol).strip().upper()

    if symbol not in fecha_emision_map:
        return {"error": "No hay fecha de emisión cargada."}

    if symbol not in fecha_vencimiento_map:
        return {"error": "No hay fecha de vencimiento cargada."}

    try:
        precio_actual = float(precio_actual)
        cer_proyectado_final = float(cer_proyectado_final)
    except Exception:
        return {"error": "Precio actual o CER proyectado inválido."}

    if precio_actual <= 0 or cer_proyectado_final <= 0:
        return {"error": "Precio actual o CER proyectado no válidos."}

    fecha_emis = pd.Timestamp(fecha_emision_map[symbol]).normalize()
    fecha_vto = pd.Timestamp(fecha_vencimiento_map[symbol]).normalize()

    f_emis_m10 = fecha_emis - 10 * ARG_BDAY
    f_vto_m10 = fecha_vto - 10 * ARG_BDAY

    cer_emis = cer_en_o_antes(cer_df, f_emis_m10)

    if cer_emis is None or cer_emis == 0:
        return {"error": "No se encontró CER en emisión - 10 hábiles."}

    coef_esperado = cer_proyectado_final / cer_emis
    vf_esperado = 100.0 * coef_esperado

    dias_a_vto = (fecha_vto - pd.Timestamp.today().normalize()).days

    tir_esperada = tir_cer_cupon_cero(precio_actual, vf_esperado, dias_a_vto, base_dias=base_dias)
    tna_esperada = tna_cer_cupon_cero(precio_actual, vf_esperado, dias_a_vto, base_dias=base_dias)

    return {
        "symbol": symbol,
        "fecha_emis_m10": f_emis_m10.date(),
        "fecha_vto_m10": f_vto_m10.date(),
        "cer_emis": cer_emis,
        "cer_final_proyectado": cer_proyectado_final,
        "coef_esperado": coef_esperado,
        "vf_esperado": vf_esperado,
        "dias_a_vto": dias_a_vto,
        "tir_esperada": tir_esperada,
        "tna_esperada": tna_esperada,
        "error": None,
    }


# =========================
# HELPERS PROYECCION CER MULTI-MES
# =========================
MESES_ES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}


def etiqueta_mes_es(fecha_ref: date) -> str:
    return f"{MESES_ES[fecha_ref.month]} {fecha_ref.year}"


def construir_tabla_supuestos_ipc(
    fecha_base: date,
    n_meses: int,
    ipc_default: float = 3.0
) -> pd.DataFrame:
    """
    Construye la tabla de supuestos de IPC para proyectar CER
    desde fecha_base hacia adelante por n_meses tramos.
    """
    filas = []

    fecha_base_ts = pd.Timestamp(fecha_base).date()

    for i in range(n_meses):
        fecha_inicio = (pd.Timestamp(fecha_base_ts) + relativedelta(months=i)).date()
        fecha_fin = (pd.Timestamp(fecha_base_ts) + relativedelta(months=i + 1)).date()

        # El tramo fecha_inicio -> fecha_fin usa el IPC del mes anterior a fecha_inicio
        mes_ipc_ref = (pd.Timestamp(fecha_inicio) - relativedelta(months=1)).date()

        filas.append({
            "Mes IPC": etiqueta_mes_es(mes_ipc_ref),
            "IPC estimado (%)": ipc_default,
            "Fecha inicio tramo": fecha_inicio.strftime("%d/%m/%Y"),
            "Fecha fin tramo": fecha_fin.strftime("%d/%m/%Y"),
        })

    return pd.DataFrame(filas)


def proyectar_cer_multi_tramos(
    fecha_cer_conocido: date,
    cer_conocido: float,
    supuestos_ipc_df: pd.DataFrame,
    fecha_objetivo: date,
) -> dict:
    """
    Proyecta CER usando múltiples tramos mensuales.
    Cada tramo distribuye diariamente el IPC supuesto correspondiente.
    """
    try:
        cer_actual = float(cer_conocido)
    except Exception:
        return {"error": "CER conocido inválido."}

    if cer_actual <= 0:
        return {"error": "CER conocido debe ser mayor a 0."}

    if supuestos_ipc_df is None or supuestos_ipc_df.empty:
        return {"error": "No hay supuestos de IPC cargados."}

    fecha_obj = pd.Timestamp(fecha_objetivo).date()

    detalle = []

    for _, row in supuestos_ipc_df.iterrows():
        try:
            fecha_inicio = pd.to_datetime(row["Fecha inicio tramo"], dayfirst=True).date()
            fecha_fin = pd.to_datetime(row["Fecha fin tramo"], dayfirst=True).date()
            ipc_pct = float(row["IPC estimado (%)"])
        except Exception:
            return {"error": "Hay un supuesto de IPC inválido en la tabla."}

        dias_tramo = (fecha_fin - fecha_inicio).days
        if dias_tramo <= 0:
            return {"error": "Se detectó un tramo inválido en la tabla de supuestos."}

        factor_total = 1 + ipc_pct / 100.0
        factor_diario = factor_total ** (1 / dias_tramo)

        detalle.append({
            "fecha_inicio": fecha_inicio,
            "fecha_fin": fecha_fin,
            "ipc_pct": ipc_pct,
            "factor_total": factor_total,
            "factor_diario": factor_diario,
            "cer_inicio": cer_actual,
        })

        # Si la fecha objetivo cae dentro de este tramo
        if fecha_inicio <= fecha_obj <= fecha_fin:
            dias_hasta_obj = (fecha_obj - fecha_inicio).days
            cer_obj = cer_actual * (factor_diario ** dias_hasta_obj)

            return {
                "fecha_inicio_global": pd.Timestamp(fecha_cer_conocido).date(),
                "fecha_objetivo": fecha_obj,
                "fecha_max_proyectable": fecha_fin,
                "cer_proyectado_obj": cer_obj,
                "detalle": detalle,
                "error": None,
            }

        # Si todavía no llegamos al objetivo, cerramos este tramo completo
        cer_actual = cer_actual * factor_total

    fecha_max = pd.to_datetime(supuestos_ipc_df.iloc[-1]["Fecha fin tramo"], dayfirst=True).date()

    return {
        "error": (
            f"La fecha objetivo ({fecha_obj.strftime('%d/%m/%Y')}) queda fuera del horizonte proyectado. "
            f"Extendé los supuestos de IPC hasta al menos {fecha_obj.strftime('%d/%m/%Y')}. "
            f"Hoy llegás hasta {fecha_max.strftime('%d/%m/%Y')}."
        )
    }

# =========================
# HELPERS INFLACION IMPLICITA
# =========================
def meses_necesarios_hasta_fecha(fecha_base: date, fecha_objetivo: date) -> int:
    """
    Devuelve cuántos tramos mensuales hacen falta para cubrir una fecha objetivo.
    """
    fb = pd.Timestamp(fecha_base).date()
    fo = pd.Timestamp(fecha_objetivo).date()

    meses = 0
    cursor = fb

    while cursor < fo:
        cursor = (pd.Timestamp(cursor) + relativedelta(months=1)).date()
        meses += 1

        if meses > 240:
            break

    return max(meses, 1)


def tir_esperada_cer_con_inflacion_plana(
    symbol_cer: str,
    inflacion_mensual_pct: float,
    fecha_cer_conocido: date,
    cer_conocido: float,
    fecha_emision_map: dict,
    fecha_vencimiento_map: dict,
    cer_df: pd.DataFrame,
    df_cer: pd.DataFrame,
) -> float | None:
    """
    Calcula la TIR esperada de un bono CER cupón cero usando una inflación
    mensual plana repetida en todos los tramos hasta vto - 10 hábiles.
    """
    symbol_cer = str(symbol_cer).strip().upper()

    if symbol_cer not in fecha_vencimiento_map:
        return None

    row_bono = df_cer[df_cer["symbol"].astype(str).str.upper() == symbol_cer]
    if row_bono.empty:
        return None

    precio_actual = pd.to_numeric(row_bono.iloc[0]["c"], errors="coerce")
    if pd.isna(precio_actual) or float(precio_actual) <= 0:
        return None

    fecha_vto = pd.Timestamp(fecha_vencimiento_map[symbol_cer]).normalize()
    fecha_objetivo_bono = (fecha_vto - 10 * ARG_BDAY).date()

    n_meses = meses_necesarios_hasta_fecha(fecha_cer_conocido, fecha_objetivo_bono)

    df_supuestos = construir_tabla_supuestos_ipc(
        fecha_base=fecha_cer_conocido,
        n_meses=n_meses,
        ipc_default=inflacion_mensual_pct
    )
    df_supuestos["IPC estimado (%)"] = float(inflacion_mensual_pct)

    resultado_bono = proyectar_cer_multi_tramos(
        fecha_cer_conocido=fecha_cer_conocido,
        cer_conocido=cer_conocido,
        supuestos_ipc_df=df_supuestos,
        fecha_objetivo=fecha_objetivo_bono,
    )

    if resultado_bono.get("error"):
        return None

    rendimiento_bono = rendimiento_esperado_cer_cupon_cero(
        symbol=symbol_cer,
        precio_actual=precio_actual,
        cer_proyectado_final=resultado_bono["cer_proyectado_obj"],
        fecha_emision_map=fecha_emision_map,
        fecha_vencimiento_map=fecha_vencimiento_map,
        cer_df=cer_df,
    )

    if rendimiento_bono.get("error"):
        return None

    return rendimiento_bono.get("tir_esperada")


def inflacion_implicita_par(
    ticker_fija: str,
    ticker_cer: str,
    df_tf: pd.DataFrame,
    df_cer: pd.DataFrame,
    fecha_cer_conocido: date,
    cer_conocido: float,
    fecha_emision_map: dict,
    fecha_vencimiento_map: dict,
    cer_df: pd.DataFrame,
    lower: float = -5.0,
    upper: float = 15.0,
    tol: float = 1e-4,
    max_iter: int = 80,
) -> dict:
    """
    Busca la inflación mensual plana que iguala la TIR del CER con la TIR
    actual del instrumento de tasa fija equivalente.
    """
    ticker_fija = str(ticker_fija).strip().upper()
    ticker_cer = str(ticker_cer).strip().upper()

    row_fija = df_tf[df_tf["symbol"].astype(str).str.upper() == ticker_fija]
    if row_fija.empty:
        return {"error": f"No se encontró {ticker_fija} en tasa fija."}

    tir_fija = pd.to_numeric(row_fija.iloc[0]["TIR (%)"], errors="coerce")
    dias_fija = pd.to_numeric(row_fija.iloc[0]["dias_a_vencimiento"], errors="coerce")

    if pd.isna(tir_fija):
        return {"error": f"No hay TIR válida para {ticker_fija}."}

    def f(ipc_mensual):
        tir_cer = tir_esperada_cer_con_inflacion_plana(
            symbol_cer=ticker_cer,
            inflacion_mensual_pct=ipc_mensual,
            fecha_cer_conocido=fecha_cer_conocido,
            cer_conocido=cer_conocido,
            fecha_emision_map=fecha_emision_map,
            fecha_vencimiento_map=fecha_vencimiento_map,
            cer_df=cer_df,
            df_cer=df_cer,
        )
        if tir_cer is None:
            return None
        return tir_cer - float(tir_fija)

    f_low = f(lower)
    f_high = f(upper)

    if f_low is None or f_high is None:
        return {"error": f"No se pudo evaluar el par {ticker_fija} / {ticker_cer}."}

    if f_low == 0:
        raiz = lower
    elif f_high == 0:
        raiz = upper
    elif f_low * f_high > 0:
        return {
            "error": (
                f"No se encontró raíz para {ticker_fija} / {ticker_cer} "
                f"en el rango [{lower}%, {upper}%]."
            )
        }
    else:
        a, b = lower, upper
        raiz = None

        for _ in range(max_iter):
            m = (a + b) / 2
            fm = f(m)

            if fm is None:
                return {"error": f"No se pudo evaluar el punto medio para {ticker_fija} / {ticker_cer}."}

            if abs(fm) < tol:
                raiz = m
                break

            if f_low * fm < 0:
                b = m
                f_high = fm
            else:
                a = m
                f_low = fm

            raiz = m

    tir_cer_final = tir_esperada_cer_con_inflacion_plana(
        symbol_cer=ticker_cer,
        inflacion_mensual_pct=raiz,
        fecha_cer_conocido=fecha_cer_conocido,
        cer_conocido=cer_conocido,
        fecha_emision_map=fecha_emision_map,
        fecha_vencimiento_map=fecha_vencimiento_map,
        cer_df=cer_df,
        df_cer=df_cer,
    )

    return {
        "ticker_fija": ticker_fija,
        "ticker_cer": ticker_cer,
        "tir_fija": float(tir_fija),
        "tir_cer_eq": tir_cer_final,
        "inflacion_implicita_mensual": raiz,
        "dias": int(dias_fija) if pd.notna(dias_fija) else None,
        "error": None,
    }


# =========================
# A3 / PRIMARY - FUTUROS DOLAR
# =========================

A3_BASE_URL = "https://api.remarkets.primary.com.ar"

# Ideal: guardar user y pass en variables de entorno
A3_USERNAME = "seguraseba221314"
A3_PASSWORD = "igiouC8$"

DLR_CONTRATOS = [
    "DLR/ABR26",
    "DLR/AGO26",
    "DLR/ENE27",
    "DLR/FEB27",
    "DLR/JUL26",
    "DLR/JUN26",
    "DLR/MAR26",
    "DLR/MAR27",
    "DLR/MAY26",
    "DLR/NOV26",
    "DLR/OCT26",
]

def a3_get_token():
    if not A3_USERNAME or not A3_PASSWORD:
        raise ValueError("Faltan credenciales A3_USERNAME / A3_PASSWORD.")

    url = f"{A3_BASE_URL}/auth/getToken"
    headers = {
        "X-Username": A3_USERNAME,
        "X-Password": A3_PASSWORD,
    }

    r = requests.post(url, headers=headers, timeout=20)
    r.raise_for_status()

    token = r.headers.get("X-Auth-Token")
    if not token:
        raise ValueError("No se recibió X-Auth-Token.")
    return token


def a3_get_market_data(token: str, symbol: str, market_id: str = "ROFX") -> dict:
    url = f"{A3_BASE_URL}/rest/marketdata/get"
    headers = {
        "X-Auth-Token": token
    }
    params = {
        "marketId": market_id,
        "symbol": symbol,
        "entries": "BI,OF,LA,TV",
        "depth": 1
    }

    r = requests.get(url, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def cargar_dlr_spot(token: str) -> dict:
    data = a3_get_market_data(token, "DLR/SPOT")
    row = a3_parse_md("DLR/SPOT", data)
    return row

def a3_parse_md(symbol: str, data: dict) -> dict:
    md = data.get("marketData", {}) or {}

    bi_list = md.get("BI", []) or []
    of_list = md.get("OF", []) or []
    la = md.get("LA")
    tv = md.get("TV", 0)

    bid = bi_list[0].get("price") if bi_list else None
    ask = of_list[0].get("price") if of_list else None
    last = la.get("price") if isinstance(la, dict) else None

    return {
        "Contrato": symbol,
        "Último": last,
        "Bid": bid,
        "Ask": ask,
        "Volumen": tv
    }


MESES_MAP = {
    "ENE": 1,
    "FEB": 2,
    "MAR": 3,
    "ABR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AGO": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DIC": 12,
}

def ultimo_dia_habil_mes(anio: int, mes: int) -> date:
    ultimo_dia = calendar.monthrange(anio, mes)[1]
    f = pd.Timestamp(year=anio, month=mes, day=ultimo_dia)

    while f.weekday() >= 5 or f in feriados_arg:
        f -= pd.Timedelta(days=1)

    return f.date()

def parsear_vencimiento_dlr(symbol: str):
    """
    Ej: DLR/ABR26 -> último día hábil de abril 2026
    """
    try:
        base = symbol.replace("DLR/", "")
        mes_txt = base[:3]
        anio_txt = base[3:]

        mes = MESES_MAP[mes_txt]
        anio = 2000 + int(anio_txt)

        return ultimo_dia_habil_mes(anio, mes)
    except Exception:
        return None

def dias_habiles_al_vto(vto):
    if vto is None:
        return None

    hoy = pd.Timestamp.today().normalize()
    vto_ts = pd.Timestamp(vto)

    if vto_ts <= hoy:
        return 0

    return len(pd.date_range(start=hoy + ARG_BDAY, end=vto_ts, freq=ARG_BDAY))

@st.cache_data(ttl=10)
def cargar_futuros_dolar_snapshot():
    token = a3_get_token()

    # Spot
    try:
        spot_row = cargar_dlr_spot(token)
    except Exception:
        spot_row = {
            "Contrato": "DLR/SPOT",
            "Último": None,
            "Bid": None,
            "Ask": None,
            "Volumen": None
        }

    # Futuros
    rows = []

    for symbol in DLR_CONTRATOS:
        try:
            data = a3_get_market_data(token, symbol)
            rows.append(a3_parse_md(symbol, data))
        except Exception:
            rows.append({
                "Contrato": symbol,
                "Último": None,
                "Bid": None,
                "Ask": None,
                "Volumen": None
            })

    df = pd.DataFrame(rows)

    for col in ["Último", "Bid", "Ask", "Volumen"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Vto"] = df["Contrato"].apply(parsear_vencimiento_dlr)
    df["Días al vto"] = df["Vto"].apply(dias_habiles_al_vto)

    df = df.sort_values(by=["Días al vto", "Contrato"], na_position="last").reset_index(drop=True)

    return spot_row, df

# =========================
# MAIN APP (CON PESTAÑAS)
# =========================

st.title("Monitor de Renta Fija")

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

tab_curvas, tab_leg, tab_corpos, tab_cer_proj, tab_carry, tab_futuros = st.tabs(
    ["Curvas", "Spread Legislación", "Corporativos", "Proyección CER", "Carry Trade", "Futuros Dólar"]
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

        styler_tf = df_display.style.format({
            "Precio": "{:,.2f}",
            "TNA (%)": "{:,.2f}",
            "TIR (%)": "{:,.2f}",
            "TEM (%)": "{:,.2f}"
        })

        if "Precio" in df_display.columns:
            styler_tf = styler_tf.map(color_precio, subset=["Precio"])

        if "TIR (%)" in df_display.columns:
            styler_tf = styler_tf.map(color_tir, subset=["TIR (%)"])

        st.dataframe(
            styler_tf,
            use_container_width=True,
            height=height_tf
        )

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

        styler_cer = df_cer_display.style.format({
            "Precio": "{:,.2f}",
            "Volumen": "{:,.0f}",
            "% Var": "{:,.2f}",
            "TIR CER (%)": "{:,.2f}"
        })

        if "Precio" in df_cer_display.columns:
            styler_cer = styler_cer.map(color_precio, subset=["Precio"])

        if "TIR CER (%)" in df_cer_display.columns:
            styler_cer = styler_cer.map(color_tir, subset=["TIR CER (%)"])

        st.dataframe(
            styler_cer,
            use_container_width=True,
            height=height_cer
        )

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
# TAB: PROYECCION CER
# =========================
with tab_cer_proj:
    st.subheader("Rendimiento esperado bono CER")

    st.caption(
        "Proyección diaria del CER usando un IPC mensual estimado, "
        "distribuido con capitalización diaria hasta el 15 del mes siguiente."
    )

    # -------------------------s
    # TOMAR ULTIMO CER DESDE EXCEL
    # -------------------------
    if cer_df is not None and not cer_df.empty:
        ultima_fila_cer = cer_df.sort_values("fecha").iloc[-1]
        fecha_cer_default = pd.to_datetime(ultima_fila_cer["fecha"]).date()
        cer_conocido_default = float(ultima_fila_cer["cer"])
    else:
        fecha_cer_default = date.today()
        cer_conocido_default = 1000.0

    # -------------------------
    # UNIVERSO CER CUPON CERO PARA CALCULADORA
    # -------------------------
    tickers_cer_calc = [
        s for s in FECHA_VENCIMIENTO.keys()
        if s in FECHA_EMISION and s not in CER_ESPECIALES_CON_FLUJOS
    ]
    tickers_cer_calc = sorted(tickers_cer_calc)

    st.caption(
        "CER tomado automáticamente desde Excel (último dato disponible): "
        f"{fecha_cer_default.strftime('%d/%m/%Y')} → {cer_conocido_default:,.4f}"
    )

    col_in_1, col_in_2 = st.columns(2)

    with col_in_1:
        st.metric(
            "Fecha CER base",
            fecha_cer_default.strftime("%d/%m/%Y")
        )

        st.metric(
            "CER base",
            f"{cer_conocido_default:,.4f}"
        )

        # variables internas (sin input del usuario)
        fecha_cer_conocido = fecha_cer_default
        cer_conocido = cer_conocido_default
    

    with col_in_2:
        ticker_cer = st.selectbox(
            "Ticker CER",
            options=tickers_cer_calc,
            index=0 if tickers_cer_calc else None
        )

        if ticker_cer in FECHA_VENCIMIENTO:
            fecha_vto_ticker = pd.Timestamp(FECHA_VENCIMIENTO[ticker_cer]).normalize()
            fecha_objetivo_bono_auto = (fecha_vto_ticker - 10 * ARG_BDAY).date()
            meses_proyeccion = meses_necesarios_hasta_fecha(
                fecha_cer_conocido,
                fecha_objetivo_bono_auto
            )
        else:
            fecha_objetivo_bono_auto = fecha_cer_conocido
            meses_proyeccion = 1

        st.metric("Fecha objetivo bono", fecha_objetivo_bono_auto.strftime("%d/%m/%Y"))
        st.metric("Meses a proyectar", f"{meses_proyeccion}")

    # -------------------------
    # TABLA DE SUPUESTOS IPC
    # -------------------------
    firma_supuestos = f"{fecha_cer_conocido.isoformat()}_{meses_proyeccion}"

    if st.session_state.get("firma_supuestos_ipc") != firma_supuestos:
        st.session_state["firma_supuestos_ipc"] = firma_supuestos
        st.session_state["df_supuestos_ipc"] = construir_tabla_supuestos_ipc(
            fecha_base=fecha_cer_conocido,
            n_meses=meses_proyeccion,
            ipc_default=3.0
        )

    st.markdown("### Supuestos de inflación")
    st.caption(
        "Cada tramo CER usa el IPC del mes indicado. "
        "Podés editar solo la columna 'IPC estimado (%)'.  "
        "Presionar 'Aplicar cambios IPC' para ver rendimiento según la estimación."
    )


    df_supuestos_ipc_edit = st.data_editor(
        st.session_state["df_supuestos_ipc"],
        use_container_width=True,
        hide_index=True,
        disabled=["Mes IPC", "Fecha inicio tramo", "Fecha fin tramo"],
        key="editor_supuestos_ipc"
    )

    if st.button("Aplicar cambios IPC"):
        st.session_state["df_supuestos_ipc"] = df_supuestos_ipc_edit.copy()

    # resultado_cer se usa ahora solo como "horizonte proyectable"
    # para no romper la estructura del código más abajo
    fecha_objetivo_bono_preview = fecha_objetivo_bono_auto

    resultado_cer = proyectar_cer_multi_tramos(
        fecha_cer_conocido=fecha_cer_conocido,
        cer_conocido=cer_conocido,
        supuestos_ipc_df=st.session_state["df_supuestos_ipc"],
        fecha_objetivo=fecha_objetivo_bono_preview,
    )
    

    if resultado_cer.get("error"):
        st.warning(resultado_cer["error"])
    else:
        # =========================
        # RENDIMIENTO ESPERADO BONO CER
        # =========================
        st.markdown("### Rendimiento esperado del bono CER")

        if ticker_cer and df_cer is not None and not df_cer.empty:
            row_bono = df_cer[df_cer["symbol"].astype(str).str.upper() == ticker_cer]

            if row_bono.empty:
                st.warning(f"No se encontró precio actual para {ticker_cer} en el universo CER.")
            else:
                precio_actual_bono = pd.to_numeric(row_bono.iloc[0]["c"], errors="coerce")

                fecha_objetivo_bono = fecha_objetivo_bono_auto

                # validar que la fecha objetivo del bono esté dentro del horizonte proyectable
                if (
                    fecha_objetivo_bono < resultado_cer["fecha_inicio_global"]
                    or fecha_objetivo_bono > resultado_cer["fecha_max_proyectable"]
                ):
                    st.warning(
                        f"Para {ticker_cer}, la fecha objetivo relevante ({fecha_objetivo_bono.strftime('%d/%m/%Y')}) "
                        f"queda fuera del horizonte actualmente proyectable "
                        f"({resultado_cer['fecha_inicio_global'].strftime('%d/%m/%Y')} a "
                        f"{resultado_cer['fecha_max_proyectable'].strftime('%d/%m/%Y')})."
                    )
                
                else:
                    resultado_bono = proyectar_cer_multi_tramos(
                        fecha_cer_conocido=fecha_cer_conocido,
                        cer_conocido=cer_conocido,
                        supuestos_ipc_df=st.session_state["df_supuestos_ipc"],
                        fecha_objetivo=fecha_objetivo_bono,
                    )

                       

                    if resultado_bono.get("error"):
                        st.warning(resultado_bono["error"])
                    else:
                        rendimiento_bono = rendimiento_esperado_cer_cupon_cero(
                            symbol=ticker_cer,
                            precio_actual=precio_actual_bono,
                            cer_proyectado_final=resultado_bono["cer_proyectado_obj"],
                            fecha_emision_map=FECHA_EMISION,
                            fecha_vencimiento_map=FECHA_VENCIMIENTO,
                            cer_df=cer_df,
                        )

                        if rendimiento_bono.get("error"):
                            st.warning(rendimiento_bono["error"])
                        else:
                            # =========================
                            # FILA 1 — INFO GENERAL
                            # =========================
                            c1, c2, c3 = st.columns(3)

                            c1.metric("Ticker", ticker_cer)
                            c2.metric("Precio", f"{precio_actual_bono:,.2f}")
                            c3.metric("Días a vencimiento", f"{rendimiento_bono['dias_a_vto']}")


                            st.markdown("---")

                            # =========================
                            # FILA 3 — RESULTADO (FOCO)
                            # =========================
                            c8, c9 = st.columns(2)

                            c8.metric(
                                "TIR esperada",
                                f"{rendimiento_bono['tir_esperada']:,.2f}%"
                                if rendimiento_bono["tir_esperada"] is not None else "-"
                            )

                            c9.metric(
                                "TNA esperada",
                                f"{rendimiento_bono['tna_esperada']:,.2f}%"
                                if rendimiento_bono["tna_esperada"] is not None else "-"
                            )
                            
        # =========================
        # CURVA DE INFLACION IMPLICITA
        # =========================
        st.markdown("---")
        st.markdown("### Curva de inflación implícita CER vs tasa fija")

        resultados_implicita = []

        for par in PARES_INFLACION_IMPLICITA:
            out = inflacion_implicita_par(
                ticker_fija=par["ticker_fija"],
                ticker_cer=par["ticker_cer"],
                df_tf=df_tf,
                df_cer=df_cer,
                fecha_cer_conocido=fecha_cer_conocido,
                cer_conocido=cer_conocido,
                fecha_emision_map=FECHA_EMISION,
                fecha_vencimiento_map=FECHA_VENCIMIENTO,
                cer_df=cer_df,
            )

            if out.get("error") is None:
                resultados_implicita.append({
                    "Par": f"{par['ticker_fija']} / {par['ticker_cer']}",
                    "Ticker fija": out["ticker_fija"],
                    "Ticker CER": out["ticker_cer"],
                    "Días": out["dias"],
                    "TIR fija (%)": out["tir_fija"],
                    "Inflación implícita mensual (%)": out["inflacion_implicita_mensual"],
                    "TIR CER eq (%)": out["tir_cer_eq"],
                })

        if not resultados_implicita:
            st.info("No se pudieron calcular pares de inflación implícita con los supuestos actuales.")
        else:
            df_implicita = pd.DataFrame(resultados_implicita).sort_values("Días").reset_index(drop=True)

            st.dataframe(
                df_implicita.style.format({
                    "TIR fija (%)": "{:,.2f}",
                    "Inflación implícita mensual (%)": "{:,.3f}",
                    "TIR CER eq (%)": "{:,.2f}",
                }),
                use_container_width=True,
                hide_index=True
            )

            fig_imp = go.Figure()
            fig_imp.add_trace(go.Scatter(
                x=df_implicita["Días"],
                y=df_implicita["Inflación implícita mensual (%)"],
                mode="lines+markers+text",
                text=df_implicita["Par"],
                textposition="top center",
                name="Inflación implícita mensual"
            ))

            fig_imp.update_layout(
                title="Curva de inflación implícita",
                xaxis_title="Días a vencimiento",
                yaxis_title="Inflación implícita mensual (%)",
                template="plotly_white",
                hovermode="closest"
            )

            st.plotly_chart(fig_imp, use_container_width=True)

                            
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

        df_show_styled = df_show.style.format({
            "Precio AL": "{:,.2f}",
            "Precio GD": "{:,.2f}",
            "Spread %": "{:,.2f}%"
        }).map(color_spread, subset=["Spread %"])

        st.dataframe(
            df_show_styled,
            use_container_width=True,
            hide_index=True,
            height=min(500, 40 + 35 * len(df_show))
        )

    st.markdown("---")
    st.subheader("Soberanos USD")

    if st.button("Refrescar soberanos USD", key="refresh_soberanos_leg"):
        soberanos_usd_lista.clear()

    try:
        df_sob = soberanos_usd_lista().copy()

        if df_sob.empty:
            st.info("No se encontraron soberanos USD en la API de bonos.")
        else:
            df_sob["precio"] = pd.to_numeric(df_sob["precio"], errors="coerce").round(2)
            df_sob["años_al_vto"] = pd.to_numeric(df_sob["años_al_vto"], errors="coerce").round(2)
            df_sob["tir"] = pd.to_numeric(df_sob["tir"], errors="coerce") * 100
            df_sob["tir"] = df_sob["tir"].round(2)
            df_sob["vencimiento"] = pd.to_datetime(df_sob["vencimiento"]).dt.strftime("%d-%m-%Y")

            df_sob = df_sob.rename(columns={
                "bono": "Bono",
                "symbol": "Ticker",
                "precio": "Precio",
                "vencimiento": "Vencimiento",
                "años_al_vto": "Años al vto",
                "tir": "TIR (%)"

            })

            st.dataframe(
                df_sob,
                use_container_width=True,
                hide_index=True
            )

    st.markdown("---")
    st.subheader("Flujos del bono")

    bono_sel = st.selectbox(
        "Seleccionar bono para ver flujos",
        options=["AL29", "AL30", "AL35", "AL38", "AL41"],
        index=1,
        key="bono_flujos_soberano"
    )

    df_flujos = tabla_flujos_bono(bono_sel, vn=100.0)

    if df_flujos.empty:
        st.info("No hay flujos futuros para este bono.")
    else:
        df_flujos_show = df_flujos.copy()

        df_flujos_show["fecha"] = pd.to_datetime(df_flujos_show["fecha"]).dt.strftime("%d-%m-%Y")
        df_flujos_show["tasa_anual"] = (pd.to_numeric(df_flujos_show["tasa_anual"], errors="coerce") * 100).round(4)
        df_flujos_show["outstanding_previo"] = pd.to_numeric(df_flujos_show["outstanding_previo"], errors="coerce").round(4)
        df_flujos_show["interes"] = pd.to_numeric(df_flujos_show["interes"], errors="coerce").round(6)
        df_flujos_show["amort_pct"] = (pd.to_numeric(df_flujos_show["amort_pct"], errors="coerce") * 100).round(4)
        df_flujos_show["amort"] = pd.to_numeric(df_flujos_show["amort"], errors="coerce").round(6)
        df_flujos_show["flujo"] = pd.to_numeric(df_flujos_show["flujo"], errors="coerce").round(6)

        df_flujos_show = df_flujos_show.rename(columns={
            "fecha": "Fecha",
            "outstanding_previo": "VN previo",
            "tasa_anual": "Tasa anual (%)",
            "interes": "Interés",
            "amort_pct": "Amort (%)",
            "amort": "Amortización",
            "flujo": "Flujo total"
        })

        df_flujos_show = df_flujos_show[
            ["Fecha", "VN previo", "Tasa anual (%)", "Interés", "Amort (%)", "Amortización", "Flujo total"]
        ]

        st.dataframe(
            df_flujos_show,
            use_container_width=True,
            hide_index=True
        )

        

    except Exception as e:
        st.error(f"Error cargando soberanos USD: {e}")



# =========================
# TAB 5: CORPORATIVOS
# =========================
# =========================
# TAB 5: CORPORATIVOS
# =========================
with tab_corpos:
    st.subheader("Bonos corporativos")

        # =========================
    # CALCULADORA RÁPIDA CORPORATIVOS
    # =========================
    with st.expander("Calculadora rápida", expanded=True):
        col_calc_1, col_calc_2 = st.columns(2)

        with col_calc_1:
            pesos_disponibles = st.number_input(
                "Pesos disponibles",
                min_value=0.0,
                value=51000000.0,
                step=100000.0,
                format="%.4f"
            )

            px_usd_mesa = st.number_input(
                "Px USD mesa",
                min_value=0.0,
                value=1.06,
                step=0.0001,
                format="%.4f"
            )

            arancel_pct = st.number_input(
                "Arancel (%)",
                min_value=0.0,
                value=1.0,
                step=0.01,
                format="%.4f"
            )

        with col_calc_2:
            fx_offer = st.number_input(
                "Offer FX (ver MEP o CCL)",
                min_value=0.0,
                value=1430.0,
                step=1.0,
                format="%.4f"
            )

            monto_por_vn = st.number_input(
                "Monto a cobrar por VN",
                min_value=0.0,
                value=1.04,
                step=0.0001,
                format="%.4f"
            )

        # Cálculos
        px_usd_cliente = px_usd_mesa * (1 + arancel_pct / 100.0)
        px_ars_cliente = px_usd_cliente * fx_offer

        vn_a_cobrar = None
        fx_implicito = None

        if px_ars_cliente > 0:
            vn_a_cobrar = pesos_disponibles / px_ars_cliente

        if monto_por_vn > 0 and vn_a_cobrar is not None and vn_a_cobrar > 0:
            usd_a_cobrar = vn_a_cobrar * monto_por_vn
            if usd_a_cobrar > 0:
                fx_implicito = pesos_disponibles / usd_a_cobrar

        prima_fx = None

        if fx_implicito is not None and fx_offer > 0:
            prima_fx = (fx_implicito / fx_offer - 1) * 100

        st.markdown("### Resultado calculadora")
        c1, c2, c3 = st.columns(3)
        c4, c5 = st.columns(2)

        with c1:
            st.metric("Px USD cliente", f"{px_usd_cliente:,.4f}")

        with c2:
            st.metric("Px ARS cliente", f"{px_ars_cliente:,.4f}")

        with c3:
            if vn_a_cobrar is not None:
                st.metric("VN a comprar", f"{vn_a_cobrar:,.4f}")
            else:
                st.metric("VN a comprar", "-")

        with c4:
            if fx_implicito is not None:
                st.metric("FX implícito", f"{fx_implicito:,.4f}")
            else:
                st.metric("FX implícito", "-")

        with c5:
            if prima_fx is not None:
                st.metric("Prima sobre FX (%)", f"{prima_fx:,.2f}%")
            else:
                st.metric("Prima sobre FX (%)", "-")       

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
        # ORDEN DE COLUMNAS
        # -------------------------
        columnas_preferidas = [
            "Ticker",
            "Emisor",
            "Industria",
            "Ley",
            "Moneda Pago",
            "Precio Dirty (MEP)",
            "Cupón",
            "Vencimiento",
            "Próx. Cupón",
            "Prox. Cupón",
            "Próximo Cupón",
            "Calificación Fix",
        ]

        columnas_existentes_preferidas = [c for c in columnas_preferidas if c in df_corpos_show.columns]
        columnas_restantes = [c for c in df_corpos_show.columns if c not in columnas_existentes_preferidas]

        df_corpos_show = df_corpos_show[columnas_existentes_preferidas + columnas_restantes]

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

        # -------------------------
        # FORMATO VISUAL TABLA
        # -------------------------
        columnas_formato_2d = [
            "Precio Dirty (MEP)",
        ]

        columnas_formato_pct = [
            "Cupón",
        ]

        for col in columnas_formato_2d:
            if col in df_corpos_show.columns:
                df_corpos_show[col] = pd.to_numeric(df_corpos_show[col], errors="coerce").round(2)

        # Orden opcional por vencimiento
        if "Vencimiento" in df_corpos_show.columns:
            try:
                df_corpos_show = df_corpos_show.sort_values("Vencimiento", ascending=True)
            except Exception:
                pass

        # Armar estilos
        styler_corpos = df_corpos_show.style

        # Formatos numéricos
        formato_dict = {}
        if "Precio Dirty (MEP)" in df_corpos_show.columns:
            formato_dict["Precio Dirty (MEP)"] = "{:,.2f}"

        if formato_dict:
            styler_corpos = styler_corpos.format(formato_dict)

        # Colores
        if "Precio Dirty (MEP)" in df_corpos_show.columns:
            styler_corpos = styler_corpos.map(color_precio_dirty, subset=["Precio Dirty (MEP)"])

        if "Vencimiento" in df_corpos_show.columns:
            styler_corpos = styler_corpos.map(color_vencimiento, subset=["Vencimiento"])

        st.dataframe(
            styler_corpos,
            use_container_width=True,
            hide_index=True,
            height=min(900, 40 + 35 * len(df_corpos_show))
        )


# =========================
# TAB 6: FUTUROS DOLAR
# =========================
with tab_futuros:
    st.markdown("## Futuros de dólar A3")

    col1, col2 = st.columns([1, 4])

    with col1:
        if st.button("Refrescar futuros"):
            cargar_futuros_dolar_snapshot.clear()

    try:
        spot_row, df_fut = cargar_futuros_dolar_snapshot()

        # ===== SPOT =====
        st.markdown("### Dólar Spot")

        spot_ultimo = pd.to_numeric(spot_row.get("Último"), errors="coerce")

        st.metric(
            "Último",
            f"{spot_ultimo:.2f}" if pd.notnull(spot_ultimo) else "-"
        )

        st.markdown("---")

        # ===== FUTUROS =====
        for col in ["Último", "Bid", "Ask"]:
            df_fut[col] = pd.to_numeric(df_fut[col], errors="coerce").round(2)

        df_fut["Volumen"] = pd.to_numeric(df_fut["Volumen"], errors="coerce").fillna(0)
        df_fut["Días al vto"] = pd.to_numeric(df_fut["Días al vto"], errors="coerce")

        df_fut["Vto"] = df_fut["Vto"].apply(
            lambda x: x.strftime("%d-%m-%Y") if pd.notnull(x) else "-"
        )

        df_fut = df_fut[["Contrato", "Vto", "Días al vto", "Último", "Bid", "Ask", "Volumen"]]

        df_show = df_fut.fillna("-")

        st.caption("Ordenado por días hábiles al vencimiento (último día hábil de cada mes).")

        st.dataframe(
            df_show,
            use_container_width=True,
            hide_index=True
        )

        df_validos = df_fut[
            (df_fut["Último"].notna()) |
            (df_fut["Bid"].notna()) |
            (df_fut["Ask"].notna()) |
            (df_fut["Volumen"] > 0)
        ].copy()

        st.caption(f"Contratos con algún dato visible: {len(df_validos)} / {len(df_fut)}")

    except Exception as e:
        st.error(f"Error cargando futuros A3: {e}")

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


#cd "C:\Users\msegu\OneDrive\Desktop\mi-curva-tasa-fija"
"""
git add .
git commit -m "update app corporativos"
git push

"""

"""
git pull 
"""