# ==========================================================
# 📊 LANZAMIENTOS — SCRIPT FINAL FINAL FINAL
# Incluye: IA SKU + IA Mercado + Parque Vehicular + Penetración
# Semáforo, Score Inteligente, Recomendación Corporativa,
# 80/20 detalle, Top6 (solo en vista general), layout ejecutivo.
# Además: logos animados arriba que cambian de lugar cada 5s.
# ==========================================================

import pandas as pd
import numpy as np
import calendar
import ast
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objs as go

# ---------------- CONFIG ----------------
FILE = "TENDENCIA.xlsx"
SHEET = "VENTA"

# ---------------- LOAD DATA ----------------
df = pd.read_excel(FILE, sheet_name=SHEET)

base_cols = [
    "ITEM", "LINEA", "FAMILIA", "NOMBRE CORTO",
    "FECHA_LANZAMIENTO", "STATUS", "PRODUCCION MES", "PARQUE VEHICULAR"
]

for c in base_cols:
    if c not in df.columns:
        df[c] = np.nan

# Ensure STATUS uppercase safely
df["STATUS"] = df["STATUS"].astype(str).str.upper().str.strip()

# Date/value columns are all others
cols_fecha = [c for c in df.columns if c not in base_cols]
for c in cols_fecha:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

# Long format
df_long = df.melt(
    id_vars=base_cols,
    value_vars=cols_fecha,
    var_name="FECHA_MES",
    value_name="VENTAS"
)

df_long["FECHA_MES"] = pd.to_datetime(df_long["FECHA_MES"], errors="coerce")
df_long["FECHA_LANZAMIENTO"] = pd.to_datetime(df_long["FECHA_LANZAMIENTO"], errors="coerce")

df_long = df_long[df_long["FECHA_MES"].notna()]
df_long = df_long[df_long["FECHA_MES"] >= "2020-01-01"]
df_long["VENTAS"] = df_long["VENTAS"].fillna(0)

df_long["AÑO"] = df_long["FECHA_MES"].dt.year
df_long["MES"] = df_long["FECHA_MES"].dt.month

df_long["MESES_DESDE_LANZ"] = (
    (df_long["FECHA_MES"].dt.year - df_long["FECHA_LANZAMIENTO"].dt.year) * 12 +
    (df_long["FECHA_MES"].dt.month - df_long["FECHA_LANZAMIENTO"].dt.month)
)

df_long["CATEGORIA_LANZ"] = np.where(
    df_long["MESES_DESDE_LANZ"].between(0, 7),
    "LANZAMIENTO",
    "NORMAL"
)

# ---------------- UI COLORS ----------------
WHITE = "#FFFFFF"
BG = "#F5F6F8"
TEXT = "#1C1C1E"
PRIMARY = "#0050D4"
ACCENT = "#FF8C00"
IA_BG = "#0B2149"
IA_TEXT = "#FFFFFF"

# ---------------- APP ----------------
app = dash.Dash(__name__)
app.title = "Análisis de Lanzamientos"

# ---------------- LAYOUT ----------------
app.layout = html.Div([
    # Logos container (positioned relative para que los absolute se ubiquen correctamente)
    html.Div([
        html.Img(id="logo_left", src="/assets/bruck.png", style={
            "height": "60px",
            "transition": "opacity 1s, left 1s, right 1s",
            "position": "absolute",
            "left": "20px",
            "top": "10px",
            "opacity": 1
        }),
        html.Img(id="logo_right", src="/assets/interauto.png", style={
            "height": "60px",
            "transition": "opacity 1s, left 1s, right 1s",
            "position": "absolute",
            "right": "20px",
            "top": "10px",
            "opacity": 1
        })
    ], style={"position": "relative", "height": "90px", "marginBottom": "6px"}),

    # Interval that triggers every 5 seconds (5000 ms)
    dcc.Interval(id="interval-logos", interval=5000, n_intervals=0),

    html.H2(
        "Análisis de Lanzamientos",
        style={"textAlign": "center", "color": TEXT, "marginTop": "10px", "fontWeight": "800"}
    ),

    html.Div([
        html.Div(id="resumen_ejecutivo", style={"width": "25%", "float": "left", "padding": "8px"}),

        html.Div([
            html.Div([
                dcc.Dropdown(
                    id="linea",
                    options=[{"label": x, "value": x} for x in sorted(df_long["LINEA"].dropna().unique())],
                    placeholder="LINEA",
                    style={"width": "170px", "display": "inline-block", "marginRight": "8px"}
                ),
                dcc.Dropdown(
                    id="categoria",
                    options=[
                        {"label": "TODOS", "value": "TODOS"},
                        {"label": "LANZAMIENTO", "value": "LANZAMIENTO"},
                        {"label": "NORMAL", "value": "NORMAL"},
                    ],
                    value="TODOS",
                    style={"width": "170px", "display": "inline-block", "marginRight": "8px"}
                ),
                dcc.Dropdown(
                    id="sku",
                    placeholder="ITEM",
                    style={"width": "220px", "display": "inline-block"}
                )
            ], style={
                "background": WHITE, "padding": "8px", "borderRadius": "6px",
                "marginBottom": "10px", "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"
            }),

            html.Div(id="analisis_ia", style={"marginBottom": "12px"}),

            dcc.Graph(id="grafico", style={"height": "520px"}),

            html.Div(id="cards_below"),
            html.Div(id="detalle_container")
        ], style={"width": "72%", "float": "right", "padding": "6px"})
    ]),
    html.Div(style={"clear": "both"})
], style={"background": BG, "minHeight": "100vh", "paddingBottom": "40px"})

# ---------------- HELPERS ----------------
def apply_filters(df, linea, categoria):
    d = df.copy()
    if linea:
        d = d[d["LINEA"] == linea]
    if categoria and categoria != "TODOS":
        d = d[d["CATEGORIA_LANZ"] == categoria]
    return d

# ---------------- SKU OPTIONS ----------------
@app.callback(
    Output("sku", "options"),
    Output("sku", "value"),
    Input("linea", "value"),
    Input("categoria", "value")
)
def upd_sku(linea, categoria):
    d = apply_filters(df_long, linea, categoria)
    items = sorted(d["ITEM"].dropna().unique())
    return [{"label": x, "value": x} for x in items], None

# ---------------- MAIN CALLBACK ----------------
@app.callback(
    Output("grafico", "figure"),
    Output("cards_below", "children"),
    Output("resumen_ejecutivo", "children"),
    Output("analisis_ia", "children"),
    Input("linea", "value"),
    Input("categoria", "value"),
    Input("sku", "value")
)
def update_all(linea, categoria, sku):
    d = apply_filters(df_long, linea, categoria)

    # =============== SI SKU SELECCIONADO ===============
    if sku:
        ch = d[d["ITEM"] == sku].sort_values("FECHA_MES")
        if ch.empty:
            return go.Figure(), "", "", ""

        fig = go.Figure(go.Bar(
            x=ch["FECHA_MES"], y=ch["VENTAS"],
            marker_color=np.where(ch["CATEGORIA_LANZ"] == "LANZAMIENTO", ACCENT, PRIMARY)
        ))

        fecha_ini = ch["FECHA_LANZAMIENTO"].iloc[0]
        fecha_mad = fecha_ini + pd.DateOffset(months=8)

        fig.add_shape(type="line", xref="x", yref="paper",
                      x0=fecha_ini, x1=fecha_ini, y0=0, y1=1,
                      line=dict(color=ACCENT, width=2, dash="dot"))
        fig.add_annotation(x=fecha_ini, y=1.02, xref="x", yref="paper",
                           text="Lanzamiento", showarrow=False, font=dict(color=ACCENT))

        fig.add_shape(type="line", xref="x", yref="paper",
                      x0=fecha_mad, x1=fecha_mad, y0=0, y1=1,
                      line=dict(color="#999", width=2, dash="dot"))
        fig.add_annotation(x=fecha_mad, y=1.02, xref="x", yref="paper",
                           text="Madurez", showarrow=False, font=dict(color="#666"))

        fig.update_layout(title=f"📊 Evolución Ventas — {sku}", title_x=0.5,
                          paper_bgcolor=WHITE, plot_bgcolor=WHITE)

        ventas_total = ch["VENTAS"].sum()
        meses = ch.shape[0]
        prom = ventas_total / meses if meses > 0 else 0
        mejor_idx = ch["VENTAS"].idxmax()
        mejor_mes_row = ch.loc[mejor_idx] if not ch.empty else None
        mejor_mes_fecha = mejor_mes_row["FECHA_MES"].strftime("%B %Y") if mejor_mes_row is not None else "N/D"
        mejor_mes_val = int(mejor_mes_row["VENTAS"]) if mejor_mes_row is not None else 0
        crecimiento = ch["VENTAS"].iloc[-1] - ch["VENTAS"].iloc[0] if meses > 1 else 0
        meses_desde_lanz = int(ch["MESES_DESDE_LANZ"].iloc[-1])

        texto_item_ia = (
            f"SKU: {sku}\n"
            f"Ventas totales: {ventas_total:,.0f} pzas\n"
            f"Promedio mensual: {prom:,.0f}\n"
            f"Meses desde lanzamiento: {meses_desde_lanz}\n"
            f"Mejor mes: {mejor_mes_fecha} — {mejor_mes_val:,} pzas\n"
            f"Tendencia: {'En crecimiento' if crecimiento > 0 else 'En caída' if crecimiento < 0 else 'Estable'}\n"
            "\n"
            "📌 Insight adicional:\n"
            "El SKU presenta ventas acumuladas relevantes y una tendencia a la baja.\n"
            "Revisar rotación en plazas clave y generar impulso comercial.\n"
        )

        parque = 0
        if "PARQUE VEHICULAR" in ch.columns:
            try:
                parque = int(ch["PARQUE VEHICULAR"].iloc[0]) if pd.notna(ch["PARQUE VEHICULAR"].iloc[0]) else 0
            except Exception:
                parque = 0

        penetracion = (ventas_total / parque * 100) if (parque > 0) else 0.0

        if penetracion > 5:
            semaforo = "Alto dominio"; sem_color = "#2ECC71"
        elif penetracion > 2:
            semaforo = "En desarrollo"; sem_color = "#F1C40F"
        else:
            semaforo = "Oportunidad alta"; sem_color = "#E74C3C"

        prom_norm = min(prom / 200.0, 1.0)
        tendencia_factor = 1.0 if crecimiento > 0 else (0.5 if crecimiento == 0 else 0.2)
        penetr_norm = min(penetracion / 10.0, 1.0)
        score = round((penetr_norm*0.5 + tendencia_factor*0.3 + prom_norm*0.2) * 100, 1)

        if score >= 80:
            recomendacion = "Reforzar disponibilidad e inventario."
        elif score >= 60:
            recomendacion = "Acciones comerciales y visibilidad."
        else:
            recomendacion = "Revisar estrategia y rotación."

        col1 = html.Div([
            html.H4("Insight SKU", style={"color": "#FFD700", "marginBottom": "8px"}),
            html.Pre(texto_item_ia, style={"color": IA_TEXT, "whiteSpace": "pre-wrap", "fontSize": "14px"})
        ], style={"background": IA_BG, "padding": "16px","borderRadius": "10px","flex": "1","minWidth": "280px"})

        col2 = html.Div([
            html.H4("IA Mercado & Score", style={"color": "#00E5FF", "marginBottom": "8px"}),
            html.Pre(
                f"Parque Vehicular: {parque:,}\n"
                f"Penetración: {penetracion:.2f}%\n"
                f"Estado: {semaforo}\n"
                f"Score: {score:.0f}/100",
                style={"color": IA_TEXT, "whiteSpace": "pre-wrap", "fontSize": "14px"}
            ),
            html.Div(recomendacion, style={"color": IA_TEXT, "marginTop": "6px"})
        ], style={"background": IA_BG,"padding": "16px","borderRadius": "10px","flex": "1","minWidth": "280px"})

        ia_block = html.Div([col1, col2], style={
            "display": "flex","flexWrap": "wrap","gap": "12px","width": "100%","marginBottom": "14px"
        })

        top_cards = ""

        items = df_long[df_long["FECHA_LANZAMIENTO"].notna()].drop_duplicates("ITEM")
        items["YEAR"] = items["FECHA_LANZAMIENTO"].dt.year
        df_sum = items.groupby("YEAR").agg(
            Lanzados=("ITEM", "count"),
            Baja=("STATUS", lambda x: x.isin(["BAJA", "DESCONTINUADO"]).sum())
        ).reset_index()
        total_row = {"YEAR": "TOTAL","Lanzados": int(df_sum["Lanzados"].sum()),"Baja": int(df_sum["Baja"].sum())}
        df_sum = pd.concat([df_sum, pd.DataFrame([total_row])], ignore_index=True)

        resumen_table = dash_table.DataTable(
            data=df_sum.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_sum.columns],
            style_header={"backgroundColor": PRIMARY, "color": WHITE, "fontWeight": "700"},
            style_cell={"textAlign": "center", "padding": "6px", "fontSize": "12px"},
            page_size=10
        )

        return fig, top_cards, resumen_table, ia_block

    # =============== VISTA GENERAL SIN SKU ===============
    tot = d.groupby("FECHA_MES")["VENTAS"].sum().reset_index()
    fig_total = go.Figure(go.Bar(x=tot["FECHA_MES"], y=tot["VENTAS"], marker_color=PRIMARY))
    fig_total.update_layout(title="📊 Total General", title_x=0.5, paper_bgcolor=WHITE, plot_bgcolor=WHITE)

    lanz = apply_filters(df_long, linea, "LANZAMIENTO")
    if lanz.empty:
        top_cards = ""
    else:
        top = lanz.groupby(lanz["FECHA_MES"].dt.month)["VENTAS"].sum().sort_values(ascending=False).head(6)
        cards = []
        pos = 1
        for m, val in top.items():
            dm = lanz[lanz["FECHA_MES"].dt.month == m]
            total_sku = dm["ITEM"].nunique()
            cards.append(html.Div([
                html.Div(f"#{pos}", style={"fontWeight": "900","color": PRIMARY,"fontSize": "18px"}),
                html.Div(calendar.month_name[int(m)], style={"fontSize": "15px", "fontWeight": "700"}),
                html.Div(f"Total: {int(val):,}", style={"fontSize": "13px"}),
                html.Div(f"# SKU: {total_sku}", style={"fontSize": "12px","marginBottom": "6px"}),
                html.Button("VER DETALLE", id={"type": "btn_mes", "index": int(m)},
                            style={"background": PRIMARY, "color": WHITE, "border": "none",
                                   "borderRadius": "6px", "padding": "6px 10px"})
            ], style={"background": WHITE,"padding": "10px","width": "15%","textAlign": "center",
                      "borderRadius": "8px","boxShadow": "0 2px 6px rgba(0,0,0,0.08)"}))
            pos += 1
        top_cards = html.Div(cards, style={"display": "flex","gap": "10px","justifyContent": "center","flexWrap": "wrap"})

    items = df_long[df_long["FECHA_LANZAMIENTO"].notna()].drop_duplicates("ITEM")
    items["YEAR"] = items["FECHA_LANZAMIENTO"].dt.year
    df_sum = items.groupby("YEAR").agg(
        Lanzados=("ITEM", "count"),
        Baja=("STATUS", lambda x: x.isin(["BAJA", "DESCONTINUADO"]).sum())
    ).reset_index()
    total_row = {"YEAR": "TOTAL","Lanzados": int(df_sum["Lanzados"].sum()),"Baja": int(df_sum["Baja"].sum())}
    df_sum = pd.concat([df_sum, pd.DataFrame([total_row])], ignore_index=True)

    resumen_table = dash_table.DataTable(
        data=df_sum.to_dict("records"),
        columns=[{"name": c, "id": c} for c in df_sum.columns],
        style_header={"backgroundColor": PRIMARY,"color": WHITE,"fontWeight": "700"},
        style_cell={"textAlign": "center","padding": "6px","fontSize": "12px"}
    )

    lanz = d[d["CATEGORIA_LANZ"] == "LANZAMIENTO"]
    if lanz.empty:
        analisis_text = "Sin datos de lanzamientos en el periodo seleccionado."
    else:
        mes_top = lanz.groupby(lanz["FECHA_LANZAMIENTO"].dt.to_period("M")).agg(
            piezas=("VENTAS", "sum"), items=("ITEM", "nunique")
        ).reset_index()
        mes_top["FECHA_LANZAMIENTO"] = mes_top["FECHA_LANZAMIENTO"].dt.to_timestamp()
        top_mes = mes_top.sort_values("piezas", ascending=False).iloc[0]

        total_items = d["ITEM"].nunique() if not d.empty else 0
        tendencia_general = (
            "En crecimiento" if (not tot.empty and tot["VENTAS"].iloc[-1] > tot["VENTAS"].iloc[0]) else
            "En caída" if (not tot.empty and tot["VENTAS"].iloc[-1] < tot["VENTAS"].iloc[0]) else
            "Estable"
        )

        analisis_text = (
            f"Mejor mes: {top_mes['FECHA_LANZAMIENTO'].strftime('%B %Y')} — {int(top_mes['piezas']):,} pzas ({int(top_mes['items'])} items)\n"
            f"\n"
            f"Lanzamientos totales analizados: {total_items}\n"
            f"Crecimiento general: {tendencia_general}\n"
            f"Observación: Comportamiento alineado a estacionalidad del mercado."
        )

    analisis_card = html.Div([
        html.H4("IA — Análisis General", style={"color": "#FFD700"}),
        html.Pre(analisis_text, style={"color": IA_TEXT, "whiteSpace": "pre-wrap"})
    ], style={"background": IA_BG,"padding": "16px","borderRadius": "10px","width": "60%","margin": "0 auto"})

    return fig_total, top_cards, resumen_table, analisis_card

# ---------------- DETALLE TOP 6 ----------------
@app.callback(
    Output("detalle_container", "children"),
    Input({"type": "btn_mes", "index": ALL}, "n_clicks"),
    State("linea", "value"), State("categoria", "value")
)
def detalle(clicks, linea, categoria):
    if not any(clicks):
        return ""
    triggered = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    mes = int(ast.literal_eval(triggered)["index"])
    d = apply_filters(df_long, linea, categoria)
    d = d[(d["CATEGORIA_LANZ"] == "LANZAMIENTO") & (d["FECHA_MES"].dt.month == mes)]
    if d.empty:
        return html.Div("No hay datos para este mes.", style={"padding": "12px", "background": WHITE})
    df_det = d.groupby(["ITEM", "LINEA", "FAMILIA", "NOMBRE CORTO"])["VENTAS"].sum().reset_index().sort_values("VENTAS", ascending=False)
    df_det["ACUM"] = df_det["VENTAS"].cumsum()
    total = df_det["VENTAS"].sum()
    df_det["ACUM_VAL"] = df_det["ACUM"] / total * 100
    df_det["ACUM_PCT"] = df_det["ACUM_VAL"].round(1).astype(str) + "%"

    table = dash_table.DataTable(
        data=df_det.to_dict("records"),
        columns=[
            {"name": "ITEM", "id": "ITEM"},
            {"name": "LINEA", "id": "LINEA"},
            {"name": "FAMILIA", "id": "FAMILIA"},
            {"name": "NOMBRE", "id": "NOMBRE CORTO"},
            {"name": "VENTAS", "id": "VENTAS"},
            {"name": "ACUM", "id": "ACUM"},
            {"name": "ACUM %", "id": "ACUM_PCT"},
        ],
        sort_action="native",
        page_size=25,
        style_header={"backgroundColor": PRIMARY, "color": WHITE, "fontWeight": "bold"},
        style_cell={"padding": "6px","textAlign": "center","fontSize": "12px"},
        style_data_conditional=[
            {"if": {"filter_query": "{ACUM_VAL} <= 80"}, "backgroundColor": "#C6F6D5", "color": "#1B5E20", "fontWeight": "bold"},
            {"if": {"filter_query": "{ACUM_VAL} > 80"}, "backgroundColor": "#FFCDD2", "color": "#B71C1C", "fontWeight": "bold"}
        ]
    )

    return html.Div([
        html.H4(f"Detalle — {calendar.month_name[mes]}", style={"color": PRIMARY}),
        table
    ], style={"padding": "10px", "background": WHITE, "borderRadius": "10px", "boxShadow": "0 2px 6px rgba(0,0,0,0.10)"})


# ---------------- LOGOS: interval callback para alternar cada 5s ----------------
@app.callback(
    Output("logo_left", "src"),
    Output("logo_right", "src"),
    Output("logo_left", "style"),
    Output("logo_right", "style"),
    Input("interval-logos", "n_intervals")
)
def rotate_logos(n_intervals):
    # Alterna cada 5 segundos (n_intervals incrementa por el dcc.Interval)
    t = n_intervals % 2

    bruck = "/assets/bruck.png"
    inter = "/assets/interauto.png"

    if t == 0:
        # estado A: bruck izquierda (opaco), interauto derecha (semi-desvanecido)
        left_src = bruck
        right_src = inter
        left_style = {
            "height": "60px",
            "transition": "opacity 1s, left 1s, right 1s",
            "position": "absolute",
            "left": "20px",
            "top": "10px",
            "opacity": 1
        }
        right_style = {
            "height": "60px",
            "transition": "opacity 1s, left 1s, right 1s",
            "position": "absolute",
            "right": "20px",
            "top": "10px",
            "opacity": 0.25
        }
    else:
        # estado B: interauto izquierda (opaco), bruck derecha (semi-desvanecido)
        left_src = inter
        right_src = bruck
        left_style = {
            "height": "60px",
            "transition": "opacity 1s, left 1s, right 1s",
            "position": "absolute",
            "left": "20px",
            "top": "10px",
            "opacity": 1
        }
        right_style = {
            "height": "60px",
            "transition": "opacity 1s, left 1s, right 1s",
            "position": "absolute",
            "right": "20px",
            "top": "10px",
            "opacity": 0.25
        }

    return left_src, right_src, left_style, right_style


server = app.server  # Necesario para PythonAnywhere

if __name__ == "__main__":
    app.run(debug=True)
