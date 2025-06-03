import pandas as pd
import numpy as np
import plotly.graph_objects as go
from glob import glob

# loading data
file_paths = sorted(glob("mig5-114-2025-*"))
dataframes = [pd.read_csv(path, sep='\t', engine='python', header=None) for path in file_paths]
df = pd.concat(dataframes, ignore_index=True)
df = df.drop([0,1,2,3,5,6,8,9,10,11], axis='rows').reset_index(drop=True)

# processing raw data
rotameter_pct    = df[0].values
p_ambient        = df[1].values
p_sonic_gauge    = df[2].values
temp_C           = df[3].values
pos_closed_in    = df[4].values
pos_current_in   = df[5].values
p_orifice_abs    = df[6].values
dp_orifice_taps  = df[[7, 8, 9, 10, 12]].astype(float).values
p_turbine_abs    = df[13].values
f_turbine_kHz    = df[14].values / 1000.0
p_laminar_abs    = df[15].values
dp_laminar_inH2O = df[16].values
p_venturi_abs    = df[17].values
dp_venturi_taps  = df[[18, 19, 20, 21, 23]].astype(float).values

# constants
mu_air     = 1.861e-5   # Pa·s
rho_air    = 1.1614     # kg/m^3
gamma_air  = 1.4
R_air      = 286.9      # J/kg·K
g          = 9.81       # m/s²
D_sonic         = 0.0079248
phi = np.radians(8.88) # Sonic nozzle half angle [rad]

# venturi geometry
D_venturi     = 0.0520446
d_throat      = 0.0258064
A1_venturi    = np.pi * (D_venturi / 2)**2
A2_venturi    = np.pi * (d_throat / 2)**2

# pressure differential
dp0_venturi = dp_venturi_taps[:, 0]
m_dot_ideal = A2_venturi * np.sqrt(2 * rho_air * dp0_venturi / (1 - (A2_venturi / A1_venturi)**2))

# Reynolds number at inlet
Re_venturi = (m_dot_ideal * D_venturi) / (A1_venturi * mu_air)

# compressibility correction (Y factor)
p1 = p_venturi_abs
p2 = p1 - dp0_venturi
Y_venturi = np.sqrt(
    (p2 / p1)**(2 / gamma_air) * (gamma_air / (gamma_air - 1)) *
    ((1 - (p2 / p1)**((gamma_air - 1) / gamma_air)) / (1 - (p2 / p1))) *
    ((1 - (A2_venturi / A1_venturi)**2) / (1 - (A2_venturi / A1_venturi)**2 * (p2 / p1)**(2 / gamma_air)))
)

# discharge coefficient (assumed or interpolated)
Cd_vals = np.array([0.92, 0.93, 0.94, 0.945, 0.948, 0.95, 0.952, 0.955, 0.957, 0.96])
Cd_interp = np.interp(np.arange(len(m_dot_ideal)), np.linspace(0, len(m_dot_ideal)-1, len(Cd_vals)), Cd_vals)
m_dot_actual = m_dot_ideal * Cd_interp * Y_venturi

# pressure recovery profile
tap_locs = np.array([0, 0.5 * D_venturi, 1.0 * D_venturi, 1.5 * D_venturi, 0.1450])
recovery = (p_venturi_abs[:, None] - dp_venturi_taps) / p_venturi_abs[:, None]

fig = go.Figure()
for i in range(len(df)):
    fig.add_trace(go.Scatter(
        x=tap_locs,
        y=recovery[i],
        mode='lines+markers',
        name=f"{int(rotameter_pct[i])}% flow"
    ))

fig.update_layout(
    title=dict(
        text="Venturi Meter: Pressure Recovery Profile",
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Tap Position from Throat (m)",
    yaxis_title="Recovery Fraction",
    legend_title="Flow Rate (%)",
    template="plotly_white",
    height=600,
    width=900
)
fig.show()


# ORIFICE PLATE
# orifice geometry
D_orifice     = 0.0520446  # same as venturi inlet
d_orifice     = 0.0254     # orifice diameter
A1_orifice    = np.pi * (D_orifice / 2)**2
A2_orifice    = np.pi * (d_orifice / 2)**2

# ideal mass flow rate through the orifice (assuming incompressible)
dp0_orifice      = dp_orifice_taps[:, 0]
mdot_ideal_orif  = A2_orifice * np.sqrt(2 * rho_air * dp0_orifice / (1 - (A2_orifice / A1_orifice)**2))

# pressure upstream and downstream of orifice
P1_orifice = p_orifice_abs
P2_orifice = P1_orifice - dp0_orifice

# compressibility correction factor for orifice flow (Y)
Y_orifice = 1 - (0.41 + 0.35 * (d_orifice / D_orifice)**4) * (1 - P2_orifice / P1_orifice) / gamma_air

# experimental discharge coefficient, using Venturi flow as reference
Cd_orifice = m_dot_actual / (Y_orifice * mdot_ideal_orif)

# fig 2: Orifice Plate Discharge Coefficient vs. Reynolds Number
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=Re_venturi,
    y=Cd_orifice,
    mode='markers',
    marker=dict(size=8, color='navy'),
    name="Experimental Cd"
))

fig.update_layout(
    title=dict(
        text="Orifice Plate: Discharge Coefficient vs. Reynolds Number",
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Reynolds Number (Re)",
    yaxis_title="Discharge Coefficient (Cd)",
    template="plotly_white",
    height=500,
    width=800
)
fig.show()


# fig 3: Orifice Plate Pressure Recovery vs. Tap Location
# tap positions downstream of the orifice, in meters
tap_locs_orifice = D_orifice * np.array([0.5, 1, 1.5, 2, 4])

# pressure recovery calculation at each tap
recovery_orifice = (p_orifice_abs[:, None] - dp_orifice_taps) / p_orifice_abs[:, None]

fig = go.Figure()
for i in range(len(df)):
    fig.add_trace(go.Scatter(
        x=tap_locs_orifice,
        y=recovery_orifice[i],
        mode='lines+markers',
        name=f"{int(rotameter_pct[i])}% flow"
    ))

fig.update_layout(
    title=dict(
        text="Orifice Plate: Pressure Recovery vs. Tap Location",
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Tap Position from Plate (m)",
    yaxis_title="Recovery Fraction",
    legend_title="Flow Rate (%)",
    template="plotly_white",
    height=600,
    width=900
)
fig.show()


# SONIC NOZZLE
# nozzle opening height in meters
h_sonic = (pos_current_in - pos_closed_in) * 0.0254

# effective throat area based on cone geometry
A_sonic = (np.pi / 4) * (
    D_sonic**2 - (D_sonic - 2 * h_sonic * np.tan(phi))**2
)

temp_K = temp_C + 273.15

# theoretical choked flow through a converging nozzle
mdot_ideal_sonic = A_sonic * 1e5 * np.sqrt(
    (2 / (R_air * temp_K)) *
    (gamma_air / (gamma_air + 1)) *
    (2 / (gamma_air + 1))**(2 / (gamma_air - 1))
)

# discharge coefficient for sonic nozzle (reference = Venturi)
Cd_sonic = m_dot_actual / mdot_ideal_sonic

# fig 4: Sonic Nozzle Discharge Coefficient vs. Reynolds Number
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=Re_venturi,
    y=Cd_sonic,
    mode='markers',
    marker=dict(size=8, symbol='circle', color='darkgreen'),
    name="Sonic Cd"
))

fig.update_layout(
    title=dict(
        text="Sonic Nozzle: Discharge Coefficient vs. Reynolds Number",
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Reynolds Number (Re)",
    yaxis_title="Discharge Coefficient (Cd)",
    template="plotly_white",
    height=500,
    width=800
)
fig.show()


# LAMINAR FLOW METER
# convert Venturi mass flow rate to CFM
cfm_conversion = 0.00047194745  # (m³/s) → CFM
venturi_flow_cfm = (m_dot_actual * rho_air) / cfm_conversion

# linear fit on experimental data
coeffs_laminar = np.polyfit(dp_laminar_inH2O, venturi_flow_cfm, 1)
fit_line_laminar = np.polyval(coeffs_laminar, dp_laminar_inH2O)

# theoretical model: 40 CFM at 8 in H₂O → slope = 5 CFM/in
theory_line = 5 * dp_laminar_inH2O

# fig 5: Laminar Flow Meter CFM vs. Pressure Drop
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=dp_laminar_inH2O,
    y=venturi_flow_cfm,
    mode='markers',
    marker=dict(size=8, color='darkblue'),
    name="Experimental"
))

fig.add_trace(go.Scatter(
    x=dp_laminar_inH2O,
    y=fit_line_laminar,
    mode='lines',
    line=dict(width=2, color='royalblue'),
    name="Fitted Experimental"
))

fig.add_trace(go.Scatter(
    x=dp_laminar_inH2O,
    y=theory_line,
    mode='lines',
    line=dict(dash='dash', width=2, color='gray'),
    name="Theoretical (40 CFM @ 8 in H₂O)"
))

fig.update_layout(
    title=dict(
        text="Laminar Flow Meter: Flowrate vs. Pressure Drop",
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Normalized Pressure Drop (in H₂O)",
    yaxis_title="Flowrate (CFM)",
    legend_title="Legend",
    template="plotly_white",
    height=500,
    width=800
)

fig.show()


# TURBINE FLOW METER
# linear regression between turbine frequency and flowrate
coeffs_turbine = np.polyfit(f_turbine_kHz, venturi_flow_cfm, 1)
fit_line_turbine = np.polyval(coeffs_turbine, f_turbine_kHz)

# theoretical line: 40 CFM at 1 kHz -> slope = 40
theory_line_turbine = 40 * f_turbine_kHz


# fig 6: Turbine Frequency vs. Flowrate
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=f_turbine_kHz,
    y=venturi_flow_cfm,
    mode='markers',
    marker=dict(size=8, color='darkred'),
    name="Experimental"
))

fig.add_trace(go.Scatter(
    x=f_turbine_kHz,
    y=fit_line_turbine,
    mode='lines',
    line=dict(width=2, color='firebrick'),
    name="Fitted Experimental"
))

fig.add_trace(go.Scatter(
    x=f_turbine_kHz,
    y=theory_line_turbine,
    mode='lines',
    line=dict(dash='dash', width=2, color='gray'),
    name="Theoretical (40 CFM @ 1 kHz)"
))

fig.update_layout(
    title=dict(
        text="Turbine Flow Meter: Frequency vs. Flowrate",
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Turbine Frequency (kHz)",
    yaxis_title="Flowrate (CFM)",
    legend_title="Legend",
    template="plotly_white",
    height=500,
    width=800
)

fig.show()


# ROTAMETER
# linear regression between rotameter % and Venturi-derived flowrate
coeffs_rotameter = np.polyfit(rotameter_pct, venturi_flow_cfm, 1)
fit_line_rotameter = np.polyval(coeffs_rotameter, rotameter_pct)

# fig 7: Volume Flow Rate vs. Rotameter Percentage
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=rotameter_pct,
    y=venturi_flow_cfm,
    mode='markers',
    marker=dict(size=8, color='teal'),
    name="Experimental"
))

fig.add_trace(go.Scatter(
    x=rotameter_pct,
    y=fit_line_rotameter,
    mode='lines',
    line=dict(width=2, color='darkcyan'),
    name="Fitted Experimental"
))

fig.update_layout(
    title=dict(
        text="Volume Flow Rate vs. Rotameter Percentage",
        x=0.5,
        xanchor='center'
    ),
    xaxis_title="Rotameter Indication (%)",
    yaxis_title="Flowrate (CFM)",
    legend_title="Legend",
    template="plotly_white",
    height=500,
    width=800
)

fig.show()


# VENTURI RESULTS TABLE
flowrate_cfm_venturi = (m_dot_actual * rho_air) / cfm_conversion

# printing table
venturi_table = pd.DataFrame({
    "Test #":            np.arange(1, len(df) + 1),
    "Reynolds Number":   np.round(Re_venturi, 0),
    "Compressibility Ya": np.round(Y_venturi, 4),
    "Ideal Mass Flow (kg/s)":  np.round(m_dot_ideal, 5),
    "Actual Mass Flow (kg/s)":    np.round(m_dot_actual, 5)
})

print("\n=== Venturi Results Table ===")
print(venturi_table.to_string(index=False))


# ORIFICE RESULTS TABLE
orifice_table = pd.DataFrame({
    "Test #":               np.arange(1, len(df) + 1),
    "Reynolds number": np.round(Re_venturi,0),
    "Compressibility": np.round(Y_orifice, 4),
    "Ideal Mass Flow (kg/s)": np.round(mdot_ideal_orif, 5),
    "Cd (Orifice)":         np.round(Cd_orifice, 4)
})

print("\n=== Orifice Results Table ===")
print(orifice_table.to_string(index=False))


# SONIC NOZZLE RESULTS TABLE
sonic_table = pd.DataFrame({
    "Test #":                np.arange(1, len(df) + 1),
    "Reynolds": np.round(Re_venturi, 0),
    "Ideal Mass Flow (kg/s)": np.round(mdot_ideal_sonic, 5),
    "Cd (Sonic Nozzle)":      np.round(Cd_sonic, 4)
})

print("\n=== Sonic Nozzle Results Table ===")
print(sonic_table.to_string(index=False))


# CALIBRATION CURVE COEFFICIENTS (Least-Squares Fit)
# laminar flow meter
coeffs_laminar = np.polyfit(dp_laminar_inH2O, venturi_flow_cfm, 1)
slope_laminar, intercept_laminar = coeffs_laminar
slope_per_inH2O = slope_laminar / 8.0  # Normalize to per inch of H₂O

print("\n--- Laminar Flow Meter Calibration ---")
print(f"Slope       (CFM per ΔP/8 in)     = {slope_laminar:.3f}")
print(f"Intercept   (CFM at 0 ΔP)          = {intercept_laminar:.3f}")
print(f"Slope       (CFM per 1 in H₂O)     = {slope_per_inH2O:.3f}")

# turbine flow meter
coeffs_turbine = np.polyfit(f_turbine_kHz, venturi_flow_cfm, 1)
slope_turbine, intercept_turbine = coeffs_turbine

print("\n--- Turbine Flow Meter Calibration ---")
print(f"Slope       (CFM per kHz)          = {slope_turbine:.3f}")
print(f"Intercept   (CFM at 0 kHz)         = {intercept_turbine:.3f}")

# rotameter
coeffs_rotameter = np.polyfit(rotameter_pct, venturi_flow_cfm, 1)
slope_rotameter, intercept_rotameter = coeffs_rotameter

print("\n--- Rotameter Calibration ---")
print(f"Slope       (CFM per 1%)           = {slope_rotameter:.3f}")
print(f"Intercept   (CFM at 0%)            = {intercept_rotameter:.3f}")
