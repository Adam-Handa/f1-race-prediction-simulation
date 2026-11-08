"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  🏎️  F1 2026 MEXICO CITY GP — AI RACE PREDICTION SIMULATOR                 ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║                                                                              ║
║  An ML-powered race outcome predictor with a full animated simulation        ║
║  of drivers racing around the real Autódromo Hermanos Rodríguez circuit.     ║
║                                                                              ║
║  Features:                                                                   ║
║    • Random Forest ML model trained on 2026 F1 season data                   ║
║    • Real circuit layout loaded from FastF1 telemetry (GPS coordinates)      ║
║    • 22 animated driver dots with team-colored labels                        ║
║    • Interactive speed controls (1×, 2×, 5×)                                 ║
║    • Toggleable telemetry overlay (Speed trace + Gear map)                   ║
║    • Color-coded speed track map                                             ║
║    • F1 broadcast-style live timing tower                                    ║
║    • DRS zones, sector markings, and corner annotations                      ║
║                                                                              ║
║  Dependencies: fastf1, matplotlib, numpy, pandas, scikit-learn               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import numpy as np
import pandas as pd
import fastf1
from sklearn.ensemble import RandomForestRegressor
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.collections import LineCollection
from matplotlib import patheffects

# ═══════════════════════════════════════════════════════════════════════════════
#  THEME — Dark F1 Broadcast Style
# ═══════════════════════════════════════════════════════════════════════════════
BG_MAIN    = '#080808'
BG_PANEL   = '#0f0f0f'
BG_CARD    = '#1a1a1a'
F1_RED     = '#E10600'
SECTOR_1   = '#E10600'   # Red
SECTOR_2   = '#00BFFF'   # Cyan/Blue
SECTOR_3   = '#FFD700'   # Yellow
DRS_GREEN  = '#33FF33'
TXT_WHITE  = '#FFFFFF'
TXT_GRAY   = '#999999'
TXT_DIM    = '#444444'
TRACK_EDGE = '#282828'
TRACK_SURF = '#3c3c3c'

# ═══════════════════════════════════════════════════════════════════════════════
#  1. CACHE & SETUP
# ═══════════════════════════════════════════════════════════════════════════════
cache_dir = 'cache_folder'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# ═══════════════════════════════════════════════════════════════════════════════
#  2. ML MODEL — Train on past 2026 races
# ═══════════════════════════════════════════════════════════════════════════════
print("━" * 60)
print("  🏎️  F1 2026 MEXICO CITY GP — AI RACE PREDICTOR")
print("━" * 60)
print("\n[1/7] Gathering training data from 2026 season...")

training_data = []
past_races = ['Australian Grand Prix', 'Japanese Grand Prix',
              'Miami Grand Prix', 'United States Grand Prix']

for race in past_races:
    try:
        s = fastf1.get_session(2026, race, 'R')
        s.load(telemetry=False, weather=False, messages=False)
        df = s.results[['GridPosition', 'Position']].dropna()
        training_data.append(df)
        print(f"       ✓ {race}")
    except Exception as e:
        print(f"       ✗ {race}: {e}")

train_df = pd.concat(training_data)

print("\n[2/7] Training Random Forest model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_df[['GridPosition']], train_df['Position'])
print("       ✓ Model ready (100 estimators)")

# ═══════════════════════════════════════════════════════════════════════════════
#  3. 2026 DRIVER GRID — All 22 drivers with official team colors
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Loading 2026 driver grid...")

drivers_2026 = [
    {'Abbreviation': 'ANT', 'Driver': 'Kimi Antonelli',    'GridPosition': 1,  'Team': 'Mercedes',        'Color': '#00D2BE'},
    {'Abbreviation': 'VER', 'Driver': 'Max Verstappen',    'GridPosition': 2,  'Team': 'Red Bull Racing',  'Color': '#3671C6'},
    {'Abbreviation': 'HAD', 'Driver': 'Isack Hadjar',      'GridPosition': 3,  'Team': 'Red Bull Racing',  'Color': '#3671C6'},
    {'Abbreviation': 'LEC', 'Driver': 'Charles Leclerc',   'GridPosition': 4,  'Team': 'Ferrari',          'Color': '#E8002D'},
    {'Abbreviation': 'HAM', 'Driver': 'Lewis Hamilton',    'GridPosition': 5,  'Team': 'Ferrari',          'Color': '#E8002D'},
    {'Abbreviation': 'RUS', 'Driver': 'George Russell',    'GridPosition': 6,  'Team': 'Mercedes',        'Color': '#00D2BE'},
    {'Abbreviation': 'PIA', 'Driver': 'Oscar Piastri',     'GridPosition': 7,  'Team': 'McLaren',          'Color': '#FF8000'},
    {'Abbreviation': 'NOR', 'Driver': 'Lando Norris',      'GridPosition': 8,  'Team': 'McLaren',          'Color': '#FF8000'},
    {'Abbreviation': 'LAW', 'Driver': 'Liam Lawson',       'GridPosition': 9,  'Team': 'RB',               'Color': '#6692FF'},
    {'Abbreviation': 'GAS', 'Driver': 'Pierre Gasly',      'GridPosition': 10, 'Team': 'Alpine',           'Color': '#0093CC'},
    {'Abbreviation': 'HUL', 'Driver': 'Nico Hülkenberg',   'GridPosition': 11, 'Team': 'Sauber',           'Color': '#52E252'},
    {'Abbreviation': 'ALO', 'Driver': 'Fernando Alonso',   'GridPosition': 12, 'Team': 'Aston Martin',     'Color': '#229971'},
    {'Abbreviation': 'BEA', 'Driver': 'Oliver Bearman',    'GridPosition': 13, 'Team': 'Haas',             'Color': '#B6BABD'},
    {'Abbreviation': 'LIN', 'Driver': 'Arvid Lindblad',    'GridPosition': 14, 'Team': 'RB',               'Color': '#6692FF'},
    {'Abbreviation': 'OCO', 'Driver': 'Esteban Ocon',      'GridPosition': 15, 'Team': 'Haas',             'Color': '#B6BABD'},
    {'Abbreviation': 'STR', 'Driver': 'Lance Stroll',      'GridPosition': 16, 'Team': 'Aston Martin',     'Color': '#229971'},
    {'Abbreviation': 'SAI', 'Driver': 'Carlos Sainz',      'GridPosition': 17, 'Team': 'Williams',         'Color': '#64C4FF'},
    {'Abbreviation': 'ALB', 'Driver': 'Alex Albon',        'GridPosition': 18, 'Team': 'Williams',         'Color': '#64C4FF'},
    {'Abbreviation': 'PER', 'Driver': 'Sergio Pérez',      'GridPosition': 19, 'Team': 'Cadillac',         'Color': '#C0C0C0'},
    {'Abbreviation': 'BOT', 'Driver': 'Valtteri Bottas',   'GridPosition': 20, 'Team': 'Sauber',           'Color': '#52E252'},
    {'Abbreviation': 'COL', 'Driver': 'Franco Colapinto',  'GridPosition': 21, 'Team': 'Alpine',           'Color': '#0093CC'},
    {'Abbreviation': 'BOR', 'Driver': 'Gabriel Bortoleto', 'GridPosition': 22, 'Team': 'Cadillac',         'Color': '#C0C0C0'},
]

grid_df = pd.DataFrame(drivers_2026)
NUM_DRIVERS = len(grid_df)

# ═══════════════════════════════════════════════════════════════════════════════
#  4. PREDICT RACE OUTCOME
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] Predicting race outcome...")
grid_df['PredictedPosition'] = model.predict(grid_df[['GridPosition']])
grid_df = grid_df.sort_values('PredictedPosition').reset_index(drop=True)
print("       ✓ Predictions generated for 22 drivers")

# ═══════════════════════════════════════════════════════════════════════════════
#  5. LOAD REAL CIRCUIT — Autódromo Hermanos Rodríguez from FastF1
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] Loading real Mexico City circuit from FastF1...")

telemetry_data = None
circuit_info = None

for year in [2024, 2023, 2022]:
    for gp_name in ['Mexico City Grand Prix', 'Mexican Grand Prix', 'Mexico']:
        for session_type in ['Q', 'R']:
            try:
                trk_session = fastf1.get_session(year, gp_name, session_type)
                trk_session.load()
                fastest = trk_session.laps.pick_fastest()
                telemetry_data = fastest.get_telemetry()
                circuit_info = trk_session.get_circuit_info()
                print(f"       ✓ Loaded from {year} {gp_name} ({session_type})")
                break
            except Exception as e:
                continue
        if telemetry_data is not None:
            break
    if telemetry_data is not None:
        break

# ── Process circuit coordinates ──
if telemetry_data is not None and len(telemetry_data) > 0:
    # Clean data
    telemetry_data = telemetry_data.dropna(subset=['X', 'Y']).reset_index(drop=True)

    # Rotation matrix from circuit_info
    rot_deg = circuit_info.rotation if circuit_info else 0
    theta = np.radians(rot_deg)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta),  np.cos(theta)]])

    raw_xy = np.array([telemetry_data['X'].values,
                        telemetry_data['Y'].values], dtype=float)
    rotated = rot_mat @ raw_xy
    track_x = rotated[0]
    track_y = rotated[1]

    # Telemetry channels
    telem_speed = telemetry_data['Speed'].values.astype(float)
    telem_gear  = telemetry_data['nGear'].values.astype(float) if 'nGear' in telemetry_data.columns else np.ones(len(track_x))
    telem_dist  = telemetry_data['Distance'].values.astype(float) if 'Distance' in telemetry_data.columns else np.linspace(0, 4304, len(track_x))

    # Corner annotations (rotated)
    corner_data = []
    if circuit_info is not None and circuit_info.corners is not None:
        for _, c in circuit_info.corners.iterrows():
            cxy = rot_mat @ np.array([float(c['X']), float(c['Y'])])
            corner_data.append({'n': int(c['Number']), 'x': cxy[0], 'y': cxy[1]})
else:
    print("       ⚠ FastF1 data unavailable, using approximated circuit...")
    # Hardcoded approximation of Mexico City circuit (Autódromo Hermanos Rodríguez)
    # Key waypoints traced from the official circuit map
    waypoints = np.array([
        # Start/Finish straight (top, left to right)
        [-4500, 1500], [-3500, 1500], [-2500, 1500], [-1500, 1500],
        [-500, 1500], [500, 1500], [1500, 1550], [2500, 1600],
        [3200, 1600], [3800, 1550],
        # Turn 1 (sharp right)
        [4200, 1400], [4400, 1100],
        # Turn 2-3 (chicane)
        [4300, 800], [4000, 600], [4200, 350],
        # Straight down to Turn 4
        [4100, 0], [3900, -400], [3700, -700],
        # Turn 4-5
        [3500, -1000], [3200, -1200],
        # Turn 6 (hairpin)
        [2800, -1500], [2500, -1400],
        # Turn 7-8
        [2200, -1100], [1900, -900], [1600, -800],
        # Turn 9-10 (esses)
        [1200, -700], [800, -600], [500, -500],
        # Turn 11
        [0, -300], [-400, -200],
        # Turn 12 (into stadium)
        [-1000, 0], [-1500, 200],
        # Turn 13
        [-2000, 500], [-2200, 700],
        # Turn 14-15 (stadium section)
        [-2500, 900], [-2800, 800], [-3200, 700],
        # Turn 16
        [-3500, 800], [-3800, 1000],
        # Turn 17 (Peraltada onto main straight)
        [-4200, 1200], [-4500, 1500],   # ← must match first point to close the circuit
    ])
    # Smooth with cubic interpolation (periodic = closed loop)
    from scipy.interpolate import CubicSpline
    t_param = np.linspace(0, 1, len(waypoints))
    t_fine = np.linspace(0, 1, 600, endpoint=False)  # endpoint=False avoids duplicating closure point
    cs_x = CubicSpline(t_param, waypoints[:, 0], bc_type='periodic')
    cs_y = CubicSpline(t_param, waypoints[:, 1], bc_type='periodic')
    track_x = cs_x(t_fine)
    track_y = cs_y(t_fine)

    # Simulated telemetry
    telem_dist = np.linspace(0, 4304, len(track_x))
    telem_speed = 200 + 80 * np.sin(np.linspace(0, 8 * np.pi, len(track_x))) + np.random.randn(len(track_x)) * 5
    telem_speed = np.clip(telem_speed, 80, 340)
    telem_gear = np.clip(np.round(telem_speed / 50), 1, 8)

    # Corner annotations for the approximated circuit
    corner_waypoints = [
        (10, 'Turn 1'),  (12, 'Turn 2-3'), (17, 'Turn 4'), (19, 'Turn 5'),
        (20, 'Turn 6'),  (22, 'Turn 7'),   (24, 'Turn 8'), (26, 'Turn 9-10'),
        (29, 'Turn 11'), (31, 'Turn 12'),  (33, 'Turn 13'), (35, 'Turn 14'),
        (37, 'Turn 15'), (39, 'Turn 16'),  (41, 'Turn 17'),
    ]
    corner_data = []
    for wp_i, label in corner_waypoints:
        if wp_i < len(waypoints):
            t_val = wp_i / (len(waypoints) - 1)
            corner_data.append({'n': int(label.split()[-1].split('-')[0]),
                                'x': float(cs_x(t_val)), 'y': float(cs_y(t_val))})
    rot_mat = np.eye(2)

n_track = len(track_x)
print(f"       ✓ Circuit loaded: {n_track} coordinate points")

# ═══════════════════════════════════════════════════════════════════════════════
#  6. SIMULATE 71-LAP RACE PROGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Simulating race progression (71 laps)...")

TOTAL_LAPS = 71
FRAMES_PER_LAP = 50
TOTAL_FRAMES = TOTAL_LAPS * FRAMES_PER_LAP

np.random.seed(42)

start_order = grid_df['GridPosition'].values.astype(float)
end_order = np.arange(1, NUM_DRIVERS + 1, dtype=float)

# Smooth position curves with mid-race drama
position_history = np.zeros((NUM_DRIVERS, TOTAL_LAPS + 1))
for i in range(NUM_DRIVERS):
    base = np.linspace(start_order[i], end_order[i], TOTAL_LAPS + 1)
    env = np.sin(np.linspace(0, np.pi, TOTAL_LAPS + 1)) * 2.8
    noise = np.random.randn(TOTAL_LAPS + 1) * env
    noise[0] = noise[-1] = 0
    position_history[i] = base + noise

# Convert to rankings each lap
rank_history = np.zeros_like(position_history)
for lap in range(TOTAL_LAPS + 1):
    rank_history[:, lap] = position_history[:, lap].argsort().argsort() + 1

# Pre-compute every driver's track index for all frames
driver_track_idx = np.zeros((NUM_DRIVERS, TOTAL_FRAMES), dtype=int)
for i in range(NUM_DRIVERS):
    cum = 0.0
    for f in range(TOTAL_FRAMES):
        cl = f // FRAMES_PER_LAP
        frac = (f % FRAMES_PER_LAP) / FRAMES_PER_LAP
        rk = rank_history[i, cl] * (1 - frac) + rank_history[i, min(cl + 1, TOTAL_LAPS)] * frac
        spd = n_track / FRAMES_PER_LAP * (1 + (NUM_DRIVERS / 2 - rk) * 0.0025)
        cum += spd
        driver_track_idx[i, f] = int(cum) % n_track

print("       ✓ Race simulation computed")

# ═══════════════════════════════════════════════════════════════════════════════
#  7. BUILD THE INTERACTIVE VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[7/7] Launching interactive race simulator...\n")

# ── Global animation state ──
state = {
    'frame': 0,
    'speed_mult': 1,
    'paused': False,
    'show_telemetry': False,
    'show_speed_map': False,
}

# ── Figure ──
fig = plt.figure(figsize=(19, 10.5), facecolor=BG_MAIN)
fig.canvas.manager.set_window_title('F1 2026 Mexico City GP — AI Race Prediction Simulator')

# ── Layout axes ──
# Top title bar
ax_title = fig.add_axes([0.0, 0.93, 1.0, 0.07])
ax_title.set_facecolor(BG_MAIN)
ax_title.axis('off')

# Circuit map
ax_circuit = fig.add_axes([0.01, 0.30, 0.58, 0.62])
ax_circuit.set_facecolor(BG_MAIN)
ax_circuit.set_aspect('equal')
ax_circuit.axis('off')

# Standings tower
ax_tower = fig.add_axes([0.61, 0.30, 0.38, 0.62])
ax_tower.set_facecolor(BG_PANEL)
ax_tower.axis('off')

# Telemetry panels (start hidden)
ax_telem_speed = fig.add_axes([0.06, 0.04, 0.40, 0.22])
ax_telem_speed.set_facecolor(BG_CARD)
ax_telem_gear  = fig.add_axes([0.56, 0.04, 0.40, 0.22])
ax_telem_gear.set_facecolor(BG_CARD)
ax_telem_speed.set_visible(False)
ax_telem_gear.set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
#  TITLE BAR
# ═══════════════════════════════════════════════════════════════════════════════
title_shadow = [patheffects.withStroke(linewidth=3, foreground='black')]

ax_title.text(0.01, 0.55, '🏎️', fontsize=22, transform=ax_title.transAxes, va='center')
ax_title.text(0.035, 0.55, 'F1 2026  MEXICO CITY GP', fontsize=16, fontweight='bold',
              color=TXT_WHITE, transform=ax_title.transAxes, va='center',
              fontfamily='monospace', path_effects=title_shadow)
ax_title.text(0.035, 0.12, 'AI RACE PREDICTION SIMULATOR  •  Autódromo Hermanos Rodríguez',
              fontsize=8, color=TXT_GRAY, transform=ax_title.transAxes, va='center',
              fontfamily='monospace')

# Red accent line under title
ax_title.plot([0.0, 1.0], [0.0, 0.0], color=F1_RED, linewidth=3,
              transform=ax_title.transAxes, clip_on=False)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONTROL BUTTONS
# ═══════════════════════════════════════════════════════════════════════════════
btn_style = {'color': BG_CARD, 'hovercolor': '#333333'}
btn_active_color = F1_RED

# Speed controls
ax_b1x  = fig.add_axes([0.30, 0.945, 0.04, 0.035])
ax_b2x  = fig.add_axes([0.345, 0.945, 0.04, 0.035])
ax_b5x  = fig.add_axes([0.39, 0.945, 0.04, 0.035])
ax_bpause = fig.add_axes([0.435, 0.945, 0.05, 0.035])

btn_1x    = Button(ax_b1x, '1×',  **btn_style)
btn_2x    = Button(ax_b2x, '2×',  **btn_style)
btn_5x    = Button(ax_b5x, '5×',  **btn_style)
btn_pause = Button(ax_bpause, '⏸ Pause', **btn_style)

# View controls
ax_btelem = fig.add_axes([0.55, 0.945, 0.09, 0.035])
ax_bsmap  = fig.add_axes([0.645, 0.945, 0.10, 0.035])

btn_telem = Button(ax_btelem, '📊 Telemetry', **btn_style)
btn_smap  = Button(ax_bsmap, '🗺️ Speed Map', **btn_style)

# Style all buttons
for b in [btn_1x, btn_2x, btn_5x, btn_pause, btn_telem, btn_smap]:
    b.label.set_color(TXT_WHITE)
    b.label.set_fontsize(8)
    b.label.set_fontfamily('monospace')
    b.label.set_fontweight('bold')

# Highlight 1x as default active
btn_1x.color = btn_active_color
ax_b1x.set_facecolor(btn_active_color)

# Speed label
ax_title.text(0.275, 0.55, 'SPEED:', fontsize=8, color=TXT_GRAY,
              transform=ax_title.transAxes, va='center', fontfamily='monospace')

# ═══════════════════════════════════════════════════════════════════════════════
#  DRAW THE CIRCUIT
# ═══════════════════════════════════════════════════════════════════════════════

# Sector boundaries (approximate: 33% / 66% of track points)
s1_end = n_track // 3
s2_end = 2 * n_track // 3

# Track edge (dark outline)
ax_circuit.plot(track_x, track_y, color=TRACK_EDGE, linewidth=14,
                solid_capstyle='round', zorder=1)

# Sector-colored track surface
for start, end, color in [(0, s1_end, SECTOR_1), (s1_end, s2_end, SECTOR_2), (s2_end, n_track, SECTOR_3)]:
    seg_x = track_x[start:end + 1] if end < n_track else np.append(track_x[start:], track_x[0])
    seg_y = track_y[start:end + 1] if end < n_track else np.append(track_y[start:], track_y[0])
    ax_circuit.plot(seg_x, seg_y, color=color, linewidth=6, solid_capstyle='round',
                    zorder=2, alpha=0.55)

# Track racing line (center)
ax_circuit.plot(track_x, track_y, color='#555555', linewidth=2, solid_capstyle='round',
                zorder=3, alpha=0.4)

# Start/Finish line marker
sf_i = 0
ax_circuit.plot(track_x[sf_i], track_y[sf_i], 's', color=TXT_WHITE, markersize=8,
                markeredgecolor='black', markeredgewidth=1.5, zorder=20)
ax_circuit.annotate('S/F', xy=(track_x[sf_i], track_y[sf_i]), fontsize=6,
                    color=TXT_WHITE, fontweight='bold', ha='center', va='bottom',
                    xytext=(0, 12), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=BG_CARD, edgecolor=TXT_DIM, alpha=0.9),
                    zorder=21)

# DRS zone highlights (approximate positions on track)
drs_zones = [
    (0, int(n_track * 0.10)),                          # DRS Zone main straight
    (int(n_track * 0.12), int(n_track * 0.20)),        # DRS Zone after T3
]
for dz_start, dz_end in drs_zones:
    dz_x = track_x[dz_start:dz_end]
    dz_y = track_y[dz_start:dz_end]
    ax_circuit.plot(dz_x, dz_y, color=DRS_GREEN, linewidth=3, alpha=0.5,
                    solid_capstyle='round', zorder=4)
    mid = (dz_start + dz_end) // 2
    ax_circuit.annotate('DRS', xy=(track_x[mid], track_y[mid]),
                        fontsize=5, color=DRS_GREEN, fontweight='bold',
                        ha='center', va='center',
                        xytext=(0, -15), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='#0a2a0a',
                                  edgecolor=DRS_GREEN, alpha=0.8),
                        zorder=22)

# Corner numbers
for cd in corner_data:
    ax_circuit.annotate(f"{cd['n']}", xy=(cd['x'], cd['y']),
                        fontsize=5.5, color=TXT_GRAY, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(boxstyle='circle,pad=0.2', facecolor=BG_CARD,
                                  edgecolor=TXT_DIM, alpha=0.85),
                        zorder=22)

# Sector legend
sector_legend_y = 0.03
for i, (label, color) in enumerate([('S1', SECTOR_1), ('S2', SECTOR_2), ('S3', SECTOR_3)]):
    ax_circuit.text(0.02 + i * 0.06, sector_legend_y, f'■ {label}', transform=ax_circuit.transAxes,
                    fontsize=7, color=color, fontweight='bold', fontfamily='monospace')

# ═══════════════════════════════════════════════════════════════════════════════
#  SPEED-COLORED TRACK MAP (overlay, toggled by button)
# ═══════════════════════════════════════════════════════════════════════════════
# Pre-build the LineCollection for speed map
_pts = np.array([track_x, track_y]).T.reshape(-1, 1, 2)
_segs = np.concatenate([_pts[:-1], _pts[1:]], axis=1)

# Map telemetry speed to track points (they may differ in length)
speed_on_track = np.interp(np.linspace(0, 1, n_track),
                           np.linspace(0, 1, len(telem_speed)),
                           telem_speed)

speed_lc = LineCollection(_segs, cmap='RdYlGn', linewidth=7, zorder=5)
speed_lc.set_array(speed_on_track[:-1])
speed_lc.set_clim(speed_on_track.min(), speed_on_track.max())
speed_lc_ref = ax_circuit.add_collection(speed_lc)
speed_lc.set_visible(False)

# ═══════════════════════════════════════════════════════════════════════════════
#  DRIVER DOTS & LABELS
# ═══════════════════════════════════════════════════════════════════════════════
colors_list = grid_df['Color'].tolist()
abbrevs     = grid_df['Abbreviation'].tolist()
teams       = grid_df['Team'].tolist()
drivers_names = grid_df['Driver'].tolist()

driver_dots = []
driver_labels = []
for i in range(NUM_DRIVERS):
    c = colors_list[i]
    dot, = ax_circuit.plot([], [], 'o', color=c, markersize=7,
                           markeredgecolor='white', markeredgewidth=0.6, zorder=30)
    driver_dots.append(dot)

    lbl = ax_circuit.text(0, 0, '', fontsize=4.5, fontweight='bold', color=TXT_WHITE,
                          ha='center', va='bottom', zorder=31,
                          bbox=dict(boxstyle='round,pad=0.08', facecolor=c,
                                    alpha=0.9, edgecolor='none'))
    driver_labels.append(lbl)

# ═══════════════════════════════════════════════════════════════════════════════
#  TIMING TOWER (right panel — broadcast-style standings)
# ═══════════════════════════════════════════════════════════════════════════════

# Header
ax_tower.text(0.50, 0.98, 'RACE STANDINGS', transform=ax_tower.transAxes,
              fontsize=12, fontweight='bold', color=F1_RED, ha='center', va='top',
              fontfamily='monospace')
ax_tower.plot([0.05, 0.95], [0.96, 0.96], color=F1_RED, linewidth=2,
              transform=ax_tower.transAxes, clip_on=False)

lap_text = ax_tower.text(0.50, 0.94, 'LAP  0 / 71', transform=ax_tower.transAxes,
                         fontsize=9, color=TXT_GRAY, ha='center', va='top',
                         fontfamily='monospace')

# Progress bar background
ax_tower.plot([0.05, 0.95], [0.925, 0.925], color=TXT_DIM, linewidth=4,
              transform=ax_tower.transAxes, clip_on=False, solid_capstyle='round')
progress_line, = ax_tower.plot([0.05, 0.05], [0.925, 0.925], color=F1_RED, linewidth=4,
                               transform=ax_tower.transAxes, clip_on=False,
                               solid_capstyle='round')

# Position rows
ROW_HEIGHT = 0.037
tower_rows = []
for p in range(NUM_DRIVERS):
    y = 0.89 - p * ROW_HEIGHT
    # Position number
    pos_txt = ax_tower.text(0.06, y, '', transform=ax_tower.transAxes,
                            fontsize=7.5, fontweight='bold', color=TXT_WHITE,
                            va='top', fontfamily='monospace')
    # Driver abbreviation
    abbr_txt = ax_tower.text(0.14, y, '', transform=ax_tower.transAxes,
                             fontsize=7.5, fontweight='bold', color=TXT_WHITE,
                             va='top', fontfamily='monospace')
    # Team
    team_txt = ax_tower.text(0.28, y, '', transform=ax_tower.transAxes,
                             fontsize=6.5, color=TXT_GRAY, va='top', fontfamily='monospace')
    # Gap
    gap_txt = ax_tower.text(0.92, y, '', transform=ax_tower.transAxes,
                            fontsize=7, color=TXT_GRAY, va='top', fontfamily='monospace',
                            ha='right')
    # Color indicator
    ind_txt = ax_tower.text(0.03, y, '▌', transform=ax_tower.transAxes,
                            fontsize=8, va='top')

    tower_rows.append({'pos': pos_txt, 'abbr': abbr_txt, 'team': team_txt,
                       'gap': gap_txt, 'ind': ind_txt})

# ═══════════════════════════════════════════════════════════════════════════════
#  TELEMETRY PANELS (toggleable)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_telemetry():
    """Draw Speed and Gear telemetry traces."""
    # Speed trace
    ax_telem_speed.clear()
    ax_telem_speed.set_facecolor(BG_CARD)
    ax_telem_speed.plot(telem_dist, telem_speed, color='#00FF88', linewidth=1.2, alpha=0.9)
    ax_telem_speed.fill_between(telem_dist, telem_speed, alpha=0.15, color='#00FF88')
    ax_telem_speed.set_ylabel('Speed (km/h)', fontsize=8, color=TXT_GRAY, fontfamily='monospace')
    ax_telem_speed.set_xlabel('Distance (m)', fontsize=7, color=TXT_DIM, fontfamily='monospace')
    ax_telem_speed.set_title('SPEED TRACE', fontsize=9, fontweight='bold', color=TXT_WHITE,
                             fontfamily='monospace', pad=8)
    ax_telem_speed.tick_params(colors=TXT_DIM, labelsize=6)
    ax_telem_speed.set_xlim(telem_dist[0], telem_dist[-1])
    ax_telem_speed.set_ylim(0, max(telem_speed) * 1.05)
    ax_telem_speed.grid(axis='y', color=TXT_DIM, alpha=0.3, linewidth=0.5)
    for spine in ax_telem_speed.spines.values():
        spine.set_color(TXT_DIM)

    # Gear trace
    ax_telem_gear.clear()
    ax_telem_gear.set_facecolor(BG_CARD)

    gear_colors = {1: '#3366FF', 2: '#00AAFF', 3: '#00DDAA', 4: '#66DD00',
                   5: '#BBDD00', 6: '#FFAA00', 7: '#FF6600', 8: '#FF0033'}
    for g in range(1, 9):
        mask = telem_gear == g
        if mask.any():
            ax_telem_gear.fill_between(telem_dist, 0, g, where=mask,
                                        color=gear_colors.get(g, '#888'),
                                        alpha=0.6, step='mid')
    ax_telem_gear.plot(telem_dist, telem_gear, color=TXT_WHITE, linewidth=0.8, alpha=0.7)
    ax_telem_gear.set_ylabel('Gear', fontsize=8, color=TXT_GRAY, fontfamily='monospace')
    ax_telem_gear.set_xlabel('Distance (m)', fontsize=7, color=TXT_DIM, fontfamily='monospace')
    ax_telem_gear.set_title('GEAR MAP', fontsize=9, fontweight='bold', color=TXT_WHITE,
                            fontfamily='monospace', pad=8)
    ax_telem_gear.tick_params(colors=TXT_DIM, labelsize=6)
    ax_telem_gear.set_xlim(telem_dist[0], telem_dist[-1])
    ax_telem_gear.set_ylim(0, 9)
    ax_telem_gear.set_yticks(range(1, 9))
    ax_telem_gear.grid(axis='y', color=TXT_DIM, alpha=0.3, linewidth=0.5)
    for spine in ax_telem_gear.spines.values():
        spine.set_color(TXT_DIM)

# Pre-draw (but hidden)
_draw_telemetry()

# Telemetry position markers (vertical lines)
telem_marker_spd = ax_telem_speed.axvline(x=0, color=F1_RED, linewidth=1.5, alpha=0.8, zorder=10)
telem_marker_gear = ax_telem_gear.axvline(x=0, color=F1_RED, linewidth=1.5, alpha=0.8, zorder=10)

# ═══════════════════════════════════════════════════════════════════════════════
#  BUTTON CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════════

def _reset_speed_btns():
    for ax_b, b in [(ax_b1x, btn_1x), (ax_b2x, btn_2x), (ax_b5x, btn_5x)]:
        b.color = BG_CARD
        ax_b.set_facecolor(BG_CARD)

def on_1x(event):
    state['speed_mult'] = 1
    _reset_speed_btns()
    btn_1x.color = btn_active_color
    ax_b1x.set_facecolor(btn_active_color)
    fig.canvas.draw_idle()

def on_2x(event):
    state['speed_mult'] = 2
    _reset_speed_btns()
    btn_2x.color = btn_active_color
    ax_b2x.set_facecolor(btn_active_color)
    fig.canvas.draw_idle()

def on_5x(event):
    state['speed_mult'] = 5
    _reset_speed_btns()
    btn_5x.color = btn_active_color
    ax_b5x.set_facecolor(btn_active_color)
    fig.canvas.draw_idle()

def on_pause(event):
    state['paused'] = not state['paused']
    btn_pause.label.set_text('▶ Play' if state['paused'] else '⏸ Pause')
    fig.canvas.draw_idle()

def on_telemetry(event):
    state['show_telemetry'] = not state['show_telemetry']
    vis = state['show_telemetry']
    ax_telem_speed.set_visible(vis)
    ax_telem_gear.set_visible(vis)
    # Adjust circuit panel height
    if vis:
        ax_circuit.set_position([0.01, 0.30, 0.58, 0.62])
        ax_tower.set_position([0.61, 0.30, 0.38, 0.62])
        btn_telem.label.set_text('📊 Hide Telem')
    else:
        ax_circuit.set_position([0.01, 0.05, 0.58, 0.87])
        ax_tower.set_position([0.61, 0.05, 0.38, 0.87])
        btn_telem.label.set_text('📊 Telemetry')
    fig.canvas.draw_idle()

def on_speed_map(event):
    state['show_speed_map'] = not state['show_speed_map']
    speed_lc.set_visible(state['show_speed_map'])
    if state['show_speed_map']:
        btn_smap.label.set_text('🗺️ Hide Map')
    else:
        btn_smap.label.set_text('🗺️ Speed Map')
    fig.canvas.draw_idle()

btn_1x.on_clicked(on_1x)
btn_2x.on_clicked(on_2x)
btn_5x.on_clicked(on_5x)
btn_pause.on_clicked(on_pause)
btn_telem.on_clicked(on_telemetry)
btn_smap.on_clicked(on_speed_map)

# ═══════════════════════════════════════════════════════════════════════════════
#  ANIMATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def update(_anim_frame):
    """Called every frame by FuncAnimation. Uses internal frame counter with speed control."""
    if state['paused']:
        return []

    # Advance by speed multiplier
    state['frame'] = min(state['frame'] + state['speed_mult'], TOTAL_FRAMES - 1)
    frame = state['frame']

    current_lap = min(frame // FRAMES_PER_LAP + 1, TOTAL_LAPS)
    frac = (frame % FRAMES_PER_LAP) / FRAMES_PER_LAP
    lap_idx = min(frame // FRAMES_PER_LAP, TOTAL_LAPS)

    # ── Lap counter & progress bar ──
    lap_text.set_text(f'LAP  {current_lap} / {TOTAL_LAPS}')
    progress_pct = frame / max(TOTAL_FRAMES - 1, 1)
    progress_line.set_xdata([0.05, 0.05 + 0.90 * progress_pct])

    # ── Current rankings ──
    rankings = []
    for i in range(NUM_DRIVERS):
        if lap_idx < TOTAL_LAPS:
            rk = rank_history[i, lap_idx] * (1 - frac) + rank_history[i, lap_idx + 1] * frac
        else:
            rk = rank_history[i, -1]
        rankings.append((rk, i))
    rankings.sort()

    # ── Update driver positions on track ──
    for i in range(NUM_DRIVERS):
        idx = driver_track_idx[i, frame]
        x, y = track_x[idx], track_y[idx]
        driver_dots[i].set_data([x], [y])
        driver_labels[i].set_position((x, y))
        driver_labels[i].set_text(abbrevs[i])

    # ── Update timing tower ──
    for pos, (rk_val, drv_i) in enumerate(rankings):
        if pos >= len(tower_rows):
            break
        row = tower_rows[pos]
        row['pos'].set_text(f'P{pos + 1}')
        row['abbr'].set_text(abbrevs[drv_i])
        row['team'].set_text(teams[drv_i])
        row['ind'].set_color(colors_list[drv_i])
        row['abbr'].set_color(colors_list[drv_i])

        if pos == 0:
            row['gap'].set_text('INTERVAL')
            row['gap'].set_color(TXT_WHITE)
        else:
            gap_val = (rk_val - rankings[0][0]) * 1.3
            row['gap'].set_text(f'+{gap_val:.1f}s')
            row['gap'].set_color(TXT_GRAY)

    # ── Update telemetry markers ──
    if state['show_telemetry']:
        lead_idx = rankings[0][1]
        trk_pos = driver_track_idx[lead_idx, frame]
        marker_dist = telem_dist[trk_pos % len(telem_dist)]
        telem_marker_spd.set_xdata([marker_dist])
        telem_marker_gear.set_xdata([marker_dist])

    return []


# Launch animation (blit=False for full widget compatibility)
ani = animation.FuncAnimation(fig, update, frames=TOTAL_FRAMES,
                              interval=33, blit=False, repeat=False)

plt.show()

# ═══════════════════════════════════════════════════════════════════════════════
#  FINAL RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("  🏁  2026 MEXICO CITY GP — FINAL PREDICTED RESULTS  🏁")
print("═" * 62)
for i, row in grid_df.iterrows():
    p = i + 1
    m = {1: '🥇', 2: '🥈', 3: '🥉'}.get(p, '  ')
    print(f"  {m} P{p:<3} {row['Abbreviation']}  {row['Driver']:<22s}  ({row['Team']})")
print("═" * 62)
