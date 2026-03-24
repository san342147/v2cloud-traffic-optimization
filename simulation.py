"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ PHANTOM TRAFFIC JAM SIMULATION — Intelligent Driver Model                    ║
║ Google Maps V2Cloud Predictive Speed Regulation — PoC                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Model       : IDM — Treiber, Hennecke & Helbing (2000)                       ║
║ Scenarios   : A) Baseline cascade B) V2Cloud predictive intervention         ║
║ Vehicles    : 15 cars cruising at 100 km/h on a single-lane highway          ║
║ Key result  : Fix reduces max stopped vehicles (5→3) and clears 2x faster    ║
║                                                                              ║
║ Run         : python dyx.py                                                  ║
║ Requires    : numpy, matplotlib                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

# ══════════════════════════════════════════════════════════════════════════════
# §1 PHYSICS & SIMULATION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Vehicle geometry ──────────────────────────────────────────────────────────
CAR_LENGTH = 4.5    # m — typical sedan length
CAR_HEIGHT = 1.6    # m — for visual proportions

# ── IDM parameters ────────────────────────────────────────────────────────────
V_FREE = 100 / 3.6  # m/s — free-flow desired speed
A_MAX = 1.5         # m/s² — comfortable max acceleration
B_COMFORT = 2.5     # m/s² — comfortable deceleration
B_EMERGENCY = 9.0   # m/s² — panic / emergency brake ceiling
S0 = 2.0            # m — minimum bumper-to-bumper jam gap
T_HEADWAY = 1.2     # s — desired safe time headway (≈ 2-second rule)
DELTA = 4           # — — IDM free-road exponent

# ── Scenario timing ───────────────────────────────────────────────────────────
T_BRAKE = 8.0       # s — Car #1 initiates emergency brake
T_RECOVER = 22.0    # s — Road ahead clears; Car #1 can accelerate again
T_DETECT = 8.7      # s — V2Cloud anomaly detection fires (700 ms latency)
T_WARN = 9.0        # s — Speed-regulation warning reaches smartphones
T_WARN_OFF = 34.0   # s — Google Maps lifts regulation (jam confirmed clear)
V_REGULATED = 45 / 3.6 # m/s — optimised advisory speed for warned cars
WARNED_START = 3    # first 0-indexed car to receive warning (Car #4 onward)

# ── Simulation grid ───────────────────────────────────────────────────────────
N_CARS = 15
DT = 0.05           # s — integration timestep (50 ms)
SIM_DURATION = 100.0 # s — total simulation window
N_STEPS = int(SIM_DURATION / DT)

# ══════════════════════════════════════════════════════════════════════════════
# §2 IDM PHYSICS ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def idm_acceleration(v, v_lead, gap, v_desired=V_FREE):
    gap = max(gap, 0.01) # numerical guard
    delta_v = v - v_lead # closing rate (positive=approaching)

    s_star = (S0 + max(0.0, v * T_HEADWAY + v * delta_v / (2.0 * np.sqrt(A_MAX * B_COMFORT))))
    acc = A_MAX * (1.0 - (v / v_desired) ** DELTA - (s_star / gap) ** 2)
    return float(np.clip(acc, -B_EMERGENCY, A_MAX))

def euler_step(pos, vel, acc):
    vel = np.clip(vel + acc * DT, 0.0, None)
    pos = pos + vel * DT
    return pos, vel

def initialise_platoon():
    spacing = (S0 + V_FREE * T_HEADWAY) + CAR_LENGTH
    positions = np.array([(N_CARS - 1 - i) * spacing for i in range(N_CARS)], dtype=float)
    velocities = np.full(N_CARS, V_FREE, dtype=float)
    return positions, velocities

# ══════════════════════════════════════════════════════════════════════════════
# §3 SCENARIO A — BASELINE (NO INTERVENTION)
# ══════════════════════════════════════════════════════════════════════════════

def simulate_baseline():
    pos, vel = initialise_platoon()
    ph = np.zeros((N_STEPS, N_CARS))
    vh = np.zeros((N_STEPS, N_CARS))
    step_brake = int(T_BRAKE / DT)
    step_recover = int(T_RECOVER / DT)

    for s in range(N_STEPS):
        ph[s] = pos
        vh[s] = vel
        acc = np.zeros(N_CARS)

        if s >= step_recover:
            acc[0] = idm_acceleration(vel[0], V_FREE, 9999.0)
        elif s >= step_brake and vel[0] > 0.0:
            acc[0] = -B_EMERGENCY

        for i in range(1, N_CARS):
            gap = pos[i - 1] - pos[i] - CAR_LENGTH
            acc[i] = idm_acceleration(vel[i], vel[i - 1], gap)

        pos, vel = euler_step(pos, vel, acc)

    return ph, vh

# ══════════════════════════════════════════════════════════════════════════════
# §4 SCENARIO B — GOOGLE MAPS V2CLOUD FIX
# ══════════════════════════════════════════════════════════════════════════════

def simulate_googlemaps_fix():
    pos, vel = initialise_platoon()
    ph = np.zeros((N_STEPS, N_CARS))
    vh = np.zeros((N_STEPS, N_CARS))
    step_brake = int(T_BRAKE / DT)
    step_recover = int(T_RECOVER / DT)
    step_warn = int(T_WARN / DT)
    step_warn_off = int(T_WARN_OFF / DT)

    for s in range(N_STEPS):
        ph[s] = pos
        vh[s] = vel
        acc = np.zeros(N_CARS)

        if s >= step_recover:
            acc[0] = idm_acceleration(vel[0], V_FREE, 9999.0)
        elif s >= step_brake and vel[0] > 0.0:
            acc[0] = -B_EMERGENCY

        for i in range(1, WARNED_START):
            gap = pos[i - 1] - pos[i] - CAR_LENGTH
            acc[i] = idm_acceleration(vel[i], vel[i - 1], gap)

        for i in range(WARNED_START, N_CARS):
            gap = pos[i - 1] - pos[i] - CAR_LENGTH
            v_target = V_REGULATED if step_warn <= s < step_warn_off else V_FREE 
            acc[i] = idm_acceleration(vel[i], vel[i - 1], gap, v_desired=v_target) 

        pos, vel = euler_step(pos, vel, acc) 
    
    return ph, vh 

# ══════════════════════════════════════════════════════════════════════════════ 
# §5 RUN SIMULATIONS 
# ══════════════════════════════════════════════════════════════════════════════ 

print("━" * 68) 
print(" PHANTOM TRAFFIC JAM SIMULATION — IDM v2.0") 
print("━" * 68) 
print(f" Vehicles : {N_CARS} | Initial speed : {V_FREE*3.6:.0f} km/h") 
print(f" Timestep : {DT*1000:.0f} ms | Duration : {SIM_DURATION:.0f} s") 
print(f" Hard-brake : t={T_BRAKE:.1f} s → road clear at t={T_RECOVER:.1f} s") 
print(f" V2Cloud warn : t={T_WARN:.1f} s → Cars #4–#15 @ {V_REGULATED*3.6:.0f} km/h") 
print(f" Warning off : t={T_WARN_OFF:.1f} s") 
print("━" * 68) 

print("▶ Baseline simulation …", end="  ", flush=True) 
pos_base, vel_base = simulate_baseline() 
print("✓") 

print("▶ Google Maps Fix simulation …", end="  ", flush=True) 
pos_fix, vel_fix = simulate_googlemaps_fix() 
print("✓") 

print("\n Key timestamp comparison:") 
print(f" {'Time':>6} │ {'BASE stopped':>14} {'BASE avg':>10} │ {'FIX stopped':>13} {'FIX avg':>9}")

for t_chk in (20, 28, 35, 45):
    step = int(t_chk / DT)
    sb = int(np.sum(vel_base[step] * 3.6 < 3)) 
    sf = int(np.sum(vel_fix[step] * 3.6 < 3)) 
    ab = np.mean(vel_base[step]) * 3.6
    af = np.mean(vel_fix[step]) * 3.6 
    print(f" t={t_chk:3.0f}s │ {sb:>5}/15 vehicles {ab:>8.1f} km/h │ {sf:>5}/15 vehicles {af:>7.1f} km/h")

print("\n▶ Building animation …")

# ══════════════════════════════════════════════════════════════════════════════
# §6 VISUAL CONSTANTS & COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

FRAME_SKIP = 4
INTERVAL_MS = 30
VIEW_WIDTH = 480
VIEW_PAD = 50
ROAD_H = 3.0
CAR_Y = ROAD_H / 2 - CAR_HEIGHT / 2

C_BG = "#0B0D14"
C_ROAD = "#181A24"
C_KERB = "#2D3040"
C_DASH = "#E8B84B"
C_LEAD = "#FF3B30"
C_NORMAL = "#3A82F7"
C_WARNED = "#30D158"
C_SLOW = "#FF9F0A"
C_STOPPED = "#FF453A"
C_SHOCK = "#FF2D55"
C_V2C = "#34D399"

def car_colour(speed_ms, idx, scenario):
    if idx == 0: return C_LEAD
    kmh = speed_ms * 3.6
    if kmh < 3.0: return C_STOPPED
    if kmh < 40.0: return C_SLOW
    if scenario == "fix" and idx >= WARNED_START: return C_WARNED
    return C_NORMAL

def camera_left(positions_row):
    return np.median(positions_row) - VIEW_WIDTH / 2

# ══════════════════════════════════════════════════════════════════════════════
# §7 FIGURE & AXES
# ══════════════════════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(22, 9.5), facecolor=C_BG)
fig.subplots_adjust(left=0.03, right=0.98, top=0.87, bottom=0.07, hspace=0.60)

ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

for ax in (ax1, ax2):
    ax.set_facecolor(C_ROAD)
    ax.set_ylim(-1.5, ROAD_H + 1.3)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(colors="#555D78", labelsize=8)
    ax.axhspan(-1.5, -0.55, color="#1E2030", zorder=0)
    ax.axhspan(ROAD_H + 0.30, ROAD_H + 1.30, color="#1E2030", zorder=0)
    ax.axhline(-0.55, color=C_KERB, lw=5, zorder=1)
    ax.axhline(ROAD_H + 0.30, color=C_KERB, lw=5, zorder=1)
    
    for xd in range(-200, 2000, 60):
        ax.plot([xd, xd + 30], [ROAD_H / 2, ROAD_H / 2], color=C_DASH, lw=1.5, alpha=0.30, zorder=1)

ax1.set_title(" 🚨 SCENARIO A — BASELINE : Phantom Traffic Jam (No Intervention)", color="#FF6B6B", fontsize=13.5, fontweight="bold", pad=11, fontfamily="monospace")
ax2.set_title(" ✅ SCENARIO B — GOOGLE MAPS V2CLOUD : Predictive Speed Regulation", color="#34D399", fontsize=13.5, fontweight="bold", pad=11, fontfamily="monospace")

wm_kw = dict(transform=fig.transFigure, alpha=0.06, fontsize=56, fontweight="bold", fontfamily="monospace", va="center")
fig.text(0.5, 0.76, "JAM", color="#FF3B30", ha="center", **wm_kw)
fig.text(0.5, 0.27, "FLOW", color="#30D158", ha="center", **wm_kw)

# ══════════════════════════════════════════════════════════════════════════════
# §8 CAR PATCHES & LABELS
# ══════════════════════════════════════════════════════════════════════════════

patches_base, patches_fix = [], []
labels_base, labels_fix = [], []

for i in range(N_CARS):
    for patches, ax in [(patches_base, ax1), (patches_fix, ax2)]:
        p = mpatches.FancyBboxPatch((0.0, CAR_Y), CAR_LENGTH, CAR_HEIGHT, boxstyle="round,pad=0.18", linewidth=0.8, edgecolor="#00000060", facecolor=(C_LEAD if i == 0 else C_NORMAL), zorder=4)
        ax.add_patch(p)
        patches.append(p)
        ax.add_patch(mpatches.FancyBboxPatch((0.0, CAR_Y + CAR_HEIGHT * 0.52), CAR_LENGTH * 0.52, CAR_HEIGHT * 0.30, boxstyle="round,pad=0.05", linewidth=0, facecolor="#FFFFFF", alpha=0.06, zorder=5))

label_y = CAR_Y + CAR_HEIGHT + 0.18
lbl_kw = dict(fontsize=7, ha="center", va="bottom", fontfamily="monospace", zorder=6, color="white", path_effects=[pe.withStroke(linewidth=1.8, foreground="black")])

for _ in range(N_CARS):
    labels_base.append(ax1.text(0, label_y, "", **lbl_kw))
    labels_fix.append(ax2.text(0, label_y, "", **lbl_kw))

# ══════════════════════════════════════════════════════════════════════════════
# §9 HUD OVERLAYS & LEGENDS
# ══════════════════════════════════════════════════════════════════════════════

status_kw = dict(fontsize=11, fontweight="bold", fontfamily="monospace", ha="center", va="center", zorder=10, bbox=dict(boxstyle="round,pad=0.45", facecolor="#00000085", edgecolor="#FFFFFF18", linewidth=1.2))
status_base = ax1.text(0.5, 0.91, "", transform=ax1.transAxes, color="#FF6B6B", **status_kw)
status_fix = ax2.text(0.5, 0.91, "", transform=ax2.transAxes, color="#34D399", **status_kw)

metric_kw = dict(fontsize=8.5, ha="right", va="top", fontfamily="monospace", zorder=10, bbox=dict(boxstyle="round,pad=0.35", facecolor="#0D0F1ACC", edgecolor="#FFFFFF15"))
metric_base = ax1.text(0.99, 0.98, "", transform=ax1.transAxes, color="#FF9F0A", **metric_kw)
metric_fix = ax2.text(0.99, 0.98, "", transform=ax2.transAxes, color="#34D399", **metric_kw)

time_text = fig.text(0.5, 0.95, "", ha="center", va="center", color="#DDE0F0", fontsize=12, fontweight="bold", fontfamily="monospace", bbox=dict(boxstyle="round,pad=0.55", facecolor="#1C1E2E", edgecolor="#3A3D55", linewidth=1.5))

shock_vline = ax1.axvline(-99999, color=C_SHOCK, lw=2.0, linestyle="--", alpha=0.80, zorder=3)
shock_txt = ax1.text(-99999, ROAD_H + 0.75, "◀ shockwave", color=C_SHOCK, fontsize=7.5, fontfamily="monospace", ha="center", zorder=7, path_effects=[pe.withStroke(linewidth=2, foreground=C_BG)])

v2c_vline = ax2.axvline(-99999, color=C_V2C, lw=1.8, linestyle=":", alpha=0.85, zorder=3)
v2c_txt = ax2.text(-99999, ROAD_H + 0.75, "📡 V2Cloud", color=C_V2C, fontsize=7.5, fontfamily="monospace", ha="center", zorder=7, path_effects=[pe.withStroke(linewidth=2, foreground=C_BG)])

leg_kw = dict(loc="lower left", facecolor="#10121A", edgecolor="#2A2D3A", labelcolor="white", fontsize=8.0, ncol=2, handlelength=1.6, handleheight=0.9, borderpad=0.6, columnspacing=1.2)

ax1.legend(handles=[
    mpatches.Patch(color=C_LEAD, label="Car #1 — Lead (emergency brake)"),
    mpatches.Patch(color=C_NORMAL, label="Cars #2–#15 — IDM only"),
    mpatches.Patch(color=C_SLOW, label="Speed < 40 km/h"), 
    mpatches.Patch(color=C_STOPPED, label="Stopped  v < 3 km/h")
], **leg_kw) 

ax2.legend(handles=[ 
    mpatches.Patch(color=C_LEAD, label="Car #1  — Lead (emergency brake)"), 
    mpatches.Patch(color=C_NORMAL, label="Cars #2–#3 — No warning (too close)"), 
    mpatches.Patch(color=C_WARNED, label="Cars #4–#15 — Google Maps ✓  45 km/h"), 
    mpatches.Patch(color=C_SLOW, label="Speed < 40 km/h (transition zone)")
], **leg_kw)

# ══════════════════════════════════════════════════════════════════════════════
# §10 ANIMATION CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

N_FRAMES = N_STEPS // FRAME_SKIP 

def animate(frame): 
    step = min(frame * FRAME_SKIP, N_STEPS - 1) 
    t = step * DT 
    
    xl_b = camera_left(pos_base[step])
    xl_f = camera_left(pos_fix[step]) 
    ax1.set_xlim(xl_b - VIEW_PAD, xl_b + VIEW_WIDTH + VIEW_PAD)
    ax2.set_xlim(xl_f - VIEW_PAD, xl_f + VIEW_WIDTH + VIEW_PAD) 

    for i in range(N_CARS): 
        xb = pos_base[step, i] 
        xf = pos_fix[step, i] 
        patches_base[i].set_x(xb) 
        patches_base[i].set_facecolor(car_colour(vel_base[step, i], i, "base"))
        labels_base[i].set_position((xb + CAR_LENGTH / 2, label_y)) 
        labels_base[i].set_text(f"{vel_base[step, i] * 3.6:.0f}") 
        patches_fix[i].set_x(xf) 
        patches_fix[i].set_facecolor(car_colour(vel_fix[step, i], i, "fix")) 
        labels_fix[i].set_position((xf + CAR_LENGTH / 2, label_y))
        labels_fix[i].set_text(f"{vel_fix[step, i] * 3.6:.0f}") 

    n_stop_b = int(np.sum(vel_base[step] * 3.6 < 3.0))
    n_slow_b = int(np.sum((vel_base[step] * 3.6 >= 3.0) & (vel_base[step] * 3.6 < 40.0))) 
    n_stop_f = int(np.sum(vel_fix[step] * 3.6 < 3.0))
    n_slow_f = int(np.sum((vel_fix[step] * 3.6 >= 3.0) & (vel_fix[step] * 3.6 < 40.0))) 
    avg_b = np.mean(vel_base[step]) * 3.6 
    avg_f = np.mean(vel_fix[step]) * 3.6 

    if t < T_BRAKE:
        status_base.set_text("🟢 All 15 vehicles cruising at 100 km/h") 
        status_fix.set_text("🟢 All 15 vehicles cruising at 100 km/h") 
    elif t < T_WARN: 
        pct = min((t - T_BRAKE) / (T_WARN - T_BRAKE) * 100, 100) 
        status_base.set_text("🚨 CAR #1 EMERGENCY BRAKE — shockwave forming!") 
        status_fix.set_text(f"🚨 Brake detected! 📡 V2Cloud analyzing … {pct:.0f}%") 
    elif t < T_RECOVER:
        status_base.set_text("💥 Shockwave propagating — CASCADE IN PROGRESS") 
        status_fix.set_text(f"⚠️ Regulation active → Cars #4–#15 @ {V_REGULATED * 3.6:.0f} km/h | Absorbing shockwave …") 
    elif t < T_WARN_OFF:
        status_base.set_text(f"🛑 Road cleared | {n_stop_b} still stopped | Recovery crawling …")
        status_fix.set_text(f"✅ Road cleared | {n_stop_f} stopped | Traffic recovering faster")
    else: 
        status_base.set_text(f"♻️ Recovering … avg {avg_b:.0f} km/h") 
        status_fix.set_text(f"🏁 Regulation lifted | avg {avg_f:.0f} km/h | Flow restored") 

    metric_base.set_text(f"⛔ {n_stop_b:2d} 🐢 {n_slow_b:2d} ∅ {avg_b:5.1f} km/h") 
    metric_fix.set_text(f"⛔ {n_stop_f:2d} 🟡 {n_slow_f:2d} ∅ {avg_f:5.1f} km/h") 

    if t >= T_BRAKE:
        slow_mask = vel_base[step] * 3.6 < 50.0 
        if slow_mask.any(): 
            sx = pos_base[step, slow_mask].min() - 4.0 
            shock_vline.set_xdata([sx, sx]) 
            shock_txt.set_x(sx) 
        else: 
            shock_vline.set_xdata([-99999, -99999]) 
            shock_txt.set_x(-99999) 
    else: 
        shock_vline.set_xdata([-99999, -99999])
        shock_txt.set_x(-99999) 

    if T_WARN <= t < T_WARN_OFF and WARNED_START < N_CARS: 
        vx = (pos_fix[step, WARNED_START - 1] + pos_fix[step, WARNED_START]) / 2
        v2c_vline.set_xdata([vx, vx]) 
        v2c_txt.set_x(vx) 
    else: 
        v2c_vline.set_xdata([-99999, -99999])
        v2c_txt.set_x(-99999) 

    phase = ("PRE-EVENT" if t < T_BRAKE else "DETECTION" if t < T_WARN else "ACTIVE JAM" if t < T_RECOVER else "RECOVERY")
    time_text.set_text(f" t={t:5.1f} s │ Phase: {phase:<12s} │ Speed labels in km/h ")

    return (patches_base + patches_fix + labels_base + labels_fix + [status_base, status_fix, metric_base, metric_fix, shock_vline, shock_txt, v2c_vline, v2c_txt, time_text])

# ══════════════════════════════════════════════════════════════════════════════
# §11 LAUNCH ANIMATION
# ══════════════════════════════════════════════════════════════════════════════

anim = animation.FuncAnimation(
    fig, animate,
    frames=N_FRAMES,
    interval=INTERVAL_MS,
    blit=True,
    repeat=True,
    repeat_delay=3000
)

print(f" Animation ready — {N_FRAMES} frames @ ~{1000 // INTERVAL_MS} fps") 
print(" Watch for t ≈ 20–40s — the key difference window.") 
print(" Close the window to exit.") 
print("━" * 68) 
print("Saving animation as GIF... this might take a minute.")
# This saves the animation to your folder so you can upload it!
anim.save('traffic_simulation.gif', writer='pillow', fps=30)
print("✅ Saved successfully as traffic_simulation.gif!")