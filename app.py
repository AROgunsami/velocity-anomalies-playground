# app.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from app_preset_lib import get_preset, show_preset_note

# -----------------------------
# Basic helpers
# -----------------------------
def make_grid(nx=301, nz=301, x_km=4.0, z_km=2.0):
    x = np.linspace(0, x_km*1000, nx)  # meters
    z = np.linspace(0, z_km*1000, nz)  # meters
    dx = x[1]-x[0]
    dz = z[1]-z[0]
    X, Z = np.meshgrid(x, z)
    return X, Z, dx, dz

def ricker(f, dt, nt, t0=0.0):
    t = np.arange(nt)*dt
    pi2 = (np.pi**2)
    w = (1 - 2*pi2*(f**2)*(t-t0)**2)*np.exp(-pi2*(f**2)*(t-t0)**2)
    return w

def integrate_vertical_time(v_col, dz):
    return np.sum((1.0/np.maximum(v_col, 1.0)) * dz)

# -----------------------------
# Eikonal & migration helpers
# -----------------------------
def _interp2_linear(V, xo_m, zo_m, xg_m, zg_m):
    Vx = np.empty((len(zo_m), len(xg_m)), dtype=float)
    for j in range(len(zo_m)):
        Vx[j, :] = np.interp(xg_m, xo_m, V[j, :])
    Vg = np.empty((len(zg_m), len(xg_m)), dtype=float)
    for i in range(len(xg_m)):
        Vg[:, i] = np.interp(zg_m, zo_m, Vx[:, i])
    return Vg

def _eikonal_update(a, b, s, h):
    if a > b:
        a, b = b, a
    t = a + s*h
    if (b - a) < s*h:
        disc = 2*(s*h)**2 - (b - a)**2
        if disc > 0:
            t = (a + b + np.sqrt(disc)) / 2.0
    return t

def eikonal_fast_sweeping(speed, h, src_i, src_j, n_sweeps=6):
    nz, nx = speed.shape
    T = np.full((nz, nx), np.inf, dtype=float)
    T[src_j, src_i] = 0.0
    s = 1.0 / np.maximum(speed, 1.0)
    sweep_orders = [
        (range(nz), range(nx)),
        (range(nz-1, -1, -1), range(nx)),
        (range(nz), range(nx-1, -1, -1)),
        (range(nz-1, -1, -1), range(nx-1, -1, -1)),
    ]
    for _ in range(n_sweeps):
        for rows, cols in sweep_orders:
            for j in rows:
                for i in cols:
                    if i == src_i and j == src_j:
                        continue
                    a = min(T[j, i-1] if i>0 else np.inf, T[j, i+1] if i<nx-1 else np.inf)
                    b = min(T[j-1, i] if j>0 else np.inf, T[j+1, i] if j<nz-1 else np.inf)
                    Tij = _eikonal_update(a, b, s[j, i], h)
                    if Tij < T[j, i]:
                        T[j, i] = Tij
    return T

def compute_bent_twt(V, x_km, z_km, z_ref_m, nx_eik=151, ray_stride=6):
    nz_o, nx_o = V.shape
    xo_m = np.linspace(0, x_km*1000.0, nx_o)
    zo_m = np.linspace(0, z_km*1000.0, nz_o)

    nxe = int(nx_eik)
    hx = (x_km*1000.0) / (nxe - 1)
    nze = int(np.round(z_km*1000.0 / hx)) + 1
    xg_m = np.linspace(0, x_km*1000.0, nxe)
    zg_m = np.linspace(0, (nze-1)*hx, nze)

    Vg = _interp2_linear(V, xo_m, zo_m, xg_m, zg_m)

    x_orig_km = np.linspace(0, x_km, nx_o)
    idxs = np.arange(0, nx_o, max(1, ray_stride))
    TWT_partial = np.full_like(x_orig_km, np.nan, dtype=float)

    j_ref = int(np.clip(np.round(z_ref_m / hx), 0, nze-1))

    for ix in idxs:
        src_i = int(np.clip(np.round((x_orig_km[ix]*1000.0)/hx), 0, nxe-1))
        T = eikonal_fast_sweeping(Vg, hx, src_i=src_i, src_j=0, n_sweeps=6)
        t_down = np.min(T[j_ref, :])
        TWT_partial[ix] = 2.0 * t_down

    good = np.isfinite(TWT_partial)
    if good.sum() >= 2:
        TWT_full = np.interp(x_orig_km, x_orig_km[good], TWT_partial[good])
    else:
        TWT_full = TWT_partial
    return TWT_full

def kirchhoff_time_migration(section, dt, x_km, v_mig, stride_x=4, stride_t=2, aperture_km=None):
    nt, nx = section.shape
    x = np.linspace(0.0, x_km, nx)
    if aperture_km is None:
        aperture_km = x_km
    xi = np.arange(0, nx, max(1, stride_x))
    ti = np.arange(0, nt, max(1, stride_t))
    mig = np.zeros((len(ti), len(xi)), dtype=float)

    for ix_out, ix0 in enumerate(xi):
        x0 = x[ix0]
        mask = np.abs(x - x0) <= aperture_km
        traces = section[:, mask]  # (nt, ntr)
        dx_m = (x[mask] - x0) * 1000.0
        for it_out, jt in enumerate(ti):
            t0 = jt * dt
            t_in = np.sqrt(t0*t0 + (dx_m / v_mig)**2)
            ti_f = t_in / dt
            i0 = np.floor(ti_f).astype(int)
            i1 = i0 + 1
            w = ti_f - i0
            valid = (i0 >= 0) & (i1 < nt)
            if not np.any(valid):
                continue
            cols = np.where(valid)[0]
            amp = (1 - w[valid]) * traces[i0[valid], cols] + w[valid] * traces[i1[valid], cols]
            mig[it_out, ix_out] = amp.mean() if amp.size else 0.0

    return mig, x[xi], ti*dt

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Velocity Modeling — Pull‑up & Push‑down", layout="wide")
st.title("Velocity Modeling — Pull‑up & Push‑down")
st.caption("Interactive toy model to illustrate fast/slow bodies and seabed effects on TWT and imaging.")

with st.sidebar:
    st.header("Presets")
    preset_name = st.selectbox(
        "Scenario",
        ["None (custom)", "Gas pocket (slow, push-down)", "Salt dome (fast, pull-up)",
         "Basalt flow (fast shallow)", "Shallow channel (slow shallow)", "Tilted seabed (dip)"]
    )
    preset_enable = st.checkbox("Enable preset overrides", value=False,
                                help="When ON, the preset values override manual controls below.")

    st.header("Model Setup")
    colA, colB = st.columns(2)
    with colA:
        x_km = st.slider("Model width (km)", 1.0, 10.0, 4.0, 0.5)
    with colB:
        z_km = st.slider("Model depth (km)", 0.5, 5.0, 2.0, 0.1)

    nx = st.slider("Grid NX", 201, 801, 301, 50)
    nz = st.slider("Grid NZ", 201, 801, 301, 50)

    st.divider()
    st.subheader("Background velocities (m/s)")
    v_water = st.number_input("Water velocity", 1300, 1700, 1480, 10)
    v_sed = st.number_input("Sediment velocity", 1800, 4500, 2200, 50)

    st.subheader("Bathymetry / Seabed")
    water_z = st.slider("Water depth at left (m)", 0, int(z_km*1000)-50, 300, 10)
    water_z_right = st.slider("Water depth at right (m)", 0, int(z_km*1000)-50, 300, 10)

    st.subheader("Target reflector (true depth)")
    z_ref = st.slider("Reflector depth below sea surface (m)", 200, int(z_km*1000)-50, 1500, 10)

    st.divider()
    st.subheader("Anomaly (body)")
    anomaly_kind = st.radio("Type", ["Fast (pull‑up)", "Slow (push‑down)"], index=1)
    v_anom = st.number_input("Anomaly velocity (m/s)", 1500, 6000, 3000, 50)
    col1, col2 = st.columns(2)
    with col1:
        x0_km = st.slider("Center x (km)", 0.0, x_km, x_km/2, 0.1)
        ax_km = st.slider("Half‑width aₓ (km)", 0.05, x_km, 0.5, 0.05)
    with col2:
        z0_km = st.slider("Center z (km)", 0.0, z_km, 1.0, 0.05)
        az_km = st.slider("Half‑height a_z (km)", 0.02, z_km, 0.3, 0.02)

    st.caption("The ellipse ((x−x₀)/aₓ)² + ((z−z₀)/a_z)² ≤ 1 defines the body. Set its velocity via 'Anomaly velocity'.")

    st.divider()
    st.subheader("Synthetic section")
    fdom = st.slider("Ricker dominant freq (Hz)", 5, 60, 25, 1)
    tmax = st.slider("Record length (s)", 0.5, 5.0, 2.0, 0.1)
    dt = st.select_slider("Sample rate dt (ms)", options=[0.5, 1.0, 2.0, 4.0, 8.0], value=2.0)

    st.divider()
    st.subheader("Ray‑bending (eikonal)")
    do_bending = st.checkbox("Compute bent‑ray TWT to deep reflector (slower)", value=False)
    ray_nx = st.slider("Eikonal grid NX (coarse)", 81, 241, 151, 10)
    ray_stride = st.slider("Ray compute stride (surface decimation)", 1, 12, 6, 1)

    st.divider()
    st.subheader("Poststack time migration (constant‑velocity)")
    do_mig = st.checkbox("Show migrated view (Kirchhoff)", value=False)
    v_mig = st.number_input("Migration velocity v (m/s)", 1500, 6000, int(v_sed), 50)
    aperture_km = st.slider("Migration aperture (km)", 0.1, max(0.2, float(x_km)), min(float(x_km)/3, float(x_km)), 0.1)
    stride_x = st.slider("Output x decimation (bigger = faster)", min_value=1, max_value=10, value=4, step=1)
    stride_t = st.slider("Output t decimation (bigger = faster)", min_value=1, max_value=10, value=2, step=1)

# -----------------------------
# Apply preset overrides
# -----------------------------
preset_note = None
if preset_enable and preset_name != "None (custom)":
    p = get_preset(preset_name, v_sed=float(v_sed), x_km=float(x_km), z_km=float(z_km))
    if p is not None:
        anomaly_kind = p["anomaly_kind"]
        v_anom = p["v_anom"]
        x0_km, z0_km, ax_km, az_km = p["x0_km"], p["z0_km"], p["ax_km"], p["az_km"]
        water_z, water_z_right = p["water_z"], p["water_z_right"]
        z_ref = p["z_ref"]
        preset_note = (p["note"], p.get("notice", []))

# -----------------------------
# Build model
# -----------------------------
X, Z, dx, dz = make_grid(nx, nz, x_km=x_km, z_km=z_km)

# Seabed/bathymetry as a linear plane from left to right
z_seabed = np.linspace(water_z, water_z_right, nx)  # meters
ZSB = np.tile(z_seabed, (nz,1))

# Background velocity
V = np.where(Z < ZSB, v_water, v_sed).astype(float)

# Apply anomaly ellipse (only below seabed)
ellipse = (((X - x0_km*1000)/(ax_km*1000))**2 + ((Z - z0_km*1000)/(az_km*1000))**2) <= 1.0
if anomaly_kind.startswith("Fast") or anomaly_kind.startswith("Slow"):
    V = np.where(ellipse & (Z >= ZSB), float(v_anom), V)

V = np.clip(V, 200.0, 7000.0)

# Two-way times
z_ref_arr = np.maximum(z_seabed + 5.0, min(z_ref, int(z_km*1000)-1))
z_axis = Z[:,0]
oneway_t_seabed = np.zeros(nx)
oneway_t_ref = np.zeros(nx)

for ix in range(nx):
    zsb = z_seabed[ix]
    zrf = z_ref_arr[ix]
    ksb = int(np.clip(np.searchsorted(z_axis, zsb), 1, nz-1))
    krf = int(np.clip(np.searchsorted(z_axis, zrf), 1, nz-1))
    vcol = V[:ksb, ix]
    oneway_t_seabed[ix] = integrate_vertical_time(vcol, dz)
    vcol_ref = V[:krf, ix]
    oneway_t_ref[ix] = integrate_vertical_time(vcol_ref, dz)

TWT_seabed = 2.0 * oneway_t_seabed
TWT_ref = 2.0 * oneway_t_ref

# Baseline (no anomaly) for comparison
V0 = np.where(Z < ZSB, v_water, v_sed).astype(float)
oneway_t_ref0 = np.zeros(nx)
for ix in range(nx):
    zrf = z_ref_arr[ix]
    krf = int(np.clip(np.searchsorted(z_axis, zrf), 1, nz-1))
    oneway_t_ref0[ix] = integrate_vertical_time(V0[:krf, ix], dz)
TWT_ref0 = 2.0 * oneway_t_ref0

# Bent-ray TWT (optional)
TWT_ref_bent = None
if do_bending:
    with st.spinner("Computing eikonal travel times…"):
        TWT_ref_bent = compute_bent_twt(V, x_km, z_km, float(np.mean(z_ref_arr)), nx_eik=ray_nx, ray_stride=ray_stride)

# -----------------------------
# Build a simple zero‑offset synthetic (two reflectors)
# -----------------------------
true_dt = dt/1000.0
nt = int(tmax / true_dt) + 1
section = np.zeros((nt, nx), dtype=float)

# Wavelet (compact)
wlen_s = min(0.256, tmax)
w = ricker(fdom, true_dt, int(wlen_s/true_dt)+1)

# Stamp events safely
for ix in range(nx):
    # Seabed
    isamp_s = int(np.clip(np.round(TWT_seabed[ix]/true_dt), 0, nt-1))
    if isamp_s < nt:
        i0 = max(0, isamp_s - len(w)//2)
        i1 = min(nt, isamp_s + (len(w) - len(w)//2))
        wi0 = max(0, len(w)//2 - (isamp_s - i0))
        seg_len = min(i1 - i0, len(w) - wi0)
        if seg_len > 0:
            section[i0:i0+seg_len, ix] += w[wi0:wi0+seg_len] * 0.8

    # Deep reflector
    isamp_r = int(np.clip(np.round(TWT_ref[ix]/true_dt), 0, nt-1))
    if isamp_r < nt:
        i0 = max(0, isamp_r - len(w)//2)
        i1 = min(nt, isamp_r + (len(w) - len(w)//2))
        wi0 = max(0, len(w)//2 - (isamp_r - i0))
        seg_len = min(i1 - i0, len(w) - wi0)
        if seg_len > 0:
            section[i0:i0+seg_len, ix] += w[wi0:wi0+seg_len] * 1.0

# -----------------------------
# Plots
# -----------------------------
colL, colR = st.columns([1.1, 1.0])

with colL:
    st.subheader("Velocity model")
    show_preset_note(preset_note)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(V, extent=[0, x_km, z_km, 0], aspect='auto')
    ax.plot(np.linspace(0, x_km, nx), z_seabed/1000.0, lw=1.2, label='Seabed', alpha=0.9)
    ax.axhline(np.mean(z_ref_arr)/1000.0, ls='--', lw=1.0, label='True deep reflector (avg)')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (km)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Velocity (m/s)')
    ax.legend(loc='upper right')
    st.pyplot(fig, clear_figure=True)

    st.caption("Adjust anomaly velocity relative to sediments: lower → push‑down, higher → pull‑up. Water column uses v_water.")

with colR:
    st.subheader("Two‑way time of horizons")
    x_axis_km = np.linspace(0, x_km, nx)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(x_axis_km, TWT_seabed, label='Seabed TWT', lw=1.2)
    ax2.plot(x_axis_km, TWT_ref, label='Deep reflector TWT (vertical rays)', lw=1.8)
    ax2.plot(x_axis_km, TWT_ref0, label='Deep reflector TWT (no body)', lw=1.0, linestyle='--')
    if TWT_ref_bent is not None:
        ax2.plot(x_axis_km, TWT_ref_bent, label='Deep reflector TWT (bent rays, eikonal)', lw=1.3)
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Time (s)')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

colS, colM = st.columns(2)
with colS:
    st.subheader("Zero‑offset synthetic (image)")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sec_disp = section / (np.max(np.abs(section)) + 1e-9)
    ax3.imshow(sec_disp, extent=[0, x_km, tmax, 0], aspect='auto', interpolation='bilinear')
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Time (s)')
    st.pyplot(fig3, clear_figure=True)

with colM:
    st.subheader("Migrated view (const‑v Kirchhoff)")
    if do_mig:
        with st.spinner("Migrating…"):
            mig, x_out_km, t_out = kirchhoff_time_migration(
                section, true_dt, x_km, float(v_mig),
                stride_x=int(stride_x), stride_t=int(stride_t),
                aperture_km=float(aperture_km)
            )
        figm, axm = plt.subplots(figsize=(8, 6))
        mig_disp = mig / (np.max(np.abs(mig)) + 1e-9)
        axm.imshow(mig_disp, extent=[x_out_km[0], x_out_km[-1], t_out[-1], t_out[0]], aspect='auto', interpolation='bilinear')
        axm.set_xlabel('Distance (km)')
        axm.set_ylabel('Time (s)')
        st.pyplot(figm, clear_figure=True)
    else:
        st.info("Enable migration in the sidebar to compute and display a migrated section.")

st.info(
    "**Reading the panels:**  Left: velocity model + bathymetry. Right: TWT curves; compare vertical-ray, bent-ray, and no-body. "
    "Bottom-left: unmigrated synthetic. Bottom-right: migrated (constant-v). Try under/over v_mig to see smile/frown."
)
