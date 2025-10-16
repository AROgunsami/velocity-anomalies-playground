# app.py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# -----------------------------
# Basic helpers
# -----------------------------
def make_grid(nx=301, nz=301, x_km=4.0, z_km=2.0):
    x = np.linspace(0, x_km * 1000.0, nx)  # meters
    z = np.linspace(0, z_km * 1000.0, nz)  # meters
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    X, Z = np.meshgrid(x, z)
    return X, Z, dx, dz

def ricker(f, dt, nt, t0=0.0):
    t = np.arange(nt) * dt
    pi2 = (np.pi ** 2)
    w = (1 - 2 * pi2 * (f ** 2) * (t - t0) ** 2) * np.exp(-pi2 * (f ** 2) * (t - t0) ** 2)
    return w

def integrate_vertical_time(v_col, dz):
    return np.sum((1.0 / np.maximum(v_col, 1.0)) * dz)

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
            t_in = np.sqrt(t0 * t0 + (dx_m / v_mig) ** 2)
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

    return mig, x[xi], ti * dt

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Velocity Modeling — Pull-up & Push-down", layout="wide")
st.title("Velocity Modeling — Pull-up & Push-down")
st.caption("Interactive toy model to illustrate fast/slow bodies and seabed effects on TWT and imaging.")

with st.sidebar:
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
    v_sed   = st.number_input("Sediment velocity", 1800, 4500, 2200, 50)

    st.subheader("Bathymetry / Seabed")
    water_z       = st.slider("Water depth at left (m)", 0, int(z_km * 1000) - 50, 300, 10)
    water_z_right = st.slider("Water depth at right (m)", 0, int(z_km * 1000) - 50, 300, 10)

    # Rugose seabed controls
    enable_rugosity = st.checkbox("Enable rugose seabed (add sinusoid)", value=False)
    if enable_rugosity:
        rug_amp       = st.slider("Rugosity amplitude (m)", 5, 200, 50, 5)
        rug_wlen_km   = st.slider("Rugosity wavelength (km)", 0.1, max(0.2, float(x_km)), min(1.0, float(x_km)), 0.1)
        rug_phase_deg = st.slider("Phase (deg)", 0, 360, 0, 10)
    else:
        rug_amp, rug_wlen_km, rug_phase_deg = 0.0, 1.0, 0

    st.subheader("Target reflector (true depth)")
    z_ref = st.slider("Reflector depth below sea surface (m)", 200, int(z_km * 1000) - 50, 1500, 10)

    st.divider()
    st.subheader("Anomaly (body)")
    anomaly_kind = st.radio("Type", ["Fast (pull-up)", "Slow (push-down)"], index=1)
    v_anom = st.number_input("Anomaly velocity (m/s)", 1500, 6000, 3000, 50)
    col1, col2 = st.columns(2)
    with col1:
        x0_km = st.slider("Center x (km)", 0.0, x_km, x_km / 2, 0.1)
        ax_km = st.slider("Half-width aₓ (km)", 0.05, x_km, 0.5, 0.05)
    with col2:
        z0_km = st.slider("Center z (km)", 0.0, z_km, 1.0, 0.05)
        az_km = st.slider("Half-height a_z (km)", 0.02, z_km, 0.3, 0.02)

    st.caption("The ellipse ((x−x₀)/aₓ)² + ((z−z₀)/a_z)² ≤ 1 defines the body. Set its velocity via 'Anomaly velocity'.")

    st.divider()
    st.subheader("Synthetic section")
    fdom = st.slider("Ricker dominant freq (Hz)", 5, 60, 25, 1)
    tmax = st.slider("Record length (s)", 0.5, 5.0, 2.0, 0.1)
    dt   = st.select_slider("Sample rate dt (ms)", options=[0.5, 1.0, 2.0, 4.0, 8.0], value=2.0)

    st.divider()
    st.subheader("Poststack time migration (constant-velocity)")
    do_mig      = st.checkbox("Show migrated view (Kirchhoff)", value=False)
    v_mig       = st.number_input("Migration velocity v (m/s)", 1500, 6000, int(v_sed), 50)
    aperture_km = st.slider("Migration aperture (km)", 0.1, max(0.2, float(x_km)), min(float(x_km) / 3, float(x_km)), 0.1)
    stride_x    = st.slider("Output x decimation (bigger = faster)", min_value=1, max_value=10, value=4, step=1)
    stride_t    = st.slider("Output t decimation (bigger = faster)", min_value=1, max_value=10, value=2, step=1)

# -----------------------------
# Build model
# -----------------------------
X, Z, dx, dz = make_grid(nx, nz, x_km=x_km, z_km=z_km)

# Seabed/bathymetry: linear plane + optional sinusoidal rugosity
x_m   = np.linspace(0.0, x_km * 1000.0, nx)
z_lin = np.linspace(water_z, water_z_right, nx)
if enable_rugosity and rug_amp > 0:
    phase = np.deg2rad(rug_phase_deg)
    z_seabed = z_lin + rug_amp * np.sin(2 * np.pi * x_m / (rug_wlen_km * 1000.0) + phase)
else:
    z_seabed = z_lin

# Background velocity using broadcasting (no ZSB)
V = np.where(Z < z_seabed[None, :], v_water, v_sed).astype(float)

# Apply anomaly ellipse (only below seabed)
ellipse = (((X - x0_km * 1000.0) / (ax_km * 1000.0)) ** 2 +
           ((Z - z0_km * 1000.0) / (az_km * 1000.0)) ** 2) <= 1.0
if anomaly_kind.startswith("Fast") or anomaly_kind.startswith("Slow"):
    V = np.where(ellipse & (Z >= z_seabed[None, :]), float(v_anom), V)

V = np.clip(V, 200.0, 7000.0)

# Two-way times (vertical rays)
z_ref_arr = np.maximum(z_seabed + 5.0, min(z_ref, int(z_km * 1000.0) - 1))
z_axis = Z[:, 0]
oneway_t_seabed = np.zeros(nx)
oneway_t_ref = np.zeros(nx)

for ix in range(nx):
    zsb = z_seabed[ix]
    zrf = z_ref_arr[ix]
    ksb = int(np.clip(np.searchsorted(z_axis, zsb), 1, nz - 1))
    krf = int(np.clip(np.searchsorted(z_axis, zrf), 1, nz - 1))
    vcol = V[:ksb, ix]
    oneway_t_seabed[ix] = integrate_vertical_time(vcol, dz)
    vcol_ref = V[:krf, ix]
    oneway_t_ref[ix] = integrate_vertical_time(vcol_ref, dz)

TWT_seabed = 2.0 * oneway_t_seabed
TWT_ref    = 2.0 * oneway_t_ref

# Baseline (no anomaly) for comparison (vertical rays)
V0 = np.where(Z < z_seabed[None, :], v_water, v_sed).astype(float)
oneway_t_ref0 = np.zeros(nx)
for ix in range(nx):
    zrf = z_ref_arr[ix]
    krf = int(np.clip(np.searchsorted(z_axis, zrf), 1, nz - 1))
    oneway_t_ref0[ix] = integrate_vertical_time(V0[:krf, ix], dz)
TWT_ref0 = 2.0 * oneway_t_ref0

# -----------------------------
# Build a simple zero-offset synthetic (two reflectors)
# -----------------------------
true_dt = dt / 1000.0
nt = int(tmax / true_dt) + 1
section = np.zeros((nt, nx), dtype=float)

# Wavelet (compact)
wlen_s = min(0.256, tmax)
w = ricker(fdom, true_dt, int(wlen_s / true_dt) + 1)

# Stamp events safely
for ix in range(nx):
    # Seabed
    isamp_s = int(np.clip(np.round(TWT_seabed[ix] / true_dt), 0, nt - 1))
    if isamp_s < nt:
        i0 = max(0, isamp_s - len(w) // 2)
        i1 = min(nt, isamp_s + (len(w) - len(w) // 2))
        wi0 = max(0, len(w) // 2 - (isamp_s - i0))
        seg_len = min(i1 - i0, len(w) - wi0)
        if seg_len > 0:
            section[i0:i0 + seg_len, ix] += w[wi0:wi0 + seg_len] * 0.8

    # Deep reflector
    isamp_r = int(np.clip(np.round(TWT_ref[ix] / true_dt), 0, nt - 1))
    if isamp_r < nt:
        i0 = max(0, isamp_r - len(w) // 2)
        i1 = min(nt, isamp_r + (len(w) - len(w) // 2))
        wi0 = max(0, len(w) // 2 - (isamp_r - i0))
        seg_len = min(i1 - i0, len(w) - wi0)
        if seg_len > 0:
            section[i0:i0 + seg_len, ix] += w[wi0:wi0 + seg_len] * 1.0

# -----------------------------
# Plots
# -----------------------------
colL, colR = st.columns([1.1, 1.0])

with colL:
    st.subheader("Velocity model")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(V, extent=[0, x_km, z_km, 0], aspect='auto')
    ax.plot(np.linspace(0, x_km, nx), z_seabed / 1000.0, lw=1.2, label='Seabed', alpha=0.9)
    ax.axhline(np.mean(z_ref_arr) / 1000.0, ls='--', lw=1.0, label='True deep reflector (avg)')
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Depth (km)')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Velocity (m/s)')
    ax.legend(loc='upper right')
    st.pyplot(fig, clear_figure=True)
    st.caption("Adjust anomaly velocity relative to sediments: lower → push-down, higher → pull-up. Water column uses v_water.")

with colR:
    st.subheader("Two-way time of horizons")
    x_axis_km = np.linspace(0, x_km, nx)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(x_axis_km, TWT_seabed, label='Seabed TWT', lw=1.2)
    ax2.plot(x_axis_km, TWT_ref, label='Deep reflector TWT (vertical rays)', lw=1.8)
    ax2.plot(x_axis_km, TWT_ref0, label='Deep reflector TWT (no body)', lw=1.0, linestyle='--')
    ax2.set_xlabel('Distance (km)')
    ax2.set_ylabel('Time (s)')
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.25)
    ax2.legend()
    st.pyplot(fig2, clear_figure=True)

colS, colM = st.columns(2)
with colS:
    st.subheader("Zero-offset synthetic (image)")
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    sec_disp = section / (np.max(np.abs(section)) + 1e-9)
    ax3.imshow(sec_disp, extent=[0, x_km, tmax, 0], aspect='auto', interpolation='bilinear')
    ax3.set_xlabel('Distance (km)')
    ax3.set_ylabel('Time (s)')
    st.pyplot(fig3, clear_figure=True)

with colM:
    st.subheader("Migrated view (const-v Kirchhoff)")
    if do_mig:
        with st.spinner("Migrating…"):
            mig, x_out_km, t_out = kirchhoff_time_migration(
                section, true_dt, x_km, float(v_mig),
                stride_x=int(stride_x), stride_t=int(stride_t),
                aperture_km=float(aperture_km)
            )
        figm, axm = plt.subplots(figsize=(8, 6))
        mig_disp = mig / (np.max(np.abs(mig)) + 1e-9)
        axm.imshow(mig_disp, extent=[x_out_km[0], x_out_km[-1], t_out[-1], t_out[0]],
                   aspect='auto', interpolation='bilinear')
        axm.set_xlabel('Distance (km)')
        axm.set_ylabel('Time (s)')
        st.pyplot(figm, clear_figure=True)
    else:
        st.info("Enable migration in the sidebar to compute and display a migrated section.")

st.info(
    "**Reading the panels:**  Left: velocity model + bathymetry. "
    "Right: TWT curves (vertical rays) with and without the body. "
    "Bottom-left: unmigrated synthetic. Bottom-right: migrated (constant-v). "
    "Try under/over v_mig to see smile/frown."
)
