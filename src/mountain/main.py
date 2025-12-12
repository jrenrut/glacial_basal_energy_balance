"""Simulate glacier and mountain evolution with basal energy balance.

"
This piece of ice still shows traces of its original nature: part of it has become
stone, part resisted the cold. It is a freak of winter's, more precious by reason of its
incomplete crystallization, for that the jewel contains within itself living water...

Alpine ice was becoming so hard that the sun could not melt it, and this excess of cold
was like to make it precious as diamond. But it could not imitate that stone in its
entirety for at its heart lay a drop of water which betrayed its nature. As crystal its
value is enhanced, for this liquid rock is accounted a miracle and the water enclosed
within it increases its rarity...

Do not despise this sphere of rock-crystal. Kings' palaces contain no rarer jewel, nor
are the Red Sea's pearls of greater value. It may be shapeless ice, unpolished rock, a
rough, uncarven mass, yet is it accounted among the most precious of riches.
"

- Claudian, "De Crystallo Cui Aqua Inerat", circa 400 AD
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
from typing import Optional
from mountain.utils import smooth_1d, plot_mountain

# --- CONSTANTS ---

# numerical simulation constants
CFL = 0.25  # dimensionless - CFL number for explicit diffusion stability

# general physical constants
SECONDS_PER_YEAR = 365.25 * 24 * 60 * 60  # s
G = 9.81  # m s^-2 - gravitational acceleration
R = 8.314  # J mol^-1 K^-1 - universal gas constant

# climatic constants
GAMMA = 5e-3  # K m^-1 - temperature lapse rate
CLIMATE_TEMP = 286.0  # K - surface temperature
DDF = 1e-8  # m K^-1 s^-1 - melt degree-day factor

# orograpic precipitation constants
CLIMATE_RHO_0 = 1.225  # kg m^-3 - density of air at sea level
CLIMATE_U = 10.0  # m s^-1 - wind speed
CLIMATE_V_F = 4.0  # m s^-1 - terminal speed
CLIMATE_TAU_G = 1000.0  # s - condensation timescale
CLIMATE_TAU_EV = 2000.0  # s - evaporation timescale
CLIMATE_Q_0 = 4e-3  # dimensionless - kg water per kg air
CLIMATE_H_M = 3e3  # m - scale height for moisture
RHO_WATER = 1000.0  # kg m^-3 - density of water
PRECIPITATION_SCALE = 1e-1  # dimensionless - scaling factor for precipitation
T_SPAN = 10.0  # K - temperature span for precipitation calc

# ice flow constants
RHO_ICE = 917.0  # kg m^-3 - density of ice
N_ICE = 3.0  # dimensionless - exponent in Glen's flow law
A_0 = 2.9e-13  # dimensionless - Glen's flow law pre-factor
E_0 = 60e3  # J mol^-1 - activation energy for Glen's flow law
GLEN_A_SCALE = 1e0  # dimensionless - scaling factor for Glen's flow law pre-factor
VEL_CAP = 1000.0  # m a^-1 - maximum allowed ice velocity

# ice thermal constants
T_FREEZE = 273.15 - 2.0  # K - temperature threshold for snow/rain
T_MELT = 273.15 + 1.0  # K - temperature threshold for melting
K_ICE = 2.1  # W m^-1 K^-1 - thermal conductivity of ice
L_ICE = 3.34e5  # J kg^-1 - latent heat of fusion for ice
GAMMA_CC = 9.8e-8  # K * Pa^-1, Clausius-Clapeyron slope
C_ICE = 2100  # J kg^-1 K^-1, specific heat of ice
H_EFF_SLIDE = 10  # m, effective thickness for friction heating
H_EFF_COND = 0.5  # m - minimum ice thickness to allow conduction calc
H_EFF_DEF = 5.0  # m - effective thickness for deformational heating
Q_COND_CLIP = 200.0  # W m^-2 - maximum conductive heat flux to avoid numerical issues
Q_DEF_CLIP = 50.0  # W m^-2 - maximum deformational heating to avoid numerical issues
Q_FRIC_CLIP = 50.0  # W m^-2 - maximum frictional heating to avoid numerical issues
MAX_TEMP_CHANGE = (
    10.0 / SECONDS_PER_YEAR
)  # K s^-1 - maximum temperature change per time step

# uplift constants
TAU_OROGENY = 1e6 * SECONDS_PER_YEAR  # s - characteristic orogeny timescale
Q_GEO_0 = 0.05  # W m^-2 - initial geothermal heat flux
K_CRUST = 2.5  # W m^-1 K^-1 - thermal conductivity of crust
ALPHA_CRUST = 1e6  # m^2 K^-1 - GHF correction
Z_CRUST = 0  # m - reference height for GHF correction
UPLIFT_RATE = 1e-1 / SECONDS_PER_YEAR  # m s^-1 - uniform uplift rate
NOISE_AMPLITUDE = 50  # m - amplitude of random noise in initial topography
NOISE_SMOOTHING = 3.0  # m - smoothing scale for initial topography noise

# erosion constants
L_G = 1.0  # dimensionless - exponent in erosion law
K_G = 1e-2 / SECONDS_PER_YEAR ** (L_G - 1)  # m^(1-L_G) s^(L_G -1) - erosion coefficient
MAX_EROSION_RATE = 1e2 / SECONDS_PER_YEAR  # m s^-1
F_SLIDE = 5e-1  # dimensionless - sliding scaling factor
ANGLE_OF_REPOSE = 35.0  # degrees - angle of repose for mountain
SLOPE_MAX = np.tan(np.deg2rad(ANGLE_OF_REPOSE))  # max |dz/dx|
K_REPOSE = 5e-3  # m^2 s^-1 - flux coefficient for excess slope


class Mountain:
    """Mountain and glacier evolution model with basal energy balance."""

    def __init__(
        self,
        dt: float,
        t0: float,
        t1: float,
        n_x: int,
        height: float,
        length1: float,
        length2: float,
        mountain_buffer: Optional[float] = 0.2,
    ):
        """Initialize mountain model.

        Parameters
        ----------
        dt : float
            a - time step
        t0 : float
            a - start time
        t1 : float
            a - end time
        n_x : int
            number of spatial points
        height : float
            m - initial mountain height
        length1 : float
            m - initial mountain length on windward side
        length2 : float
            m - initial mountain length on leeward side
        mountain_buffer : float, optional
            dimensionless - buffer size as fraction of mountain length, by default 0.2
        """
        # numerical parameters
        self.t = 0.0  # a - simulation time
        self.index = 0
        self.dt = dt  # a - time step
        self.t0 = t0  # a - start time
        self.t1 = t1  # a - end time
        self.nx = n_x  # number of spatial points
        self.time_array = np.arange(t0, t1 + dt, dt)
        self.nt = len(self.time_array)

        # create mountain profile - creates z_mountain, z_ice, and x arrays
        self.height = float(height)
        self.height_ref = float(height)
        self.l1 = float(length1)
        self.l2 = float(length2)
        self.create_mountain_profile(mountain_buffer)
        self.h_ice = np.maximum(self.z_ice - self.z_mountain_ref, 0.0)

        # initialize model variables
        self.Q_geo = np.full(
            self.nx, Q_GEO_0, dtype=np.float32
        )  # W m^-2 - geothermal heat flux
        self.Q_cond = np.zeros(
            self.nx, dtype=np.float32
        )  # W m^-2 - conductive heat flux
        self.Q_fric = np.zeros(
            self.nx, dtype=np.float32
        )  # W m^-2 - frictional heat flux
        self.Q_def = np.zeros(
            self.nx, dtype=np.float32
        )  # W m^-2 - deformational heat flux
        self.Q_lat = np.zeros(self.nx, dtype=np.float32)  # W m^-2 - latent heat flux
        self.Q_net = np.zeros(self.nx, dtype=np.float32)  # W m^-2 - net heat flux
        self.T_m = np.full(self.nx, T_MELT, dtype=np.float32)  # K - ice temperature
        self.T_s = CLIMATE_TEMP - GAMMA * self.z_ice  # K - surface temperature
        self.T_b = self.T_s.copy()  # K - basal temperature
        self.dmdt = np.zeros(
            self.nx, dtype=np.float32
        )  # m s^-1 - basal melt rate (disabled)
        self.dTbdt = np.zeros(
            self.nx, dtype=np.float32
        )  # K s^-1 - basal temperature change rate
        self.dedt = np.zeros(
            self.nx, dtype=np.float32
        )  # m s^-1 - erosion rate (disabled)
        self.dbdt = np.zeros(
            self.nx, dtype=np.float32
        )  # m s^-1 - surface mass balance rate
        self.dpdt = np.zeros(self.nx, dtype=np.float32)  # m s^-1 - precipitation rate
        self.dqdx_diffusion = np.zeros(
            self.nx, dtype=np.float32
        )  # m s^-1 - ice diffusion rate
        self.dqdx_advection = np.zeros(
            self.nx, dtype=np.float32
        )  # m s^-2 - flux divergence on thickness (centers)
        self.u_b = np.zeros(
            self.nx, dtype=np.float32
        )  # m s^-1 - basal ice velocity (disabled)
        self.tau_b = np.zeros(self.nx, dtype=np.float32)  # Pa - basal shear stress
        self.glen_A_faces = np.zeros(
            self.nx, dtype=np.float32
        )  # dimensionless - Glen's flow law prefactor at cell faces
        self.glen_A_centers = np.zeros(
            self.nx, dtype=np.float32
        )  # dimensionless - Glen's flow law prefactor at cell centers

        # initialize history arrays
        self.z_mountain_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.z_ice_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.z_mountain_ref_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.Q_geo_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.Q_cond_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.Q_fric_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.Q_def_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.Q_lat_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.T_m_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.T_s_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.T_b_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.dmdt_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.dedt_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.dbdt_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.dpdt_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.dqdx_diffusion_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.dqdx_advection_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.u_b_hist = np.zeros((self.nt, self.nx), dtype=np.float32)
        self.ELA_hist = np.zeros(self.nt, dtype=np.float32)

        self.run()

    def create_mountain_profile(self, mountain_buffer):
        """Create initial mountain profile.

        Parameters
        ----------
        mountain_buffer : float
            dimensionless - buffer size as fraction of mountain length
        """
        # create x array with buffer on either side of mountain
        buffer = mountain_buffer * max(self.l1, self.l2)
        x = np.linspace(-self.l1 - buffer, self.l2 + buffer, self.nx)
        self.dx = x[1] - x[0]
        z = np.zeros_like(x, dtype=float)

        # create mountain shape - triangular
        left = (x >= -self.l1) & (x <= 0.0)
        right = (x > 0.0) & (x <= self.l2)
        if np.any(left):
            z[left] = self.height * ((x[left] + float(self.l1)) / float(self.l1))
        if np.any(right):
            z[right] = self.height * ((float(self.l2) - x[right]) / float(self.l2))
        z[~(left | right)] = 0.0

        # add random noise to mountain profile
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(len(x)) * NOISE_AMPLITUDE
        sigma_pts = max(1, int(round(1.0 / self.dx)))
        noise = gaussian_filter1d(noise, sigma=sigma_pts, mode="reflect")
        z += noise
        z = smooth_1d(z, sigma=NOISE_SMOOTHING, mode="reflect")
        z = np.maximum(z, 0.0)

        # store mountain profile, create reference and ice profiles and x array
        self.x = x
        self.z_mountain = z  # m - mountain surface elevation
        self.z_mountain_ref = z.copy()  # m - mountain surface initially
        self.z_ice = z.copy()  # m - ice surface initially at mountain surface

    def uplift(self, dt):
        """Apply tectonic uplift.

        Parameters
        ----------
        dt : float
            s - time step
        """
        # Fraction of reference to add this timestep
        if self.t < TAU_OROGENY:
            frac = UPLIFT_RATE * dt / self.height_ref
        else:
            frac = UPLIFT_RATE * dt * np.exp(dt / TAU_OROGENY) / self.height_ref

        dz = self.height * frac  # m - uplift amount this timestep
        dl1 = self.l1 * frac  # m - increase in left length
        dl2 = self.l2 * frac  # m - increase in right length

        # apply uplift to mountain profile and reference and ice profiles
        left = (self.x >= -self.l1 - dl1) & (self.x <= 0)
        right = (self.x > 0) & (self.x <= self.l2 + dl2)
        if np.any(left):
            self.z_mountain[left] += dz * (
                (self.x[left] + self.l1 + dl1) / (self.l1 + dl1)
            )
            self.z_mountain_ref[left] += dz * (
                (self.x[left] + self.l1 + dl1) / (self.l1 + dl1)
            )
            self.z_ice[left] += dz * ((self.x[left] + self.l1 + dl1) / (self.l1 + dl1))
        if np.any(right):
            self.z_mountain[right] += dz * (
                (self.l2 + dl2 - self.x[right]) / (self.l2 + dl2)
            )
            self.z_mountain_ref[right] += dz * (
                (self.l2 + dl2 - self.x[right]) / (self.l2 + dl2)
            )
            self.z_ice[right] += dz * (
                (self.l2 + dl2 - self.x[right]) / (self.l2 + dl2)
            )

        # update mountain geometry parameters
        self.height += dz
        self.l1 += dl1
        self.l2 += dl2

    def update_Q_geo(self):
        """Update geothermal heat flux based on mountain height and time."""
        Q_geo = np.ones(
            self.nx, dtype=float
        )  # W m^-2 - initial geothermal flux profile
        # if within orogeny timescale, constant flux
        if self.t < TAU_OROGENY:
            Q_geo *= Q_GEO_0
        # else, exponentially decaying flux
        else:
            Q_geo *= Q_GEO_0 * np.exp(-self.t / TAU_OROGENY)

        # apply height correction
        height_correction = (K_CRUST * (self.z_mountain - Z_CRUST)) / ALPHA_CRUST
        Q_geo -= height_correction
        self.Q_geo = np.maximum(Q_geo, 0.0)  # no negative geothermal flux

    def update_dpdt(self):
        """Update orographic precipitation rate dp/dt."""
        l1 = max(self.l1, 1e-8)
        l2 = max(self.l2, 1e-8)

        # orographic forcing from Roe & Baker 2005
        R_0 = CLIMATE_RHO_0 * CLIMATE_Q_0 * CLIMATE_U * self.height / l1  # kg m^-2 s^-1
        theta_1 = (
            l1 * CLIMATE_V_F / (CLIMATE_U * max(self.height, 1e-8))
        )  # dimensionless
        theta_2 = (
            l2 * CLIMATE_V_F / (CLIMATE_U * max(self.height, 1e-8))
        )  # dimensionless
        alpha = self.height / CLIMATE_H_M  # dimensionless
        psi_1 = max(l1 / (CLIMATE_U * CLIMATE_TAU_G), 1e-3)  # dimensionless
        xi = self.height / (CLIMATE_V_F * CLIMATE_TAU_EV)  # dimensionless

        left_mask = (self.x >= l1 * (1 / psi_1 - 1)) & (self.x <= 0.0)
        right_mask = self.x >= 0.0

        z_scale_1 = (
            1 - (1 / psi_1) + (self.x / l1)
        )  # dimensionless - scaled height on windward side
        z_scale_2 = (
            1 - (1 / psi_1) + (theta_2 * self.x / l2)
        )  # dimensionless - scaled height on leeward side
        self.dpdt[left_mask] = (
            R_0
            * (theta_1 / (theta_1 - 1))
            * (
                np.exp(-alpha * z_scale_1[left_mask])
                - np.exp(-theta_1 * alpha * z_scale_1[left_mask])
            )
        )
        self.dpdt[right_mask] = (
            R_0
            * (theta_1 / (theta_1 - 1))
            * np.exp(-theta_2 * xi * self.x[right_mask] / l2)
            * np.exp(-alpha * z_scale_2[right_mask])
            * (1 - np.exp(-alpha * (1 - (1 / psi_1)) * (theta_1 - 1)))
        )
        self.dpdt = (
            np.maximum(np.where(np.isfinite(self.dpdt), self.dpdt, 0.0), 0.0)
            / RHO_WATER
            * PRECIPITATION_SCALE
        )  # m s^-1

        # update ELA
        p_mean = np.mean(self.dpdt)  # m s^-1 - mean precipitation rate
        T_ELA = T_MELT + p_mean / (DDF * SECONDS_PER_YEAR)  # K - ELA temperature
        self.ELA = (CLIMATE_TEMP - T_ELA) / GAMMA  # m - equilibrium line altitude

    def update_dbdt(self):
        """Update surface mass balance rate db/dt."""
        # gradual transition from rain to snow over T_SPAN
        freeze_frac = np.clip((T_FREEZE - self.T_s) / T_SPAN, 0.0, 1.0)  # dimensionless
        accumulation = self.dpdt * freeze_frac  # m s^-1 - accumulation rate
        # simple degree-day ablation
        ablation = DDF * np.maximum(self.T_s - T_MELT, 0.0)  # m s^-1 - ablation rate

        # mass balance = accumulation - ablation
        self.dbdt = accumulation - ablation  # m s^-1

    def update_glen_flow_factor(self):
        """Update Glen's flow law prefactor based on basal temperature."""
        # Glen's flow law prefactor as a function of temperature (K)
        T_mean_center = 0.5 * (self.T_s + self.T_b)  # K - centered temperature
        T_mean_face = 0.5 * (
            T_mean_center[1:] + T_mean_center[:-1]
        )  # K - cell face temperature
        # calculate glen prefactors for cell face and centered values
        self.glen_A_center = A_0 * np.exp(-E_0 / (R * T_mean_center)) * GLEN_A_SCALE
        self.glen_A_face = A_0 * np.exp(-E_0 / (R * T_mean_face)) * GLEN_A_SCALE

    def update(self, dt):
        """Advance model state.

        Parameters
        ----------
        dt : float
            a - time step

        """
        dt_years = float(dt)  # a - time step in years
        dt_seconds = dt_years * SECONDS_PER_YEAR  # s - time step in seconds
        self.t += dt_seconds  # s - update model time

        # update basic processes once per outer time step
        self.uplift(dt_seconds)
        self.update_Q_geo()
        self.update_dpdt()
        self.update_dbdt()
        self.update_glen_flow_factor()

        # inner time-stepping loop for stability
        remaining_dt = dt_seconds  # s - remaining time to step
        while remaining_dt > 0:
            # calculate values for flux and temperature calculations
            h_ice = np.maximum(self.z_ice - self.z_mountain, 0.0)  # m - ice thickness
            dhdx = np.gradient(self.z_ice, self.dx)  # dimensionless - ice surface slope
            ice_mask = h_ice >= 1e-8  # dimensionless - ice presence mask
            ice_mask_float = ice_mask.astype(float)
            P = RHO_ICE * G * h_ice  # Pa - basal pressure
            T_m = T_MELT - (GAMMA_CC * P)  # K - pressure-melting point at bed
            temperate_mask = (
                self.T_b >= T_m
            ) & ice_mask  # dimensionless - temperate ice mask

            # ice flux
            hmean_faces = (
                h_ice[:-1] + h_ice[1:]
            ) / 2.0  # m - ice thickness at faces (length nx-1)
            dhdx_faces = (
                np.diff(self.z_ice) / self.dx
            )  # dimensionless - ice surface slope at faces (length nx-1)
            glen_pref = (
                2.0 * self.glen_A_face * (RHO_ICE * G) ** N_ICE / float(N_ICE + 2)
            )  # dimensionless - Glen prefactor at faces (length nx-1)
            qx_faces = (
                glen_pref
                * np.abs(dhdx_faces) ** (N_ICE - 1)
                * hmean_faces ** (N_ICE + 2)
            )  # m^2 s^-1 - ice flux at faces (length nx-1)
            q_faces = (
                -qx_faces * dhdx_faces
            )  # m^2 s^-1 - ice flux at faces (length nx-1)
            q_padded = np.concatenate(([0.0], q_faces, [0.0]))  # (length nx+1)
            q_center = 0.5 * (q_padded[:-1] + q_padded[1:])  # (length nx)
            dqdx_diffusion = (
                q_padded[1:] - q_padded[:-1]
            ) / self.dx  # m s^-1 - flux divergence (length nx)

            # stable inner time step based on diffusion CFL
            max_qx_local = np.nanmax(qx_faces) if qx_faces.size > 0 else 0.0
            if max_qx_local > 0.0:
                max_stable_dt = CFL * self.dx**2 / max_qx_local
            else:
                max_stable_dt = remaining_dt
            dt_inner = min(remaining_dt, max_stable_dt)

            # basal sliding velocity
            alpha = np.clip(
                h_ice / (H_EFF_SLIDE + 1e-12), 0.0, 1.0
            )  # dimensionless - sliding activation
            safe_h_slide = (
                alpha * h_ice + (1.0 - alpha) * H_EFF_SLIDE
            )  # m - effective thickness for sliding calculation
            slide_mask = np.clip(
                temperate_mask.astype(float), 0.0, 1.0
            )  # dimensionless - sliding mask
            u_b = (
                np.clip(q_center / safe_h_slide, -VEL_CAP, VEL_CAP)
                * slide_mask
                * F_SLIDE
            )  # m s^-1 - basal sliding velocity
            u_b = smooth_1d(u_b, sigma=1.0, mode="reflect")

            # advective flux from sliding
            q_slide_center = (
                h_ice * u_b
            )  # m^2 s^-1 - sliding flux at cell centers (length nx)
            q_slide_faces = 0.5 * (
                q_slide_center[1:] + q_slide_center[:-1]
            )  # m^2 s^-1 - sliding flux at cell faces (length nx-1)
            q_slide_padded = np.concatenate(
                ([0.0], q_slide_faces, [0.0])
            )  # (length nx+1)
            dqdx_advection = (
                q_slide_padded[1:] - q_slide_padded[:-1]
            ) / self.dx  # m s^-1 - advective sliding flux (length nx)

            # frictional heat flux
            tau_b = RHO_ICE * G * h_ice * np.abs(dhdx)  # Pa - basal shear stress
            Q_fric = tau_b * np.abs(u_b) * ice_mask_float  # W m^-2 - frictional heating
            Q_fric = np.clip(Q_fric, -Q_FRIC_CLIP, Q_FRIC_CLIP)

            # conductive heat flux
            cond_mask = h_ice >= H_EFF_COND  # dimensionless - mask for conduction
            safe_h_cond = h_ice + H_EFF_COND  # m - effective thickness for conduction
            taper = np.minimum(
                1.0, h_ice / (H_EFF_COND + 1e-12)
            )  # dimensionless - tapering factor for thin ice
            Q_cond = (
                K_ICE * (self.T_s - self.T_b) / safe_h_cond
            )  # W m^-2 - conductive heat flux
            Q_cond = np.clip(
                Q_cond * taper, -Q_COND_CLIP, Q_COND_CLIP
            ) * cond_mask.astype(float)

            # deformation heat flux
            glen_pref_def = (
                self.glen_A_center * (RHO_ICE * G) ** (N_ICE + 1.0) / float(N_ICE + 2)
            )  # dimensionless - Glen prefactor for deformation (length nx)
            Q_def = (
                glen_pref_def * h_ice ** (N_ICE + 2) * np.abs(dhdx) ** (N_ICE + 1)
            )  # W m^-2 - deformational heating
            f_basal = (
                (N_ICE + 2) * H_EFF_DEF / h_ice[cond_mask]
            )  # dimensionless - basal fraction for deformation
            Q_def[cond_mask] *= f_basal
            Q_def = np.clip(Q_def, -Q_DEF_CLIP, Q_DEF_CLIP)

            # net basal heat flux
            Q_net = self.Q_geo + Q_fric + Q_cond + Q_def  # W m^-2 - net basal heat flux

            # update basal temperature where cold
            cold_mask = (self.T_b < T_m) & ice_mask
            if np.any(cold_mask):
                dTb = (dt_inner * Q_net[cold_mask]) / (
                    RHO_ICE * C_ICE * safe_h_cond[cold_mask]
                )  # K - basal temperature change
                per_step_cap = MAX_TEMP_CHANGE * dt_inner
                dTb = np.clip(dTb, -per_step_cap, per_step_cap)
                excess_heat_mask = (self.T_b[cold_mask] + dTb) > T_m[
                    cold_mask
                ]  # dimensionless - mask for excess heat
                if np.any(excess_heat_mask):
                    dTb[excess_heat_mask] = (
                        T_m[cold_mask][excess_heat_mask]
                        - self.T_b[cold_mask][excess_heat_mask]
                    )  # limit to melting point
                    excess_heat = (
                        Q_net[cold_mask][excess_heat_mask]
                        - (
                            dTb[excess_heat_mask]
                            * RHO_ICE
                            * C_ICE
                            * safe_h_cond[cold_mask][excess_heat_mask]
                        )
                        / dt_inner
                    )  # W m^-2 - excess heat available for melting
                    dmdt_excess = np.maximum(
                        excess_heat / (RHO_ICE * L_ICE), 0.0
                    )  # m s^-1 - melt rate from excess heat
                    Q_lat_excess = (
                        -dmdt_excess * RHO_ICE * L_ICE
                    )  # W m^-2 - latent heat flux from excess melting
                    Q_net[cold_mask][excess_heat_mask] -= (
                        Q_lat_excess  # remove latent heat from net flux
                    )
                self.T_b[cold_mask] += dTb  # K - update basal temperature
            T_b = np.minimum(self.T_b, T_m)

            # latent heat flux and basal melting where temperate
            Q_latent = np.zeros(self.nx, dtype=float)  # W m^-2 - latent heat flux
            dmdt = np.zeros(self.nx, dtype=float)  # m s^-
            temperate_mask = (
                T_b >= T_m
            ) & ice_mask  # dimensionless - mask for temperate basal ice
            if any(temperate_mask):
                dmdt[temperate_mask] = np.maximum(
                    Q_net[temperate_mask] / (RHO_ICE * L_ICE), 0.0
                )  # m s^-1 - melt rate from net heat flux
                Q_latent = -dmdt * RHO_ICE * L_ICE  # W m^-2 - latent heat flux

            # erosion
            dedt = K_G * np.abs(u_b) ** L_G  # m s^-1 - erosion rate
            dedt = np.minimum(dedt, MAX_EROSION_RATE)

            # update mountain heigh
            self.z_mountain = np.maximum(self.z_mountain - dedt * dt_inner, 0.0)  # m
            # enforce maximum angle of repose
            mountain_slope = (
                np.diff(self.z_mountain) / self.dx
            )  # dimensionless - mountain slope (length nx-1)
            excess = np.maximum(
                np.abs(mountain_slope) - SLOPE_MAX, 0.0
            )  # dimensionless - excess slope (length nx-1)
            q_rock_faces = (
                -K_REPOSE * excess * np.sign(mountain_slope)
            )  # m^2 s^-1 - rock flux at faces (length nx-1)
            q_rock_padded = np.concatenate(
                ([0.0], q_rock_faces, [0.0])
            )  # (length nx+1)
            dqdx_rock = (
                q_rock_padded[1:] - q_rock_padded[:-1]
            ) / self.dx  # m s^-1 - rock flux divergence (length nx)
            self.z_mountain = np.maximum(
                self.z_mountain - dqdx_rock * dt_inner, 0.0
            )  # m

            # keep ice above bed after bed change
            self.z_ice = np.maximum(self.z_ice, self.z_mountain)

            dzdt_ice = (
                self.dbdt - (dqdx_diffusion + dqdx_advection) - dmdt
            )  # m s^-1 - ice surface elevation change rate
            self.z_ice = np.maximum(
                self.z_ice + (dzdt_ice - dedt) * dt_inner, self.z_mountain
            )  # m (include erosion since it lowers bed)

            # update state
            self.h_ice = h_ice
            self.u_b = u_b
            self.tau_b = tau_b
            self.Q_cond = Q_cond
            self.Q_fric = Q_fric
            self.Q_def = Q_def
            self.Q_lat = Q_latent
            self.dmdt = dmdt
            self.dedt = dedt
            self.dbdt = self.dbdt
            self.dqdx_diffusion = -dqdx_diffusion
            self.dqdx_advection = dqdx_advection
            self.T_m = T_m
            self.T_b = T_b

            # advance inner time
            remaining_dt -= dt_inner

    def run(self):
        """Run mountain-glacier evolution simulation."""
        for i in tqdm(range(self.nt), desc="Running mountain-glacier evolution model"):
            dt = self.time_array[i] - self.t / SECONDS_PER_YEAR  # a - time step
            self.update(dt)  # advance model state

            # store history
            self.z_mountain_hist[i, :] = self.z_mountain
            self.z_ice_hist[i, :] = self.z_ice
            self.z_mountain_ref_hist[i, :] = self.z_mountain_ref
            self.Q_geo_hist[i, :] = self.Q_geo
            self.Q_cond_hist[i, :] = self.Q_cond
            self.Q_fric_hist[i, :] = self.Q_fric
            self.Q_def_hist[i, :] = self.Q_def
            self.Q_lat_hist[i, :] = self.Q_lat
            self.T_m_hist[i, :] = self.T_m
            self.T_s_hist[i, :] = self.T_s
            self.T_b_hist[i, :] = self.T_b
            self.dmdt_hist[i, :] = self.dmdt
            self.dedt_hist[i, :] = self.dedt
            self.dbdt_hist[i, :] = self.dbdt
            self.dpdt_hist[i, :] = self.dpdt
            self.dqdx_diffusion_hist[i, :] = self.dqdx_diffusion
            self.dqdx_advection_hist[i, :] = self.dqdx_advection
            self.u_b_hist[i, :] = self.u_b
            self.ELA_hist[i] = self.ELA


if __name__ == "__main__":
    # Experiment 1: glacial buzzsaw
    #  A short mountain with moderate uplift and high precipitation
    fname = "glacial_buzzsaw"

    CLIMATE_TEMP = 286.0  # K - surface temperature
    CLIMATE_Q_0 = 4e-3  # dimensionless - kg water per kg air
    PRECIPITATION_SCALE = 1e-1  # dimensionless - scaling factor for precipitation
    Q_GEO_0 = 0.05  # W m^-2 - initial geothermal heat flux
    UPLIFT_RATE = 1e-2 / SECONDS_PER_YEAR  # m s^-1 - uniform uplift rate

    mountain = Mountain(
        dt=10,
        t0=0.0,
        t1=1e4,
        n_x=501,
        height=4000.0,
        length1=20000.0,
        length2=40000.0,
        mountain_buffer=0.2,
    )

    plot_mountain(mountain, fname)

    # Experiment 2: glacial shielding
    #  A tall mountain with high uplift and low precipitation
    fname = "glacial_shielding"

    CLIMATE_TEMP = 282.0  # K - surface temperature
    CLIMATE_Q_0 = 4e-3  # dimensionless - kg water per kg air
    PRECIPITATION_SCALE = 1e-3  # dimensionless - scaling factor for precipitation
    Q_GEO_0 = 0.05  # W m^-2 - initial geothermal heat flux
    UPLIFT_RATE = 1e-1 / SECONDS_PER_YEAR  # m s^-1 - uniform uplift rate

    mountain = Mountain(
        dt=10,
        t0=0.0,
        t1=1e4,
        n_x=501,
        height=6000.0,
        length1=20000.0,
        length2=40000.0,
        mountain_buffer=0.2,
    )

    plot_mountain(mountain, fname)

    # Experiment 3: escape from glacial shielding
    #  A tall mountain with high uplift and low precipitation
    fname = "glacial_unshielding"

    CLIMATE_TEMP = 284.0  # K - surface temperature
    CLIMATE_Q_0 = 5e-2  # dimensionless - kg water per kg air
    PRECIPITATION_SCALE = 1e-3  # dimensionless - scaling factor for precipitation
    Q_GEO_0 = 0.05  # W m^-2 - initial geothermal heat flux
    UPLIFT_RATE = 1e-1 / SECONDS_PER_YEAR  # m s^-1 - uniform uplift rate

    mountain = Mountain(
        dt=10,
        t0=0.0,
        t1=1e4,
        n_x=501,
        height=6000.0,
        length1=20000.0,
        length2=40000.0,
        mountain_buffer=0.2,
    )

    plot_mountain(mountain, fname)

    # Experiment 4: escape from glacial shielding
    #  A tall mountain with high uplift and low precipitation
    fname = "reverse_buzzsaw"

    CLIMATE_TEMP = 286.0  # K - surface temperature
    CLIMATE_Q_0 = 1e-2  # dimensionless - kg water per kg air
    PRECIPITATION_SCALE = 1e-1  # dimensionless - scaling factor for precipitation
    Q_GEO_0 = 0.05  # W m^-2 - initial geothermal heat flux
    UPLIFT_RATE = 1e-2 / SECONDS_PER_YEAR  # m s^-1 - uniform uplift rate
    ANGLE_OF_REPOSE = 25.0  # degrees - angle of repose for mountain
    SLOPE_MAX = np.tan(np.deg2rad(ANGLE_OF_REPOSE))  # max |dz/dx|
    K_REPOSE = 5e-2  # m^2 s^-1 - flux coefficient for excess slope

    mountain = Mountain(
        dt=10,
        t0=0.0,
        t1=1e4,
        n_x=301,
        height=4000.0,
        length1=40000.0,
        length2=20000.0,
        mountain_buffer=0.2,
    )

    plot_mountain(mountain, fname)
