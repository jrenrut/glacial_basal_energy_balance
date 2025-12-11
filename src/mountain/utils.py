"""Helper functions for Mountain class."""

import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

pio.templates.default = None  # keep your layout clean
LATEX_FONT = "Latin Modern Roman"

SECONDS_PER_YEAR = 365.25 * 24 * 3600.0
T_MELT = 273.15  # K


def gaussian_kernel(sigma, radius=None):
    """
    Normalized 1D Gaussian kernel.

    Parameters
    ----------
    sigma : float
        Standard deviation.
    radius : int or None
        Kernel half-width. If None uses ceil(3*sigma).

    Returns
    -------
    k : ndarray
        1D kernel that sums to 1.
    """
    sigma = float(sigma)
    if sigma <= 0 or np.isclose(sigma, 0.0):
        return np.array([1.0], dtype=float)
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    return k


def smooth_1d(orig, sigma, radius=None, mode="reflect"):
    """
    Smooth a 1D array with a Gaussian kernel.

    Parameters
    ----------
    orig : array_like
        1D array to smooth.
    sigma : float
        Kernel sigma.
    radius : int or None
        Kernel half-width.
    mode : str
        Padding mode.

    Returns
    -------
    new : ndarray
        Smoothed array, same length as orig.
    """
    orig = np.asarray(orig, dtype=float)
    if orig.size == 0:
        return orig.copy()
    k = gaussian_kernel(sigma, radius=radius)
    pad = len(k) // 2
    padded = np.pad(orig, pad, mode=mode)  # type: ignore
    new = np.convolve(padded, k, mode="valid")
    return new


def viz_smooth(orig, sigma=2.0, clip_pct=99.0):
    """
    Smooth and clip array for visualization.

    Parameters
    ----------
    orig : array_like
        1D array to smooth and clip.
    sigma : float
        Smoothing sigma in points.
    clip_pct : float
        Percentile for clipping.
    Returns
    -------
    new : ndarray
        Smoothed and clipped array.
    """
    orig = np.asarray(orig, dtype=float)
    new = smooth_1d(orig, sigma=sigma, mode="reflect")
    vmin, vmax = np.percentile(new, [100 - clip_pct, clip_pct])
    new = np.clip(new, vmin, vmax)
    return new


def plot_mountain(mountain, fname=""):
    """
    Create show and save mountain plots.

    Parameters
    ----------
    mountain : Mountain
        Mountain object with history data.
    fname : str
        String to append to filename.

    """
    x_km = mountain.x / 1000.0  # km for plotting

    # frames for animation
    frames_mtn, frames_rate, frames_flux, frames_temp = [], [], [], []
    for k in range(mountain.nt):
        # mountain and ice profile
        frames_mtn.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x_km,
                        y=mountain.z_ice_hist[k] / 1000,
                        fill="tozeroy",
                        fillcolor="#17BECF",
                        mode="lines",
                        name="Ice",
                        line=dict(color="#17BECF"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=mountain.z_mountain_hist[k] / 1000,
                        fill="tozeroy",
                        fillcolor="saddlebrown",
                        mode="lines",
                        name="Mountain",
                        line=dict(color="saddlebrown"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=mountain.z_mountain_ref_hist[k] / 1000,
                        mode="lines",
                        name="Uneroded Reference",
                        line=dict(color="black", dash="dash"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=[mountain.ELA_hist[k] / 1000] * len(mountain.x),
                        mode="lines",
                        name="Equilibrium Line Altitude",
                        line=dict(color="green", dash="dash"),
                    ),
                ],
                name=str(k),
            )
        )
        # process rates
        frames_rate.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(mountain.dpdt_hist[k]) * SECONDS_PER_YEAR,
                        mode="lines",
                        name="Precipitation",
                        line=dict(color="green"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(mountain.dbdt_hist[k]) * SECONDS_PER_YEAR,
                        mode="lines",
                        name="Mass Balance Rate",
                        line=dict(color="blue"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(np.abs(mountain.u_b_hist[k])) * SECONDS_PER_YEAR,
                        mode="lines",
                        name="Sliding Velocity",
                        line=dict(color="purple"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(mountain.dedt_hist[k]) * SECONDS_PER_YEAR,
                        mode="lines",
                        name="Erosion Rate",
                        line=dict(color="brown"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(mountain.dmdt_hist[k]) * SECONDS_PER_YEAR,
                        mode="lines",
                        name="Melt Rate",
                        line=dict(color="cyan"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(np.abs(mountain.dqdx_diffusion_hist[k]))
                        * SECONDS_PER_YEAR,
                        mode="lines",
                        name="Vertical Ice Diffusion",
                        line=dict(color="magenta"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(np.abs(mountain.dqdx_advection_hist[k]))
                        * SECONDS_PER_YEAR,
                        mode="lines",
                        name="Horizontal Ice Advection",
                        line=dict(color="darkorange"),
                    ),
                ],
                name=str(k),
            )
        )
        # basal heat fluxes
        frames_flux.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x_km,
                        y=mountain.Q_geo_hist[k],
                        mode="lines",
                        name="Geothermal Heat Flux",
                        line=dict(color="orange"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(mountain.Q_fric_hist[k]),
                        mode="lines",
                        name="Friction Heat Flux",
                        line=dict(color="red"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(mountain.Q_cond_hist[k]),
                        mode="lines",
                        name="Conductive Heat Flux",
                        line=dict(color="blue"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(mountain.Q_def_hist[k]),
                        mode="lines",
                        name="Deformational Heat Flux",
                        line=dict(color="purple"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(mountain.Q_lat_hist[k]),
                        mode="lines",
                        name="Latent Heat Flux",
                        line=dict(color="cyan"),
                    ),
                ],
                name=str(k),
            )
        )
        # temperatures
        frames_temp.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=x_km,
                        y=mountain.T_s_hist[k],
                        mode="lines",
                        name="Surface Temperature",
                        line=dict(color="red"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=mountain.T_m_hist[k],
                        mode="lines",
                        name="Pressure Melting Temperature",
                        line=dict(color="orange"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=viz_smooth(mountain.T_b_hist[k]),
                        mode="lines",
                        name="Basal Temperature",
                        line=dict(color="blue"),
                    ),
                    go.Scatter(
                        x=x_km,
                        y=[T_MELT] * len(x_km),
                        mode="lines",
                        name="Melt Temperature",
                        line=dict(color="black", dash="dash"),
                    ),
                ],
                name=str(k),
            )
        )

    # mountain and ice profile figure
    mtn_fig = go.Figure(
        data=[
            go.Scatter(
                x=x_km,
                y=mountain.z_ice_hist[0] / 1000,
                fill="tozeroy",
                fillcolor="lightblue",
                mode="lines",
                name="Ice",
                line=dict(color="lightblue"),
            ),
            go.Scatter(
                x=x_km,
                y=mountain.z_mountain_hist[0] / 1000,
                fill="tozeroy",
                fillcolor="saddlebrown",
                mode="lines",
                name="Mountain",
                line=dict(color="saddlebrown"),
            ),
            go.Scatter(
                x=x_km,
                y=mountain.z_mountain_ref_hist[0] / 1000,
                mode="lines",
                name="Uneroded Reference",
                line=dict(color="black", dash="dash"),
            ),
            go.Scatter(
                x=x_km,
                y=[mountain.ELA_hist[0] / 1000] * len(x_km),
                mode="lines",
                name="Equilibrium Line Altitude",
                line=dict(color="green", dash="dash"),
            ),
        ],
        frames=frames_mtn,
    )
    mtn_fig.update_layout(
        title="Mountain + Ice Time Evolution",
        xaxis_title="Distance [km]",
        yaxis_title="Elevation [km]",
        sliders=[
            {
                "steps": [
                    dict(
                        args=[
                            [str(k)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        label=f"{mountain.time_array[k]:.0f}",
                        method="animate",
                    )
                    for k in range(mountain.nt)
                ],
                "currentvalue": {"prefix": "Year: "},
            }
        ],
        height=350,
        template="plotly_white",
    )
    mtn_fig.update_layout(
        font=dict(family=LATEX_FONT, size=14),
        title_font=dict(family=LATEX_FONT, size=16),
    )
    mtn_fig.update_xaxes(
        title_font=dict(family=LATEX_FONT), tickfont=dict(family=LATEX_FONT)
    )
    mtn_fig.update_yaxes(
        title_font=dict(family=LATEX_FONT), tickfont=dict(family=LATEX_FONT)
    )
    mtn_fig.update_layout(legend=dict(font=dict(family=LATEX_FONT, size=12)))
    mtn_fig.show()

    # process rates figure
    rate_fig = go.Figure(
        data=[
            go.Scatter(
                x=x_km,
                y=viz_smooth(mountain.dpdt_hist[0]) * SECONDS_PER_YEAR,
                mode="lines",
                name="Precipitation",
                line=dict(color="green"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(mountain.dbdt_hist[0]) * SECONDS_PER_YEAR,
                mode="lines",
                name="Mass Balance Rate",
                line=dict(color="blue"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(np.abs(mountain.u_b_hist[0])) * SECONDS_PER_YEAR,
                mode="lines",
                name="Sliding Velocity",
                line=dict(color="purple"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(mountain.dedt_hist[0]) * SECONDS_PER_YEAR,
                mode="lines",
                name="Erosion Rate",
                line=dict(color="brown"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(mountain.dmdt_hist[0]) * SECONDS_PER_YEAR,
                mode="lines",
                name="Melt Rate",
                line=dict(color="cyan"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(np.abs(mountain.dqdx_diffusion_hist[0]))
                * SECONDS_PER_YEAR,
                mode="lines",
                name="Vertical Ice Diffusion",
                line=dict(color="magenta"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(np.abs(mountain.dqdx_advection_hist[0]))
                * SECONDS_PER_YEAR,
                mode="lines",
                name="Horizontal Ice Advection",
                line=dict(color="darkorange"),
            ),
        ],
        frames=frames_rate,
    )
    rate_fig.update_layout(
        title="Process Rate Time Evolution",
        xaxis_title="Distance [km]",
        yaxis_title="Value [m/yr]",
        sliders=[
            {
                "steps": [
                    dict(
                        args=[
                            [str(k)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        label=f"{mountain.time_array[k]:.0f}",
                        method="animate",
                    )
                    for k in range(mountain.nt)
                ],
                "currentvalue": {"prefix": "Year: "},
            }
        ],
        height=350,
        template="plotly_white",
    )
    rate_fig.update_layout(
        font=dict(family=LATEX_FONT, size=14),
        title_font=dict(family=LATEX_FONT, size=16),
    )
    rate_fig.update_xaxes(
        title_font=dict(family=LATEX_FONT), tickfont=dict(family=LATEX_FONT)
    )
    rate_fig.update_yaxes(
        title_font=dict(family=LATEX_FONT), tickfont=dict(family=LATEX_FONT)
    )
    rate_fig.update_layout(legend=dict(font=dict(family=LATEX_FONT, size=12)))
    rate_fig.show()

    # basal heat fluxes figure
    flux_fig = go.Figure(
        data=[
            go.Scatter(
                x=x_km,
                y=mountain.Q_geo_hist[0],
                mode="lines",
                name="Geothermal Heat Flux",
                line=dict(color="orange"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(mountain.Q_fric_hist[0]),
                mode="lines",
                name="Friction Heat Flux",
                line=dict(color="red"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(mountain.Q_cond_hist[0]),
                mode="lines",
                name="Conductive Heat Flux",
                line=dict(color="blue"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(mountain.Q_def_hist[0]),
                mode="lines",
                name="Deformational Heat Flux",
                line=dict(color="purple"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(mountain.Q_lat_hist[0]),
                mode="lines",
                name="Latent Heat Flux",
                line=dict(color="cyan"),
            ),
        ],
        frames=frames_flux,
    )
    flux_fig.update_layout(
        title="Heat Fluxes Time Evolution",
        xaxis_title="Distance [km]",
        yaxis_title="Heat Flux [W/m²]",
        sliders=[
            {
                "steps": [
                    dict(
                        args=[
                            [str(k)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        label=f"{mountain.time_array[k]:.0f}",
                        method="animate",
                    )
                    for k in range(mountain.nt)
                ],
                "currentvalue": {"prefix": "Year: "},
            }
        ],
        height=350,
        template="plotly_white",
    )
    flux_fig.update_layout(
        font=dict(family=LATEX_FONT, size=14),
        title_font=dict(family=LATEX_FONT, size=16),
    )
    flux_fig.update_xaxes(
        title_font=dict(family=LATEX_FONT), tickfont=dict(family=LATEX_FONT)
    )
    flux_fig.update_yaxes(
        title_font=dict(family=LATEX_FONT), tickfont=dict(family=LATEX_FONT)
    )
    flux_fig.update_layout(legend=dict(font=dict(family=LATEX_FONT, size=12)))
    flux_fig.show()

    # temperatures figure
    temp_fig = go.Figure(
        data=[
            go.Scatter(
                x=x_km,
                y=mountain.T_s_hist[0],
                mode="lines",
                name="Surface Temperature",
                line=dict(color="red"),
            ),
            go.Scatter(
                x=x_km,
                y=mountain.T_m_hist[0],
                mode="lines",
                name="Pressure Melting Temperature",
                line=dict(color="orange"),
            ),
            go.Scatter(
                x=x_km,
                y=viz_smooth(mountain.T_b_hist[0]),
                mode="lines",
                name="Basal Temperature",
                line=dict(color="blue"),
            ),
            go.Scatter(
                x=x_km,
                y=[T_MELT] * len(x_km),
                mode="lines",
                name="Melt Temperature",
                line=dict(color="black", dash="dash"),
            ),
        ],
        frames=frames_temp,
    )
    temp_fig.update_layout(
        title="Temperature Time Evolution",
        xaxis_title="Distance [km]",
        yaxis_title="Temperature [K]",
        sliders=[
            {
                "steps": [
                    dict(
                        args=[
                            [str(k)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        label=f"{mountain.time_array[k]:.0f}",
                        method="animate",
                    )
                    for k in range(mountain.nt)
                ],
                "currentvalue": {"prefix": "Year: "},
            }
        ],
        height=350,
        template="plotly_white",
    )
    temp_fig.update_layout(
        font=dict(family=LATEX_FONT, size=14),
        title_font=dict(family=LATEX_FONT, size=16),
    )
    temp_fig.update_xaxes(
        title_font=dict(family=LATEX_FONT), tickfont=dict(family=LATEX_FONT)
    )
    temp_fig.update_yaxes(
        title_font=dict(family=LATEX_FONT), tickfont=dict(family=LATEX_FONT)
    )
    temp_fig.update_layout(legend=dict(font=dict(family=LATEX_FONT, size=12)))
    temp_fig.show()

    # save figures as html
    mtn_fig.write_html(f"mountain_{fname}.html", include_plotlyjs="cdn")
    rate_fig.write_html(f"rate_{fname}.html", include_plotlyjs="cdn")
    flux_fig.write_html(f"flux_{fname}.html", include_plotlyjs="cdn")
    temp_fig.write_html(f"temp_{fname}.html", include_plotlyjs="cdn")
