# Coupling of Tectonic Uplift and Climate via Glacial Basal Energy Balance

## Orographic Forcing

Orographic precipitation is a function of multiple climatic variables, in addition to mountain geography. This interactive figure demonstrates how precipiration rate changes based on mountain shape.

<iframe src="mountain_precipitation_widget.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 1 - Galcial Buzzsaw

### Parameters

#### Tectonic & Climatic

| Variable Name | Symbol | Meaning | Value | Units |
|---|---|---|---|---|
| CLIMATE_TEMP | $T_0$ | Surface temperature | $286.0$ | $\text{K}$ |
| CLIMATE_Q_0 | $q_0$ | Specific humidity | $4\times 10^{-3}$ | $\text{kg}$ water / $\text{kg}$ air |
| PRECIPITATION_SCALE | $\alpha_p$ | Precipitation scaling factor | $1\times 10^{-1}$ | - |
| Q_GEO_0 | $Q_{geo}$ | Initial geothermal heat flux | $0.05$ | $\text{W}\,\text{m}^{-2}$ |
| UPLIFT_RATE | $\dot{U}$ | Uniform uplift rate | $1\times 10^{-2}$ | $\text{m}\,\text{a}^{-1}$ |

#### Simulation

| Parameter | Value | Description |
|---|---|---|
| `dt` | `10` | Time step |
| `t0` | `0.0` | Start time |
| `t1` | `1e4` | End time |
| `n_x` | `501` | Grid points |
| `height` | `4000.0` | Maximum elevation (m) |
| `length1` | `20000.0` | Windward length (m) |
| `length2` | `40000.0` | Leeward length (m) |

### Result

<iframe src="experiment_1/mountain_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_1/flux_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_1/temp_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_1/rate_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 2 - Glacial Shielding

### Parameters

#### Tectonic & Climatic

| Variable Name | Symbol | Meaning | Value | Units |
|---|---|---|---|---|
| CLIMATE_TEMP | $T_0$ | Surface temperature | $282.0$ | $\text{K}$ |
| CLIMATE_Q_0 | $q_0$ | Specific humidity | $4\times 10^{-3}$ | $\text{kg}$ water / $\text{kg}$ air |
| PRECIPITATION_SCALE | $\alpha_p$ | Precipitation scaling factor | $1\times 10^{-3}$ | - |
| Q_GEO_0 | $Q_{geo}$ | Initial geothermal heat flux | $0.05$ | $\text{W}\,\text{m}^{-2}$ |
| UPLIFT_RATE | $\dot{U}$ | Uniform uplift rate | $1\times 10^{-1}$ | $\text{m}\,\text{a}^{-1}$ |

#### Simulation

| Parameter | Value | Description |
|---|---|---|
| `dt` | `10` | Time step |
| `t0` | `0.0` | Start time |
| `t1` | `1e4` | End time |
| `n_x` | `501` | Grid points |
| `height` | `6000.0` | Maximum elevation (m) |
| `length1` | `20000.0` | Windward length (m) |
| `length2` | `40000.0` | Leeward length (m) |

### Result

<iframe src="experiment_2/mountain_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_2/flux_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_2/temp_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_2/rate_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 3 - Reduced Shielding

### Parameters

#### Tectonic & Climatic

| Variable Name | Symbol | Meaning | Value | Units |
|---|---|---|---|---|
| CLIMATE_TEMP | $T_0$ | Surface temperature | $284.0$ | $\text{K}$ |
| CLIMATE_Q_0 | $q_0$ | Specific humidity | $5\times 10^{-2}$ | $\text{kg}$ water / $\text{kg}$ air |
| PRECIPITATION_SCALE | $\alpha_p$ | Precipitation scaling factor | $1\times 10^{-3}$ | - |
| Q_GEO_0 | $Q_{geo}$ | Initial geothermal heat flux | $0.05$ | $\text{W}\,\text{m}^{-2}$ |
| UPLIFT_RATE | $\dot{U}$ | Uniform uplift rate | $1\times 10^{-1}$ | $\text{m}\,\text{a}^{-1}$ |

#### Simulation

| Parameter | Value | Description |
|---|---|---|
| `dt` | `10` | Time step |
| `t0` | `0.0` | Start time |
| `t1` | `1e4` | End time |
| `n_x` | `501` | Grid points |
| `height` | `6000.0` | Maximum elevation (m) |
| `length1` | `20000.0` | Windward length (m) |
| `length2` | `40000.0` | Leeward length (m) |

### Result

<iframe src="experiment_3/mountain_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_3/flux_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_3/temp_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_3/rate_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>
