# Coupling of Tectonic Uplift and Climate via Glacial Basal Energy Balance

Below are several experiments run using my `Mountain` class. Computational cost limited me to about 10k years of simulation time, so I increased uplift, erosion, and precipitation rates in order to observe large-scale mountain-climate interactions.

# Coupling of Tectonic Uplift and Climate via Glacial Basal Energy Balance

<!-- Enable MathJax for $...$ and $$...$$ rendering on GitHub Pages -->
<script>
window.MathJax = {
	tex: {
		inlineMath: [['$', '$'], ['\\(', '\\)']],
		displayMath: [['$$','$$'], ['\\[','\\]']]
	},
	options: {
		skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
	}
};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" id="MathJax-script" async></script>

## Orographic Forcing

Orographic precipitation is a function of multiple climatic variables, in addition to mountain geography. This interactive figure demonstrates how precipiration rate changes based on mountain shape.

<iframe src="mountain_precipitation_widget.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 1 - Galcial Buzzsaw

### Parameters

#### Tectonic & Climatic

| Variable Name | Symbol | Meaning | Value | Units |
|---|---|---|---|---|
| CLIMATE_TEMP | $$T_0$$ | Surface temperature | $$286.0$$ | $$\text{K}$$ |
| CLIMATE_Q_0 | $$q_0$$ | Specific humidity | $$4\times 10^{-3}$$ | $$\text{kg}$$ water / $$\text{kg}$$ air |
| PRECIPITATION_SCALE | $$\alpha_p$$ | Precipitation scaling factor | $$1\times 10^{-1}$$ | - |
| Q_GEO_0 | $$Q_{geo}$$ | Initial geothermal heat flux | $$0.05$$ | $$\text{W}\,\text{m}^{-2}$$ |
| UPLIFT_RATE | $$\dot{U}$$ | Uniform uplift rate | $$1\times 10^{-2}$$ | $$\text{m}\,\text{a}^{-1}$$ |

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

This experiment demonstrates the glacial buzzsaw effect. Because uplift is moderate and precipiation is high, ice quickly builds up and erodes the mountain. When the critical angle of repose of the mountain is reached, it fails to grow. Interstingly, the peak begins to shift to the right, since erosion is acting more strongly on the windward side. This shows how a directional orographic forcing can cause significant changes to a landscape.

<iframe src="experiment_1/mountain_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_1/flux_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_1/temp_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_1/rate_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 2 - Glacial Shielding

### Parameters

#### Tectonic & Climatic

| Variable Name | Symbol | Meaning | Value | Units |
|---|---|---|---|---|
| CLIMATE_TEMP | $$T_0$$ | Surface temperature | $$282.0$$ | $$\text{K}$$ |
| CLIMATE_Q_0 | $$q_0$$ | Specific humidity | $$4\times 10^{-3}$$ | $$\text{kg}$$ water / $$\text{kg}$$ air |
| PRECIPITATION_SCALE | $$\alpha_p$$ | Precipitation scaling factor | $$1\times 10^{-3}$$ | - |
| Q_GEO_0 | $$Q_{geo}$$ | Initial geothermal heat flux | $$0.05$$ | $$\text{W}\,\text{m}^{-2}$$ |
| UPLIFT_RATE | $$\dot{U}$$ | Uniform uplift rate | $$1\times 10^{-1}$$ | $$\text{m}\,\text{a}^{-1}$$ |

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

In this experiment, uplift is faster and the background climate is colder and drier. This results in a thinner layer of ice on the mountain, leading to conductive heat flux being dominant. The temperature plot shows that the basal temperature remains below the pressure melting temperature on much of the mountain. In these cells, the ice is stuck fast to the bedrock and no erosion takes place. Thus the peak does not deviat from the uneroded reference.

<iframe src="experiment_2/mountain_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_2/flux_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_2/temp_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_2/rate_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 3 - Reduced Shielding

### Parameters

#### Tectonic & Climatic

| Variable Name | Symbol | Meaning | Value | Units |
|---|---|---|---|---|
| CLIMATE_TEMP | $$T_0$$ | Surface temperature | $$284.0$$ | $$\text{K}$$ |
| CLIMATE_Q_0 | $$q_0$$ | Specific humidity | $$5\times 10^{-2}$$ | $$\text{kg}$$ water / $$\text{kg}$$ air |
| PRECIPITATION_SCALE | $$\alpha_p$$ | Precipitation scaling factor | $$1\times 10^{-3}$$ | - |
| Q_GEO_0 | $$Q_{geo}$$ | Initial geothermal heat flux | $$0.05$$ | $$\text{W}\,\text{m}^{-2}$$ |
| UPLIFT_RATE | $$\dot{U}$$ | Uniform uplift rate | $$1\times 10^{-1}$$ | $$\text{m}\,\text{a}^{-1}$$ |

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

This is a more moderate glacial shielding scenario. The background climate is slightly warmer and wetter, which will limit conductive heat flux and encourage basal melt.

<iframe src="experiment_3/mountain_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_3/flux_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_3/temp_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_3/rate_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 4 - Enhanced Buzzsaw & A Different Geometry

### Parameters

#### Tectonic & Climatic

| Variable Name | Symbol | Meaning | Value | Units |
|---|---|---|---|---|
| CLIMATE_TEMP | $$T_0$$ | Surface temperature | $$286.0$$ | $$\text{K}$$ |
| CLIMATE_Q_0 | $$q_0$$ | Specific humidity | $$1\times 10^{-2}$$ | $$\text{kg}$$ water / $$\text{kg}$$ air |
| PRECIPITATION_SCALE | $$\alpha_p$$ | Precipitation scaling factor | $$1\times 10^{-1}$$ | - |
| Q_GEO_0 | $$Q_{geo}$$ | Initial geothermal heat flux | $$0.05$$ | $$\text{W}\,\text{m}^{-2}$$ |
| UPLIFT_RATE | $$\dot{U}$$ | Uniform uplift rate | $$1\times 10^{-2}$$ | $$\text{m}\,\text{a}^{-1}$$ |
| ANGLE_OF_REPOSE | $$\theta_r$$ | Maximum mountain slope | $$25^{\circ}$$ | $$\text{deg}$$ |
| K_REPOSE | $$K_r$$ | Slope diffusion | $$5\times 10^{-2}$$ | $$\text{m}^2\,\text{s}^{-1}$$ |


#### Simulation

| Parameter | Value | Description |
|---|---|---|
| `dt` | `10` | Time step |
| `t0` | `0.0` | Start time |
| `t1` | `1e4` | End time |
| `n_x` | `301` | Grid points |
| `height` | `4000.0` | Maximum elevation (m) |
| `length1` | `40000.0` | Windward length (m) |
| `length2` | `20000.0` | Leeward length (m) |

### Result

Here the initial geometry is reversed so the windward side is longer than the leeward slope. Additionally, there is a wet climate, and a lower angle of repose and faster slope diffusion response. This experiment demonstrates an enhanced buzzsaw. There seems to be an unrealistic amount of ice reaching the base. Partially this is because the orographic forcing does not take into account change in mountain height. Also since the slope by the peak is high and the ice is temperate, any ice that accretes there will quickly advect downslope. Interstingly, the peak in this experiment shifts to the left, since erosion is more powerful on the leeward slope.

<iframe src="experiment_1/mountain_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_1/flux_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_1/temp_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

<iframe src="experiment_1/rate_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>
