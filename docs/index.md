# Coupling of Tectonic Uplift and Climate via Glacial Basal Energy Balance

<!-- Sidebar Table of Contents -->
<nav id="toc"></nav>

<style>
	/* Sticky sidebar TOC */
	#toc {
		position: fixed;
		top: 100px;
		right: 20px;
		width: 260px;
		max-height: 70vh;
		overflow: auto;
		padding: 12px 14px;
		border: 1px solid #ddd;
		background: #ffffff;
		box-shadow: 0 2px 6px rgba(0,0,0,0.06);
		font-family: Courier New, monospace;
		font-size: 14px;
		z-index: 999;
	}
	#toc h3 { margin: 0 0 8px; font-size: 15px; }
	#toc ul { list-style: none; padding-left: 0; margin: 0; }
	#toc li { margin: 6px 0; }
	#toc a { text-decoration: none; color: #0366d6; }
	#toc a:hover { text-decoration: underline; }
	/* Indentation for nested headings */
	#toc ul ul { padding-left: 14px; }
	/* Keep main content from hiding under TOC on small screens */
	@media (max-width: 1000px) {
		#toc { position: static; width: auto; max-height: none; margin: 10px 0; }
	}
</style>

Below are several experiments run using my `Mountain` class. Computational cost limited me to about 10k years of simulation time, so I increased uplift, erosion, and precipitation rates in order to observe large-scale mountain-climate interactions.

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

Orographic precipitation is a function of multiple climatic variables, in addition to mountain geography (Roe & Baker, 2005). This interactive figure demonstrates how precipiration rate changes based on mountain shape.

<iframe src="mountain_precipitation_widget.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 1 - Galcial Buzzsaw

This experiment demonstrates the glacial buzzsaw effect with moderate uplift and high precipitation.

### Parameterization

#### Tectonic & Climatic

| Variable Name | Symbol | Meaning | Value | Units |
|---|---|---|---|---|
| CLIMATE_TEMP | $$T_0$$ | Surface temperature | $$286.0$$ | $$\text{K}$$ |
| CLIMATE_Q_0 | $$q_0$$ | Specific humidity | $$4\times 10^{-3}$$ | $$\text{kg}$$ water / $$\text{kg}$$ air |
| PRECIPITATION_SCALE | $$\alpha_p$$ | Precipitation scaling factor | $$1\times 10^{-1}$$ | - |
| Q_GEO_0 | $$Q_{geo}$$ | Initial geothermal heat flux | $$0.05$$ | $$\text{W}\,\text{m}^{-2}$$ |
| UPLIFT_RATE | $$\dot{U}$$ | Uniform uplift rate | $$1\times 10^{-2}$$ | $$\text{m}\,\text{a}^{-1}$$ |
| ANGLE_OF_REPOSE | $$\theta_r$$ | Maximum mountain slope | $$35^{\circ}$$ | $$\text{deg}$$ |
| K_REPOSE | $$K_r$$ | Slope diffusion | $$5\times 10^{-3}$$ | $$\text{m}^2\,\text{s}^{-1}$$ |

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

Because uplift is moderate and precipiation is high, ice quickly builds up and erodes the mountain. When the critical angle of repose of the mountain is reached, it fails to grow. Interstingly, the peak begins to shift to the right, since erosion is acting more strongly on the windward side. This shows how a directional orographic forcing can cause significant changes to a landscape.

<iframe src="experiment_1/flux_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

Frictional heat flux is observed to be dominant.

<iframe src="experiment_1/temp_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

Basal temperature begins at the surface temperature, but due to net positive heat flux, soon reaches melting temperature across much of the mountain profile.

<iframe src="experiment_1/rate_glacial_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 2 - Glacial Shielding

In this experiment, uplift is faster and the background climate is colder and drier. This results in a thinner layer of ice on the mountain, leading to conductive heat flux being dominant and a shielding effect taking place. Additionally, the mountian geometry has a greater initial height compared to Experiment 1.

### Parameterization

#### Tectonic & Climatic

| Variable Name | Symbol | Meaning | Value | Units |
|---|---|---|---|---|
| CLIMATE_TEMP | $$T_0$$ | Surface temperature | $$282.0$$ | $$\text{K}$$ |
| CLIMATE_Q_0 | $$q_0$$ | Specific humidity | $$4\times 10^{-3}$$ | $$\text{kg}$$ water / $$\text{kg}$$ air |
| PRECIPITATION_SCALE | $$\alpha_p$$ | Precipitation scaling factor | $$1\times 10^{-3}$$ | - |
| Q_GEO_0 | $$Q_{geo}$$ | Initial geothermal heat flux | $$0.05$$ | $$\text{W}\,\text{m}^{-2}$$ |
| UPLIFT_RATE | $$\dot{U}$$ | Uniform uplift rate | $$1\times 10^{-1}$$ | $$\text{m}\,\text{a}^{-1}$$ |
| ANGLE_OF_REPOSE | $$\theta_r$$ | Maximum mountain slope | $$35^{\circ}$$ | $$\text{deg}$$ |
| K_REPOSE | $$K_r$$ | Slope diffusion | $$5\times 10^{-3}$$ | $$\text{m}^2\,\text{s}^{-1}$$ |

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

Ice height is noticeably lower in this experiment. Only towards the end of the simulation is erosion noticeable around the ELA. The peak and most of the higher slopes do not deviate from the uneroded reference.

<iframe src="experiment_2/flux_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

Conductive heat flux away from the ice base is observed to be dominant. Conductive heat flux is inversely proportional to ice thickness so this is to be expected in a cold, dry climate.

<iframe src="experiment_2/temp_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

Basal temperature remains below the pressure melting temperature on much of the mountain, so the ice is frozen to the bed.

<iframe src="experiment_2/rate_glacial_shielding.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 3 - Reduced Shielding

This is a more moderate glacial shielding scenario. The background climate is slightly warmer and wetter, which will limit conductive heat flux and encourage basal melt.

### Parameterization

#### Tectonic & Climatic

| Variable Name | Symbol | Meaning | Value | Units |
|---|---|---|---|---|
| CLIMATE_TEMP | $$T_0$$ | Surface temperature | $$284.0$$ | $$\text{K}$$ |
| CLIMATE_Q_0 | $$q_0$$ | Specific humidity | $$5\times 10^{-2}$$ | $$\text{kg}$$ water / $$\text{kg}$$ air |
| PRECIPITATION_SCALE | $$\alpha_p$$ | Precipitation scaling factor | $$1\times 10^{-3}$$ | - |
| Q_GEO_0 | $$Q_{geo}$$ | Initial geothermal heat flux | $$0.05$$ | $$\text{W}\,\text{m}^{-2}$$ |
| UPLIFT_RATE | $$\dot{U}$$ | Uniform uplift rate | $$1\times 10^{-1}$$ | $$\text{m}\,\text{a}^{-1}$$ |
| ANGLE_OF_REPOSE | $$\theta_r$$ | Maximum mountain slope | $$35^{\circ}$$ | $$\text{deg}$$ |
| K_REPOSE | $$K_r$$ | Slope diffusion | $$5\times 10^{-3}$$ | $$\text{m}^2\,\text{s}^{-1}$$ |

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

Ice height is slightly thicker compared to Experiment 2. Erosion acts quickly, however around the summit the shielding effect still dominates. The maximum angle of repose can be clearly observed on the windward slope at the end of the simulation.

<iframe src="experiment_3/flux_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>

Frictional and deformational heat flux dominate downslope, while conductive heat flux dominates around the summit.

<iframe src="experiment_3/temp_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>

Basal temperature remains below the pressure melting temperature around the summit, but reaches melting temperature much earlier downslope compared to Experiment 2.

<iframe src="experiment_3/rate_glacial_unshielding.html" width="100%" height="600" frameborder="0"></iframe>

## Experiment 4 - Enhanced Buzzsaw & A Different Geometry

Here the initial geometry is reversed compared to Experiment 1, so the windward side is longer than the leeward slope. Additionally, there is a wet climate, and a lower angle of repose and faster slope diffusion response. This experiment demonstrates an enhanced buzzsaw effect.

### Parameterization

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

<iframe src="experiment_4/mountain_reverse_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

There seems to be an unrealistic amount of ice reaching the base. Partially this is because the orographic forcing does not take into account change in mountain height. Also since the slope by the peak is high and the ice is temperate, any ice that accretes there will quickly advect downslope. Interstingly, the peak in this experiment shifts to the left, since erosion is more powerful on the leeward slope.

<iframe src="experiment_4/flux_reverse_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

Frictional heat flux is observed to dominate.

<iframe src="experiment_4/temp_reverse_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

Basal temperature begins at the surface temperature, but due to net positive heat flux, soon reaches melting temperature across much of the mountain profile.

<iframe src="experiment_4/rate_reverse_buzzsaw.html" width="100%" height="600" frameborder="0"></iframe>

<script>
// Build sidebar TOC from h2/h3 headings
(function() {
	function slugify(text) {
		return text.toLowerCase()
			.replace(/[^a-z0-9\s-]/g, '')
			.trim()
			.replace(/\s+/g, '-')
			.replace(/-+/g, '-');
	}

	const toc = document.getElementById('toc');
	if (!toc) return;

	const headings = Array.from(document.querySelectorAll('h2, h3'));
	if (headings.length === 0) return;

	const container = document.createElement('div');
	const title = document.createElement('h3');
	title.textContent = 'Contents';
	container.appendChild(title);

	const ul = document.createElement('ul');
	let currentUl = ul;
	let lastLevel = 2;

	headings.forEach(h => {
		const level = parseInt(h.tagName.substring(1), 10);
		if (!h.id || h.id.length === 0) {
			h.id = slugify(h.textContent || h.innerText || 'section');
		}
		// Nesting logic: create sublist for h3 under last h2
		if (level > lastLevel) {
			const sub = document.createElement('ul');
			currentUl.lastElementChild && currentUl.lastElementChild.appendChild(sub);
			currentUl = sub;
		} else if (level < lastLevel) {
			// move back to parent list
			currentUl = ul;
		}
		lastLevel = level;

		const li = document.createElement('li');
		const a = document.createElement('a');
		a.href = '#' + h.id;
		a.textContent = h.textContent || h.innerText || '';
		li.appendChild(a);
		currentUl.appendChild(li);
	});

	container.appendChild(ul);
	toc.appendChild(container);
})();
</script>
