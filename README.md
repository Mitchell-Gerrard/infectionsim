

# InfectionSim — a chill particle SIR toy

This is a small, student-friendly particle simulation that recreates SIR dynamics by simulating individuals moving around in 2D and passing infection on contact. It's a quick and dirty model useful for learning, demos, or messing around with parameter sweeps.

What it does
- Simulates N agents wandering on a 2D torus (wrap-around area).
- Finds nearby contacts with a KD-tree and transmits infection probabilistically.
- Marks recovered agents after a configurable recovery time.
- Aggregates S/I/R counts and fits a simple SIR ODE to estimate `beta` and `gamma`.

Outputs you should see after running
- `infection_simulation.gif` — animation of agents (blue = susceptible, red = infected, green = recovered) with S/I/R plot on the side.
- `sir_fit_results.png` — single-run fit diagnostics (data vs fitted SIR).
- `multi_run_sir_fit.png` — summary fit across multiple runs.

Quick start

1. Make a virtualenv and install the libs:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the demo script:

```bash
python infectionsim/infectionsim.py
```

That will run a single simulation, save an animation as `infection_simulation.gif`, and generate fit plots.

Play with it
- Edit `infectionsim/infectionsim.py` near the bottom to change `population_size`, `area_size_x`, `initial_infected`, `total_time`, `units`, `dt`, and `records_interval`.
- Lower `population_size` or raise `records_interval` if your machine chokes.
- The `infection_func(distance)` at the top decides infection probability vs distance — tweak it to change transmission behavior.

Developer notes
- The simulation records positions and status at regular intervals and then:
	- animates agent positions and S/I/R curves
	- fits an SIR ODE using least-squares to estimate `beta` and `gamma`
- Uses `scipy.spatial.KDTree` for efficient neighbor queries and `scipy.integrate.odeint` for SIR integration.

Requirements
- See `requirements.txt` for the minimal set of Python packages used by the script.

License / attribution

Feel free to use or adapt this for coursework or demos. If you end up publishing anything based on this, a short acknowledgement is appreciated.

