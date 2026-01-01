import numpy as np
import matplotlib.pyplot as plt
import cmasher as cm
from matplotlib.animation import FuncAnimation
import random
import time as ti
from scipy.spatial import KDTree
from tqdm import trange
from scipy.integrate import odeint
from scipy.optimize import least_squares
import concurrent.futures
def subsample_frames(*arrays, step):
    """Subsample all arrays by the same step along the first axis, always including the last frame."""
    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        idxs = list(range(0, len(arr), step))
        if idxs[-1] != len(arr) - 1:
            idxs.append(len(arr) - 1)
        result.append(arr[idxs])
    return result
class InfectionSim:
    def __init__(self,infection_func, recovery_time):
        self.infection_func = infection_func #function that determines infection spread
        self.recovery_time = recovery_time #time for an infected individual to recover in days
    def run_simulation(self, population_size, area_size_x, area_size_y, initial_infected, total_time,units,dt = 30,records_interval=10):
        # Initialize population


        if units == 'seconds':
             time_step = 1
        elif units == 'minutes':
             time_step = 60
        elif units == 'hours':
            time_step = (60*60)
        elif units == 'days':
            time_step = (24*60*60)
        else:
            raise ValueError("Invalid time unit")

        total_dt_seconds = total_time * time_step / dt
        day_conversion = time_step / (records_interval * dt)
        speed = 1.4  # units per 10th of a second
        speed_varriation = 0.6 # varriation in speed per 10th of a second
        steps = int(total_dt_seconds)
        positions = np.random.rand(population_size, 2) * [area_size_x, area_size_y]
        status = np.zeros(population_size)  # 0: susceptible, 1
        infected_indices = random.sample(range(population_size), initial_infected)
        status[infected_indices] = 1

        #recovered_time_steps = np.random.normal(self.recovery_time * (24*60*60) * (1/dt), 0.1 * self.recovery_time * (24*60*60) * (1/dt), population_size).astype(int)
        define_revcovered = 1e10
        all_positions = np.zeros((int(steps//day_conversion), population_size, 2))
        all_status = np.zeros((int(steps//day_conversion), population_size))
        contact_interval = 15
        for step in range(steps):
            
            speed_distribution = np.abs(np.random.normal(speed, speed_varriation, population_size)) * dt
            angles = np.random.uniform(0, 2 * np.pi, population_size)
            displacements = np.column_stack((speed_distribution * np.cos(angles),
                                             speed_distribution * np.sin(angles)))
            positions += displacements
            positions = np.mod(positions, [area_size_x, area_size_y])
            if step % contact_interval == 0:
                tree = KDTree(positions)
                result=tree.query_pairs(r=2.0)
                for i, j in result:
                    if status[i] >= 1 and status[j] >= 1:
                        continue
                    if (status[i] >= 1 and status[i] != define_revcovered) or (status[j] >= 1 and status[j] != define_revcovered):
                        distance = np.linalg.norm(positions[i] - positions[j])
                        infection_probability = self.infection_func(distance)
                        if random.random() < infection_probability:
                            if status[i] == 0:
                                status[i] = 1
                            if status[j] == 0:
                                status[j] = 1
            p_recover = 1 - np.exp(-(1/self.recovery_time)* (dt / time_step))
            infected = (status > 0) & (status < define_revcovered)
            recover_now = infected & (np.random.rand(population_size) < p_recover)
            status[recover_now] = define_revcovered


            status[(status > 0) & (status != define_revcovered)] += 1

            if step % day_conversion == 0:
                all_positions[int(step//day_conversion)] = positions
                all_status[int(step//day_conversion)] = status
        S=np.sum(all_status == 0, axis=1)
        I=np.sum((all_status > 0) & (all_status != define_revcovered), axis=1)
        R=np.sum(all_status == define_revcovered, axis=1)
        return all_positions, S, I, R, all_status, define_revcovered
    def multi_run_simulation(self, num_runs, population_size, area_size_x, area_size_y, initial_infected, total_time, units, dt=30, records_interval=10):
        population_size = population_size*np.ones(num_runs,dtype=int)
        print("Starting multi-run simulation with", num_runs, "runs...")
        all_positions = []
        Ss = []
        Is = []
        Rs = []
        all_statuss = []
        recovered_time_stepss = []
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
                results = [executor.submit(self.run_simulation, int(population_size[i]), area_size_x, area_size_y, initial_infected, total_time, units, dt, records_interval) for i in range(num_runs)]
                for f in concurrent.futures.as_completed(results):
                    positions, S, I, R, all_status, recovered_time_steps = f.result()
                    all_positions.append(positions)
                    Ss.append(S)
                    Is.append(I)
                    Rs.append(R)
                    recovered_time_stepss.append(recovered_time_steps)
                    all_statuss.append(all_status)
        all_positions = np.asarray(all_positions)
        Ss = np.asarray(Ss)
        Is = np.asarray(Is)
        Rs = np.asarray(Rs)
        all_statuss = np.asarray(all_statuss)
        recovered_time_stepss = np.asarray(recovered_time_stepss)
        betas = []
        gammas = []
        for i in range(num_runs):
            fit_results = sim.fiting_SIR(Ss[i], Is[i], Rs[i], recovered_time_stepss[i], dt=1/records_interval)  # dt in days
            recovered_time_steps = int(self.recovery_time * (24*60*60) * (1/dt))
            plt.plot(Ss[i], color='blue', linestyle='--', alpha=0.3)
            plt.plot(Is[i], color='red', linestyle='--', alpha=0.3)
            plt.plot(Rs[i], color='green', linestyle='--', alpha=0.3)
            betas.append(fit_results['beta_fit'])
            gammas.append(fit_results['gamma_fit'])
        # Plot average S/I/R curves
        gamma=np.median(gammas)
        beta=np.median(betas)
        time=np.arange(len(Ss[0]))/records_interval # time in days
        N0 = population_size[0]
        fitted = self.simulate((beta, gamma), time, Ss[0], Is[0], Rs[0], N0)
        S_fit, I_fit, R_fit = fitted[:, 0], fitted[:, 1], fitted[:, 2]

        plt.plot( I_fit, 'r--', label=f'I (fit) beta={beta:.4f}', linewidth=2)
        plt.plot( S_fit, 'b--', label=f'S (fit) gamma={gamma:.4f}', linewidth=2)
        plt.plot(R_fit, 'g--', label='R (fit)', linewidth=2)
   
        plt.legend()
        plt.xlabel(f'Time ({units})')
        plt.ylabel('Number of Individuals')
        plt.title('S / I / R counts over time for multiple runs')
 
        plt.tight_layout()
        plt.savefig('multi_run_sir_fit.png')
        plt.close()
                
        return all_positions, S, I, R, all_status, recovered_time_steps
    def sir_rhs(self, y, t, beta, gamma, N0):
        S_, I_, R_ = y
        dSdt = -beta * S_ * I_ / N0
        dIdt = beta * S_ * I_ / N0 - gamma * I_
        dRdt = gamma * I_
        return [dSdt, dIdt, dRdt]

    def simulate(self, beta_gamma, time, S, I, R, N0):
        beta, gamma = beta_gamma
        y0 = [S[0], I[0], R[0]]
        sol = odeint(self.sir_rhs, y0, time, args=(beta, gamma, N0))
        return sol
    def plot_fit_results(self, fit_results, units):
        time = fit_results['time']
        S = fit_results['S']
        I = fit_results['I']
        R = fit_results['R']
        S_fit = fit_results['S_fit']
        I_fit = fit_results['I_fit']
        R_fit = fit_results['R_fit']
        N = len(time)
        step = max(1, N // 30)
        plt.figure(figsize=(10, 6))
        plt.plot(time[::step], S[::step], 'b.', label='S (data)', alpha=0.3)
        plt.plot(time[::step], I[::step], 'r.', label='I (data)', alpha=0.3)
        plt.plot(time[::step], R[::step], 'g.', label='R (data)', alpha=0.3)
        plt.plot(time, S_fit, 'b--', label=f'S (fit), gamma = {fit_results["gamma_fit"]:.4f}', linewidth=1)
        plt.plot(time, I_fit, 'r--', label=f'I (fit), beta = {fit_results["beta_fit"]:.4f}', linewidth=1)
        plt.plot(time, R_fit, 'g--', label='R (fit)', linewidth=1)
        plt.xlabel(f'Time ({units})')
        plt.ylabel('Number of Individuals')
        plt.title('S / I / R counts over time with SIR model fit')
        plt.legend()
        plt.tight_layout()
        plt.savefig('sir_fit_results.png')
        plt.close()    
    def animate_simulation(self, all_positions, S, I, R, all_status, recovered_time_steps, area_size_x, area_size_y, total_time, units, records_interval, fpss=10):
        drop_factor = 5
        # Subsample frames to speed up animation
        if drop_factor > 1:
            all_positions, S, I, R, all_status = subsample_frames(
                np.asarray(all_positions), np.asarray(S), np.asarray(I), np.asarray(R), np.asarray(all_status), step=drop_factor
            )
        else:
            all_positions = np.asarray(all_positions)
            S = np.asarray(S)
            I = np.asarray(I)
            R = np.asarray(R)
            all_status = np.asarray(all_status)
        frames = all_positions.shape[0]

        if frames == 0:
            raise ValueError("all_positions has zero frames")

        # precompute counts per frame
        
        susceptible = S
        infected = I
        recovered = R
        x_all = np.linspace(0, total_time, frames)
  
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.set_xlim(0, area_size_x)
        ax1.set_ylim(0, area_size_y)
        ax1.set_title('Infection Simulation')
        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)

        # scatter init
        first_pos = all_positions[0]
        scat = ax1.scatter(first_pos[:, 0], first_pos[:, 1], s=10)

        # S/I/R lines init
        ax2.set_xlim(0, total_time)
        ax2.set_ylim(0, np.max(S + I + R) * 1.1)
        ax2.set_title('S / I / R counts')
        line_s, = ax2.plot([], [], color='blue', label='Susceptible')
        line_i, = ax2.plot([], [], color='red', label='Infected')
        line_r, = ax2.plot([], [], color='green', label='Recovered')
        ax2.legend(loc='upper right')

        colors_map = np.array(['blue', 'red', 'green'])

        def init():
            scat.set_offsets(all_positions[0])
            line_s.set_data([], [])
            line_i.set_data([], [])
            line_r.set_data([], [])
            time_text.set_text('')
            return scat, line_s, line_i, line_r, time_text

        def update(frame):
            pos = all_positions[frame]
            status= all_status[frame]

            # map status -> color index: 0=sus,1=inf,2=rec
            idx = np.zeros_like(status, dtype=int)
            idx[status > 0] = 1
            idx[status >= recovered_time_steps] = 2

            scat.set_offsets(pos)
            scat.set_color(colors_map[idx])

            # update S/I/R lines up to current frame
            x = x_all[: frame + 1]
            line_s.set_data(x, susceptible[: frame + 1])
            line_i.set_data(x, infected[: frame + 1])
            line_r.set_data(x, recovered[: frame + 1])

            time_text.set_text(f'Time: {x[-1]:.2f} {units}')
            return scat, line_s, line_i, line_r, time_text

        ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, repeat=False)
        ani.save('infection_simulation.gif', writer='pillow', fps=fpss)
        plt.close(fig)
        return 'infection_simulation.gif'

    def fiting_SIR(self,S, I, R, recovered_time_steps, dt):
        # Convert and compute aggregated S,I,R time series
        S = np.asarray(S).astype(float)
        I = np.asarray(I).astype(float)
        R = np.asarray(R).astype(float)
        N = S + I + R
        time = np.arange(len(S)) * float(dt)
        # Quick per-step estimators (noisy) for initial guess
        eps = 1e-12
        dS = np.diff(S)
        dR = np.diff(R)
        S_mid = S[:-1]
        I_mid = I[:-1]
        valid = (I_mid > 150) & (S_mid > 150)

        gamma_steps = np.full_like(dR, np.nan)
        beta_steps = np.full_like(dS, np.nan)
        gamma_steps[valid] = dR[valid] / (I_mid[valid] * dt + eps)
        beta_steps[valid] = -dS[valid] * (N[:-1][valid]) / (S_mid[valid] * I_mid[valid] * dt + eps)

        # median robust estimators (positive values only)
        gamma_med = np.nanmean(gamma_steps[np.isfinite(gamma_steps) & (gamma_steps > 0)])
        beta_med = np.nanmean(beta_steps[np.isfinite(beta_steps) & (beta_steps > 0)])
        if not np.isfinite(gamma_med):
            gamma_med = 0.1
        if not np.isfinite(beta_med):
            beta_med = 0.1

        # Define SIR ODEs and model simulation
        N0 = float(N[0]) if len(N) > 0 else 1.0

        def sir_rhs(y, t, beta, gamma):
            S_, I_, R_ = y
            dSdt = -beta * S_ * I_ / N0
            dIdt = beta * S_ * I_ / N0 - gamma * I_
            dRdt = gamma * I_
            return [dSdt, dIdt, dRdt]

        def simulate(beta_gamma):
            beta, gamma = beta_gamma
            y0 = [S[0], I[0], R[0]]
            sol = odeint(sir_rhs, y0, time, args=(beta, gamma))
            return sol

        # residuals: fit infected curve (I) which is usually most informative
        def residuals(bg):
            sol = simulate(bg)
            I_m = sol[:, 1]
            # Normalize both model and data to fractions of N0 for shape-based fitting
            return (I_m / N0 - I / N0)

        # prepare initial guess and ensure it's within bounds to avoid least_squares error
        x0 = np.array([float(beta_med), float(gamma_med)], dtype=float)
        lower = np.array([1e-9, 1e-9], dtype=float)
        upper = np.array([10.0, 10.0], dtype=float)
        # replace NaN/inf and clip into (lower, upper)
        x0 = np.nan_to_num(x0, nan=0.1, posinf=upper - 1e-6, neginf=lower + 1e-6)
        x0 = np.clip(x0, lower + 1e-12, upper - 1e-12)
        res = least_squares(residuals, x0, bounds=(lower, upper), xtol=1e-4, ftol=1e-4,loss='soft_l1')
        beta_fit, gamma_fit = res.x
        fitted = simulate([beta_fit, gamma_fit])
        S_fit, I_fit, R_fit = fitted[:, 0], fitted[:, 1], fitted[:, 2]

        # Plot comparison I (data vs fit)
        '''
        try:
            plt.figure()
            plt.plot(time, I, 'r.', label='I (data)')
            plt.plot(time, I_fit, 'k-', label=f'I (fit) beta={beta_fit:.4f}, gamma={gamma_fit:.4f}')
            plt.xlabel('time')
            plt.ylabel('Infected')
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception:
            pass
        '''

        return {
            'time': time,
            'S': S, 'I': I, 'R': R, 'N': N,
            'beta_med': float(beta_med), 'gamma_med': float(gamma_med),
            'beta_fit': float(beta_fit), 'gamma_fit': float(gamma_fit),
            'S_fit': S_fit, 'I_fit': I_fit, 'R_fit': R_fit,
            'opt_result': res
        }
def infection_func(distance):
    
    return 0.05 # Infection radius of 2 units

if __name__ == "__main__":


    sim = InfectionSim(infection_func, recovery_time=14)  # recovery time of 7 days
    all_positions, S, I, R, all_status, recovered_time_steps = sim.run_simulation(population_size=3500, area_size_x=2000, area_size_y=2000,
                       initial_infected=10, total_time=100, units='days', dt = 15, records_interval=10)
    gif_name = sim.animate_simulation(all_positions, S, I, R, all_status, recovered_time_steps, area_size_x=2000, area_size_y=2000,
                             total_time=100, units='days',records_interval=10, fpss=70)
    fit_results = sim.fiting_SIR(S, I, R, recovered_time_steps, dt=1/10)  # dt in days
    sim.plot_fit_results(fit_results, units='days')
    print(f"Fitting complete. Estimated beta: {fit_results['beta_fit']}, gamma: {fit_results['gamma_fit']}.")
    print(f"Simulation complete. Animation saved as {gif_name}.")
    all_positions, S, I, R, all_status, recovered_time_steps = sim.multi_run_simulation(num_runs=8, population_size=3500, area_size_x=2000, area_size_y=2000,
                       initial_infected=10, total_time=100, units='days', dt = 15, records_interval=10)

