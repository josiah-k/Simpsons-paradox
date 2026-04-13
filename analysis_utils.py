import numpy as np
from cell_model_mothermachine_divNoise import Cell_Population
import pandas as pd
import matplotlib.pyplot as plt


def simulate_and_plot(tau_r: float,
                      tau_u: float,
                      sigma_u: float,
                      ribo_div_noise: float,
                      k_n0: float,
                      num_cells: int = 3000,
                      n_steps: int = 1500,
                      save_data: bool = False,
                      save_fig: bool = False,
                      return_vals: bool = False):


    figure = plt.figure()
    colors=['lightblue','green','gold','red','purple','k','gray']


    f_i = np.array([0.005, 0.01, 0.0368, 0.0726, 0.1084, 0.1442])

    df_act, df_gr, df_phiR, df_pRm, df_vol, df_phiU = [pd.DataFrame([]) for _ in range(6)]

    for fi,j in zip(f_i,range(len(f_i))):
        simulator = Cell_Population(k_n0, ribo_div_noise, fi, tau_u, sigma_u, tau_r, num_cells_init=num_cells)
        simulator.initialize()
        act, kappa, phiR, phiRmax, vol, phiU = [np.array([]) for _ in range(6)]

        for _ in range(n_steps):
            gr, activity, phi_R, phiR_max, volume, phi_U = simulator.simulate_population()
            # note: when simulating more than one cell, index does not represent time, as cell data is concatenated together for each timestep
            kappa = np.concatenate([kappa, gr])
            act = np.concatenate([act, activity])
            phiR = np.concatenate([phiR, phi_R])
            phiRmax = np.concatenate([phiRmax, phiR_max])
            vol = np.concatenate([vol, volume])
            phiU = np.concatenate([phiU, phi_U])


        num_bins_x = 7
        # Bin the data in x and y directions
        x_bins = pd.qcut(act, num_bins_x, labels=False)
        # Compute mean x and y for each bin
        df = pd.DataFrame({'act '+str(fi): act, 'gr '+str(fi): kappa, 'x_bin': x_bins})
        binned_data = df.groupby('x_bin').agg({'act '+str(fi): 'mean', 'gr '+str(fi): 'mean'}).reset_index()

        # Create dfs for saving data
        df_act_poor = pd.concat((df_act, df['act '+str(fi)]), axis=1)
        df_gr_poor = pd.concat((df_gr, df['gr '+str(fi)]), axis=1)
        df_phiR = pd.concat((df_phiR, pd.DataFrame(phiR, columns=['phiR '+str(fi)])), axis=1)
        df_pRm = pd.concat((df_pRm, pd.DataFrame(phiRmax, columns=['phiRmax '+str(fi)])), axis=1)
        df_vol = pd.concat((df_vol, pd.DataFrame(phiR, columns=['vol '+str(fi)])), axis=1)
        df_phiU = pd.concat((df_phiU, pd.DataFrame(phiR, columns=['phiU '+str(fi)])), axis=1)

        # plt.scatter(activity, growth_rate, alpha=0.1, color=colors[j])
        plt.plot(binned_data['act '+str(fi)],binned_data['gr '+str(fi)], color=colors[j])
        plt.scatter(binned_data['act '+str(fi)],binned_data['gr '+str(fi)], color=colors[j])
        plt.scatter(np.mean(act), np.mean(kappa), color='k')

    if save_data:
        df_act_poor.to_csv(f'activity_kn0_{k_n0}_riboDivNoise_{ribo_div_noise}.csv')
        df_gr_poor.to_csv(f'growthRate_kn0_{k_n0}_riboDivNoise_{ribo_div_noise}.csv')


    plt.xlabel('unnecessary protein activity (h$^{-1}$)')
    plt.ylabel('growth rate (h$^{-1}$)')
    plt.show()
    if save_fig:
        figure.savefig(f'simpsons_paradox_kn0_{k_n0}_riboDivNoise_{ribo_div_noise}.pdf', dpi=300, bbox_inches='tight')

    if return_vals:
        return df_act_poor, df_gr_poor, df_phiR, df_pRm, df_vol, df_phiU