import numpy as np
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os as os
from smfret.src.processing_scripts import simulation_scripts as aa

def plot_state_simulations(plot_export, data):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='Time', y='State')
    plt.xlabel('Time (s)')
    plt.ylabel('')
    plt.yticks(range(5), [f'State {i}' for i in range(5)])
    plt.savefig(f'{plot_export}/simulated_states.svg', dpi=300)
    plt.show()

def simple_sim_test(plot_export, num_states=5, rate_up=0.4, rate_down=0.11, simulation_time=360, time_step=1, seed=None):
    data = aa.simulate_state_transitions(num_states=num_states, rate_up=rate_up, rate_down=rate_down, simulation_time=simulation_time, 
                                time_step=time_step, seed=seed)

    plot_state_simulations(plot_export, data)
    data = aa.plot_binary_state(data, plot_export, threshold_state=1, noise_level=.05)
    mean_dwell_time, concatenated_data, fits = aa.analyze_dwell_times(plot_export=plot_export, threshold_state=1, num_simulations=300, num_states=num_states, rate_up=rate_up, rate_down=rate_down,
                    simulation_time=360, time_step=1)

    average_state_df = aa.calculate_average_state_per_simulation(concatenated_data, plot_export)

    # Plot the average state per simulation
    plt.figure(figsize=(10, 6))
    sns.histplot(data=average_state_df, x='Average State', color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Average State')
    plt.ylabel('Frequency')
    plt.title('Average State Per Simulation')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'{plot_export}/test_simulation.svg', dpi=300)
    plt.show()

def plot_average_state_hist(combined_average_state_df):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=combined_average_state_df, x='Average State', hue='num_states', palette='viridis', edgecolor='black', alpha=0.7, bins=30)
    plt.xlabel('Average State')
    plt.ylabel('Frequency')
    plt.title('Average State Distribution Across Simulations (Grouped by Num States)')
    plt.tight_layout()
    plt.show()

def scatter_plot_rate_constant(fits_data_melted, plot_export, palette='BuPu'):
    plt.figure(figsize=(3, 5))
    sns.scatterplot(
    data=fits_data_melted, 
    x='Rate Type', 
    y='Mean Rate Constant', 
    hue='num_states', 
    palette=palette, 
    s=100, 
    edgecolor='black')
    plt.ylabel(r'Mean rate constant (s$^{-1}$)')
    plt.legend(title='Num States', loc='upper right')
    # Adjust ticks to be closer to the middle
    plt.gca().set_xticks([0, 1])  # Assuming 'Rate Type' has two categories
    plt.gca().set_xticklabels([r'$k_{\mathit{1}}$', r'$k_{\mathit{2}}$'], ha='center')
    plt.gca().tick_params(axis='x', pad=10)  # Adjust padding for better centering
    plt.xlabel('')
    # Adjust x-axis and y-axis limits to add padding around the data
    x_min, x_max = plt.gca().get_xlim()
    y_min, y_max = plt.gca().get_ylim()
    plt.gca().set_xlim(x_min - 0.5, x_max + 0.5)
    plt.gca().set_ylim(y_min - 0.0, y_max + 0.1)
    plt.tight_layout()
    plt.savefig(f'{plot_export}/mean_double_exponential_rates_scatter_vertical.svg', dpi=300)
    plt.show()

def plot_bar_by_num_states_and_rate_up(data, x, y, hue, palette, xlabel, ylabel, legend_title, filename, plot_export, figsize=(10, 6)):
    """
    Plots a barplot of y vs x, grouped by hue, with custom labels and saves the figure.

    Parameters:
        data (pd.DataFrame): DataFrame containing the data to plot.
        x (str): Column name for x-axis.
        y (str or int): Column name or index for y-axis.
        hue (str): Column name for hue grouping.
        palette (dict or str): Color palette for hue.
        xlabel (str): Label for x-axis.
        ylabel (str): Label for y-axis.
        legend_title (str): Title for the legend.
        filename (str): Filename for saving the plot (without extension).
        plot_export (str): Directory to save the plot.
        figsize (tuple): Figure size.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        palette=palette,
        edgecolor='black'
    )
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend(title=legend_title, fontsize=14, title_fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    fig.savefig(f'{plot_export}/{filename}.svg', dpi=300)
    plt.show()

def plot_kinetic_map(results_df, plot_export, palette='Greys'):
    for rate_up in results_df['rate_up'].unique():
        subset = results_df[results_df['rate_up'] == rate_up]
    
        # Plot k1 with num_states as hue
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=subset, x='rate_down', y='k1', hue='num_states', palette=palette)
        plt.title(f'k1 vs Rate Down (Rate Up = {rate_up})')
        plt.xlabel('Rate Down')
        plt.ylabel('k1 (Rate Constant)')
        plt.legend(title='Num States')
        plt.tight_layout()
        plt.savefig(f'{plot_export}/k1_analysis_rate_up_{rate_up}.svg', dpi=300)
        plt.show()
    
        # Plot k2 with num_states as hue
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=subset, x='rate_down', y='k2', hue='num_states', palette=palette)
        plt.title(f'k2 vs Rate Down (Rate Up = {rate_up})')
        plt.xlabel('Rate Down')
        plt.ylabel('k2 (Rate Constant)')
        plt.legend(title='Num States')
        plt.tight_layout()
        plt.savefig(f'{plot_export}/k2_analysis_rate_up_{rate_up}.svg', dpi=300)
        plt.show()

        # Plot k2 with num_states as hue
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=subset, x='rate_down', y='prop_fast', hue='num_states', palette=palette)
        plt.title(f'proportion of k1 (Rate Up = {rate_up})')
        plt.xlabel('Rate Down')
        plt.ylabel('proportion of k1')
        plt.legend(title='Num States')
        plt.tight_layout()
        plt.savefig(f'{plot_export}/k2_analysis_rate_up_{rate_up}.svg', dpi=300)
        plt.show()

def master_simplot(results, plot_export, rate_up_palette='BuPu', other_pal='BuPu'):
    (combined_average_state_df, 
    combined_concat, 
    proportion_time_combined, 
    transitions_x_to_y_rate, 
    mean_dwell_times_by_num_states_rate_up, 
    fits_data_melted) = aa.df_processing_for_plots(results, state_x=2, state_y=1)

    plot_average_state_hist(combined_average_state_df)
    average_state_df = aa.calculate_average_state_per_simulation2(combined_concat, plot_export, palette='BuPu')

    plot_bar_by_num_states_and_rate_up(
    data=proportion_time_combined,
    x='num_states',
    y=0,
    hue='rate_up',
    palette=rate_up_palette,
    xlabel='# possible states',
    ylabel='Proportion of time in state 0 or 1',
    legend_title='Rate Up',
    filename='percent_time_state_0_1_combined_rate_up',
    plot_export=plot_export)

    plot_bar_by_num_states_and_rate_up(
    data=transitions_x_to_y_rate,
    x='num_states',
    y=0,
    hue='rate_up',
    palette=rate_up_palette,
    xlabel='# possible states',
    ylabel=r'$k_{2\to1}$ (s$^{-1}$)',
    legend_title='Rate Up',
    filename='average_transitions_2_to_1_rate_rate_up',
    plot_export=plot_export)

    plot_bar_by_num_states_and_rate_up(
    data=mean_dwell_times_by_num_states_rate_up,
    x='num_states',
    y='mean_dwell_time',
    hue='rate_up',
    palette=rate_up_palette,
    xlabel='# possible states',
    ylabel='Mean dwell time (s)',
    legend_title='Rate Up',
    filename='mean_dwell_time_by_num_states_rate_up',
    plot_export=plot_export)

    scatter_plot_rate_constant(fits_data_melted, plot_export, palette=other_pal)
    return combined_average_state_df,combined_concat, proportion_time_combined, transitions_x_to_y_rate, mean_dwell_times_by_num_states_rate_up, fits_data_melted
