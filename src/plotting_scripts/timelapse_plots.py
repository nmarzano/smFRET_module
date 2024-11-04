import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from smfret.src.processing_scripts import timelapse_hist_processing as ps
from smfret.src.Utilities import Data_analysis as util
from scipy import optimize, signal
import numpy as np
from scipy import optimize

def plot_data_col(df, thresh, fit_type, xlim, ylim, save_loc, palette, data):
    """Plots the proportion of time each molecule spends below a threshold (defined previously in 1A-plot-histogram) as the mean +- SE as a function of time. This function
    is predominantly designed to collate timelapse data from different experiments and present it within a single plot. It also fits a generic curve to the data, which has been 
    defined above in fit_curve_to_plot function. 

    Args:
        df (dataframe): collated dataset of all treatments to compare
        fit_type (func): the type of fit you wish to plot. If you want a different function, define elsewhere and call the function here to plot.
        xlim (float): minimum x-axis value used to define fit
        ylim (float): maximum x-axis value used to define fit
        data (str, optional): changes if you want to plot normalised data 'normalised' or raw data 'Mean'. Defaults to 'Mean'.
    """
    fig, ax = plt.subplots()
    fit_dict = {}
    for (protein, treatment), dfs in df.groupby(['protein', 'treatment']):
        if data == 'Mean':
            fitx, fity, fit_A, fit_B, fit_A_error, fit_B_error = ps.fit_curve_to_plot(dfs, fit_type, xlim, ylim, data)
        else:
            fitx, fity, fit_A, fit_B, fit_A_error, fit_B_error = ps.fit_curve_to_plot(dfs, fit_type, xlim, ylim, data)
        plt.errorbar(dfs['timepoint_plot'],dfs[f'{data}'],dfs['Std_error'], fmt='none', capsize=3, ecolor='black')
        sns.scatterplot(data=dfs, x='timepoint_plot', y=data, color=palette[protein], edgecolor='black')
        plt.plot(fitx, fity,'k', color='black')
        fit_dict[(protein, treatment)] = (fit_A, fit_B, fit_A_error, fit_B_error)        
    plt.ylabel(f'Proportion of time spent < {thresh} FRET (mean)')
    plt.xlabel('Time (min)')
    plt.legend(title='')
    plt.savefig(f'{save_loc}/timelapse_proportion_mol_below_thresh.svg', dpi=600)
    plt.show()
    return fit_dict

def plot_data_col_sep(df, thresh, fit_type, xlim, ylim, save_loc, palette, data, markersize):
    """Plots the proportion of time each molecule spends below a threshold (defined previously in 1A-plot-histogram) as the mean +- SE as a function of time. This function
    is predominantly designed to collate timelapse data from different experiments and present it within a single plot. It also fits a generic curve to the data, which has been 
    defined above in fit_curve_to_plot function. 

    Args:
        df (dataframe): collated dataset of all treatments to compare
        fit_type (func): the type of fit you wish to plot. If you want a different function, define elsewhere and call the function here to plot.
        xlim (float): minimum x-axis value used to define fit
        ylim (float): maximum x-axis value used to define fit
        data (str, optional): changes if you want to plot normalised data 'normalised' or raw data 'Mean'. Defaults to 'Mean'.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True, sharex=True)
    fit_dict = {}
    for (protein, treatment), dfs in df.groupby(['protein', 'treatment']):
        if data == 'Mean':
            fitx, fity, fit_A, fit_B, fit_A_error, fit_B_error = ps.fit_curve_to_plot(dfs, fit_type, xlim, ylim, data)
        else:
            fitx, fity, fit_A, fit_B, fit_A_error, fit_B_error = ps.fit_curve_to_plot(dfs, fit_type, xlim, ylim, data)

        if treatment=='KJE':
            ax1.errorbar(dfs['timepoint_plot'],dfs[f'{data}'],dfs['Std_error'], fmt='none', capsize=3, ecolor='black')
            sns.scatterplot(data=dfs, x='timepoint_plot', y=data, color=palette[protein], edgecolor='black', ax=ax1, s=markersize)
            ax1.plot(fitx, fity,'k', color='black')
        else:
            ax2.errorbar(dfs['timepoint_plot'],dfs[f'{data}'],dfs['Std_error'], fmt='none', capsize=3, ecolor='black')
            sns.scatterplot(data=dfs, x='timepoint_plot', y=data, color=palette[protein], edgecolor='black', ax=ax2, s=markersize)
            ax2.plot(fitx, fity,'k', color='black')


        fit_dict[(protein, treatment)] = (fit_A, fit_B, fit_A_error, fit_B_error)        

# Remove individual y-axis labels
    ax1.set_ylabel('')
    ax2.set_ylabel('')

    # Add a centered y-axis label across both subplots
    fig.text(0.04, 0.5, f'Proportion of time spent < {thresh} FRET (mean)', va='center', rotation='vertical')        
    plt.xlabel('Time (min)')
    plt.legend(title='')
    plt.savefig(f'{save_loc}/timelapse_proportion_mol_below_thresh.svg', dpi=600)
    plt.show()
    return fit_dict
# -------------------------------- MASTER FUNCTION -----------------------------------------

def master_timelapse_func(filt_dfs, thresh=0.3, xlim_min=6, xlim_max=60, output_folder='Experiment_1-description/python_results', palette='BuPu', data_type='Mean', markersize=5, split=True):
    filt_dfs['protein'] = filt_dfs['treatment_name'].str.split('_').str[0]
    filt_dfs['treatment'] = filt_dfs['treatment_name'].str.split('_').str[1]
    timepoint_plotdata = ps.concat_df_fit_data(filt_dfs, output_folder)
    if split==False:
        fit_dict = plot_data_col(timepoint_plotdata, thresh, ps.guess_exponential, xlim_min, xlim_max, output_folder, palette, data_type)
    else:
        fit_dict = plot_data_col_sep(timepoint_plotdata, thresh, ps.guess_exponential, xlim_min, xlim_max, output_folder, palette, data_type, markersize)
    return filt_dfs, timepoint_plotdata, fit_dict



