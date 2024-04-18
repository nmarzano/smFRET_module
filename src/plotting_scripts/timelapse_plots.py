import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.processing_scripts import timelapse_hist_processing as ps
from src.Utilities import Data_analysis as util

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
    for treatment, dfs in df.groupby('treatment'):
        if data == 'Mean':
            fitx, fity = ps.fit_curve_to_plot(dfs, fit_type, xlim, ylim, data)
        else:
            fitx, fity = ps.fit_curve_to_plot(dfs, fit_type, xlim, ylim, data)
        sns.scatterplot(data=dfs, x='timepoint', y=data, hue='treatment', palette=palette)
        plt.errorbar(dfs['timepoint'],dfs[f'{data}'],dfs['Std_error'], fmt='none', capsize=3, ecolor='black')
        plt.plot(fitx, fity,'k', color=palette[f'{treatment}'])
    plt.ylabel(f'Proportion of time spent < {thresh} FRET (mean)')
    plt.xlabel('Time (min)')
    plt.legend(title='')
    plt.savefig(f'{save_loc}/timelapse_proportion_mol_below_thresh.svg', dpi=600)
    plt.show()
    return 

# -------------------------------- MASTER FUNCTION -----------------------------------------

def master_timelapse_func(data, thresh=0.3, xlim_min=6, xlim_max=60, output_folder='Experiment_1-description/python_results', palette='BuPu', data_type='Mean'):
    test = []
    for data_name, data_path in data.items():
        data = util.file_reader(data_path, 'other')
        data['treatment']=data_name
        test.append(data)
    final = pd.concat(test).reset_index()
    final = final.iloc[:, 2:]

    plot_data_col(final, thresh, ps.guess_exponential, xlim_min, xlim_max, output_folder, palette, data_type)
    return final

