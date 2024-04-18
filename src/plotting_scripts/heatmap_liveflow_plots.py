import matplotlib.pyplot as plt
import seaborn as sns
import math 
import pandas as pd
import glob as glob
import os as os
from src.Utilities import Data_analysis as util
from src.processing_scripts import heatmap_liveflow_processing as ps


def plot_heatmap(df, gridsize, bins_hex, save_loc, plot_type='hex'):
    for treatment, dfs in df.groupby('treatment_name'):
        if plot_type == 'hex':
            plt.rcParams['svg.fonttype']='none'
            sns.set(style="whitegrid")
            g = sns.JointGrid(data=dfs, x='time', y='FRET', xlim=(0,130), ylim=(0, 1))
            g.plot_joint(plt.hexbin, gridsize=(gridsize, gridsize), cmap='ocean_r', mincnt=0, bins=bins_hex)
            g.plot_marginals(sns.histplot, kde=True, bins=20)
            plt.savefig(f'{save_loc}/Heatmap_{treatment}_{plot_type}.svg', dpi=600)
        if plot_type == 'kde':
            plt.rcParams['svg.fonttype']='none'
            sns.set_style("whitegrid", {'grid.linestyle':'--', 'axes.linewidth':20, 'axes.color':'black', 'axes.edgecolor': 'black', 'font.size':10})
            fig = sns.jointplot(data=dfs, x='time', y='FRET', xlim=(0,300), ylim=(0, 1), alpha=0.05, color='#2AA6CF', marginal_kws=dict(bins=20, kde=True))   
            fig.plot_joint(sns.kdeplot, cmap='mako') 
            fig.ax_joint.spines['top'].set_visible(True)
            fig.ax_joint.spines['right'].set_visible(True)
            fig.ax_marg_x.spines['left'].set_visible(True)
            fig.ax_marg_x.spines['top'].set_visible(True)
            fig.ax_marg_x.spines['right'].set_visible(True)
            fig.ax_marg_y.spines['top'].set_visible(True)
            fig.ax_marg_y.spines['right'].set_visible(True)
            fig.ax_marg_y.spines['bottom'].set_visible(True)
            plt.savefig(f'{save_loc}/Heatmap_{treatment}_{plot_type}.svg', dpi=600)
        plt.show()
    return


def plot_average_FRET_over_time(df, filt,  save_loc='',  ci='sd', x_axis='time', subplot=False):
    sns.set(style="ticks", font_scale=1)
    filt_data = df[df['treatment_name'].isin(filt)]
    if x_axis == 'time':
        fig, ax = plt.subplots()
        sns.lineplot(data=filt_data, x=x_axis, y='FRET', ci=ci, hue='treatment_name')
        plt.xlim(0, 200, 10)
        ax.axvline(x=10, linestyle='-', color='grey')
        ax.axvspan(0, 10, facecolor='grey', alpha=.2)   
        plt.ylim(0, 1, 10)
        plt.xlabel('Time (s)')
        plt.legend(title='', fontsize='small')
        plt.savefig(f'{save_loc}/Average_Heatmap_{filt}.svg', dpi=600)
    if (x_axis == 'normalised_to_event' and subplot == True):
        nrow = math.ceil(len(filt)/2)
        fig, axes = plt.subplots(nrow, 2, figsize=(8, 3*nrow), sharex=True, sharey=True)
        axes = axes.flatten()
        for i, treatment in enumerate(filt):
            sns.lineplot(data=df[df['treatment_name'] == treatment], x=x_axis, y='FRET', ci=ci, hue='treatment_name', ax=axes[i])
            axes[i].axvline(x=0, linestyle='--', color='grey')
            plt.xlim(-30, 30, 10)
            plt.ylim(0, 1, 10)
            axes[i].set_xlabel('Time (s)')
            axes[i].set_title(f"{treatment} (n={str(df[df['treatment_name'] == treatment]['molecule_number'].nunique())})")
            axes[i].get_legend().remove()
        if len(filt) < len(axes):
            axes[-1].remove()
            axes[3].set_xlabel('Time (s)')
        plt.savefig(f'{save_loc}/Traces_normalised_to_first_trans.svg', dpi=600)
        plt.show()
    elif x_axis == 'normalised_to_event':
        for treatment, df in filt_data.groupby('treatment_name'):
            sns.lineplot(data=df[df['treatment_name'] == treatment], x=x_axis, y='FRET', ci=ci, hue='treatment_name')
            plt.xlim(-30, 30, 10)
            plt.axvline(x=0, linestyle='--', color='grey')
            plt.ylim(0, 1, 10)
            plt.xlabel('Time (s)')
            plt.legend(title='', fontsize='small', loc='upper left')
            plt.savefig(f'{save_loc}/Traces_normalised_to_first_trans_{treatment}.svg', dpi=600)
            plt.show()
    plt.show()
    return


def plot_first_specified_transition(df, trans_type, save_loc):
    plot1 = plt.figure()
    sns.set(style='ticks', font_scale=1)
    sns.violinplot(data=df, y='cum_sum', x='treatment', scale='width')
    sns.stripplot(data=df, y='cum_sum', x='treatment', color ='black', alpha=0.5)
    plt.ylabel('Time for RNA binding (s)')
    plt.xlabel('')
    plot1.savefig(f'{save_loc}/time_until_first_{trans_type}_transition.svg', dpi=600)
    plt.show()


def master_heatmap_processing_func(data_paths, input_folder='Experiment_1-description/python_results', exposure=0.200, FRET_thresh=0.5, transition_type='low_to_high', time_thresh=15, injection_time=15):
    output_folder = f'{input_folder}/Heatmaps-and-first-transitions'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_rate = 1/exposure  ##### used when importing heatmap data to turn frame number to a unit of time
    compiled_df = []
    for data_name, (label, data_path) in data_paths.items():
        imported_data = util.file_reader(data_path, 'heatmap', frame_rate)
        cleaned_raw = ps.remove_outliers(imported_data, 'hist', "raw")    #### add "idealized" after imported_data to get idealized histograms
        cleaned_raw["treatment_name"]=data_name
        compiled_df.append(cleaned_raw)
    compiled_df = pd.concat(compiled_df)   #### .rename(columns={1:"test", 3:"test2"}) ## can rename individually if needed
    compiled_df.columns = ["frames", "donor", "acceptor", "FRET", "idealized FRET", 'molecule_number', 'time', "treatment_name"]
    compiled_df.to_csv(f'{output_folder}/compiled_df.csv')
    
    compiled_filt=[]
    for treatment, df in compiled_df.groupby('treatment_name'):
        treatment_df = compiled_df[compiled_df['treatment_name'] == treatment]
        treatment_df2 = treatment_df.filter(items=['idealized FRET','molecule_number'])
        treatment_df3 = ps.calculate_dwell_time(treatment_df2)
        treatment_transitions = ps.generate_transitions(treatment_df3)
        treatment_cleaned_transitions = ps.remove_outliers2(treatment_transitions)
        treatment_cleaned_transitions['time (s)'] = treatment_cleaned_transitions['Time'] * exposure
        treatment_cumsum = ps.filter_FRET_trans_if(treatment_cleaned_transitions, FRET_thresh, transition_type) 
        treatment_first_transition = ps.select_first_transition(treatment_cumsum, time_thresh, injection_time)
        treatment_first_transition['treatment'] = treatment
        compiled_filt.append(treatment_first_transition)
    col = pd.concat(compiled_filt)
    col.to_csv(f'{output_folder}/col.csv')

    normalised_data = ps.normalise_to_event(compiled_df, col, FRET_thresh, transition_type='low_to_high')
    normalised_data.to_csv(f'{output_folder}/normalised_data.csv')
    return normalised_data

# -------------------------------- MASTER FUNCTION -----------------------------------------

def master_plot_flowin_func(input_folder='Experiment_1-description/python_results', transition_type='low_to_high', gridsize=100, binshex=80):
    output_folder = f'{input_folder}/Heatmaps-and-first-transitions'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    compiled_df = pd.read_csv(f'{output_folder}/compiled_df.csv')
    col = pd.read_csv(f'{output_folder}/col.csv')
    normalised_data = pd.read_csv(f'{output_folder}/normalised_data.csv')
    no_filt = compiled_df['treatment_name'].unique().tolist()
    plot_heatmap(compiled_df, gridsize, binshex, save_loc=output_folder, plot_type='kde')
    plot_first_specified_transition(col, transition_type, save_loc=output_folder)
    plot_average_FRET_over_time(normalised_data, no_filt, save_loc=output_folder, ci='sd', x_axis='time') ##### change 'no_filt' to 'to_filt' to include only datasets that were mentioned above
    plot_average_FRET_over_time(normalised_data, no_filt, save_loc=output_folder, ci='sd', x_axis='normalised_to_event', subplot=False)  #### replace with True to plot subplots
    print(col.groupby('treatment')['cum_sum'].mean())
    print(col.groupby('treatment')['cum_sum'].sem())
