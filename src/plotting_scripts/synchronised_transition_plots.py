import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from smfret.src.processing_scripts import synchronised_transition_processing as ps



def plot_synchronised_transition(dfs, index_to_plot, exposure_seconds, order, frame_from_trans, save_loc, palette, label=''):
    """plots the FRET values either side of a transition type of interest

    Args:
        dfs (df): dataframe containing the raw FRET values (generated after the calculate dwells function)
        index_to_plot (list): list of index values that met the criteria defined in 'filt_df_to_plot' that will the be mapped to df for plotting of raw FRET values
        exposure_seconds (float): exposure in seconds used to convert frames to a unit of time
        list_to_drop (list): list containing the name of all treatments to be dropped
        frame_from_trans (int, optional): should be the same as 'min_dwell_before' variable in the 'filt_df_to_plot' function, basically sets the xlims. Defaults to 80.
    """
    combined_mini=[]
    for df in index_to_plot:
        print(df)
        lower = df - frame_from_trans
        upper = df + (frame_from_trans+1)
        upper = min(upper, len(dfs))
            # Calculate the length of the range that will be generated by np.arange
        range_length = upper - lower
        # Calculate time values based on the index of interest and the exposure time
        time_from_trans = np.arange(-(frame_from_trans/(1/exposure_seconds)), 
                                    -(frame_from_trans/(1/exposure_seconds))+exposure_seconds*range_length, 
                                    exposure_seconds)[:range_length]
        mini_df = dfs.iloc[lower:upper].reset_index()
        mini_df['time_from_trans'] = time_from_trans
        combined_mini.append(mini_df)
    combined_mini = pd.concat(combined_mini)
    fig, axes = plt.subplots()
    sns.set(style='ticks')
    sns.lineplot(data=combined_mini, x='time_from_trans', y='FRET', hue='treatment_name', palette=palette, hue_order=order)
    plt.xlabel('Time (s)')
    plt.legend(title='',loc='best')
    plt.ylim(0, 0.8)
    fig.savefig(f'{save_loc}/synchronised_release{"_"+label}.svg', dpi=600)
    plt.show()




def plot_synchronised_transition2(dfs, index_to_plot, index_to_plot2, exposure_seconds, order, frame_from_trans, save_loc, palette, add_time=0, label = ''):
    combined_mini = []
    for df in index_to_plot:
        print(df)
        lower = df - frame_from_trans
        upper = df + (frame_from_trans+1) + (add_time/exposure_seconds)
        upper = int(min(upper, len(dfs)))
            # Calculate the length of the range that will be generated by np.arange
        range_length = int(upper - lower)
        # Calculate time values based on the index of interest and the exposure time
        time_from_trans = np.arange(-(frame_from_trans/(1/exposure_seconds)), 
                                    -(frame_from_trans/(1/exposure_seconds))+exposure_seconds*range_length, 
                                    exposure_seconds)[:range_length]
        mini_df = dfs.iloc[lower:upper].reset_index()
        mini_df['time_from_trans'] = time_from_trans
        combined_mini.append(mini_df)
    combined_mini = pd.concat(combined_mini)
    filt_data = combined_mini
    combined_mini2 = []
    for df in index_to_plot2:
        print(df)
        lower = df - frame_from_trans
        upper = df + (frame_from_trans+1) + (add_time/exposure_seconds)
        upper = int(min(upper, len(dfs)))
            # Calculate the length of the range that will be generated by np.arange
        range_length = int(upper - lower)
        # Calculate time values based on the index of interest and the exposure time
        time_from_trans = np.arange(-(frame_from_trans/(1/exposure_seconds)), 
                                    -(frame_from_trans/(1/exposure_seconds))+exposure_seconds*range_length, 
                                    exposure_seconds)[:range_length]
        mini_df2 = dfs.iloc[lower:upper].reset_index()
        mini_df2['time_from_trans'] = time_from_trans
        combined_mini2.append(mini_df2)
    combined_mini2 = pd.concat(combined_mini2)
    filt_data2 = combined_mini2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4), sharex=True, sharey=True)
    sns.set(style = 'ticks')
    sns.lineplot(data = filt_data, x = 'time_from_trans', y = 'FRET', hue = 'treatment_name', palette = palette, hue_order = order, ax=ax2)
    sns.lineplot(data = filt_data2, x = 'time_from_trans', y = 'FRET', hue = 'treatment_name', palette = palette, hue_order = order, ax=ax1)

    ax1.set_xlabel('Time (s)')
    ax2.set_xlabel('Time (s)')
    # plt.xlim(-(frame_from_trans/(1/exposure_seconds)),(frame_from_trans/(1/exposure_seconds)+add_time))
    # plt.legend(title = '',loc = 'best')
    ax1.legend('',loc='upper left')
    ax2.legend('')
    fig.savefig(f'{save_loc}/synchronised_release{"_"+label}.svg', dpi = 600)
    plt.show()
    return filt_data, filt_data2


# -------------------- Code to plot the percentage transition data -------------------------------

def plot_summary_transition(save_loc, order, df, palette, filt=True):
    melted_data = df.melt(id_vars=['treatment', 'repeat'])
    # Ensure the order of the 'treatment' column
    melted_data['treatment'] = pd.Categorical(melted_data['treatment'], categories=order, ordered=True)
    fig, ax = plt.subplots(figsize=(3, 6))
    sns.set_style('ticks',{'grid.linestyle':'--', 'font_scale': 1.5})
    if filt == False:
        sns.barplot(data=melted_data, y='value', x='variable', hue='treatment', palette=palette, hue_order=order, edgecolor='black', order=order)
    else:
        ax = sns.barplot(data=melted_data[melted_data['variable']=='% DnaK release are consecutive'], y='value', x='treatment', palette='BuPu', hue_order=order, edgecolor='black', fill=False, order=order, capsize=.2, errcolor='black')
        sns.scatterplot(data=melted_data[melted_data['variable']=='% DnaK release are consecutive'], y='value', x='treatment', hue='treatment', palette='BuPu', hue_order=order, edgecolor='black', ax=ax, s=200)

    plt.xlabel('')
    plt.xticks(rotation=45)
    plt.legend(title='')
    plt.ylabel('Proportion of transitions (%)')
    fig.savefig(f'{save_loc}/consecutive_transition_summary.svg', dpi=600)
    plt.show()

def plot_consec_DnaK_release_with_filter(dataframe, consecutive_trans, nonconsecutive_trans, FRET_before, FRET_after, palette, save_loc, datatype='Proportion'):
    helpplease = []
    for x, df in enumerate(range(0, 401)):
        if datatype == 'Proportion':
            dfs = ps.prop_DnaK_release_events_are_consecutive(dataframe, x, consecutive_trans, nonconsecutive_trans, FRET_before, FRET_after)
        else:
            dfs = ps.ratio_consecutive_to_nonconsecutive(dataframe, x, consecutive_trans, nonconsecutive_trans, FRET_before, FRET_after)
        dfs['frames_to_thresh'] = x
        helpplease.append(dfs)
        helpplease_df = pd.concat(helpplease).reset_index()
    sns.lineplot(data=helpplease_df, x='frames_to_thresh', y='prop_consecutive_dnaK_release', hue='treatment', palette=palette)
    plt.xlabel('Threshold prior to DnaK release (frames)')
    plt.ylabel(f'{datatype} of transitions (consecutive:non-consecutive)')
    plt.legend(title='')
    plt.savefig(f'{save_loc}/consecutive_transition_over_frame_threshold_{datatype}.svg', dpi=600)
    plt.show()
    return

def plot_FRET_after_release(plot_export, df, order, palette='BuPu'):
    fig, ax = plt.subplots()
    sns.violinplot(data=df, y='FRET_after', x='treatment_name', split=True, hue='consec', scale='width', palette=palette, inner='quart', order=order)
    plt.xlabel('')
    plt.legend(title='')
    plt.ylabel('FRET state after FRET increase')
    plt.xticks(rotation=45)
    plt.savefig(f'{plot_export}/FRET_state_after_increase.svg', dpi=600)
    plt.show()

# -------------------------------- MASTER FUNCTION -----------------------------------------
    
def master_plot_synchronised_transitions(order, output_folder='Experiment_1-description/python_results', exposure=0.2, frames_to_plot=50, FRET_before=0.3,FRET_after=0.3, datatype='Proportion', filt=True, palette='BuPu', add_time=0):
    plot_export = f'{output_folder}/synchronised_transitions/'
    if not os.path.exists(plot_export):
        os.makedirs(plot_export)
    compiled_data = pd.read_csv(f'{output_folder}/Cleaned_FRET_histogram_data.csv')

    calculated_transitions = []
    for treatment, df in compiled_data.groupby('treatment_name'):
        dwell_df = ps.calculate_dwells(df)
        transition_df = ps.generate_transitions_sync(dwell_df)
        calculated_transitions.append(transition_df)
    calculated_transitions_df = pd.concat(calculated_transitions)
    calculated_transitions_df.reset_index(inplace=True)
    calculated_transitions_df.drop('index', axis=1, inplace=True)

    font = {'weight' : 'normal', 'size'   : 12 }
    plt.rcParams['font.sans-serif'] = "Arial"
    plt.rcParams['font.family'] = "sans-serif"
    plt.rc('font', **font)
    plt.rcParams['svg.fonttype'] = 'none'

    dnak_stable_release = ps.filt_df_to_plot(calculated_transitions_df, FRET_before, FRET_after,'low_to_high', frames_to_plot)
    plot_synchronised_transition(calculated_transitions_df, dnak_stable_release, exposure, order, frames_to_plot, plot_export, palette, label='release')

    dnak_stable_binding = ps.filt_df_to_plot(calculated_transitions_df, FRET_before, FRET_after, 'high_to_low', frames_to_plot)
    plot_synchronised_transition(calculated_transitions_df, dnak_stable_binding, exposure, order, frames_to_plot, plot_export, palette, label='binding')

    col = []
    for  treatment, df in calculated_transitions_df.groupby('treatment_name'):
        transition_data = df[df['transition_point']==True]
        transition_data['FRET_increase'] = transition_data['FRET_before'] < transition_data['FRET_after'] 
        consecutive_identified = ps.determine_first_transition_in_sequence(transition_data)
        col.append(consecutive_identified)
    consecutive_data = pd.concat(col)

    consecutive_trans, nonconsecutive_trans, percent_trans_meet_criteria_df = ps.concat_trans_proportion(
    consecutive_data, 
    calculated_transitions_df, 
    FRET_before, 
    FRET_after)


    consecutive_from_dnak_release = ps.filt_df_to_plot(consecutive_trans, FRET_before, FRET_after,'low_to_high', frames_to_plot)
    nonconsecutive_from_dnak_release = ps.filt_df_to_plot(nonconsecutive_trans, FRET_before, FRET_after,'low_to_high', frames_to_plot)

    plot_synchronised_transition(calculated_transitions_df, consecutive_from_dnak_release, exposure, order, frames_to_plot, plot_export, palette, label='consecutive_transitions')
    plot_synchronised_transition(calculated_transitions_df, nonconsecutive_from_dnak_release, exposure, order, frames_to_plot, plot_export, palette, label='non-consecutive_transition')


    plot_summary_transition(plot_export, order, percent_trans_meet_criteria_df, palette, filt=filt)

    plot_consec_DnaK_release_with_filter(calculated_transitions_df, consecutive_trans, nonconsecutive_trans, FRET_before, FRET_after, palette, save_loc=plot_export, datatype=datatype)


    first_consecutive_transition = calculated_transitions_df.iloc[consecutive_from_dnak_release]
    first_nonconsecutive_transition = calculated_transitions_df.iloc[nonconsecutive_from_dnak_release]
    first_consecutive_transition['consec'] = True
    first_nonconsecutive_transition['consec'] = False
    combined_consec_nonconsec = first_consecutive_transition.append(first_nonconsecutive_transition)

    plot_FRET_after_release(plot_export, combined_consec_nonconsec, order)

    filt_data, filt_data2 = plot_synchronised_transition2(dfs=calculated_transitions_df, 
                                                          index_to_plot=consecutive_from_dnak_release, 
                                                          index_to_plot2=nonconsecutive_from_dnak_release,
                                                          exposure_seconds=exposure, 
                                                          order=order, 
                                                          frame_from_trans=frames_to_plot, 
                                                          save_loc=plot_export, 
                                                          palette=palette,  
                                                          label='consecutive_transitions', 
                                                          add_time=add_time)

    return percent_trans_meet_criteria_df, calculated_transitions_df, consecutive_from_dnak_release, nonconsecutive_from_dnak_release, filt_data, filt_data2, combined_consec_nonconsec

