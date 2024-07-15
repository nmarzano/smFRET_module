import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import Utilities.Data_analysis as uda
import os
from smfret.src.processing_scripts import residence_time_processing as ps 


def plot_violin(data, save_loc, palette='BuPu', scale="y_axis"):
    if scale=="y_axis":
        plt.rcParams['svg.fonttype'] = 'none'
        plot1 = plt.figure()
        sns.set(style="ticks", font_scale=1)
        sns.violinplot(
            data=data, 
            x="transition_name", 
            y="y_axis",
            palette=palette, 
            hue="treatment", 
            log_scale=True,
            cut=0)
        plt.ylabel("Residence time (s)")
        plt.xlabel("Transition class")
        plt.legend(title='',loc="upper right")
        plot1.savefig(f"{save_loc}/Violin_plot_normal.svg", dpi=600)
        plt.show()
    if scale=="y_axis_log10":
        plt.rcParams['svg.fonttype'] = 'none'
        plot2 = plt.figure()
        sns.set(style="ticks", font_scale=1)
        sns.violinplot(
            data=data, 
            x="transition_name", 
            y="y_axis_log10",
            palette=palette, 
            hue="treatment", 
            log_scale=True)
        plt.ylabel("Log residence time (s)")
        plt.xlabel("Transition class")
        plt.legend(title='',loc="upper left", bbox_to_anchor=(1,1), ncol =1)
        plot2.savefig(f"{save_loc}/Violin_plot_log.svg", dpi=600)
        plt.show()
    if scale=='split':     
        f, (ax_top, ax_bottom)=plt.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw={'hspace':0.05})
        sns.set(style='ticks')
        sns.violinplot(x="transition_name", y="y_axis", hue="treatment",data=data, ax=ax_top, palette=palette, scale='width')
        sns.violinplot(x="transition_name", y="y_axis", hue="treatment",data=data, ax=ax_bottom, cut=0, palette=palette, scale='width')
        ax_top.set_ylim(bottom=40)   # those limits are fake
        ax_bottom.set_ylim(0,40)
        ax = ax_top
        d = .015  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
        ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
        ax2 = ax_bottom
        kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
        ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
        #remove one of the legend
        ax_bottom.legend_.remove()
        ax_top.set_xlabel('')
        plt.xlabel('')
        ax_top.tick_params(bottom=False)
        ax_top.set_ylabel('')
        ax_bottom.set_ylabel('')
        f.text(0.04, 0.5, 'Residence time (s)', ha='center', va='center', rotation='vertical')
        ax_top.legend(title='')
        f.savefig(f"{save_loc}/Violin_plot_splitaxis.svg", dpi=600)
        plt.show()

def plot_bar_with_sem(df, summary_df, order, save_loc, y_axis='y_axis', palette='mako'):
    # -------- prepare order of datasets so that sem are correctly mapped onto the correct dataset ------
    list_to_order = list(np.arange(0, len(order), 1))
    dict_to_order = dict(zip(order, list_to_order))
    summary_df['plot_order'] = summary_df['treatment'].map(dict_to_order)
    collated_sorted = summary_df.sort_values(['plot_order', 'transition_type'])
    sorted_df = df.sort_values(['treatment', 'transition_type'])
    # ----------- now plot the figure -------------------
    fig, ax = plt.subplots()
    sns.set(style='ticks', font_scale=1)
    sns.barplot(x='transition_name', y=y_axis, data=sorted_df, hue= 'treatment', palette=palette, ci =None, hue_order=order, edgecolor='black')
    x_coords = [p.get_x() + 0.5 * p.get_width() for p in ax.patches]
    y_coords = [p.get_height() for p in ax.patches]
    ax.errorbar(x=x_coords, y=y_coords, yerr=collated_sorted["sem"], fmt="none",  elinewidth=2, capsize=4, color='black')
    plt.ylabel('Residence time (s)')
    plt.xlabel('')
    plt.legend(title='')
    fig.savefig(f'{save_loc}/mean_residence_withSEM_{y_axis}.svg', dpi=600)
    plt.show()


def plot_residence_time_of_class(df, binwidth, transition, save_log, plot_type='KDE', log=False):
    num_subplots = df['treatment'].nunique()
    if plot_type=='KDE':
        for transition, dfs in df.groupby('transition_type'):
            filt_trans = dfs[dfs['transition_type']== transition]
            sns.kdeplot(data=filt_trans, x='y_axis', hue='treatment', fill=False,  log_scale=True, common_norm=False, palette='mako')
            plt.title(transition)
            plt.show()
    if plot_type=='individual':
        for treatment, dfs in df.groupby('treatment'):
            fig, axes = plt.subplots(num_subplots, 1, figsize=(8, 18), sharex=True)
            axes = axes.flatten()
            for i, transition in enumerate(list(dfs['transition_type'].unique())):
                fig = sns.histplot(data=dfs[dfs['transition_type']==transition], x='y_axis', binwidth=binwidth, kde=True, stat='density', log_scale=log, ax=axes[i])
                axes[i].set_xlabel("Residence time before transition to 'bound' state (s)")
                axes[i].set_title(f'{treatment} and {transition}')
                plt.xlim(0, 50)
            plt.savefig(f'{save_log}/residence_time_histogram_{plot_type}.svg', dpi=600)
            plt.show()
    if plot_type=='collated':
        fig, axes = plt.subplots(num_subplots, 1, sharex=True)
        axes = axes.flatten()
        for i, treatment in enumerate(list(df['treatment'].unique())):
            dfs = df[df['treatment']==treatment]
            df2 = dfs[dfs['transition_type']==transition]
            fig = sns.histplot(
                data=df2, 
                hue='treatment', 
                x='y_axis',
                binwidth=binwidth, 
                stat='density', 
                log_scale=log, 
                ax=axes[i], 
                common_norm=False, 
                fill=False, 
                palette='mako')
            axes[i].set_xlabel("Residence time before transition to 'bound' state (s)")
            axes[i].set_title(f'{treatment}', loc='center')
            plt.xlim(0, 200)
        plt.savefig(f'{save_log}/residence_time_histogram_{plot_type}.svg', dpi=600)
        plt.show()
    if plot_type=='cum_dwell':
        fig, axes = plt.subplots(num_subplots, 1, sharex=True)
        axes = axes.flatten()
        for i, treatment in enumerate(list(df['treatment'].unique())):
            dfs = df[df['treatment']==treatment]
            df2 = dfs[dfs['transition_type']==transition]
            fig = sns.histplot(
                data=df2, 
                hue='treatment', 
                x='CumulativeTime(s)',
                binwidth=binwidth, 
                kde=True, 
                stat='density', 
                log_scale=log, 
                ax=axes[i], 
                common_norm=False, 
                fill=False, 
                palette='mako')
            axes[i].set_xlabel("Residence time before transition to 'bound' state (s)")
            axes[i].set_title(f'{treatment}')
            axes[i].legend('')
            plt.xlim(0, 200)
        plt.savefig(f'{save_log}/residence_time_histogram_{plot_type}.svg', dpi=600)
        plt.show()
    plt.show



def cumulative_hist_residence_plot(filt_df, fits_df, palette, save_loc, binwidth=5, xlim=300):
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
    treatment_list = sorted(list(filt_df['transition_type'].unique()))
    for i, transition in enumerate(treatment_list):
        sns.histplot(data=filt_df[filt_df['transition_type']==transition], 
                    x='CumulativeTime(s)', 
                    stat='density', 
                    cumulative=True, 
                    hue='treatment', 
                    fill=False, 
                    common_norm=False, 
                    element='step', 
                    binwidth=binwidth, 
                    palette=palette,
                    ax=axes[i])
        sns.lineplot(data=fits_df[fits_df['transition_type']==transition], x='x_bins', y='fit', hue='treatment', palette=palette, ax = axes[i])
    plt.xlabel('Residence time (s)')
    plt.ylabel('Cumulative density')
    plt.xlim(-5, xlim)
    plt.tight_layout()
    fig.savefig(f'{save_loc}/cumulative_histogram_residencetime.svg', dpi=600)
    plt.show()


def master_residence_time_func(output_folder, data_paths_violin, order, palette='BuPu', FRET_thresh=0.2, binwidth=10, cumulative_hist_binwidth=2, fit_xlim=300, plot_xlim=300):
    plot_export = f'{output_folder}/Residence_time_plots/'
    if not os.path.exists(plot_export):
        os.makedirs(plot_export)
    cumulative_dwell = pd.read_csv(f'{output_folder}/cumulative_dwell.csv')

    test=[]
    for data_name, data_path in data_paths_violin.items():
        data = uda.file_reader(data_path, 'other')
        compiled_data = ps.compiled(data, data_name, FRET_thresh)
        test.append(compiled_data)
    final = pd.concat(test)
    final["y_axis_log10"]=np.log10(final['y_axis']) ## if need to plot in log scale

    dict_for_label={
    f"< {FRET_thresh} to < {FRET_thresh}":'$T_{low-low}$', 
    f"< {FRET_thresh} to > {FRET_thresh}":'$T_{low-high}$',
    f"> {FRET_thresh} to > {FRET_thresh}":'$T_{high-high}$',
    f"> {FRET_thresh} to < {FRET_thresh}":'$T_{high-low}$'
    }

    final['transition_name'] = final['transition_type'].map(dict_for_label)
    plot_violin(final, plot_export, palette, 'split')

    # ---------------------- Generate and collate summary statistics of residence times ---------------------
    mean = final.groupby(['treatment', 'transition_type']).mean()
    sem =  final.groupby(['treatment', 'transition_type']).sem()
    final_drop = final.drop('transition_name', axis=1)
    N = final_drop.groupby(['treatment', 'transition_type']).count()
    collated = pd.concat([mean,sem, N], axis=1)
    collated.drop([col for col in collated.columns.tolist() if 'y_axis_log10' in col], axis=1, inplace=True)
    collated.columns = ['mean_residence_time', 'sem', 'n']
    collated.reset_index(inplace=True)
    collated.to_csv(f"{output_folder}/summary.csv", index=False)
    collated_filt = collated[(collated['transition_type']==f'< {FRET_thresh} to > {FRET_thresh}')|(collated['transition_type']==f'> {FRET_thresh} to < {FRET_thresh}')]
    final_filt = final[(final['transition_type']==f'< {FRET_thresh} to > {FRET_thresh}')|(final['transition_type']==f'> {FRET_thresh} to < {FRET_thresh}')]


    # -------------------- plot residence time as a conventional histogram ------------------------


    plot_bar_with_sem(final, collated, order,plot_export, 'y_axis',palette)
    plot_bar_with_sem(final_filt, collated_filt, order,plot_export, 'y_axis', palette)
    plot_residence_time_of_class(final, binwidth, f'< {FRET_thresh} to > {FRET_thresh}', plot_export, 'collated', False)

    # -------------------- Prepare cumulative dwell time data for plotting (i.e., convert to two-state system) ---------------------


    dict_for_label_cum_dwell={
    'low-low':'$T_{low-low}$', 
    'low-high':'$T_{low-high}$',
    'high-high':'$T_{high-high}$',
    'high-low':'$T_{high-low}$'
    }

    cumulative_dwell = cumulative_dwell[cumulative_dwell['treatment_name'].isin(order)]
    cumulative_dwell['transition_name'] = cumulative_dwell['transition_type'].map(dict_for_label_cum_dwell)
    mean = cumulative_dwell.groupby(['treatment_name', 'transition_type'])['CumulativeTime(s)'].mean()
    sem =  cumulative_dwell.groupby(['treatment_name', 'transition_type'])['CumulativeTime(s)'].sem()
    cumulative_dwell_drop = cumulative_dwell.drop('transition_name', axis=1)
    N = cumulative_dwell_drop.groupby(['treatment_name', 'transition_type'])['CumulativeTime(s)'].count()
    col_cum_dwell = pd.concat([mean,sem, N], axis=1)
    col_cum_dwell.drop([col for col in col_cum_dwell.columns.tolist() if 'y_axis_log10' in col], axis=1, inplace=True)
    col_cum_dwell.columns = ['mean_residence_time', 'sem', 'n']
    col_cum_dwell.reset_index(inplace=True)
    cumulative_dwell.columns = ['molecule', 'FRET_before', 'FRET_after', 'Time', 'number_of_frames', 'treatment', 'transition_type', 'shift', 'is_in_sequence', 'CumulativeTime', 'CumulativeTime(s)', 'transition_name']
    col_cum_dwell.columns = ['treatment', 'transition_type', 'mean_residence_time', 'sem', 'n']
    col_cum_dwell.to_csv(f"{output_folder}/summary_cum_dwell.csv", index=False)
    col_cum_dwell_filt = col_cum_dwell[(col_cum_dwell['transition_type']=='low-high')|(col_cum_dwell['transition_type']=='high-low')]
    cumulative_dwell_filt = cumulative_dwell[(cumulative_dwell['transition_type']=='low-high')|(cumulative_dwell['transition_type']=='high-low')]

    # -------------------- plot cumulative residence time as a conventional histogram ------------------------

    plot_bar_with_sem(cumulative_dwell_filt, col_cum_dwell_filt, order,plot_export,'CumulativeTime(s)', palette)
    plot_residence_time_of_class(cumulative_dwell_filt, binwidth, 'low-high', plot_export, 'cum_dwell', False)
    
    
    fits_df, halftime_summary = ps.cumulative_residence_fitting(cumulative_dwell_filt, 
                                                                    plot_export, 
                                                                    bin_width=cumulative_hist_binwidth, 
                                                                    xlim = fit_xlim, 
                                                                    func=ps.one_phase_association)


    cumulative_hist_residence_plot(cumulative_dwell_filt, 
                               fits_df, 
                               palette='BuPu', 
                               save_loc=plot_export,
                               binwidth=cumulative_hist_binwidth, 
                               xlim=plot_xlim)

    return final, collated, cumulative_dwell_filt, halftime_summary
