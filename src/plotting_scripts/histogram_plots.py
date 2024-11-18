import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from smfret.src.Utilities import Data_analysis as util
import pandas as pd
from smfret.src.processing_scripts import timelapse_hist_processing as ps


def ridgeline_plot(df, save_loc, dict_key, font, palette='BuPu'):
    # Sets order for ridgeline plot
    ridgeline_keys = list(range(1,len(dict_key)+1))
    str_ridgeline_keys = list(map(str, ridgeline_keys))
    data_paths_ridgeline =  dict(zip(str_ridgeline_keys,dict_key))

    n_colors = len(data_paths_ridgeline)
    pal = sns.color_palette(palette=palette, n_colors=n_colors)
    matplotlib.rc('font', **font)
    plt.rcParams['svg.fonttype'] = 'none'
    g = sns.FacetGrid(df, row='treatment_name', hue='treatment_name', aspect=10, height=5, palette=pal)
    # then we add the densities kdeplots for each condition
    g.map(sns.kdeplot, 'FRET',
      bw_adjust=1, clip_on=False,
      fill=True, alpha=1, linewidth=10)
    # here we add a black line that represents the contour of each kdeplot
    g.map(sns.kdeplot, 'FRET', 
      bw_adjust=1, clip_on=False, 
      color="black", linewidth=10)
    # here we add a horizontal line for each plot
    g.map(plt.axhline, y=0,
      lw=2, clip_on=False)
    # we loop over the FacetGrid figure axes (g.axes.flat) and add the condition as text with the right color
    # notice how ax.lines[-1].get_color() enables you to access the last line's color in each matplotlib.Axes
    for i, ax in enumerate(g.axes.flat):
        ax.text(-0.5, .5, data_paths_ridgeline[f'{i+1}'],
            fontweight='bold', fontsize=100,
            color=ax.lines[-1].get_color())
        ax.set_facecolor((0, 0, 0, 0))  ### removes background so KDE plots are not obstructed when stacked
    # we use matplotlib.Figure.subplots_adjust() function to get the subplots to overlap
    g.fig.subplots_adjust(hspace=-0.7)
    # eventually we remove axes titles, yticks and spines
    g.set_titles("")
    g.set(yticks=([]))
    g.despine(bottom=True, left=True)
    plt.setp(ax.get_xticklabels(), fontsize=100, fontweight='bold')
    plt.xlabel('FRET', fontweight='bold', fontsize=100)
    plt.xlim(-.4, 1.2)
    g.savefig(f'{save_loc}/Histogram-ridgeline.svg', dpi=600)
    plt.show()

# ---------------------- Code to plot regular histogram -------------------------------

def plot_hist_type(df, order, labels, save_loc, kind='bar'):
    plot_hist, ax = plt.subplots()
    sns.set_style("ticks",{'font_scale':1} )
    if kind == 'kde':
        sns.kdeplot(
            data=df, 
            palette='BuPu', 
            x="FRET",
            hue="treatment_name",
            hue_order=order,
            common_norm=False, 
            fill=True, 
            linewidth=1.5, 
            alpha=0.25)
    if kind == 'bar':
        sns.histplot(
            data=df, 
            palette='BuPu_r', 
            x="FRET",
            hue="treatment_name",
            hue_order=order,
            common_norm=False, 
            stat='density',
            binwidth=0.05,
            fill=False, 
            kde=True,
            linewidth=1.5, 
            alpha=0.25, 
            element='step')
    plt.xlim(0, 1, 10)
    plt.xlabel("FRET")
    plt.legend([labels[treatment] for treatment in reversed(order)], loc='best', fontsize=12, title='')    
    [x.set_linewidth(2) for x in ax.spines.values()]
    [x.set_color('black') for x in ax.spines.values()]
    plot_hist.savefig(f'{save_loc}/Histogram_{kind}.svg', dpi=600)
    plt.show()

# --------------------- code to calculate and plot the proportion of time each molecules spends below a defined threshold ------------------------

def plot_time_below_thresh(df, order, thresh, save_loc, swarmplot=False, palette='BuPu'):
    fig, ax = plt.subplots()
    sns.violinplot(data=df,
                y='FRET_time_below_thresh',
                x='treatment_name',
                palette=palette,
                scale='width',
                order=order)
    if swarmplot:
        sns.swarmplot(data=df,
                    y='FRET_time_below_thresh',
                    x='treatment_name',
                    color='grey',
                    order=order,
                    alpha=0.25)
    plt.xlabel('')
    plt.ylabel(f'Proportion of time spent < {thresh} FRET')
    fig.savefig(f'{save_loc}/Time_below_thresh_per_molecule.svg', dpi=600)
    plt.xticks(rotation=45)
    plt.show()

# -------------------------------- MASTER FUNCTION -----------------------------------------

def master_histogram_func(data_paths, output_folder="Experiment_X-description/python_results", thresh=0.2, swarmplot=False):
    if isinstance(data_paths, dict):    # --------- the first item in the tuple will be the name that goes into the graph legend -------------
        dict_key = list(data_paths.keys())
        # -------- Data from all data sets in the dict will be imported and concatenated into a single dataframe. Outliers wil be removed -----------
        compiled_df = []
        for data_name, (label, data_path) in data_paths.items():
            imported_data = util.file_reader(data_path, 'hist')
            cleaned_raw = util.remove_outliers(imported_data, 'hist') # add "idealized" after imported_data to get idealized histograms
            cleaned_raw["treatment_name"] = data_name
            compiled_df.append(cleaned_raw)
        compiled_df = pd.concat(compiled_df)   #### .rename(columns = {1:"test", 3:"test2"}) ## can rename individually if needed
        compiled_df.columns = ["frames", "donor", "acceptor", "FRET", "idealized FRET", 'molecule_number', "treatment_name"]
        compiled_df.to_csv(f'{output_folder}/Cleaned_FRET_histogram_data.csv', index=False)
        labels = {data_name:label for data_name, (label, data_path) in data_paths.items()}
        order = list(reversed(dict_key))

    else:
        compiled_df = pd.read_csv(f'{output_folder}/Cleaned_FRET_histogram_data.csv')
        labels = {val: val for val in compiled_df['treatment_name'].unique()}
        order = list(reversed(labels.keys()))

    # ------------------------- To plot ridgeline histograms -------------------------

    font = {'weight' : 'normal', 'size'   : 12 }
    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"
    matplotlib.rc('font', **font)
    plt.rcParams['svg.fonttype'] = 'none'
    ridgeline_plot(compiled_df, output_folder, labels, font, 'BuPu')


    plot_hist_type(compiled_df, order, labels, output_folder,kind='bar')
    filt_df = []
    order_normal = list(reversed(order))
    for (treatment, molecule), df in compiled_df.groupby(['treatment_name', 'unique_id']):
        mol_total = len(df)
        mol_below_thresh = len(df[df['FRET']<thresh])
        percent_below_thresh = (mol_below_thresh/mol_total)*100
        df['FRET_time_below_thresh'] = percent_below_thresh
        df = df.iloc[[0]]
        filt_df.append(df)
    filt_dfs = pd.concat(filt_df)
    filt_dfs.to_csv(f'{output_folder}/filt_dfs.csv')
    plot_time_below_thresh(filt_dfs, order_normal, thresh, save_loc=output_folder, palette='BuPu', swarmplot=swarmplot)
    return compiled_df, filt_dfs


