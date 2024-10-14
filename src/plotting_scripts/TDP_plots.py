import pandas as pd
import numpy as np
from smfret.src.processing_scripts import TDP_processing as ps
import seaborn as sns
import matplotlib.pyplot as plt
import os as os

def tdp_plot(treatment):
    plt.rcParams['svg.fonttype']='none'
    plot1 = plt.figure(figsize=(6, 6))
    plot1 = sns.JointGrid(data=treatment, x=treatment["FRET_before"], y=treatment["FRET_after"], xlim=(0,1), ylim=(0, 1))
    plot1.plot_joint(sns.kdeplot, cmap="BuPu", shade=bool, cbar=False, cbar_kws={'format': '%.0f%%', 'ticks': [0, 100]}, thresh=0.05, gridsize=100)
    plot1.plot_joint(sns.kdeplot, thresh=0.05, gridsize=100, color='black')
    plot1.ax_joint.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plot1.ax_joint.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plot1.plot_marginals(sns.histplot, kde=True, bins=10, color='black', fill=False)
    sns.set(style='ticks', font_scale=1.5)
    plt.show()
    return plot1

# -------------------------------- MASTER FUNCTION -----------------------------------------

def master_tdp_cleanup_func(output_folder='Figure3b-overhangs_9-10-11-22-only_col/python_results', exposure=0.2, FRET_value_for_classes=0.5,FRET_to_filt_tdp=0.3):
    compiled_data = pd.read_csv(f'{output_folder}/Cleaned_FRET_histogram_data.csv')

    compiled_filt = []
    for treatment, df in compiled_data.groupby('treatment_name'):
        treatment_df = compiled_data[compiled_data['treatment_name']==treatment]
        treatment_df2 = treatment_df.filter(items=['idealized FRET','unique_id'])
        treatment_dwell = ps.calculate_dwell_time(treatment_df2)
        treatment_trans = ps.generate_transitions(treatment_dwell)
        treatment_cleaned = ps.remove_outliers_tdp(treatment_trans)
        treatment_cleaned["treatment_name"] = treatment
        compiled_filt.append(treatment_cleaned)
    compiled_TDP = pd.concat(compiled_filt)
    compiled_TDP['repeat'] = compiled_TDP['Molecule'].str.split('_').str[-1]
    compiled_TDP.to_csv(f'{output_folder}/TDP_cleaned.csv', index=False)


    # -------------- Generate a dataset whereby only molecules that go below a defined threshold are included ---------------------


    filtered_data = ps.filter_tdp(compiled_TDP, FRET_to_filt_tdp)  ##### number refers to the FRET threshold to filter data. Will only include molecules that go below this set threshold. Will default to 0.3 
    filtered_data.to_csv(f'{output_folder}/TDP_cleaned_filt.csv', index=False)

    proportion_mol_below_thresh = pd.DataFrame(ps.count_filt_mol(compiled_TDP, FRET_value_for_classes).groupby('treatment_name')['proportion_mol_below_thresh'].mean())
    proportion_mol_below_thresh['norm_percent_mol'] = proportion_mol_below_thresh['proportion_mol_below_thresh'] - proportion_mol_below_thresh['proportion_mol_below_thresh'].iloc[0]
    proportion_mol_below_thresh.reset_index()
    proportion_mol_below_thresh.to_csv(f'{output_folder}/mol_below_{FRET_value_for_classes}.csv', index=None)

    # -------------- Define multiple conditions and corresponding values -----------------------
    conditions = [
    (compiled_TDP['FRET_before'] < FRET_value_for_classes) & (compiled_TDP['FRET_after'] > FRET_value_for_classes),
    (compiled_TDP['FRET_before'] < FRET_value_for_classes) & (compiled_TDP['FRET_after'] < FRET_value_for_classes),
    (compiled_TDP['FRET_before'] > FRET_value_for_classes) & (compiled_TDP['FRET_after'] > FRET_value_for_classes),
    (compiled_TDP['FRET_before'] > FRET_value_for_classes) & (compiled_TDP['FRET_after'] < FRET_value_for_classes),
]

    values = ['low-high', 'low-low', 'high-high', 'high-low']

    # ------------- Create a new column based on the conditions and values ----------------
    compiled_TDP['transition_type'] = np.select(conditions, values)

    combined_data = []
    for (treatment, molecule), df in compiled_TDP.groupby(['treatment_name', 'Molecule']):
        test = ps.determine_if_in_sequence(df)
        test2 = ps.determine_cumulative_sum_in_sequence(test, exposure)
        combined_data.append(test2)
    cumulative_dwell_transitions = pd.concat(combined_data)
    cumulative_dwell_transitions.to_csv(f'{output_folder}/cumulative_dwell.csv', index=False)
    return compiled_TDP


def master_TDP_plot(input_folder='Experiment_1-description/python_results', filt=True):
    output_folder = f'{input_folder}/TDP_plots'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # --------- import cleaned data -----------------
    if filt == False:
        filename = f'{input_folder}/TDP_cleaned.csv'
    else:
        filename = f'{input_folder}/TDP_cleaned_filt.csv'
    TDP = pd.read_csv(filename, header="infer")
    for treatment, df in TDP.groupby('treatment_name'):
        treatments = TDP[TDP["treatment_name"] == treatment]
        tdp_plot(treatments).savefig(f"{output_folder}/TDP_plot_{treatment}.svg", dpi=600)
