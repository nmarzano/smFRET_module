import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from lmfit import models
import src.Utilities.Data_analysis as uda


def fit_3gauss_dif_constrained_nativespont(df, treatment, save_loc, mu_1, sigma_1, amplitude_1, gamma_1, mu_2, sigma_2, amplitude_2, mu_3, sigma_3, amplitude_3, gamma_3):
    filt_df = df[df['treatment_name'] == treatment]
    bins = np.arange(-0.21, 1.1, 0.025) 
    inds = np.digitize(filt_df['FRET'].astype(float), bins)
    xdata, ydata = np.unique(inds, return_counts=True)
    ydata = ydata[1:-1] #### trim off outside range bins at the end
    xdata = [np.mean(bins[x : x + 2]) for x in range(len(bins)- 1)]  ##### convert bin edges to bin centres, therefore end up with one less bin
    sns.lineplot(xdata, ydata)

    model_1 = models.SkewedGaussianModel(prefix='m1_')
    model_2 = models.GaussianModel(prefix='m2_')
    model_3 = models.SkewedGaussianModel(prefix='m3_')
    model = model_1 + model_2 + model_3 
   
    model_1.set_param_hint('m1_center', vary=False)

    model_2.set_param_hint('m2_sigma', vary=False)

    model_2.set_param_hint('m2_center', vary=False)
    model_3.set_param_hint('m3_gamma', vary=False)
    model_3.set_param_hint('m3_sigma', vary=False)
    model_3.set_param_hint('m3_center', vary=False)


    params_1 = model_1.make_params(center=mu_1, sigma=sigma_1, amplitude=amplitude_1, gamma=gamma_1, min=0)
    params_2 = model_2.make_params(center=mu_2, sigma=sigma_2, amplitude=amplitude_2, min=0)
    params_3 = model_3.make_params(center=mu_3, sigma=sigma_3, amplitude=amplitude_3, gamma=gamma_3, min=0)
    params = params_1.update(params_2)
    params = params.update(params_3)

    output = model.fit((ydata/np.max(ydata)), params, x=xdata)
    fig = output.plot(data_kws={'markersize': 3})

    paramaters = {name:output.params[name].value for name in output.params.keys()}
    fitx = np.arange(-0.2, 1.2, 0.025)

    fit1 = model_1.eval(x=fitx, center=paramaters['m1_center'], amplitude=abs(paramaters['m1_amplitude']), sigma=paramaters['m1_sigma'], gamma=paramaters['m1_gamma'])
    fit2 = model_2.eval(x=fitx, center=paramaters['m2_center'], amplitude=abs(paramaters['m2_amplitude']), sigma=paramaters['m2_sigma'], fwhm=paramaters['m2_fwhm'])
    fit3 = model_3.eval(x=fitx, center=paramaters['m3_center'], amplitude=abs(paramaters['m3_amplitude']), sigma=paramaters['m3_sigma'], gamma=paramaters['m3_gamma'])

    sns.lineplot(fitx, fit1)
    sns.lineplot(fitx, fit2)
    sns.lineplot(fitx, fit3)
    plt.show()

    # Calculate area under the curve for each gaussian
    aoc_m1 = paramaters['m1_amplitude']
    aoc_m2 = paramaters['m2_amplitude']
    aoc_m3 = paramaters['m3_amplitude']
    sum_aoc = aoc_m1 + aoc_m2 + aoc_m3 

    aoc_m1_percent_of_total = (aoc_m1/sum_aoc)*100
    aoc_m2_percent_of_total = (aoc_m2/sum_aoc)*100
    aoc_m3_percent_of_total = (aoc_m3/sum_aoc)*100
    list_of_gaus_proportion = [aoc_m1_percent_of_total, aoc_m2_percent_of_total, aoc_m3_percent_of_total]
    labels_of_gaus_proportion = ['m1', 'm2', 'm3']
    proportion_df = pd.DataFrame([labels_of_gaus_proportion, list_of_gaus_proportion])
    proportion_df.columns = proportion_df.iloc[0]
    proportion_df = proportion_df.drop(0)
    proportion_df['treatment'] = treatment
    proportion_df.to_csv(f'{save_loc}/gaussian_proportions_for_{treatment}.csv')
    return proportion_df


def fit_2gauss_dif_constrained_nativespont(df, treatment, save_loc, mu_1, sigma_1, amplitude_1, gamma_1, mu_2, sigma_2, amplitude_2):
    filt_df = df[df['treatment_name'] == treatment]
    bins = np.arange(-0.21, 1.1, 0.025) 
    inds = np.digitize(filt_df['FRET'].astype(float), bins)
    xdata, ydata = np.unique(inds, return_counts=True)
    ydata = ydata[1:-1] #### trim off outside range bins at the end
    xdata = [np.mean(bins[x : x + 2]) for x in range(len(bins)- 1)]  ##### convert bin edges to bin centres, therefore end up with one less bin
    sns.lineplot(xdata, ydata)

    model_1 = models.SkewedGaussianModel(prefix='m1_')
    model_2 = models.GaussianModel(prefix='m2_')

    model = model_1 + model_2
   
    model_1.set_param_hint('m1_center', vary=False)
    model_2.set_param_hint('m2_sigma', vary=False)
    model_2.set_param_hint('m2_center', vary=False)



    params_1 = model_1.make_params(center=mu_1, sigma=sigma_1, amplitude=amplitude_1, gamma=gamma_1, min=0)
    params_2 = model_2.make_params(center=mu_2, sigma=sigma_2, amplitude=amplitude_2, min=0)
    params = params_1.update(params_2)

    output = model.fit((ydata/np.max(ydata)), params, x=xdata)
    fig = output.plot(data_kws={'markersize': 3})

    paramaters = {name:output.params[name].value for name in output.params.keys()}
    fitx = np.arange(-0.2, 1.2, 0.025)

    fit1 = model_1.eval(x=fitx, center=paramaters['m1_center'], amplitude=abs(paramaters['m1_amplitude']), sigma=paramaters['m1_sigma'], gamma=paramaters['m1_gamma'])
    fit2 = model_2.eval(x=fitx, center=paramaters['m2_center'], amplitude=abs(paramaters['m2_amplitude']), sigma=paramaters['m2_sigma'], fwhm=paramaters['m2_fwhm'])

    sns.lineplot(fitx, fit1)
    sns.lineplot(fitx, fit2)
    plt.show()

    # Calculate area under the curve for each gaussian
    aoc_m1 = paramaters['m1_amplitude']
    aoc_m2 = paramaters['m2_amplitude']


    sum_aoc = aoc_m1 + aoc_m2 

    aoc_m1_percent_of_total = (aoc_m1/sum_aoc)*100
    aoc_m2_percent_of_total = (aoc_m2/sum_aoc)*100
    list_of_gaus_proportion = [aoc_m1_percent_of_total, aoc_m2_percent_of_total]
    labels_of_gaus_proportion = ['m1', 'm2']
    proportion_df = pd.DataFrame([labels_of_gaus_proportion, list_of_gaus_proportion])
    proportion_df.columns = proportion_df.iloc[0]
    proportion_df = proportion_df.drop(0)
    proportion_df['treatment'] = treatment
    proportion_df.to_csv(f'{save_loc}/gaussian_proportions_for_{treatment}.csv')
    return proportion_df

# -------------------------------- MASTER FUNCTION -----------------------------------------

def fit_gauss_master(input_folder, gauss_num='two',timepoint=None):
    output_folder = f"{input_folder}/GaussianFits"  ### modify for each experiment
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = f'{input_folder}/Cleaned_FRET_histogram_data.csv'
    compiled_df = pd.read_csv(filename, header="infer")
    treatment_list = list(compiled_df['treatment_name'].unique())
    collated = []
    if gauss_num == 'three':
        for i, df in enumerate(treatment_list):
            treatment = fit_3gauss_dif_constrained_nativespont(compiled_df, df, output_folder, 0.05, .1, 1, 1, .63, .1, .05, .95, .22, .03, -2.7)
            collated.append(treatment)
        collated_df = pd.concat(collated)
    else:
        for i, df in enumerate(treatment_list):
            treatment = fit_2gauss_dif_constrained_nativespont(compiled_df, df, output_folder, 0.05, .1, 1, 1, .63, .1, .05)
            collated.append(treatment)
        collated_df = pd.concat(collated)
    # ----------- Set timepoints to plot based on your conditions --------------
    # ----------- Cleans up data and melts it for plotting ---------------------
    if timepoint:
        collated_df['timepoint'] = collated_df['treatment'].map(timepoint)
        collated_df.drop('treatment', axis=1, inplace=True)
        collated_df.columns = ['DnaK-bound', 'Native', 'Misfolded', 'timepoint']
        test = pd.melt(collated_df, id_vars='timepoint', value_name='pop_percent')
        test['pop_percent'] = test['pop_percent'].astype(float)
        test.to_csv(f'{output_folder}/collated_populations.csv')
        return test, collated_df, output_folder
    return collated_df, output_folder


def plot_gauss_timelapse(data_from_exp, output_folder):
    data_col = []
    for data_name, data_path in data_from_exp.items():
        data = uda.file_reader(data_path, 'other')
        data['treatment'] = data_name
        data_col.append(data)
    final = pd.concat(data_col).reset_index()
    final = final.iloc[:, 2:]


    for pop, df in final.groupby('variable'):
        fig, ax = plt.subplots()
        sns.set_style('ticks',{'font_scale': 1.5})
        sns.lineplot(data=df, x='timepoint', y='pop_percent', hue='treatment', palette='BuPu', marker='o')
        plt.legend(title='', loc='best')
        plt.ylim(0, 100)
        plt.xlabel('Time (min)')
        plt.ylabel(f'{pop} (% of total)')
        [x.set_linewidth(1.5) for x in ax.spines.values()]
        [x.set_color('black') for x in ax.spines.values()]
        fig.savefig(f'{output_folder}/Proportion_of_{pop}.svg', dpi=600)
        plt.show()
    return final
