import pandas as pd
import numpy as np
import scipy.optimize

def compiled(df, data_name, FRET_thresh):
    """Will filter transitions dependent on a threshold defined above as FRET_thresh to calculate residenc time for each transition class

    Args:
        df (dataframe): dataset containing the residence times  for each treatment
        data_name (string): treatment name  

    Returns:
        dataframe: compiles all transition classes (with residence times) from all treatments together
    """
    violin_data_lowtolow = pd.DataFrame(df[f"< {FRET_thresh} to < {FRET_thresh}"])
    violin_data_lowtolow.columns = ["y_axis"]
    violin_data_lowtolow["transition_type"] = f"< {FRET_thresh} to < {FRET_thresh}"
    violin_data_lowtolow["treatment"] = data_name

    violin_data_lowtohigh = pd.DataFrame(df[f"< {FRET_thresh} to > {FRET_thresh}"])
    violin_data_lowtohigh.columns = ["y_axis"]
    violin_data_lowtohigh["transition_type"] = f"< {FRET_thresh} to > {FRET_thresh}"
    violin_data_lowtohigh["treatment"] = data_name

    violin_data_hightohigh = pd.DataFrame(df[f"> {FRET_thresh} to > {FRET_thresh}"])
    violin_data_hightohigh.columns = ["y_axis"]
    violin_data_hightohigh["transition_type"] = f"> {FRET_thresh} to > {FRET_thresh}"
    violin_data_hightohigh["treatment"] = data_name

    violin_data_hightolow = pd.DataFrame(df[f"> {FRET_thresh} to < {FRET_thresh}"])
    violin_data_hightolow.columns = ["y_axis"]
    violin_data_hightolow["transition_type"] = f"> {FRET_thresh} to < {FRET_thresh}"
    violin_data_hightolow["treatment"] = data_name
    return pd.concat(
        [
            violin_data_lowtolow,
            violin_data_lowtohigh,
            violin_data_hightohigh,
            violin_data_hightolow,
        ]
    )


def one_phase_association(x, Y0, Plateau, K):
    return Y0 + (Plateau-Y0)*(1-np.exp(-K*x))


def cumulative_residence_fitting(dfs, output_folder, bin_width, xlim,func=one_phase_association):
    """Function is used to fit cumulative histogram data with a one-phase association curve. The script will create bins from the raw data and create a cumulative histogram, which is then
    used to fit the curve to the data. Will return the fit (with half time, plateua, etc) and a an Rsquared value to provide a measure of goodness of fit. 

    Args:
        dfs (df): dataframe containing raw data to be used for fitting. 
        output_folder (str): where to save data.
        bin_width (float): bin_width used to calculate the fit. Recommended to use smaller bin_widths (especially if data is tightly distributed at low values), but note
                            smaller bin_widths will reduce the number of datapoints in each bin.
        xlim (float): value used to determine how far the fit will extend to. Recommended to extend to max possible bin value.
        func (float, optional): decide here what fit to use. Defaults to one_phase_association. Can call another fit as long as it has been previously defined in a function.

    Returns:
        df: returns the fits and also the summary data (containing the half-time for each treatment and residence time state).
    """
    data = []
    summary = []
    for (treatment, transition), df in dfs.groupby(['treatment', 'transition_type']):
        bin_width = bin_width
        bin_edges = range(0, xlim+1, bin_width)

        # Bin the 'CumulativeTime(s)' column
        bins = pd.cut(df['CumulativeTime(s)'], bins=bin_edges, right=False)  # Set right=False to include the rightmost edge

        # Count the number of values in each bin
        bin_counts = bins.value_counts().sort_index()

        # Calculate cumulative count
        cumulative_counts = bin_counts.cumsum()

        bin_edges_array = bin_edges[:-1] 
        # Exclude the last edge to match the length of bin_counts
        cumulative_counts_array = cumulative_counts.values
        cumulative_counts_array = cumulative_counts_array/cumulative_counts_array.max()
        # plt.show()

        # perform the fit
        p0 = (20, 100, 0.1) # start with values near those we expect
        params, cv = scipy.optimize.curve_fit(func, bin_edges_array, cumulative_counts_array, p0)
        Y0, Plateau, K = params
        tauSec = (1 / K) 
        half_time = np.log(2)/K

        # Calculate standard errors from the covariance matrix
        se_half_time = np.sqrt(np.diag(cv))[2] * np.log(2) / K**2

        # Calculate confidence interval for the half-time (assuming normal distribution)
        alpha = 0.05  # significance level
        z_score = scipy.stats.norm.ppf(1 - alpha / 2)  # two-tailed z-score
        ci_half_time = (half_time - z_score * se_half_time, half_time + z_score * se_half_time)
        n_value = cumulative_counts.max()


        print("Estimated Half-Time:", half_time)
        print("Standard Error of Half-Time:", se_half_time)
        print("95% Confidence Interval of Half-Time:", ci_half_time)

        # determine quality of the fit
        squaredDiffs = np.square(cumulative_counts_array - func(bin_edges_array, Y0, Plateau, K))
        squaredDiffsFromMean = np.square(cumulative_counts_array - np.mean(cumulative_counts_array))
        rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

        # inspect the parameters
        print(f"R² = {rSquared}")
        print(f"Y = {Y0}")
        print(f'Plateau = {Plateau}')
        print(f'K = {K}')
        print(f'half-time = {half_time} s')
        print(f"Tau = {tauSec} s")

        fitted_data = func(bin_edges_array, Y0, Plateau, K)
        fitted_data_df = pd.DataFrame(fitted_data)
        x_bins = pd.DataFrame(bin_edges_array)
        test = pd.DataFrame(cumulative_counts_array)
        test = pd.concat([test, fitted_data_df, x_bins],axis=1)
        test.columns = ['Cumative_hist', 'fit', 'x_bins']
        test['treatment'] = treatment
        test['transition_type'] = transition
        data.append(test)


        col = [half_time,se_half_time, n_value, treatment, transition]
        col_halftime_df = pd.DataFrame([col], columns=['mean', 'sem', 'n', 'treatment', 'transition'])
        summary.append(col_halftime_df)



    fits_df = pd.concat(data, ignore_index=True)
    halftime_summary = pd.concat(summary, ignore_index=True)
    halftime_summary.to_csv(f'{output_folder}/halftime_summary.csv')
    return fits_df, halftime_summary