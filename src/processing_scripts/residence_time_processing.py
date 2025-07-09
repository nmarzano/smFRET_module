import pandas as pd
import numpy as np
import scipy.optimize
import seaborn as sns
import matplotlib.pyplot as plt

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

def MLE(x,params):
    k1,k2,frac = params
    if not (0 < frac < 1):
        return np.inf
    yPred = frac * k1 * np.exp(k1 * x) + (1 - frac) * k2 * np.exp(k2 * x)
    # Avoid zero or negative predictions
    yPred = np.clip(yPred, 1e-12, None)
    negLL = -np.sum(np.log(yPred))
    return negLL

def double_expon(x, a,b,c,d):
    return a*np.exp(b*x)+c*np.exp(d*x)

def cumulative_residence_fitting(dfs, output_folder, bin_width, xlim, func=one_phase_association):
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
        print(treatment)
        print(transition)
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
        fits_output = []
        if func == 'both':
            print('single-fit first')
            p0 = (20, 100, 0.1)  # initial guess
            params, cv = scipy.optimize.curve_fit(one_phase_association, bin_edges_array, cumulative_counts_array, p0)
            Y0, Plateau, K = params
            tauSec = (1 / K)
            half_time = np.log(2) / K
            # Calculate standard error of half-time
            se_half_time = np.sqrt(np.diag(cv))[2] * np.log(2) / K**2
            # 95% confidence interval
            alpha = 0.05
            z_score = scipy.stats.norm.ppf(1 - alpha / 2)
            ci_half_time = (half_time - z_score * se_half_time, half_time + z_score * se_half_time)
            n_value = cumulative_counts.max()
            print(treatment)
            print(transition)
            print('bruh')
            print("Estimated Half-Time:", half_time)
            print("Standard Error of Half-Time:", se_half_time)
            print("95% Confidence Interval of Half-Time:", ci_half_time)
            # R² of single fit
            squaredDiffs = np.square(cumulative_counts_array - one_phase_association(bin_edges_array, Y0, Plateau, K))
            squaredDiffsFromMean = np.square(cumulative_counts_array - np.mean(cumulative_counts_array))
            rSquared = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)

            fitted_data = one_phase_association(bin_edges_array, Y0, Plateau, K)
            fitted_data_df = pd.DataFrame(fitted_data)
            bin_edges = np.arange(0, xlim + bin_width, bin_width)
            bin_centers = bin_edges[:-1] + bin_width / 2
            res_single = cumulative_counts_array - fitted_data
            x_bins = pd.DataFrame(bin_edges_array)
            test_single = pd.DataFrame(cumulative_counts_array)
            test_single = pd.concat([test_single, fitted_data_df, x_bins, pd.DataFrame(res_single)], axis=1)
            test_single.columns = ['Cumative_hist_sing', 'fit_sing', 'x_bins_sing', 'residuals_sing']
            test_single['treatment'] = treatment
            test_single['transition_type'] = transition
            data.append(test_single)
            print(f"MLE (CDF) fitting for {treatment} {transition}")
            # MLE part
            dwell_times = df["CumulativeTime(s)"].values
            dwell_times = dwell_times[dwell_times > 0]
            def CDF_mixture(x, k1, k2, frac):
                return frac * (1 - np.exp(-k1 * x)) + (1 - frac) * (1 - np.exp(-k2 * x))
            def MLE_cdf(params, x):
                k1, k2, frac = params
                if not (0 < frac < 1):
                    return np.inf
                if k1 <= 0 or k2 <= 0:
                    return np.inf
                pdf_vals = frac * k1 * np.exp(-k1 * x) + (1 - frac) * k2 * np.exp(-k2 * x)
                pdf_vals = np.clip(pdf_vals, 1e-12, None)
                return -np.sum(np.log(pdf_vals))
            initial_guess = [1/20, 1/300, 0.5]
            bounds = [(1e-4, 10), (1e-4, 10), (0.01, 0.99)]
            result = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(dwell_times,), bounds=bounds)
            k1, k2, frac = result.x
            print(f"Fitted: k1={k1:.4f}, k2={k2:.4f}, frac={frac:.4f}")
                        # ----------- BOOTSTRAP TO ESTIMATE STANDARD ERRORS -----------
            n_bootstrap = 500
            bootstrap_frac = []
            bootstrap_k1 = []
            bootstrap_k2 = []
            for i in range(n_bootstrap):
                boot_sample = np.random.choice(dwell_times, size=len(dwell_times), replace=True)
                result_boot = scipy.optimize.minimize(MLE_cdf, initial_guess, args=(boot_sample,), bounds=bounds)
                if result_boot.success:
                    k1_boot, k2_boot, frac_boot = result_boot.x
                    bootstrap_frac.append(frac_boot)
                    bootstrap_k1.append(k1_boot)
                    bootstrap_k2.append(k2_boot)
            bootstrap_frac = np.array(bootstrap_frac)
            bootstrap_k1 = np.array(bootstrap_k1)
            bootstrap_k2 = np.array(bootstrap_k2)
            # Calculate bootstrap statistics
            def bootstrap_summary(values, name):
                mean = np.mean(values)
                std_error = np.std(values, ddof=1)
                ci_lower = np.percentile(values, 2.5)
                ci_upper = np.percentile(values, 97.5)
                print(f"Bootstrap {name}: mean={mean:.4f}, std error={std_error:.4f}, 95% CI=({ci_lower:.4f}, {ci_upper:.4f})")
                return mean, std_error, ci_lower, ci_upper
            frac_mean_bootstrap, frac_std_error, frac_ci_lower, frac_ci_upper = bootstrap_summary(bootstrap_frac, 'frac')
            k1_mean_bootstrap, k1_std_error, k1_ci_lower, k1_ci_upper = bootstrap_summary(bootstrap_k1, 'k1')
            k2_mean_bootstrap, k2_std_error, k2_ci_lower, k2_ci_upper = bootstrap_summary(bootstrap_k2, 'k2')
            # ----------- END BOOTSTRAP -----------
            bin_edges = np.arange(0, xlim + bin_width, bin_width)
            bin_centers = bin_edges[:-1] + bin_width / 2
            hist, _ = np.histogram(dwell_times, bins=bin_edges)
            cum_counts = np.cumsum(hist)
            cum_norm = cum_counts / cum_counts[-1]
            fitted_cdf = CDF_mixture(bin_centers, k1, k2, frac)
            # R² of double fit
            ss_res = np.sum((cum_norm - fitted_cdf) ** 2)
            ss_tot = np.sum((cum_norm - np.mean(cum_norm)) ** 2)
            r_squared_double = 1 - ss_res / ss_tot
            print(f'{treatment} R2 is {r_squared_double}')
            # Residuals plot
            res = cum_norm - fitted_cdf
            fig = plt.figure(figsize=(6, 2))
            plt.scatter(bin_centers, res_single, label='single_fit', s=8, c='#f05423')
            plt.scatter(bin_centers, res, label='double_fit', s=8, c='#00aeef')
            plt.axhline(0, color='gray', linestyle='--')
            plt.title(f"Residuals for {treatment} {transition} ")
            plt.xlabel("Resident time (s)")
            plt.ylabel("Residual")
            plt.xlim(0, 100)
            plt.ylim(-0.25, 0.25)
            plt.legend()
            plt.show()
            fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.png", dpi=300)
            fig.savefig(f"{output_folder}/residuals_{treatment}_{transition}.svg", dpi=600)
            # Half-times
            half_time_fast = np.log(2) / k1
            half_time_slow = np.log(2) / k2
            fit_df = pd.DataFrame({
                'x_bins': bin_centers,
                'fit': fitted_cdf,
                'Cumative_hist': cum_norm,
                'treatment': treatment,
                'transition_type': transition
            })
            data.append(fit_df)
            col_df = pd.DataFrame([[half_time, se_half_time, n_value, treatment, transition, K, rSquared,
                        half_time_fast, half_time_slow, k1, k2, frac, frac_std_error, r_squared_double,
                        k1_std_error, k2_std_error]],
                    columns=['mean', 'sem', 'n', 'treatment', 'transition', 'K', 'r_squared',
                            'half_time_fast', 'half_time_slow', 'k1', 'k2', 'frac_fast', 'frac_std_error', 'R2',
                            'k1_std_error', 'k2_std_error'])
            summary.append(col_df)
            
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            # Bootstrap plot for frac
            sns.histplot(bootstrap_frac, bins=30, kde=True, ax=axs[0], color="#939990FF", edgecolor='black')
            axs[0].axvline(frac, color='black', linestyle='--', label='Original Fit')
            axs[0].set_title('Bootstrap Distribution of frac')
            axs[0].set_xlabel('frac')
            axs[0].legend()
            # Bootstrap plot for k1
            sns.histplot(bootstrap_k1, bins=30, kde=True, ax=axs[1], color="#386B20FF", edgecolor='black')
            axs[1].axvline(k1, color='black', linestyle='--', label='Original Fit')
            axs[1].set_title('Bootstrap Distribution of k1')
            axs[1].set_xlabel('k1 (1/s)')
            axs[1].legend()
            # Bootstrap plot for k2
            sns.histplot(bootstrap_k2, bins=30, kde=True, ax=axs[2], color="#81b868df", edgecolor='black')
            axs[2].axvline(k2, color='black', linestyle='--', label='Original Fit')
            axs[2].set_title('Bootstrap Distribution of k2')
            axs[2].set_xlabel('k2 (1/s)')
            axs[2].legend()
            plt.tight_layout()
            plt.show()
            # Optional: save the figure
            fig.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.png", dpi=300)   
            fig.savefig(f"{output_folder}/bootstrap_distributions_{treatment}_{transition}.svg", dpi=600)   
        else:
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


            col = [half_time, se_half_time, n_value, rSquared, treatment, transition]
            col_halftime_df = pd.DataFrame([col], columns=['mean', 'sem', 'n', 'r_squared', 'treatment', 'transition'])
            summary.append(col_halftime_df)

    fits_df = pd.concat(data, ignore_index=True)
    halftime_summary = pd.concat(summary, ignore_index=True)
    return fits_df, halftime_summary
