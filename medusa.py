import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from typing import Union, List

def create_list_of_files(extension, directory=os.getcwd()):
    """
    Finds all files with the specified extension in the given directory.

    Args:
        directory (str): The directory path. Default current working directory.
        extension (str): The desired file extension (e.g., '.txt', '.py').

    Returns:
        list: A list of filenames with the specified extension.
    """
    files = [file for file in os.listdir(directory) if file.endswith(extension)]
    files.sort()
    return files

def create_csv(input_files):
    """
    Creates csv files from a list of MCNP output file. CSV files are created in
    the working directory with the columns:
    Energy, Intensity, and error.

    Args:
        input_files (list): A list of file names which are used for creating csv files.

    Returns:
        None: It creates csv files in the folder the code is run.
    """
    for file in input_files:
        # read file
        lines = []
        with open(file, 'r') as fp:
        # read and store all the lines into the lines list
            lines = fp.readlines()
        the_line = lines.index(' cell  40                                                                                                                              \n')

        with open(file[:-3]+'csv', 'w') as fp:
        #iterate each line
            for number, line in enumerate(lines):
                if number in range(the_line+2, the_line+304):
                    line = line[4:14]+'\t'+line[17:28]+'\t'+line[29:]
                    fp.write(line)

def broadening(csv_files, c0, c1, c2):
    """
    Calculates the Gaussian broadening of the tally probabilities that obtained form MCNP, and modifies
    the csv files by wiriting these broadened values as a new column.

    Args:
        csv_files (list): List of string that contains the name of csv files 
                          which are extracted from an MCNP RhoC detector simulation output file.
        c0, c1, c2 (float): Coefficients that determine the desired energy resolution.

    Returns:
        None: It modifies the csv files by adding a cloumn that contains the broadened values.
    """
    for file in csv_files:
        df = pd.read_csv(file, sep='\t', header=None, names=['Energy', 'Probabilities', 'Errors'])
        #PH_UnknownFWHM2ResolutionFactor = 1.5329
        result = np.zeros(len(df['Probabilities']), float)
        for i in range(len(df['Probabilities'])):
            if df['Probabilities'][i] != 0:
                integral = 0
                E0 = i / 100
                w = c0 + c1 * np.sqrt(E0 + E0 * E0 * c2) # width
                for j in range(len(df['Probabilities'])):
                    E = j / 100 # energy in MeV
                    if w > 0:
                        gauss = np.exp(-((E - E0) / w)**2) / (np.sqrt(np.pi) * w)
                    else: gauss = 0
                    integral = integral + gauss
                    result[j] = result[j] + df['Probabilities'][i] * gauss / 100
        df['Broadened'] = result
        df.to_csv(file, index=False, header=False, sep='\t')

def source_activity(measurement_date=datetime.today().strftime('%Y/%m/%d'), strength_date='2024/1/15', half_life=2605):
    """
    Calculates source activity for a certain measurement date.

    Args:
        measurement_date (str): The date that the desired source activity is determined. It must be
                                given in the format of YYYY/MM/DD as string. Default: th day function is run.
        strength_date (str): The default value is the strength date of the Na-22 source. It was taken
                            from the provider of the source. It must be given in the format of YYYY/MM/DD.
        half_life (float or int): The default value is the half-life of a Na-22 source. It must be given in years.

    Returns:
        float: Source activitiy.
    """
    time_difference = datetime.strptime(measurement_date, "%Y/%m/%d") - datetime.strptime(strength_date, "%Y/%m/%d")
    return 2**(-((time_difference.days)/365.25)/half_life)

def particle_per_second(activity, branching_ratio=2.7985, source_strength=1049000):
    """
    Calculates how many particles a sources radiates per second.

    Args:
        activity (float): Activitiy of the gamma-ray source. The activity can be passed
                          directly as a float for a certain source element or the source_activity function
                          can be passed with its arguments.
        branching_ratio (float): The default value is for Na-22 isotope.
        source_strength (int or float): The default value is for Na-22 isotope. It is taken from provider.

    Returns:
        float: The number of particles radiated by the soruce per second.
    """
    return activity*branching_ratio*source_strength

def convert_probs_to_counts(csv_files, particle_per_second, live_time):
    """
    Converts the each broadened probability of tally to particle counts.

    Args:
        csv_files (list): List of string that contains the name of csv files 
                          which are extracted from an MCNP RhoC detector simulation output file.
                          The probabilities of the tally should be broadened first.
        particle_per_second (Float): Look at the function named particle_per_second or pass an
                                     explicit particle per second value for your source element.
        live_time (Float): The duration of measurement that will be compared to this simluation.
    
    Returns:
        None: It modifies the csv files by adding a cloumn that contains calculated counts.
    """
    for file in csv_files:
        df = pd.read_csv(file, sep='\t', header=None, names=['Energies', 'Probabilities',
                                                             'Errors', 'Broadenings'])
        df['Counts'] = df['Broadenings'] * particle_per_second * live_time
        df.to_csv(file, index=False, header=False, sep='\t')

def csv_to_dfs(csv_files: Union[str, List[str]], *args, titels=['Energies', 'Probs', 'Errors', 'Broadened', 'Counts']):
    """
    Creates dataframe(s) from a given csv file(s). The csv files must be in the same column format
    and contain same type of data.

    Args:
        csv_files (str or list): String or list of strings that specifies the csv files
                                 from which dataframes will be created. If user creates only
                                 one dataframe but still wants to store it in a dictionary than
                                 it must be given as a list with only one element.
        titels (list): List of strings that specifies the column titels of the dataframes.
        *args (str or list): String or list of strings that specifies the names of the dataframes
                             as keys of a dictionary. If it is not specified the names will be
                             df1, df2, etc. If the user creates only one dataframe but still wants
                             to store it in a dictionary than it must be given as a string.
    """
    if isinstance(csv_files, str):
        df = pd.read_csv(csv_files, sep='\t', header=None, names=titels)
        return df
    elif isinstance(csv_files, list):
        dfs = {}
        for i, file in enumerate(csv_files, start=1):
            if not args:
                df_name = f"df{i}"
                dfs[df_name] = pd.read_csv(file, sep='\t', header=None, names=titels)
            else:
                dfs[args[i-1]] = pd.read_csv(file, sep='\t', header=None, names=titels)
        return dfs

def plot_spectrum(x, y, labels, savefig_name, title, xlabel, ylabel, colors=None, fig_size=(8,4), x_lims=False, y_lims=False,
                  yscale='linear', plot_type='scatter', grid_major=False, grid_minor=False, **kwargs):
    """
    Plots the spectra obtained from MCNP simulations in one plot, on top of each other.

    Args:
        x (list, array or dataframe): Data series for x axis. It can be a list 1D array,
                                      array column, array row, dataframe, dataframe column,
                                      or dataframe row.
        y (list): List of dataseries for y axis. If there are more than one, then the plot
                  will contain every data in one axis.
        labels (list): The list of labels to put on the legend of the plot.
        savefig_name (str): String that specifies the name of the output file as png.
        title (str): For title of the plot.
        xlabel (str): For the label of the x-axis.
        ylabel (str): For the label of the y-axis.
        colors (list): The list of colors for lines of the plot. Order of the list must be correspond to
                       the order of the data in y.
        x_lims (list): A list with two element as lowest and highest limits for x-axis. Default: False
        y_lims (list): A list with two element as lowest and highest limits for y-axis. Default: False
        yscale (str): For the scale type of the y-axis. It can be linear, log, symlog, or logit.
                      The default value is linear.
        plot_type (str): The type of the plot. It can be scatter, stackplot, line, or bar. The
                         default value is scatter.
        grid_major (bool): To put major grid lines. Default is Flase.
        grid_minor (bool): To put minor grid lines. Default is Flase.
        **kwargs (Any): All the line properties from matplotlib can be used in the same way here.
    
    Return:
        Axes: It saves the plot in a png file and shows it on your screen.
    """
    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=200)
    if plot_type == 'scatter':
        for i in range(len(y)):
            if colors:
                ax.scatter(x, y[i], color=colors[i], **kwargs)
            else:
                ax.scatter(x, y[i], **kwargs)
    elif plot_type == 'stackplot':
        for i in range(len(y)):
            if colors:
                ax.stackplot(x, y[i], color=colors[i], **kwargs)
            else:
                ax.stackplot(x, y[i], **kwargs)
    elif plot_type == 'line':
        for i in range(len(y)):
            if colors:
                ax.line(x, y[i], color=colors[i], **kwargs)
            else:
                ax.line(x, y[i], **kwargs)
    elif plot_type == 'bar':
        for i in range(len(y)):
            if colors:
                ax.bar(x, y[i], color=colors[i], **kwargs)
            else:
                ax.bar(x, y[i], **kwargs)
    else:
        raise ValueError("Invalid plot type. Choose from 'scatter', 'stackplot',  'line', or 'bar'.")
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_yscale(yscale)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_axisbelow(True)
    if grid_major:
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    else: ax.grid(visible=False, which='major')
    if grid_minor:
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.6)
    else: ax.grid(visible=False, which='minor')
    if len(y) == 1:
        plt.savefig(savefig_name, edgecolor='none', transparent=True)
    else:
        ax.legend([label for label in labels], fontsize=7, fancybox=False,
                  framealpha= 0.0, facecolor='inherit')
        plt.savefig(savefig_name, edgecolor='none', transparent=True)
    return ax

def plot_spectrumv2(dfs, which_dataframes, column_name_for_x, column_name_for_y, start_x, end_x,
                    labels, savefig_name, title, xlabel, ylabel, colors=None, fig_size=(8,4), x_lims=False, y_lims=False,
                    yscale='linear', plot_type='scatter', grid_major=False, grid_minor=False, **kwargs):
    """
    Plots the spectra obtained from MCNP simulations in one plot, on top of each other.

    Args:
        dfs (dictionary): Contains dataframe names as keys and dataframes that obtained from each csv files as values.
        which_dataframes (list): List of keys from dfs dictionary that specifies which dataframe(s) will be plotted.
        column_name_for_x (str): The column that will be used as x data of the plot.
        column_name_for_y (str): The column that will be used for y data of the plot. Just one column name is enough
                                 because it will exist in every dataframe that is to be plotted.
        start_x (int): Starting data point for x-axis. It has to be the index corresponding to that data point.
        end_x (int): The last data point for x-axis. It has to be the index corresponding to that data point.
        labels (list): The list of labels to put on the legend of the plot.
        savefig_name (str): String that specifies the name of the output file as png.
        title (str): For title of the plot.
        xlabel (str): For the label of the x-axis.
        ylabel (str): For the label of the y-axis.
        colors (list): The list of colors for lines of the plot. Order of the list must be correspond to
                       the order of the data in y.
        x_lims (list): A list with two element as lowest and highest limits for x-axis. Default: False
        y_lims (list): A list with two element as lowest and highest limits for y-axis. Default: False
        yscale (str): For the scale type of the y-axis. It can be linear, log, symlog, or logit.
                      The default value is linear.
        plot_type (str): The type of the plot. It can be scatter, stackplot, line, or bar. The
                         default value is scatter.
        grid_major (bool): To put major grid lines. Default is Flase.
        grid_minor (bool): To put minor grid lines. Default is Flase.
        **kwargs (Any): All the line properties from matplotlib can be used in the same way here.
    
    Return:
        Axes: It saves the plot in a png file and shows it on your screen.
    """

    fig, ax = plt.subplots(1, 1, figsize=fig_size, dpi=200)

    if plot_type == 'scatter':
        for i in range(len(which_dataframes)):
            if colors:
                ax.scatter(dfs[which_dataframes[0]][column_name_for_x][start_x:end_x],
                           dfs[which_dataframes[i]][column_name_for_y][start_x:end_x], color=colors[i], **kwargs)
            else:
                ax.scatter(dfs[which_dataframes[0]][column_name_for_x][start_x:end_x],
                           dfs[which_dataframes[i]][column_name_for_y][start_x:end_x], **kwargs)
    elif plot_type == 'stackplot':
        for i in range(len(which_dataframes)):
            if colors:
                ax.stackplot(dfs[which_dataframes[0]][column_name_for_x][start_x:end_x],
                           dfs[which_dataframes[i]][column_name_for_y][start_x:end_x], color=colors[i], **kwargs)
            else:
                ax.stackplot(dfs[which_dataframes[0]][column_name_for_x][start_x:end_x],
                           dfs[which_dataframes[i]][column_name_for_y][start_x:end_x], **kwargs)
    elif plot_type == 'line':
        for i in range(len(which_dataframes)):
            if colors:
                ax.line(dfs[which_dataframes[0]][column_name_for_x][start_x:end_x],
                           dfs[which_dataframes[i]][column_name_for_y][start_x:end_x], color=colors[i], **kwargs)
            else:
                ax.line(dfs[which_dataframes[0]][column_name_for_x][start_x:end_x],
                           dfs[which_dataframes[i]][column_name_for_y][start_x:end_x], **kwargs)
    elif plot_type == 'bar':
        for i in range(len(which_dataframes)):
            if colors:
                ax.bar(dfs[which_dataframes[0]][column_name_for_x][start_x:end_x],
                           dfs[which_dataframes[i]][column_name_for_y][start_x:end_x], color=colors[i], **kwargs)
            else:
                ax.bar(dfs[which_dataframes[0]][column_name_for_x][start_x:end_x],
                           dfs[which_dataframes[i]][column_name_for_y][start_x:end_x], **kwargs)
    else:
        raise ValueError("Invalid plot type. Choose from 'scatter', 'stackplot',  'line', or 'bar'.")
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_yscale(yscale)
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.set_axisbelow(True)
    if grid_major:
        ax.grid(which='major', color='#DDDDDD', linewidth=0.8)
    else: ax.grid(visible=False, which='major')
    if grid_minor:
        ax.grid(which='minor', color='#DDDDDD', linestyle=':', linewidth=0.6)
    else: ax.grid(visible=False, which='minor')
    if len(which_dataframes) == 1:
        plt.savefig(savefig_name, edgecolor='none', transparent=True)
    else:
        ax.legend([label for label in labels], fontsize=7, fancybox=False,
                  framealpha= 0.0, facecolor='inherit')
        plt.savefig(savefig_name, edgecolor='none', transparent=True)
    return ax