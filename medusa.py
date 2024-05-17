import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

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

def source_activity(calibration_date=datetime.today().strftime('%Y/%m/%d'), strength_date='2024/1/15', half_life=2605):
    """
    Calculates source activity for a certain measurement date.

    Args:
        calibration_date (str): The date that the desired source activity is determined. It must be
                                given in the format of YYYY/MM/DD as string. Default: th day function is run.
        strength_date (str): The default value is the strength date of the Na-22 source. It was taken
                            from the provider of the source. It must be given in the format of YYYY/MM/DD.
        half_life (float or int): The default value is the half-life of a Na-22 source. It must be given in years.

    Returns:
        float: Source activitiy.
    """
    time_difference = datetime.strptime(calibration_date, "%Y/%m/%d") - datetime.strptime(strength_date, "%Y/%m/%d")
    s_act = 2**(-((time_difference.days)/365.25)/half_life)
    return s_act

def particle_per_second(branching_ratio=2.7985, source_strength=1049000):
    """
    Calculates how many particles a sources radiates per second.

    Args:
        source_activity (float): Look at the function named source_activity.
        branching_ratio (float): The default value is for Na-22 isotope.
        source_strength (int or float): The default value is for Na-22 isotope.
        It is taken from provider.

    Returns:
        float: The number of particles radiated by the soruce per second.
    """
    return source_activity()*branching_ratio*source_strength

def convert_probs_to_counts(csv_files, live_time):
    """
    Converts the each broadened probability of tally to particle counts.

    Args:
        csv_files (list): List of string that contains the name of csv files 
                          which are extracted from an MCNP RhoC detector simulation output file.
                          The probabilities of the tally should be broadened first.
        particle_per_second (Float): Look at the function named particle_per_second.
        live_time (Float): The duration of measurement that will be compared to this simluation.
    
    Returns:
        None: It modifies the csv files by adding a cloumn that contains calculated counts.
    """
    for file in csv_files:
        # read file
        lines = []
        with open(file, 'r+') as fp:
            # read and store all the lines into the lines list
            lines = fp.readlines()
            #iterate each line
            for line in lines:
                line = line[:-1] + '\t' + str(float(line[11:22]) * particle_per_second() 
                                              * live_time)
                fp.write(line)

def plot_spectrum(csv_files,  savefig_name):
    # make a function creates df from csv files
    # then change the data argument of this function to df
    # thus, this can be used with any df seperately
    """
    Plot the spectra obtained from MCNP simulations in one plot, on top of each other.

    Args:
        csv_files (list): List of string that contains the name of csv files 
                          which are extracted from an MCNP RhoC detector simulation output file.
                          The files must contain converted counts as fifth coulmn.
        savefig_name (str): String that specifies the name of the output file as png.
    
    Return:
        Axes: It saves the plot in a png file and shows it on your screen.
    """
    fig, ax = plt.subplots(1, 1, dpi=200)
    for i, file in enumerate(csv_files):
        df = pd.read_csv(file, sep='\t', header=None, names=['Energy', 'Probabilities', 'Errors'])
        ax.stackplot(df['Energy'][2:151]*1000, df['Probabilities'][2:151], alpha = 0.3)
    ax.set_title('RhoC 5')
    ax.set_xlabel('Energy (KeV)')
    ax.set_ylabel('Counts')
    ax.set_xlim(0)
    ax.set_yscale('log')
    ax.minorticks_on()
    ax.tick_params(which='both', direction='in', top=True, right=True)
    if len(csv_files) == 1:
        plt.savefig(savefig_name, edgecolor='none', transparent=True)
    else:
        ax.legend([file[6:-4] for file in csv_files], fontsize=7, fancybox=False,
                  framealpha= 0.0, facecolor='inherit')
        plt.savefig(savefig_name, edgecolor='none', transparent=True)
    return ax