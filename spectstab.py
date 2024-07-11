import json
import pandas as pd
import numpy as np
import csv
import datetime
import os

class SpectrumStabilizer:
    """
    This class is written to analyze the measurement results of a RhoC survey. When it is run as a script,
    it asks the path of the JSON file that created by the RhoC during measurement and the date of the measurement,
    extracts the a0 value from the JSON file. Then it pulls the counts taken during each live time of the detector,
    live time, real time, count rates and total and writes these in a csv file as raw spectrum. From this CSV file,
    it can calculate the live time and total of the counts, wirtes the total counts to another CSV as summed counts,
    and calculates the a1 value after which it is used to calculate the stabilized spectrum. Finally it writes the
    stabilized spectrum to a CSV file.
    """
    def __init__(self): 
        # Constants:
        self.required_live_time = 180  # Required live time before attempting a stab update.
        self.required_peak_counts = 200  # Required number of counts in the peak of the (smoothed) spectrum. 
                                         # Todo: set another value. With a 1 MBq source,
                                         # the count rate in the peak is approiimately 1.8 cps
        self.peak_channel = 127.5  # Energy of the photo peak to search for in 10 keV/channel
        self.peak_valley_ratio = 1.3  # Minimal ratio between valley and peak before identifying a high content channel as the peak.
                                      # Todo: Decide which value to use here
        self.stab_start_channel = 120  # Start channel to start searching for the peak. Must be >= 4.
        self.stab_end_channel = 400  # End channel to start searching for the 1275 keV peak. Must be <= RESOLUTION - 5.

    def stabilize_spectrum(self, summed_csv, meas_date, stab_a1, stab_a0):
        """
        Calculates the a1 value needed for energy calibration and spectrum stabilization.

        Args:
            summed_csv (str): The path to the csv file containing the summation of the raw spectra counts.
            meas_date (str): The measurement date in the format of YYYMMDD to use for the name of output CSV file.
            stab_a1 (float): The a1 value.It is the slope of the mapping function that connects the channel numbers
                             of the detectors to the positions where the should be regarding to the known spectrum.
            stab_a0 (float): The a0 value. It is the intersection between tht channels of the detector and
                             the position where they should be regarding to the known spectrum.

        Returns:
            str: The full path of the created CSV file.

        Output:
            Creates a CSV file with a column that contains 512 rows of stabilized counts.
        """

        counts = np.loadtxt(summed_csv)
        channels = np.arange(1, len(counts)+1)

        new_channels = stab_a1 * channels + stab_a0

        new_counts = np.zeros(len(new_channels), float)
        for i in range(len(counts)):
            y_floor = int(np.floor(new_channels[i]))
            y_ceil = int(np.ceil(new_channels[i]))
            frac = new_channels[i] - y_floor

            if 0 <= y_floor < len(new_counts):
                new_counts[y_floor] += (1 - frac) * counts[i]

            if 0 <= y_ceil < len(new_counts) and y_ceil != y_floor:
                new_counts[y_ceil] += frac * counts[i]

        # Open a file in write mode
        stabilized_csv = f'{meas_date}_stabilized.csv'
        with open(stabilized_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write each item in the list as a new row
            for item in new_counts:
                writer.writerow([item])

        print(f'The given raw spectra is stabilized and have written in {stabilized_csv}.')

        # Get the absolute path of the CSV file
        file_path = os.path.abspath(stabilized_csv)
        return file_path

    def get_a1(self, summed_csv, live_time, stab_a0):
        """
        Calculates the a1 value needed for energy calibration and spectrum stabilization.

        Args:
            summed_csv (str): The file path of the CSV file that contains the summed counts.
            live_time (float): The summation of the live time of the spectra.
            stab_a0 (float): The a0 value. It is the intersection between tht channels of the detector and
                        the position where they should be regarding to the known spectrum.

        Returns:
            float: The a1 value. It is the slope of the mapping function that connects the channel numbers
                   of the detectors to the positions where the should be regarding to the known spectrum.
        """

        df = pd.read_csv(summed_csv, header=None)

        filtered_channels = np.zeros(self.stab_end_channel, float)

        if live_time < self.required_live_time:
            #return False  # Not enough live time, cannot update
            print('The total live time of this measurement is less than required live time!')

        # Apply 9-channel Savitzky-Golay filter. More info on the filter:
        # http://www.statistics4u.info/fundstat_eng/cc_savgol_coeff.html
        for i in range(self.stab_start_channel, self.stab_end_channel):
            filtered_channels[i] = sum((
                - 21 * df[0][i - 4],
                + 14 * df[0][i - 3],
                + 39 * df[0][i - 2],
                + 54 * df[0][i - 1],
                + 59 * df[0][i],
                + 54 * df[0][i + 1],
                + 39 * df[0][i + 2],
                + 14 * df[0][i + 3],
                - 21 * df[0][i + 4]
            ))
            # The filter requires division by 231, but this is not required when doing a peak search

        # Search for the (local) maximum
        min_counts = filtered_channels[self.stab_start_channel]  # The lowest channel content found so far
        max_counts = 0  # The highest channel content found so far
        max_channel = self.stab_start_channel  # Channel with the highest (local) counts

        for i in range(self.stab_start_channel, self.stab_end_channel):
            min_counts = min(min_counts, filtered_channels[i])
            if filtered_channels[i] > self.peak_valley_ratio * min_counts:
                # This might be a local maximum (it has channels to the left which are significantly lower)
                if filtered_channels[i] > max_counts:
                    max_counts = filtered_channels[i]
                    max_channel = i

        if max_counts < self.required_peak_counts * 231:  # * 231 because the channels in filtered_channels 
                                                     # were not divided by 231 when applying the SG filter.
            #return False  # Not enough counts in the peak, cannot update
            print('The maximum count in this spectrum is less than \nrequired peak counts multiplied by 231!')

        # The local maximum is now in channel max_channel.
        # Now do a quadratic interpolation of the spectral peak to find a maximum in between the channels.
        # More info here: https://ccrma.stanford.edu/~jos/sasp/Quadratic_Interpolation_Spectral_Peaks.html

        alpha = filtered_channels[max_channel - 1]
        beta = filtered_channels[max_channel]
        gamma = filtered_channels[max_channel + 1]

        p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)

        stab_a1 = (self.peak_channel - stab_a0) / (max_channel + p)

        return stab_a1

    def sum_raw_counts(self, spec_range, meas_date, raw_csv):
        """
        Sums the counts of RhoC 5 measurements over a specified number of spectra or overall.

        Args:
            input_json (string): file path of the JSON file.
            meas_date (string): The measurement date in "YYYYMMDD" format.
            raw_csv (string): The file path of the CSV that contains raw counts.

        Returns:
            str: The full path of the created CSV file.

        Output:
            Creates a CSV file with a column that contains 512 row of summed counts of the spectra.
        """
        summed_csv = f'{meas_date}_summed.csv'
        df = pd.read_csv(raw_csv, header=None)
        df.iloc[spec_range[0]:spec_range[1]][df.iloc[spec_range[0]:spec_range[1]].
                                           columns[-512:]].sum(axis=0).to_csv(summed_csv, sep=',', header=False, index=False)
        with open(summed_csv, 'r+') as f:
            lines = f.read().splitlines()
            f.seek(0) # Move the file pointer to the beginning.
            for line in lines:
                if line is not lines[-1]:
                    f.write(line + '\n')
                else:
                    f.write(line)
                    f.truncate()

        # Get the absolute path of the CSV file
        file_path = os.path.abspath(summed_csv)
        return file_path

    def get_live_time(self, spec_range, raw_csv):
        """
        Sums the live times of RhoC 5 measurements over a specified number of spectra or overall.

        Args:
            spec_range (tuple): The start and the end of the desired sub-range of the survey.
            raw_csv (string): The file path of the CSV that contains raw counts.

        Returns:
            float: Summation of the live times of the spectra (interval).
        """
        df = pd.read_csv(raw_csv, header=None)
        live_time = df.iloc[spec_range[0]:spec_range[1]][df.iloc[spec_range[0]:spec_range[1]].columns[1]]
        return sum(live_time)

    def get_measurement_range(self):
        """
        A JSON file that obtained from a RhoC5 survey can contain more than one measurement,
        such as different depth or different places. In this situation, user can take
        create a certain part of the measurement by giving a starting and an ending spectrum
        number.
        """
        answer = input('Do you want to take a sub-range of the survey (e/h)?')
        if answer == 'e':
            range = input('Enter the starting and ending spectrum no with a comma between them...')
            start_spec_no = int(range.split(',')[0])
            end_spec_no = int(range.split(',')[1])
            return start_spec_no, end_spec_no
        else:
            return None, None

    def json_to_csv(self, input_json, meas_date):
        """
        Extracts data of a RhoC 5 measurement from a JSON file to a CSV file.

        Args:
            input_json (string): file path of the JSON file.
            meas_date (string): The measurement date in "YYYYMMDD" format.

        Returns:
            str: The full path of the created CSV file.

        Output:
            Creates a CSV file with columns:
                - Realtime
                - Livetime
                - Total
                - Countrate
                - Count values (extracted from the "Spectrum" list)
        """

        rows = []
        with open(input_json, 'r') as json_file:
            for line in json_file:
                data = json.loads(line)
                dicts = next((item for item in data), None)
                if dicts.get('eID') == 'SPECTRO_4841':
                    rows.append(dicts.get('v'))
                else: continue

        raw_csv = f'{meas_date}_raw.csv'
        with open(raw_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write data from each dictionary
            for row in rows:
                if row != None:
                    # Eitract individual elements from the "Spectrum" list
                    spectrum_values = row['Spectrum']
                    other_values = [row['Realtime'], row['Livetime'], row['Total'], row['Countrate']]
                    writer.writerow(other_values + spectrum_values)
        print(f'Each spectra, and their realtimes, livetimes, totals, and countrates have written in {raw_csv}.')

        # Get the absolute path of the CSV file
        file_path = os.path.abspath(raw_csv)
        return file_path

    def get_a0(self, input_json):
        """
        Extracts the a0 value from the JSON file of a RhoC 5 measurement.

        Args:
            input_json (string): file path of the JSON file.

        Returns:
            float: The a0 value. It is the intersection between tht channels of the detector and
                   the position where they should be regarding to the known spectrum.
        """
        with open(input_json, 'r') as file:
            for line in file:
                data = json.loads(line)
                dicts = next((item for item in data), None)
                if dicts.get('sensorConfig', {}).get('calibration', {}).get('stabA0') is None:
                    continue
                else:
                    stab_a0 = dicts["sensorConfig"]["calibration"]["stabA0"]
                    break
        return stab_a0

    def get_meas_date(self, input_json):
        """
        Extracts the measurement date from a JSON file and returns it in "YYYYMMDD" format.

        Args:
            input_json (string): file path of the JSON file.

        Returns:
            str: The measurement date in "YYYYMMDD" format.
        """
        with open(input_json, 'r') as json_file:
            for line in json_file:
                data = json.loads(line)
                dicts = next((item for item in data), None)
                if dicts.get('v', {}).get('Date') is not None:
                    unix_timestamp = dicts['v']['Date']
                    break
        return datetime.datetime.fromtimestamp(unix_timestamp / 1000, datetime.UTC).strftime("%Y%m%d")
    
    def get_json_path(self):
        """
        Gets the absolute path of the desired JSON file.

        Returns:
            str: The absolute path of the desired JSON file.
        """

        files = [file for file in os.listdir(os.getcwd()) if file.endswith('.json')]
        files.sort()
        for i, file in enumerate(files):
            print(f'{i+1}. {file}')
        num = input('Please choose the JSON file to be analyzed with its number...')
        return os.path.abspath(files[int(num)-1])

if __name__ == "__main__":
    stabilizer = SpectrumStabilizer()
    input_json = stabilizer.get_json_path()
    meas_date = stabilizer.get_meas_date(input_json)
    stab_a0 = stabilizer.get_a0(input_json)
    spec_range = stabilizer.get_measurement_range()
    raw_csv = stabilizer.json_to_csv(input_json, meas_date)
    live_time = stabilizer.get_live_time(spec_range, raw_csv)
    summed_csv = stabilizer.sum_raw_counts(spec_range, meas_date, raw_csv)
    stab_a1 = stabilizer.get_a1(summed_csv, live_time, stab_a0)
    stabilized_csv = stabilizer.stabilize_spectrum(summed_csv, meas_date, stab_a1, stab_a0)