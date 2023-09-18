'''
Copyright (c) 2014-2017 National Technology and Engineering Solutions of Sandia,
LLC . Under the terms of Contract DE-NA0003525 with National Technology and
Engineering Solutions of Sandia, LLC, the U.S. Government retains certain rights
in this software.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
import pandas as pd
import requests
from datetime import timedelta, datetime
from bs4 import BeautifulSoup
from dateutil import tz
import os

from tracktable.domain.terrestrial import Trajectory, TrajectoryPoint
from tracktable.feature import annotations
import tracktable.core.geomath as geomath
import tracktable.algorithms.distance_geometry as dist_g
from zipfile import ZipFile

vessel_type_codes = pd.read_csv('vessel_type_codes_2018.csv')
all_possible_classes = vessel_type_codes.ship_classes_for_classification.unique()

def save_month_bbox(year, month, latmin, latmax, lonmin, lonmax, region='Hawaii', daylist=None, remove_extra_files=False):
    '''
    Extracts, concatenates, and saves the values within a month of marine cadastre data for a given
    Lat/Lon bounding box.

    Inputs:
    ------------
    year: int > 2009 (since marinecadastre years do not extend beyond this)
    month: int between 1 and 12
    latmin: float, Minimum latitude for the bounding box in degrees
    latmax: float, Maximum latitude for the bounding box in degrees
    lonmin: float, Minimum longitude for the bounding box in degrees
    lonmax: float, Maximum longitude for the bounding box in degrees
    region: str, Name denoted for the region being extracted. Default 'Hawaii'
        for the HawaiiCoast_GT dataset.
    daylist: List of days from the month to download and concatenate. Default
        None returns all days in the month.
    remove_extra_files: Boolean that indicates whether to remove
        AIS_{year}_{month}_{day}.csv (and AIS_{year}_{month}_{day}.zip if
        it hasn't been unzipped yet). Default False.
    '''
    assert month in np.arange(1, 13), f'{month} is not a number between 1-12, sorry'
    ndays = pd.Period(f'{year}-{month}-1').days_in_month
    all_month = np.arange(1, ndays+1).astype(int)

    if daylist is None:
        full_month = True
        daylist = all_month
    else:
        full_month = False
        for i in daylist:
            assert i in all_month, f"{i} is not a valid day for month {month}"
        daylist.sort()

    for day in daylist:
        filename = f'AIS_{year}_{str(month).zfill(2)}_{str(day).zfill(2)}.csv'

        if not os.path.exists(filename):
            zippath = f'AIS_{year}_{str(month).zfill(2)}_{str(day).zfill(2)}.zip'
            assert os.path.exists(zippath), f"You apparently haven't downloaded {zippath} yet, please download it from https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/index.html"
            with ZipFile(zippath, 'r') as f:
                f.extractall()
            if remove_extra_files:
                os.remove(zippath)

        #print(filename)
        month_all = pd.read_csv(filename)
        month_bbox = month_all[(month_all['LON']>=lonmin) & (month_all['LON'] <= lonmax) & (month_all['LAT']>=latmin) & (month_all['LAT'] <= latmax)]
        if day == 1:
            month_df = month_bbox
        else:
            month_df = pd.concat([month_df, month_bbox], axis=0)

        if remove_extra_files:
            os.remove(filename)

    if full_month:
        month_df.to_csv(f'{region}_{year}_{str(month).zfill(2)}.csv', index=False)
    else:
        month_df.to_csv(f'{region}_{year}_{str(month).zfill(2)}_{str(daylist[0]).zfill(2)}_to_{str(daylist[-1]).zfill(2)}.csv')

def extract_class(mmsival, potential_class):
    '''
    Checks if the web-scraped potential class for the vessel corresponding to
    mmsival corresponds to any of the classes listed in vessel_type_codes_2018.csv,
    ship_classes_for_classification.
    Inputs:
    ----------------------------------------------------------------------------
    mmsival: 9 digit int associated with the given vessel
    potential_class: str, the value scraped for the internet that can potentially
        indicate class
    Outputs:
        List of possible classes from vessel_type_codes_2018.csv that the scraped
        class might correspond to.
    '''
    pc_lower = potential_class.lower()
    class_options = []

    if pc_lower == '':
        class_options.append('nan')

    # Anti-pollution equipment
    for ap_opt in ['anti-pollution', 'anti-sea pollution']:
        if ap_opt in pc_lower:
            class_options.append('Anti-pollution equipment')

    # Cargo
    for cargo_opt in ['cargo', 'freight', 'reefer', 'bulk carrier', 'container ship', 'roro', 'ro-ro']:
        if cargo_opt in pc_lower:
            class_options.append('Cargo')

    # Diving
    if ('dive' in pc_lower) or ('diving' in pc_lower):
        class_options.append('Diving')

    # Dredging
    if 'dredg' in pc_lower:
        class_options.append('Dredging')

    # Fishing
    for fish_opt in ['fish', 'trawler', 'longline', 'siener', 'gillnet', 'tuna clip', 'crabber', 'drifter']:
        if fish_opt in pc_lower:
            class_options.append('Fishing')

    # High speed craft
    if 'high speed craft' in pc_lower:
        class_options.append('High speed craft')

    # Industrial vessel
    if 'industrial' in pc_lower:
        class_options.append('Industrial vessel')

    # Law enforcement:
    if 'law enforc' in pc_lower:
        class_options.append('Law enforcement')

    # Medical Transport
    for med_opt in ['medical', 'hospital']:
        if med_opt in pc_lower:
            class_options.append('Medical Transport')

    # Military
    if 'military' in pc_lower:
        class_options.append('Military')
    if pc_lower == 'sar':
        class_options.append('Military')

    # Mobile Offshore Drilling Unit
    if 'mobile offshore drill' in pc_lower:
        class_options.append('Mobile Offshore Drilling Unit')

    # Non-vessel
    for nv_option in ['non-vessel', 'non vessel']:
        if nv_option in pc_lower:
            class_options.append('Non-vessel')

    # Offshore supply vessel
    for osv_opt in ['offshore supply', 'offshore support', 'anchor handling', 'dive support', 'accommodation', 'crew boat', 'crew transfer']:
        if osv_opt in pc_lower:
            class_options.append('osv_opt')

    # Oil Recovery
    if 'oil recov' in pc_lower:
        class_options.append('Oil Recovery')

    # Passenger
    for pass_option in ['passenger', 'ferry']:
        if pass_option in pc_lower:
            class_options.append('Passenger')

    # Port tender
    if 'port tend' in pc_lower:
        class_options.append('Port tender')

    # Public Vessel, Unclassified
    if 'public vessel, unclassified' in pc_lower:
        class_options.append('Public Vessel, Unclassified')

    # Pilot vessel
    if 'pilot' in pc_lower:
        class_options.append('Pilot vessel')

    # Pleasure Craft/Sailing
    for sail_option in ['pleasure', 'sail', 'yacht']:
        if sail_option in pc_lower:
            class_options.append('Pleasure Craft/Sailing')

    # Research Vessel
    for res_option in ['research', 'exploration', 'survey']:
        if res_option in pc_lower:
            class_options.append('Research')

    # School Ship
    if 'school' in pc_lower:
        class_options.append('School Ship')

    # Search and Rescue Vessel
    if 'rescue' in pc_lower:
        class_options.append('Search and Rescue Vessel')

    # Tanker
    for tank_opt in ['tank', 'gas carrier', 'lng carrier', 'crude carrier', 'panamax', 'aframax', 'suezmax']:
        if tank_opt in pc_lower:
            if 'tank barge' in pc_lower:
                class_options.append('Tug Tow')
            else:
                class_options.append('Tanker')

    # Tug tow
    for tug_option in ['tug', 'tow']:
        if tug_option in pc_lower:
            class_options.append('Tug Tow')

    # Wing in ground
    for wig_option in ['wing in ground', 'wing in grnd']:
        if wig_option in pc_lower:
            class_options.append('Wing in ground')

    # Other
    for other_option in ['inland, unknown', 'isolated danger', 'navigation aid']:
        if other_option == pc_lower:
            class_options.append('Other')

    # nan
    for other_option in ['other', 'other type', 'reserved', 'unspecified']:
        if other_option == pc_lower:
            class_options.append('nan')

    # No actual information about the vessel type
    for unclear_option in ['unspecified']:
        if unclear_option in pc_lower:
            class_options.append('nan')

    final_class_options = list(set(class_options))

    if len(final_class_options) == 1:
        return(final_class_options)

    if len(final_class_options) == 0:
        options = [str(i) for i in range(len(all_possible_classes))] + ['']
        classchoice=False
        while classchoice not in options:
            class_opt_statement = ''.join([f'\n{i}: {all_possible_classes[i]}' for i in range(len(all_possible_classes))])
            input_str = f'{mmsival} scraped class: {potential_class}, please choose the appropriate corresponding class or press enter to skip: {class_opt_statement}'
            classchoice = input(input_str)
            if classchoice not in options:
                print("Sorry, you did not enter a valid option, please try again.")
        if classchoice == '':
            return(['nan'])
        else:
            return([all_possible_classes[int(classchoice)]])

    if len(final_class_options) > 1:
        options = [str(i) for i in range(len(final_class_options))] + ['']
        classchoice = False
        while classchoice not in options:
            class_opt_statement = ''.join([f'\n{i}: {final_class_options[i]}' for i in range(len(final_class_options))])
            input_str = f'{mmsival} scraped class: {potential_class}, please choose the appropriate corresponding class or press enter to skip: {class_opt_statement}'
            classchoice = input(input_str)
            if classchoice not in options:
                print("Sorry, you did not enter a valid option, please try again.")
        if classchoice == '':
            return(['nan'])
        else:
            return([final_class_options[int(classchoice)]])

class pd_trajectory(object):
    def __init__(self, traj_df, primary_dt='datetime_utc'):
        '''
        Assembles a tracktable trajectory using the points in traj_df
        Inputs:
        ----------------------------------------------------------------------
        traj_df: a pandas dataframe of the AIS data associated with a given
            trajectory (typically self.month_df of a month_AIS object as defined
            below.)
        primary_dt: the column name corresponding to the primary datetime for a
            given trajectory
        '''
        self.traj_df = traj_df.copy()
        self.npoints, self.ncol = traj_df.shape
        self.primary_dt = primary_dt
        self.get_tracktable_traj()

        self.distance_comp = False
        self.timesep_comp = False


    def get_tracktable_traj(self):
        '''
        Use the AIS latitude, longitude and time values to create a tracktable
        trajectory.
        '''
        self.traj_tt = Trajectory()
        for i in range(self.npoints):
            point = TrajectoryPoint(self.traj_df.iloc[i]['lon'], self.traj_df.iloc[i]['lat'])
            point.object_id = str(self.traj_df.index.unique()[0])
            point.timestamp = self.traj_df.iloc[i][self.primary_dt]
            self.traj_tt.append(point)


    def get_distances_km(self):
        '''
        Compute the distances between subquent tracktable points using the built
        in trajectory length functions in tracktable, which use the
        haversine/vincenty distance between points.
        '''
        # Get the cumulative distance travelled up to each point.
        self.traj_df['current_lengths_km'] = np.array([self.traj_tt[i].current_length for i in range(self.npoints)])
        # Get the distance between each pair of points
        self.traj_df['distances_km'] = np.array([np.nan] + list(self.traj_df['current_lengths_km'].iloc[1:] - self.traj_df['current_lengths_km'].iloc[:-1]))

        # Get the total of all distance travelled
        self.total_length = self.traj_df['current_lengths_km'].iloc[-1]

        # Get the distance between the first and last trajectory points
        self.end_to_end = geomath.end_to_end_distance(self.traj_tt)
        self.distance_comp = True

    def get_lin_speed(self):
        '''
        Get the computed speed using time and location. This is computed using
        tracktable distances and time steps computed in get_timesep and
        get_distances_km.
        '''
        if not self.distance_comp:
            self.get_distances_km()
        if not self.timesep_comp:
            self.get_timesep()

        # Convert time distances to hours
        tdist_hrs = self.traj_df['time_steps'] / timedelta(hours=1)
        self.traj_df['comput_speed_knots'] = self.traj_df['distances_km'] / tdist_hrs / 1.852


    def get_timesep(self):
        self.traj_df['time_steps'] = np.array([np.nan] + list(self.traj_df[self.primary_dt].iloc[1:] - self.traj_df[self.primary_dt].iloc[:-1]))
        self.total_time = self.traj_df[self.primary_dt].iloc[-1] - self.traj_df[self.primary_dt].iloc[0]
        self.timesep_comp=True

class month_AIS(object):
    def __init__(self, year, month, region='Hawaii', new_tz = 'US/Hawaii', new_tz_abbreviation='hst'):
        '''
        Object for a single month of AIS data from the given region.
        Inputs:
            month: int between 1 and 12
            year: int, year number
            region: str, region name as designated in file system. Default 'Hawaii'
                for the original HawaiiCoast_GT dataset.
            new_tz: str, local timezone name for timezone conversion. Default
                'US/Hawaii' for the original HawaiiCoast_GT dataset.
            new_tz_abbreviation: str, abbreviation for timezone for dataframe
                column naming convention. Default 'hst' for the original
                HawaiiCoast_GT dataset.
        '''
        self.region=region
        self.strmonth = str(month).zfill(2)
        self.stryear = str(year)
        filename = f"{region}_{self.stryear}_{self.strmonth}.csv"
        self.month_df = pd.read_csv(filename).set_index('MMSI')

        # Probably want to drop this column/make it unnecessary
        if 'Unnamed: 0' in self.month_df.columns:
            self.month_df.drop('Unnamed: 0', axis=1, inplace=True)

        # Adjust the original AIS columnnames to snakecase, which is more accessible for dyslexic users.
        newcolnames = {'BaseDateTime':'datetime_utc', 'LAT':'lat', 'LON':'lon',
                        'SOG':'speed_over_ground_knots', 'COG':'course_over_ground_deg',
                        'Heading':'heading_deg', 'VesselName':'vessel_name',
                        'IMO':'imo_number', 'CallSign':'call_sign',
                        'VesselType':'vessel_type_code', 'Status':'status',
                        'Length':'length_m', 'Width':'width_m', 'Draft':'draft_depth_m',
                        'Cargo':'cargo_type_code'}
        if year > 2017:
            newcolnames['TransceiverClass'] = 'transceiver_class'
        self.month_df.rename(newcolnames, axis=1, inplace=True)

        if year <= 2017:
            # Add an empty transceiver class for consistency
            self.month_df['transceiver_class'] = np.nan
        # Set the index to be the 'mmsi' and make sure that it is read as an integer
        self.month_df.set_index(self.month_df.index.astype(int), inplace=True)

        # Make the 'BaseDateTime' column (now datetime_utc) into a datetime object
        self.month_df.loc[:,'datetime_utc'] = pd.to_datetime(self.month_df['datetime_utc'], utc=True)

        # Add the localized times for incident lookup
        self.month_df.loc[:, f'datetime_{new_tz_abbreviation}'] = self.month_df['datetime_utc'].dt.tz_convert(new_tz)
        self.month_df = self.month_df.sort_values('datetime_utc')

        self.cleanup()

    def cleanup(self):
        for ind in self.month_df.index.unique():
            # Remove any entries for which there is only one point
            if len(self.month_df.loc[ind].shape) == 1:
                self.month_df.drop(ind, axis=0., inplace=True)

            # Remove any entries listing a non-valid MMSI
            elif len(str(ind)) != 9:
                self.month_df.drop(ind, axis=0, inplace=True)
            else:
                pass

    def standardize_vessel_class(self, vessel_info_df):
        '''
        Using the lookup table created by vessel_class_dict, set the vessel
        class for all unique mmsi's where the vessel class has been defined
        '''
        self.month_df.loc[:,'vessel_class'] = 'nan'
        for mmsi_val in self.month_df.index.unique():
            self.month_df.loc[mmsi_val, 'vessel_class'] = vessel_info_df.loc[mmsi_val, 'vessel_class']

    def standardize_vessel_name(self, vessel_info_df):
        '''
        Using the lookup table created by vessel_name_dict, set the vessel
        name for all unique mmsi's where the vessel name has been defined.
        Note that for vessels with multiple names (perhaps due to a name
        change over time) we do not adjust the existing data.
        '''
        for mmsi_val in self.month_df.index.unique():
            if len(vessel_info_df.loc[mmsi_val, 'vessel_names']) == 1:
                self.month_df.loc[mmsi_val, 'vessel_name'] = vessel_info_df.loc[mmsi_val, 'vessel_names'][0]

    def get_distances_and_speeds(self):
        for mmsi_val in self.month_df.index.unique():
            pd_traj_obj = pd_trajectory(self.month_df.loc[mmsi_val])
            pd_traj_obj.get_lin_speed()
            self.month_df.loc[mmsi_val, 'distances_km'] = pd_traj_obj.traj_df['distances_km']
            self.month_df.loc[mmsi_val, 'comput_speed_knots'] = pd_traj_obj.traj_df['comput_speed_knots']

    def save_df(self, foldername = ''):
        if os.path.exists(foldername):
            pass
        else:
            os.makedirs(foldername)
        self.month_df.to_csv(f'{foldername}{self.region}_{self.stryear}_{self.strmonth}.csv')

class vessel_type_names(object):
    def __init__(self, month_df_list):
        '''
        Initializes an object for finding, compiling, and searching for vessel
        types/classes and vessel names.
        Inputs:
        ------------------------------------------------------------------------
        month_df_list: a list of month_AIS objects that makes up the dataset
            from which to derive the vessel name and class info.
        '''
        self.vessel_name_dict = {}
        self.vessel_type_code_dict = {}

        for month_obj in month_df_list:
            for ind in month_obj.month_df.index.unique():
                names = month_obj.month_df.loc[ind, 'vessel_name'].unique().tolist()
                types = month_obj.month_df.loc[ind, 'vessel_type_code'].unique().tolist()
                if ind in self.vessel_name_dict.keys():
                    self.vessel_name_dict[ind] = list(set(self.vessel_name_dict[ind] + names))
                    self.vessel_type_code_dict[ind] = list(set(self.vessel_type_code_dict[ind] + types))
                else:
                    self.vessel_name_dict[ind] = names
                    self.vessel_type_code_dict[ind] = types

        self.vessel_AIS_codes = pd.read_csv('vessel_type_codes_2018.csv').set_index('vessel_type_2018')
        self.all_possible_classes = list(self.vessel_AIS_codes['ship_classes_for_classification'].unique())
        self.cleaned_extra_nans = False

    def init_vessel_classes(self):
        '''
        Assign the vessel classes associated with the type codes given in the AIS data
        '''
        self.vessel_class_dict = {}
        # Make sure type code nans have been cleaned up so that classes can start to be assigned
        if not self.cleaned_extra_nans:
            self.clean_extraneous_nans()

        for key in self.vessel_type_code_dict.keys():
            if len(self.vessel_type_code_dict[key]) == 1:
                code = self.vessel_type_code_dict[key][0]
                if str(code) == 'nan':
                    self.vessel_class_dict[key] = ['nan']
                else:
                    self.vessel_class_dict[key] = [self.vessel_AIS_codes.loc[code, 'ship_classes_for_classification']]
            else:
                self.vessel_class_dict[key] = list(set([self.vessel_AIS_codes.loc[i, 'ship_classes_for_classification'] for i in self.vessel_type_code_dict[key]]))

    def clean_multi_classes(self):
        '''
        For vessels with multiple associated vessel classes, perform a basic-user assisted cleaning
        '''
        for key in self.vessel_class_dict.keys():
            if len(self.vessel_class_dict[key]) > 1:

                # If there is more than one associated vessel class, and one is other,
                # choose the more specific one (after nans have been removed, of course)
                classes_copy = self.vessel_class_dict[key].copy()
                if 'Other' in classes_copy:
                    classes_copy.remove('Other')
                    self.vessel_class_dict[key] = classes_copy

                # Note this is separate from the original nan removal pass,
                # because one of the vessel type codes actually maps to a nan (Unspecified).
                if 'nan' in classes_copy:
                    classes_copy.remove('nan')
                    self.vessel_class_dict[key] = classes_copy

                if len(self.vessel_class_dict[key]) > 1:
                    # If the choice is between two valid classes, not other or nan, let the
                    # user pick the appropriate class.
                    classes_copy = self.vessel_class_dict[key].copy()
                    options = [str(i) for i in range(len(classes_copy))]
                    classchoice = False

                    while classchoice not in options:
                        class_opt_statement = ''.join([f'\n{i}: {classes_copy[i]}' for i in range(len(classes_copy))])
                        input_str = f'Type the number associated with your chosen class for vessel {key}: {class_opt_statement}'
                        classchoice = input(input_str)
                        if classchoice not in options:
                            print("Sorry, you did not enter a valid option, please try again.")

                    self.vessel_class_dict[key] = [self.vessel_class_dict[key][int(classchoice)]]

    def clean_extraneous_nans(self):
        '''
        For lists with a valid name and a nan (namely, an MMSI didn't list
        its name in all points but listed it somewhere in the dataset), remove the extraneous
        nan so that the only names assigned are valid.

        Perform the same process for lists with a valid type code and a nan (namely, an MMSI
        didn't list its type code in all points but listed it somewhere in the dataset), remove
        the extraneous nan so that the only names assigned are valid.
        '''
        for key in self.vessel_name_dict.keys():
            if len(self.vessel_name_dict[key]) > 1:
                # print(key, self.vessel_name_dict[key])
                if np.nan in self.vessel_name_dict[key]:
                    self.vessel_name_dict[key].remove(np.nan)
                    # print(key, self.vessel_name_dict[key])

        for key in self.vessel_type_code_dict.keys():
            if len(self.vessel_type_code_dict[key]) > 1:
                strlist = list(set([str(i) for i in self.vessel_type_code_dict[key]]))
                if len(strlist) == 1:
                    self.vessel_type_code_dict[key] = [float(strlist[0])]
                else:
                    if 'nan' in strlist:
                        strlist.remove('nan')
                        self.vessel_type_code_dict[key] = [float(i) for i in strlist]
        self.cleaned_extra_nans = True

    def choose_from_mult_names(self):
        '''
        Sometimes a vessel will broadcast different names--most often different spellings or abbreviations
        of the same name. This lets the user decide which name to keep--or whether to keep all names in the
        case of a name change.
        '''
        if not self.cleaned_extra_nans:
            self.clean_extraneous_nans()
        for key in self.vessel_name_dict.keys():
            if len(self.vessel_name_dict[key]) > 1:
                namelist = self.vessel_name_dict[key]
                options = [str(i) for i in range(len(namelist)+1)]
                namechoice = False

                while namechoice not in options:
                    name_opt_statement = ''.join([f'\n{i}: {namelist[i]}' for i in range(len(namelist))] + [f'\n{len(namelist)}: All names\n'])
                    input_str = f'Type the number associated with your chosen name(s) for vessel {key}: {name_opt_statement}'
                    namechoice = input(input_str)
                    if namechoice not in options:
                        print("Sorry, you did not enter a valid option, please try again.")
                if namechoice == str(len(namelist)):
                    pass
                else:
                    self.vessel_name_dict[key] = [self.vessel_name_dict[key][int(namechoice)]]

    def find_missing_names(self,
                            proxies = None,
                            verify = True):
        '''
        Use marinetraffic, marinevesseltraffic, and myshiptracking to check for missing vessel names.
        '''
        self.truly_missing_names = []
        for key in self.vessel_name_dict.keys():
            if self.vessel_name_dict[key] == [np.nan]:
                # First check www.marinetraffic.com
                if self.check_marinetraffic_name(key, proxies, verify):
                    pass
                else:
                    if self.check_marinevesseltraffic_name(key, proxies, verify):
                        pass
                    else:
                        if self.check_myshiptracking_name(key, proxies, verify):
                            pass

    def find_missing_classes(self,
                            proxies = None,
                            verify = True):
        self.truly_missing_classes = []
        for key in self.vessel_class_dict.keys():
            if self.vessel_class_dict[key] == ['nan']:
                mvt_class = self.check_marinevesseltraffic_class(key, proxies, verify)
                if mvt_class == ['nan']:
                    self.vessel_class_dict[key] = self.check_marinetraffic_class(key, proxies, verify)
                else:
                    self.vessel_class_dict[key] = mvt_class

    def check_marinevesseltraffic_class(self, mmsival, proxies, verify):
        '''
        Check marinvesseltraffic.com for a potential vessel class.
        '''
        url = f"https://www.marinevesseltraffic.com/vessels?vessel={mmsival}&flag=&page=1&sort=none&direction=none#table-anchor"
        headers = {
            "accept": "application/json",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Mozilla/5.0",
            "x-requested-with": "XMLHttpRequest"
        }

        response = requests.get(url, headers=headers, proxies=proxies, verify=verify)
        if response.status_code == 200 or response.status_code==301:
            try:
                strvals = str(BeautifulSoup(response.content, 'html5lib'))
                potential_class = strvals.split('class="vessel_td td_type">\n\t\t\t\t')[1].split('\n')[0]
            except:
                return(['nan'])
            return(extract_class(mmsival, potential_class))
        else:
            return(['nan'])

    def check_marinetraffic_class(self, mmsival, proxies, verify):
        '''
        Check marinetraffic.com for a potential vessel class.
        '''
        url = f"https://www.marinetraffic.com/en/ais/details/ships/mmsi:{mmsival}/"
        headers = {
            "accept": "application/json",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Mozilla/5.0",
            "x-requested-with": "XMLHttpRequest"
        }
        response = requests.get(url, headers=headers, proxies=proxies, verify=verify)
        if response.status_code == 200 or response.status_code==301:
            try:
                strvals = str(BeautifulSoup(response.content, 'html5lib'))
                potential_class = strvals.split('(')[1].split(')')[0]
            except:
                return(['nan'])
            return(extract_class(mmsival, potential_class))
        else:
            return(['nan'])

    def check_marinetraffic_name(self, mmsival, proxies, verify):
        '''
        Check marinetraffic.com for the vessel name corresponding to mmsival.
        Note marine traffic name formats DO match the USCG, so we do not require user input.
        '''
        headers = {
            "accept": "application/json",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Mozilla/5.0",
            "x-requested-with": "XMLHttpRequest"
        }
        url = f"https://www.marinetraffic.com/en/ais/details/ships/mmsi:{mmsival}/"
        response = requests.get(url, headers=headers, proxies=proxies, verify=verify)
        if response.status_code == 200 or response.status_code==301:
            strvals = str(BeautifulSoup(response.content, 'html5lib'))
            ind1 = strvals.find('<title>Ship ') + 12
            ind2 = strvals.find(' (')
            name = strvals[ind1:ind2]
            self.vessel_name_dict[mmsival] = [name]
            return(True)
        else:
            return(False)

    def check_marinevesseltraffic_name(self, mmsival, proxies, verify):
        '''
        Check marinvesseltraffic.com for the vessel name corresponding to mmsival.
        Note marinevesseltraffic does not always include formats that match the
        USCG, so names must be verified.
        '''
        url = f"https://www.marinevesseltraffic.com/vessels?vessel={mmsival}&flag=&page=1&sort=none&direction=none#table-anchor"
        headers = {
            "accept": "application/json",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Mozilla/5.0",
            "x-requested-with": "XMLHttpRequest"
        }

        response = requests.get(url, headers=headers, proxies=proxies, verify=verify)
        if response.status_code == 200 or response.status_code==301:
            strvals = str(BeautifulSoup(response.content, 'html5lib'))
            try:
                potential_name = strvals.split('href="https://www.marinevesseltraffic.com/vessels/')[1].split('/')[0]
                correct_check = input(f'{mmsival} Suggested name: {potential_name}. If this is correct, please type "y", otherwise type "n"')
                if correct_check == 'y':
                    self.vessel_name_dict[mmsival] = [potential_name]
                    return(True)
                else:
                    correct_name = input('Please type the corrected name, or press enter to skip')
                    if correct_name != '':
                        self.vessel_name_dict[mmsival] = [correct_name]
                        return(True)
                    else:
                        return(False)
            except:
                return(False)
        return(False)

    def check_myshiptracking_name(self, mmsival, proxies, verify):
        '''
        Check myshiptracking.com for the vessel name corresponding to mmsival.
        Note myshiptracking does not always include formats that match the USCG,
        so names must be verified.
        '''
        url = f"https://www.myshiptracking.com/vessels/mmsi-{mmsival}"
        headers = {
            "accept": "application/json",
            "accept-encoding": "gzip, deflate",
            "user-agent": "Mozilla/5.0",
            "x-requested-with": "XMLHttpRequest"
        }

        response = requests.get(url, headers=headers, proxies=proxies, verify=verify)
        if response.status_code == 200 or response.status_code==301:
            strvals = str(BeautifulSoup(response.content, 'html5lib'))
            try:
                potential_name = strvals.split('The current position of <strong>')[1].split('</strong>')[0]
                if (potential_name == 'Unkown Name') or (potential_name == 'Unknown Name'):
                    return(False)
                else:
                    correct_check = input(f'{mmsival} Suggested name: {potential_name}. If this is correct, please type "y", otherwise type "n"')
                    if correct_check == 'y':
                        self.vessel_name_dict[mmsival] = [potential_name]
                        return(True)
                    else:
                        correct_name = input('Please type the corrected name, or press enter to skip')
                        if correct_name != '':
                            self.vessel_name_dict[mmsival] = [correct_name]
                            return(True)
                        else:
                            return(False)
            except:
                return(False)
        else:
            return(False)

    def save_data(self):
        # Do a little reformating so that it saves as a usable csv
        save_namedict = self.vessel_name_dict.copy()
        for key in save_namedict.keys():
            if save_namedict[key] == [np.nan]:
                save_namedict[key] = ['nan']
        save_classdict = {key: self.vessel_class_dict[key][0] for key in self.vessel_class_dict.keys()}

        names = {'MMSI':save_namedict.keys(), 'vessel_names':save_namedict.values()}
        classes = {'MMSI':save_classdict.keys(), 'vessel_class':save_classdict.values()}
        info_df = pd.concat([pd.DataFrame(names).set_index('MMSI'), pd.DataFrame(classes).set_index('MMSI')], axis=1)
        info_df.to_csv('vessel_names_and_classes.csv')
