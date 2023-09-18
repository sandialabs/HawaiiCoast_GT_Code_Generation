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
from dateutil import tz
import os

# Load the incident information
incidents = pd.read_csv('hawaii_primary_trajectories_of_interest_2017_2020.csv')
incidents['ais_incident_end_bound_hst'] = pd.to_datetime(incidents['ais_incident_end_bound_hst'])
incidents['ais_incident_start_bound_hst'] = pd.to_datetime(incidents['ais_incident_start_bound_hst'])
incidents['incident_num'] = incidents['incident_num'].astype(int)

print('Incidents loaded')
region='Hawaii'
foldername = f'{region}_coast_gt'
if not os.path.exists(foldername):
    os.makedirs(foldername)

# Load the interim data with names, classes, and speeds but not incidents
Ha_dict = {}
for year in [2017, 2018, 2019, 2020]:
    for month in range(1, 13):
        yearstr = str(year)
        monthstr = str(month).zfill(2)
        Ha_dict[(year, month)] = pd.read_csv(f'{region}_name_class_speed/{region}_{yearstr}_{monthstr}.csv').set_index('MMSI')
        Ha_dict[(year, month)]['datetime_hst'] = pd.to_datetime(Ha_dict[(year, month)]['datetime_hst'])
        Ha_dict[(year, month)]['incident_num'] = np.nan

# Add the incidents
for entrynum in range(incidents.shape[0]):
    inc_entry = incidents.iloc[entrynum]
    dr = pd.date_range(inc_entry['ais_incident_start_bound_hst'], inc_entry['ais_incident_end_bound_hst'], freq='D')
    df_key_list = list(set(list((zip(dr.year, dr.month)))))
    df_key_list.sort()

    if len(df_key_list) == 1:
        year, month = df_key_list[0]
        inc_index = np.where((Ha_dict[year, month]['datetime_hst'] >= inc_entry['ais_incident_start_bound_hst']) & (Ha_dict[year, month]['datetime_hst'] <= inc_entry['ais_incident_end_bound_hst']) & (Ha_dict[year, month].index == inc_entry['MMSI']))[0]
        Ha_dict[year, month].iloc[inc_index, Ha_dict[year, month].columns.get_loc('incident_num')] = inc_entry['incident_num']
    else:
        for i in range(len(df_key_list)):
            year, month = df_key_list[i]
            if i == 0:
                inc_index = np.where((Ha_dict[year, month]['datetime_hst'] >= inc_entry['ais_incident_start_bound_hst']) & (Ha_dict[year, month].index == inc_entry['MMSI']))[0]
                Ha_dict[year, month].iloc[inc_index, Ha_dict[year, month].columns.get_loc('incident_num')] = inc_entry['incident_num']
            elif i == len(df_key_list) - 1:
                inc_index = np.where((Ha_dict[year, month]['datetime_hst'] <= inc_entry['ais_incident_end_bound_hst']) & (Ha_dict[year, month].index == inc_entry['MMSI']))[0]
                Ha_dict[year, month].iloc[inc_index, Ha_dict[year, month].columns.get_loc('incident_num')] = inc_entry['incident_num']
            else:
                Ha_dict[year, month].loc[inc_entry['MMSI'], 'incident_num'] = inc_entry['incident_num']


# Save the data with added incidents in <region>_coast_gt
for j in [2017, 2018, 2019, 2020]:
    for i in range(1, 13):
        Ha_dict[j, i].to_csv(f'{foldername}/{region}_{j}_{str(i).zfill(2)}.csv')
