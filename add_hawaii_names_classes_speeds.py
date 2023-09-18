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
import AIS_data_cleaning as AIS_dc
import os

from tracktable.domain.terrestrial import Trajectory, TrajectoryPoint
from tracktable.feature import annotations
import tracktable.core.geomath as geomath
import tracktable.algorithms.distance_geometry as dist_g

# Load the vessel_names_and_classes derived from running extract_hawaii_names_classes.py
vessel_name_type_df = pd.read_csv('vessel_names_and_classes.csv').set_index('MMSI')
vessel_name_type_df.loc[vessel_name_type_df['vessel_names'] == '[nan]', 'vessel_names'] = "['nan']"
# Change strings associated with a list to a list.
vessel_name_type_df['vessel_names'] = vessel_name_type_df['vessel_names'].apply(eval)

# Note that this process can be parallelized, it doesn't have to be done in sequence.
for i in [2017, 2018, 2019, 2020]:
    for j in range(1, 13):
        # Region, timezone, and new_tz_abbreviation should be changed if this is used to create another dataset.
        Haw_month = AIS_dc.month_AIS(i, j, region='Hawaii', new_tz = 'US/Hawaii', new_tz_abbreviation='hst')
        Haw_month.standardize_vessel_name(vessel_name_type_df)
        Haw_month.standardize_vessel_class(vessel_name_type_df)
        Haw_month.get_distances_and_speeds()
        # This foldername should be changed if this is used to create another dataset.
        Haw_month.save_df('Hawaii_name_class_speed/')
