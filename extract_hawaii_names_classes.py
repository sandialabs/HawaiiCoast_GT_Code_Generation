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

from tracktable.domain.terrestrial import Trajectory, TrajectoryPoint
from tracktable.feature import annotations
import tracktable.core.geomath as geomath
import tracktable.algorithms.distance_geometry as dist_g

# Load the data assembled via
Haw_data = {(i,j):AIS_dc.month_AIS(i, j, region='Hawaii', new_tz = 'US/Hawaii', new_tz_abbreviation='hst') for i in [2017, 2018, 2019, 2020] for j in range(1,13)}
Haw_name_type = AIS_dc.vessel_type_names(list(Haw_data.values()))
Haw_name_type.clean_extraneous_nans()

Haw_name_type.init_vessel_classes()
initial_missing_classes = []
for key in Haw_name_type.vessel_class_dict.keys():
    if Haw_name_type.vessel_class_dict[key] == ['nan']:
        initial_missing_classes.append(key)
Haw_name_type.clean_multi_classes()
Haw_name_type.find_missing_classes()

Haw_name_type.choose_from_mult_names()

'''
Note in the original creation of the HawaiiCoast_GT dataset mmsi 360516189 had
to be set by hand, but find_missing_names now has the capability to edit name
choices in place, making the following line necessary. For reproducibility
transparency we include it, commented out, below.
'''
# Haw_name_type.vessel_name_dict[360516189] = ['USS WILLIAM P. LAWRENCE']

initial_missing_names = []
for key in Haw_name_type.vessel_name_dict.keys():
    if Haw_name_type.vessel_name_dict[key] == [np.nan]:
        initial_missing_names.append(key)
Haw_name_type.find_missing_names()

Haw_name_type.save_data()
