from random import choice
import numpy as np
import pandas as pd
from telenvi import raster_tools as rt
from matplotlib import pyplot as plt
import geopandas as gpd

class GMC_Xzones:
        
    def __init__(self, project, xz_key):
        
        # Check existence and unique
        assert xz_key in project._xzones.xz_id.values, 'key not found in Xzones layer'
        assert project._xzones.value_counts('xz_id')[xz_key] == 1, f'more than 1 Xzone have the key {xz_key}'

        # Attributes
        self.project = project
        self.data = project._xzones[project._xzones.xz_id == xz_key].iloc[0]
        self.xz_pz = project.get_pzones(self.data.xz_pz_name)[0]
        self.geometry = self.data.geometry
        self.xz_key = xz_key

    def get_thumbs_overview(self, criterias=''):
        return self.xz_pz.get_thumbs_overview(criterias)
    
    def get_thumbs(self, criterias=''):
        return self.xz_pz.get_thumbs(criterias)

    def get_pairs_overview(self, criterias=''):
        return self.xz_pz.get_pairs_overview(criterias)
    
    def get_pairs(self):
        return self.xz_pz.get_pairs()
    
    def get_pairs_complete_overview(self):
        return self.get_pairs_overview()[self.get_pairs_overview().pa_status == 'complete']

    def get_pairs_complete(self):
        return [pair for pair in self.get_pairs() if pair.pa_status == 'complete']
    
    def show(self, criterias=''):
        thumb = self.get_thumbs(criterias)[0].get_geoim()
        thumb = thumb.cropFromVector(self.geometry)
        thumb.maskFromVector(self.geometry)
        thumb.show()
    
    def get_pairs_on_period_overview(self, ymin, ymax):
        pairs = self.get_pairs_overview()
        pairs['chrono_min'] = pairs.apply(lambda row: min(int(row.pa_left_date.split('-')[0]), int(row.pa_right_date.split('-')[0])), axis=1)
        pairs['chrono_max'] = pairs.apply(lambda row: max(int(row.pa_left_date.split('-')[0]), int(row.pa_right_date.split('-')[0])), axis=1)
        pairs = pairs[(pairs.chrono_min>=ymin)&(pairs.chrono_max>=ymax)]
        return pairs
    
    def get_mean_disp_on_pair(self, magn_path):
        target = gpd.GeoDataFrame([{'geometry':self.geometry}]).set_crs(epsg=2154)
        data = rt.pre_process(magn_path, geoExtent=target, geoim=True)
        data.maskFromVector(target)
        return data.mean()

    def get_disp_overview(self):
        disps = []
        pairs = self.get_pairs_complete()
        for p in pairs :
            row = pd.Series(dtype='object')
            row['L'] = p.pa_left.th_year
            row['R'] = p.pa_right.th_year
            row['D'] = self.get_mean_disp_on_pair(p.pa_magn_path)
            row['V'] = row.D/abs(row.L-row.R)        
            disps.append(row)
        return pd.DataFrame(disps)
    
    def show_mean_velocities(self, savepath=None, bounds=None):
        fig, ax = plt.subplots(figsize=(10,6.5))
        disps = self.get_disp_overview()
        for pair in disps.iloc:
            ya = pair.L
            yb = pair.R
            if ya > yb:
                color = 'black'
            else:
                color = 'red'
            meters = pair.V
            ax.plot([int(ya), int(yb)],[meters, meters], linewidth=1, color=color, alpha=0.7)
            ax.plot([int(ya), int(yb)],[meters, meters], 'bo', color=color, alpha=0.7)

        if bounds != None:
            ax.set_ybound(lower=bounds[0], upper=bounds[1])
        ax.set_xticks(np.arange(2001,2023,2))
        ax.set_title(f"Vitesses annuelles moyennes sur {self.xz_key}")
        if savepath != None:
            fig.savefig(savepath)