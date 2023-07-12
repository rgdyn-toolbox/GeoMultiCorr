import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import geopandas as gpd
from telenvi import raster_tools as rt

class Xzone:
        
    def __init__(self, session, xz_key):
        
        # Check existence and unique
        assert xz_key in session._xzones.xz_id.values, 'key not found in Xzones layer'
        assert session._xzones.value_counts('xz_id')[xz_key] == 1, f'more than 1 Xzone have the key {xz_key}'

        # Attributes
        self.session = session
        self.data = session._xzones[session._xzones.xz_id == xz_key].iloc[0]
        self.xz_pz = session.get_pzones(self.data.xz_pz_name)[0]
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
        ax.set_title(f"Vitesses annuelles moyennes consolidées sur {self.xz_key}")
        if savepath != None:
            fig.savefig(savepath)

"""
GMC ISSUE 1
disps = pd.DataFrame([
{'period':(2001,2006), 'vs':0.183350, 'vm':0.352077},
{'period':(2006,2009), 'vs':0.158703, 'vm':0.383952},
{'period':(2009,2013), 'vs':0.160116, 'vm':0.491599},
{'period':(2013,2014), 'vs':0.178329, 'vm':0.541328},
{'period':(2014,2017), 'vs':0.183350, 'vm':0.700086},
{'period':(2017,2018), 'vs':0.178329, 'vm':0.775959},
{'period':(2018,2019), 'vs':0.178329, 'vm':0.880695},
{'period':(2019,2021), 'vs':0.161985, 'vm':0.859764}])

def draw_time_series(disps=disps, color='black', bounds=None, savepath=None):

    # Create plot figure
    fig, ax = plt.subplots(figsize=(10,6.5))

    for period in disps.iloc:
        
        # Recupere les bornes de la periode d'etude
        ya = period.period[0]
        yb = period.period[1]

        # Récupère les valeurs de déplacements pour zone mouvante et stable
        mean_velocity = period.vm
        uncertainity = period.vs

        # Calcule l'incertitude
        uncertainity_max = mean_velocity + uncertainity
        uncertainity_min = mean_velocity - uncertainity

        # Affiche l'incertitude
        ax.plot([int(ya), int(yb)],[uncertainity_max, uncertainity_max], color='red')
        ax.plot([int(ya), int(yb)],[uncertainity_min, uncertainity_min], color='red')

        # Trace la serie temporelle
        ax.plot([int(ya), int(yb)],[mean_velocity, mean_velocity], color=color)
        ax.plot([int(ya), int(yb)],[mean_velocity, mean_velocity], 'bo', linewidth=0.5, color=color)

        # Verrouille les bornes inférieure et supérieure de l'axe Y du graphe
        if bounds != None:
            ax.set_ybound(lower=bounds[0], upper=bounds[1])

        # Verrouille les bornes des périodes à afficher
        ax.set_xticks(np.arange(2001,2023,2))

        # Titre le graphe
        ax.set_title(f"Vitesses annuelles moyennes consolidées dans le polygone 1")

        # Sauve si demandé
        if savepath != None:
            fig.savefig(savepath)

        # fig.show()

    return None
"""