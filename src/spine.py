from pathlib import Path
import shutil

import warnings
from tqdm import tqdm

import numpy as np
from matplotlib import pyplot as plt

import geopandas as gpd
from telenvi import raster_tools as rt
from telenvi import vector_tools as vt
from shapely.errors import ShapelyDeprecationWarning

class Spine:

    def __init__(self, session, sp_id):

        # Check existence and unique
        assert sp_id in session._spines.sp_id.values, 'key not found in spines layer'
        assert session._spines.value_counts('sp_id')[sp_id] == 1, f'more than 1 spine have the key {sp_id}'

        # Attributes
        self.session = session
        self.data = session._spines[session._spines.sp_id == sp_id].iloc[0]
        self.sp_id = sp_id
        self.sp_ge = session.get_geomorphs(self.data.sp_ge_id)[0]
        self.sp_pz = session.get_pzones(self.data.sp_pz_name)[0]
        self.geometry = self.data.geometry

    def set_ribs(self,  ribLength = None, ribStep = None, ribOrientation=None):

        # Get the inputs
        if ribLength == None:
            ribLength = self.data.sp_ri_len
        
        if ribStep == None :
            ribStep = self.data.sp_ri_step
        
        if ribOrientation == None:
            ribOrientation= self.data.sp_ri_or

        # Create new ribs
        ribGeoms = vt.serializeGeoLines(self.geometry, ribLength, ribStep, ribOrientation)
        ribs = []
        for ribIndex, ribGeom in enumerate(ribGeoms):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
                rib = gpd.GeoSeries()
                rib['geometry'] = ribGeom
                rib['rib_id']= f"{self.sp_id}_{ribIndex}"
                ribs.append(rib)
        ribs = gpd.GeoDataFrame(ribs)
        return ribs

    def plot(self):
        gpd.GeoDataFrame([{'geometry':self.geometry}]).plot()

    def __repr__(self):
        self.plot()
        return ''

    def inspect_pairs(self, pairs, ribLength = None, ribStep = None, ribOrientation=None, yMin=-15, yMax=15, note='', backmapyear=2019, backmapsensor='aerial'):
        """
        Par défaut, inspecte les paires données le long de ribs créées selon les paramètres inscrits dans la table attributaire des spines.
        """

        # Get Matplotlib Color Map
        cmap = plt.cm.rainbow(np.linspace(0, 1, len(pairs)))

        # Get the directory to store the graphics
        figspath = Path(self.session.p_root, 'profils', f"{note}_{self.sp_id}")
        if figspath.exists():
            shutil.rmtree(figspath)
        figspath.mkdir()

        # Build ribs
        ribs = self.set_ribs(ribLength, ribStep, ribOrientation)

        # Ouvre l'image d'arriere-plan
        backmappath = self.session.get_thumbs([self.sp_pz.pz_name, backmapyear, backmapsensor])[0].th_path
        backmap = rt.pre_process(backmappath, geoim=True).cropFromVector(self.sp_ge.geometry)

        # Pour chaque Rib
        for rib in tqdm(ribs.iloc, total=len(ribs)):

            # Créer une figure vide
            fig, axes = plt.subplots(ncols=5, figsize=(20,10))

            # Pour chaque composante
            for indexComponant, componant in enumerate(['m', 'x', 'y']):

                # On isole un des 3 graphiques, qui correspond à la composante
                ax = axes[indexComponant]

                # Pour chaque paire
                for indexPair, pair in enumerate(pairs):

                    # Il faut extraire les valeurs
                    # De la paire, dans sa composante, le long du rib
                    motion = pair.get_slice(rib, componant)

                    # Et tracer une courbe sur le graphique de la composante
                    ax.scatter(
                        x=[i for i in range(len(motion))], 
                        y=motion, 
                        s=5,
                        marker='+',
                        color=cmap[indexPair])

                    # Couleur de fond
                    ax.set_facecolor('black')

                    # Verrouillage des bornes de l'ordonnee
                    ax.set_ybound((yMin, yMax))

                    # Titre du graphique : nom de la composante
                    if componant == 'x':
                        titre_figure = "Deplacement vers l'Est (metres)"
                    elif componant == 'y':
                        titre_figure = "Deplacement vers le Sud (metres)"
                    else:
                        titre_figure = "Deplacement bi-directionnel (metres)"
                    ax.set_title(f"{titre_figure}")

            # On fixe la légende liant les couleurs des courbes
            # aux années de chaque paire
            axes[0].legend(sorted([f"{p.pa_left.th_year}-{p.pa_right.th_year}" for p in pairs]), loc='lower left')
            
            # Création d'un visuel glacier rocheux
            thumb = axes[3]
            thumb.imshow(backmap.array)

            # Création d'une carte du rib par rapport au glacier rocheux
            map = axes[4]
            map.set_axis_off()
            gpd.GeoDataFrame([self.sp_ge.data]).boundary.plot(ax=map)
            gpd.GeoDataFrame([{'geometry':rib.geometry}]).plot(ax=map, color='red')

            # On titre la figure entière
            fig.suptitle(f"{self.sp_pz.pz_name.upper()} | profils de mouvements sur {self.sp_ge.ge_id} | base {note}", fontsize=16)

            # Et on la sauve
            figname = f"{rib.rib_id}.png"
            fig.savefig(Path(figspath, figname))

            # On nettoie la figure
            plt.close()
