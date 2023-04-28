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
        ax.set_title(f"Vitesses annuelles moyennes consolidées sur {self.xz_key}")
        if savepath != None:
            fig.savefig(savepath)

def lecture_csv(chemin_csv) :
    return pd.read_csv(chemin_csv)

def periodes_etude(chemin_csv) : 

    lire = lecture_csv(chemin_csv)
    Liste_annees = []
    Liste_periode_virgule = []

    for annees in lire['Paire']:

        #Liste_annees contient la liste des années présentes dans le dataframe issu de tri_paires
        if annees[0 : 4] not in Liste_annees:
            Liste_annees.append(annees[0 : 4])
        if annees[0 : 4] not in Liste_annees:
            Liste_annees.append(annees[0 : 4])
    
    for annees in lire['Paire']:

        #Liste_annees contient la liste des années présentes dans le dataframe issu de tri_paires
        if annees[5 : 9] not in Liste_annees:
            Liste_annees.append(annees[5 : 9])
        if annees[5 : 9] not in Liste_annees:
            Liste_annees.append(annees[5 : 9])

    #transforme chaque année (str) en entier
    Liste_annees_ordonnees = [int(annee) for annee in Liste_annees]

    #classe la liste dans l'ordre croissant
    Liste_annees_ordonnees.sort()


    #b : liste des années de la première à l'avant dernière position de Liste_annees_ordonnees
    b = Liste_annees_ordonnees[0:len(Liste_annees_ordonnees)-1]

    #c : liste des années de la deuxième à la dernière position de Liste_annees_ordonnees
    c = Liste_annees_ordonnees[1:len(Liste_annees_ordonnees)]

    for index, year_left in enumerate(b) :

        Liste_periode_virgule.append([year_left, c[index]])

    #liste des périodes entre les années min et max du csv, séparation des années par des virgules
    return Liste_periode_virgule

def periodes_etude_separateur_tiret(Liste_periodes_virgule):
    
    Liste_periodes__tirets = []    
    for  index,periode in enumerate(Liste_periodes_virgule):

            #stockage sous forme différente avec tirets comme séparateurs
            Liste_periodes__tirets.append(f"{Liste_periodes_virgule[index][0]}-{Liste_periodes_virgule[index][1]}")

    #liste des périodes entre les années min et max du csv, séparation des années par des tirets
    return Liste_periodes__tirets


def selection_paire_avt_tri_med(chemin_csv,binf,bsup) :
    lire = lecture_csv(chemin_csv)

    #liste de toutes les années entre les 2 bornes de la période étudiée
    ye = np.arange(float(binf),float(bsup) + 1, 1)
    L = []
    for paire in lire['Paire'] :
        # Bornes de la paire
        yminp = min(int(paire[0:4]),int(paire[5:9]))
        ymaxp = max(int(paire[0:4]),int(paire[5:9]))

        # Déroule toutes les années entre les bornes de la paire
        yp = np.arange(yminp, ymaxp+1, 1)

        # Détermine si une paire est intéressante au regard de la période d'étude
        y_common = np.intersect1d(yp, ye)

        #sélection des paires avec au moins 2 années en commun, sinon sélectionne aussi 
        # les paires ayant juste en commun la borne sup de la période d'étude
        if len(y_common) > 1: 
            L.append(paire)
    return L

def tri_paires_par_ecart_med(chemin_csv,binf,bsup, seuil_med) :

    "trie paire par écart à la médiane, et sort une vitesse moyenne pour le polygone et la période étudiés"

    #entrée : paires incluses dans la période étudiée; sortie : moyenne des vitesses des paires non éliminées par écart à la médiane
    lire = lecture_csv(chemin_csv)

    #on sélectionne les paires d'années intéressantes
    paires_interessantes = selection_paire_avt_tri_med(chemin_csv, binf, bsup)
    if len(paires_interessantes) != 0 :
        liste_vit = []

        for paire in paires_interessantes:

            #on stocke la valeur de vitesse associée à chaque paire d'années 
            liste_vit.append(lire[lire['Paire'] == str(paire)]['v'])

        #on dét la médiane des vitesses des paires étudiées
        med = np.median(liste_vit)
        liste_vit_valides = []

        for paire in paires_interessantes:

            vit = float(lire[lire['Paire'] == str(paire)]['v'])

            #on conserve une valeur v si elle est inclue dans un certain écart à la médiane
            if seuil_med * med <= vit <= (1+seuil_med) * med :
                liste_vit_valides.append(vit)

        #on fait la moyenne des valeurs de vitesse restantes : vitesse moyenne de deplacement sur
        #  le polygone et la période étudiés
        return np.mean(liste_vit_valides)
    else :
        return None

def donnees_vitesse_finales_zM(chemin_csv, seuil_med):

    liste_periodes = periodes_etude(chemin_csv)
    donnees_vitesses = []
    liste_periodes_bis = periodes_etude_separateur_tiret(liste_periodes)

    #on parcourt la liste d'années
    for  periode in liste_periodes:

            #v_moyenne sur période donnée en éliminant mauvaises paires avec écart à med donné
            v_moyenne = tri_paires_par_ecart_med(chemin_csv,periode[0],periode[1],seuil_med)
            if type(v_moyenne) == None :
                donnees_vitesses.append(None)
            else : 
                donnees_vitesses.append(v_moyenne)

    #df avec vitesse moyenne sur polygone de zone mouvante sur toutes les périodes
    df = pd.DataFrame(donnees_vitesses,index = liste_periodes_bis, columns=['v_moy'])
    df.index.name="Periode"

    return df
    
def donnees_vitesse_finales_zS(chemin_csv):
        #chemin_csv associé au df de la vitesse par paire en zone stable
        
        liste_periodes = periodes_etude(chemin_csv)
        donnees_vitesses= []

        #futur index du dataframe
        liste_periodes_bis = periodes_etude_separateur_tiret(liste_periodes)
        donnees_vitesses_medianes = []


    #chemin_csv le csv de vitesses en zone stable, qui n'a pas subi le traitement de tri par différence entre zones de mouvement et zone stable
        lire = lecture_csv(chemin_csv)


        for  periode in liste_periodes:

            #sélection des paires avec période incluse dans periode
            paires_interessantes = selection_paire_avt_tri_med(chemin_csv, periode[0], periode[1])

            if len(paires_interessantes) != 0 :
                for paire in paires_interessantes:

                #on stocke la valeur de vitesse associée à chaque paire d'années 
                    donnees_vitesses.append(lire[lire['Paire'] == str(paire)]['v'])

            #contient médiane des vitesses des paires valides sur la période étudiée
            donnees_vitesses_medianes.append(np.median(donnees_vitesses))

        #contient vitesses médiane en zs par période
        df = pd.DataFrame(donnees_vitesses_medianes, index = liste_periodes_bis, columns=['v_moy'])
        df.index.name="Periode"
        
        return df  

def taux_d_acceleration(chemin_csv):

    df = donnees_vitesse_finales_zM(chemin_csv,0.5)

    #contiendra les taux d'accel associés à chaque période
    taux_accel = []

    #v_ini correspond à la vitesse initiale, ici 2001-2006
    v_ini = df['v_moy'][0]

    periodes = periodes_etude(chemin_csv)
    periodes_tiret = periodes_etude_separateur_tiret(periodes)

    (ligne,colonne) = df.shape

    for index in range(ligne):

        #tx le taux d'accel : (vf-vi)/vi * 100 pour avoir un pourcentage
        tx = int(((df['v_moy'][index] - v_ini) / v_ini) * 100)

        taux_accel.append(tx)

    #df avec le tx d'accel en fonction des périodes
    df_sortie = pd.DataFrame(data = taux_accel, index = periodes_tiret, columns = ['taux_acceleration (%)'])
    df_sortie.index_name = 'Periode'
    return df_sortie

taux_d_acceleration('/home/gaiani/Documents/STAGE/GeoMultiCorr/src/dataframe_tri_paires_1.csv')

def formatation_donnees_serieT(chemin_csv_m,chemin_csv_s) :

    #chemin_csv_m pour le chemin du df de zone en mouvement, l'autre pour zone stable
    #recupère df de vitesse par période pour chaque zone
    df_m = donnees_vitesse_finales_zM(chemin_csv_m,0.5)
    df_s = donnees_vitesse_finales_zS(chemin_csv_s)
    (ligne,colonne) = df_m.shape

    #récupère données dans df
    vs = [df_m.v_moy[period] for period in range(ligne)]
    vm = [df_s.v_moy[period] for period in range(ligne)]
    period = periodes_etude(chemin_csv_m)

    disps = pd.DataFrame({'period' : period, 'vm' : vm, 'vs' : vs}, columns = ['period','vs','vm'])
    return disps

disps = []

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
