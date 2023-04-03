from pathlib import Path
import geopandas as gpd
import gmc_thumb

class GMC_rib:

    def __init__(self, project, ri_id=None):

        # Check existence and unique
        assert ri_id in project._ribs.ri_id.values, 'key not found in ribs layer'
        assert project._ribs.value_counts('ri_id')[ri_id] == 1, f'more than 1 rib have the key {ri_id}'

        # Attributes
        self.project = project
        self.data = project._ribs[project._ribs.ri_id == ri_id].iloc[0]
        self.ri_pz = project.get_pzones(self.data.ri_pz_name)[0]
        self.ri_sp = self.ri_pz.get_sp(ri_id)[0]
        self.geometry = self.data.geometry
        self.ri_id = ri_id