class GMC_Geomorph:
    
    def __init__(self, project, ge_key):
        
        # Check existence and unique
        assert ge_key in project._geomorphs.ge_rginv_key.values, 'key not found in Geomorphs layer'
        assert project._geomorphs.value_counts('ge_rginv_key')[ge_key] == 1, f'more than 1 geomorph have the key {ge_key}'

        # Attributes
        self.project = project
        self.data = project._geomorphs[project._geomorphs.ge_rginv_key == ge_key].iloc[0]
        self.ge_pz = project.get_pzones(self.data.ge_pz_name)[0]
        self.geometry = self.data.geometry
        self.ge_key = ge_key

    def get_thumbs_overview(self, criterias=''):
        return self.ge_pz.get_thumbs_overview(criterias)
    
    def get_thumbs(self, criterias=''):
        return self.ge_pz.get_thumbs(criterias)

    def get_pairs_overview(self, criterias=''):
        return self.ge_pz.get_pairs_overview(criterias)
    
    def get_pairs(self):
        return self.ge_pz.get_pairs()
    
    def get_pairs_complete_overview(self):
        return self.get_pairs_overview()[self.get_pairs_overview().pa_status == 'complete']

    def get_pairs_complete(self):
        return [pair for pair in self.get_pairs() if pair.pa_status == 'complete']
    
    def show(self, criterias=''):
        try:
            thumb = self.get_thumbs(criterias)[0].get_geoim()
        except IndexError:
            print(f'0 thumbs for year {criterias[0]} on thi geomorph')
        thumb = thumb.cropFromVector(self.geometry)
        thumb.maskFromVector(self.project.get_geomorphs_overview(criterias))
        thumb.show()
    
    def get_pairs_on_period_overview(self, ymin, ymax):
        pairs = self.get_pairs_overview()
        pairs['chrono_min'] = pairs.apply(lambda row: min(int(row.pa_left_date.split('-')[0]), int(row.pa_right_date.split('-')[0])), axis=1)
        pairs['chrono_max'] = pairs.apply(lambda row: max(int(row.pa_left_date.split('-')[0]), int(row.pa_right_date.split('-')[0])), axis=1)
        pairs = pairs[(pairs.chrono_min>=ymin)&(pairs.chrono_max>=ymax)]
        return pairs        
        
