import json
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

def describe_column(meta):
    """
    Utility function for describing a dataset column (see below for usage)
    """
    def f(x):
        d = pd.Series(name=x.name, dtype=object)
        m = next(m for m in meta if m['name'] == x.name)
        d['Type'] = m['type']
        d['#NaN'] = x.isna().sum()
        d['Description'] = m['desc']
        if m['type'] == 'categorical':
            counts = x.dropna().map(dict(enumerate(m['cats']))).value_counts().sort_index()
            d['Statistics'] = ', '.join(f'{c}({n})' for c, n in counts.items())
        elif m['type'] == 'real' or m['type'] == 'integer':
            stats = x.dropna().agg(['mean', 'std', 'min', 'max'])
            d['Statistics'] = ', '.join(f'{s}={v :.1f}' for s, v in stats.items())
        elif m['type'] == 'boolean':
            counts = x.dropna().astype(bool).value_counts().sort_index()
            d['Statistics'] = ', '.join(f'{c}({n})' for c, n in counts.items())
        else:
            d['Statistics'] = f'#unique={x.nunique()}'
        return d
    return f

def describe_data(data, meta):
    desc = data.apply(describe_column(meta)).T
    desc = desc.style.set_properties(**{'text-align': 'left'})
    desc = desc.set_table_styles([ dict(selector='th', props=[('text-align', 'left')])])
    return desc


def display_apartments():
    apartments = pd.read_csv('data/apartments_train.csv')
    print(f'Loaded {len(apartments)} apartments')
    with open('data/apartments_meta.json') as f: 
        apartments_meta = json.load(f)
    describe_data(apartments, apartments_meta)

def display_buildings():
    buildings = pd.read_csv('data/buildings_train.csv')
    print(f'Loaded {len(buildings)} buildings')
    with open('data/buildings_meta.json') as f: 
        buildings_meta = json.load(f)
    buildings.head()
    describe_data(buildings, buildings_meta)

def apartment_to_building():
    apartments = pd.read_csv('data/apartments_train.csv')
    buildings = pd.read_csv('data/buildings_train.csv')
    print(f'All apartments have an associated building: {apartments.building_id.isin(buildings.id).all()}')
    data = pd.merge(apartments, buildings.set_index('id'), how='left', left_on='building_id', right_index=True)
    data.head()