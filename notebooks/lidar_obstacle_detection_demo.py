#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LiDAR Kliūčių Aptikimo Sistemos Demonstracija (Be Open3D)

Šis skriptas demonstruoja LiDAR kliūčių aptikimo sistemos veikimą naudojant KITTI duomenų rinkinį.
Ši versija nenaudoja Open3D bibliotekos, o vietoj to naudoja matplotlib vizualizacijai.
"""

# %% [markdown]
# # LiDAR Kliūčių Aptikimo Sistemos Demonstracija (Be Open3D)
#
# Šis notebook'as demonstruoja LiDAR kliūčių aptikimo sistemos veikimą naudojant KITTI duomenų rinkinį.

# %%
# Importuojame reikalingas bibliotekas
import os
import sys
import numpy as np
import yaml
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# %% [markdown]
# ## 1. Konfigūracijos įkėlimas
#
# Pirmiausia įkelsime konfigūracijos failą, kuriame yra visi sistemos parametrai.

# %%
def ikelti_konfig(config_path):
    """Įkelti konfigūraciją iš YAML failo."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"Klaida: Konfigūracijos failas {config_path} nerastas.")
        # Grąžiname numatytąją konfigūraciją
        return {
            'preprocessing': {
                'voxel_size': 0.1,
                'x_min': -40.0,
                'x_max': 40.0,
                'y_min': -20.0,
                'y_max': 20.0,
                'z_min': -2.5,
                'z_max': 1.0
            },
            'visualization': {
                'point_size': 2.0,
                'background_color': [0.0, 0.0, 0.0],
                'original_color': [0.5, 0.5, 0.5],
                'ground_color': [0.0, 1.0, 0.0],
                'obstacle_color': [1.0, 0.0, 0.0]
            }
        }

# Įkeliame konfigūraciją
config_path = Path('../config/pipeline_config.yaml')
config = ikelti_konfig(config_path)

# Parodome konfigūracijos struktūrą
print("Konfigūracijos struktūra:")
for key, value in config.items():
    print(f"\n{key}:")
    for param, val in value.items():
        print(f"  {param}: {val}")

# %% [markdown]
# ## 2. Taškų Debesies Įkėlimas
#
# Dabar įkelsime taškų debesį iš KITTI duomenų rinkinio. Jei neturite KITTI duomenų, galite naudoti pavyzdinį taškų debesį.

# %%
def ikelti_tasku_debesi(file_path):
    """Įkelti taškų debesį iš KITTI .bin failo."""
    try:
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    except FileNotFoundError:
        print(f"Klaida: Taškų debesies failas {file_path} nerastas.")
        return None

# Nurodome kelią iki taškų debesies failo
# Pakeiskite šį kelią į savo KITTI duomenų katalogą
tasku_debesies_kelias = Path('../data/kitti/training/velodyne/000000.bin')

# Tikriname, ar failas egzistuoja
if not tasku_debesies_kelias.exists():
    print(f"Klaida: Taškų debesies failas {tasku_debesies_kelias} neegzistuoja.")
    print("Prašome atsisiųsti KITTI duomenų rinkinį pagal instrukcijas data/README.md faile.")
    # Sukuriame dirbtinį taškų debesį demonstracijai
    print("Sukuriamas dirbtinis taškų debesis demonstracijai...")
    # Sukuriame 10000 atsitiktinių taškų
    x = np.random.uniform(-20, 20, 10000)
    y = np.random.uniform(-20, 20, 10000)
    z = np.random.uniform(-2, 1, 10000)
    intensity = np.random.uniform(0, 1, 10000)
    tasku_debesis = np.column_stack((x, y, z, intensity))
    # Pridedame kelis taškus, kurie bus žemės plokštuma
    ground_x = np.random.uniform(-20, 20, 5000)
    ground_y = np.random.uniform(-20, 20, 5000)
    ground_z = np.random.uniform(-2, -1.8, 5000)
    ground_intensity = np.random.uniform(0, 1, 5000)
    ground_points = np.column_stack((ground_x, ground_y, ground_z, ground_intensity))
    tasku_debesis = np.vstack((tasku_debesis, ground_points))
else:
    # Įkeliame taškų debesį
    tasku_debesis = ikelti_tasku_debesi(tasku_debesies_kelias)
    print(f"Įkeltas taškų debesis iš {tasku_debesies_kelias}")

print(f"Taškų debesies forma: {tasku_debesis.shape}")
print(f"Pirmi 5 taškai:\n{tasku_debesis[:5]}")

# %% [markdown]
# ## 3. Taškų Debesies Apdorojimas
#
# Dabar apdorosime taškų debesį, atlikdami paprastą filtravimą ir segmentavimą.

# %%
# Filtruojame taškų debesį pagal ROI (Region of Interest)
def filtruoti_roi(tasku_debesis, x_min, x_max, y_min, y_max, z_min, z_max):
    """Filtruoti taškų debesį pagal ROI."""
    x, y, z = tasku_debesis[:, 0], tasku_debesis[:, 1], tasku_debesis[:, 2]
    mask = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max) & (z >= z_min) & (z <= z_max)
    return tasku_debesis[mask]

# Segmentuojame žemės plokštumą pagal z koordinatę
def segmentuoti_zeme(tasku_debesis, z_threshold=-1.5):
    """Segmentuoti žemės plokštumą pagal z koordinatę."""
    z = tasku_debesis[:, 2]
    zemes_mask = z <= z_threshold
    zemes_taskai = tasku_debesis[zemes_mask]
    kliuciu_taskai = tasku_debesis[~zemes_mask]
    return zemes_taskai, kliuciu_taskai

# Klasterizuojame kliūčių taškus pagal Euklidinį atstumą
def klasterizuoti_kliutis(kliuciu_taskai, distance_threshold=0.5, min_cluster_size=10):
    """Paprastas klasterizavimas pagal Euklidinį atstumą."""
    if len(kliuciu_taskai) == 0:
        return []
    
    # Inicializuojame klasterius
    klasteriai = []
    apdoroti_taskai = np.zeros(len(kliuciu_taskai), dtype=bool)
    
    # Iteruojame per visus taškus
    for i in range(len(kliuciu_taskai)):
        if apdoroti_taskai[i]:
            continue
        
        # Inicializuojame naują klasterį
        klasterio_taskai = [i]
        apdoroti_taskai[i] = True
        
        # Augame klasterį
        j = 0
        while j < len(klasterio_taskai):
            current_idx = klasterio_taskai[j]
            current_point = kliuciu_taskai[current_idx, :3]
            
            # Ieškome kaimynų
            distances = np.linalg.norm(kliuciu_taskai[:, :3] - current_point, axis=1)
            kaimynu_idx = np.where((distances <= distance_threshold) & (~apdoroti_taskai))[0]
            
            # Pridedame kaimynus į klasterį
            klasterio_taskai.extend(kaimynu_idx)
            apdoroti_taskai[kaimynu_idx] = True
            
            j += 1
        
        # Tikriname, ar klasteris pakankamai didelis
        if len(klasterio_taskai) >= min_cluster_size:
            klasterio_taskai_idx = np.array(klasterio_taskai)
            klasterio_taskai_data = kliuciu_taskai[klasterio_taskai_idx]
            
            # Apskaičiuojame klasterio centroidą
            centroidas = np.mean(klasterio_taskai_data[:, :3], axis=0)
            
            # Apskaičiuojame klasterio dėžę
            min_bound = np.min(klasterio_taskai_data[:, :3], axis=0)
            max_bound = np.max(klasterio_taskai_data[:, :3], axis=0)
            dydis = max_bound - min_bound
            
            # Sukuriame klasterio objektą
            klasteris = {
                'points': klasterio_taskai_data,
                'centroid': centroidas,
                'bbox': {
                    'center': (min_bound + max_bound) / 2,
                    'size': dydis,
                    'type': 'axis_aligned'
                },
                'color': np.random.rand(3),
                'size': len(klasterio_taskai)
            }
            
            klasteriai.append(klasteris)
    
    return klasteriai

# Apdorojame taškų debesį
pradžios_laikas = time.time()

# Filtruojame ROI
x_min, x_max = config['preprocessing']['x_min'], config['preprocessing']['x_max']
y_min, y_max = config['preprocessing']['y_min'], config['preprocessing']['y_max']
z_min, z_max = config['preprocessing']['z_min'], config['preprocessing']['z_max']
filtruotas_debesis = filtruoti_roi(tasku_debesis, x_min, x_max, y_min, y_max, z_min, z_max)
filtravimo_laikas = time.time() - pradžios_laikas
print(f"Filtravimas: {filtravimo_laikas:.3f} sekundžių")
print(f"Filtruoto debesies forma: {filtruotas_debesis.shape}")

# Segmentuojame žemės plokštumą
pradžios_laikas = time.time()
žemės_taškai, kliūčių_taškai = segmentuoti_zeme(filtruotas_debesis)
segmentavimo_laikas = time.time() - pradžios_laikas
print(f"Žemės plokštumos segmentavimas: {segmentavimo_laikas:.3f} sekundžių")
print(f"Žemės taškų skaičius: {žemės_taškai.shape[0]}")
print(f"Kliūčių taškų skaičius: {kliūčių_taškai.shape[0]}")

# Klasterizuojame kliūtis
pradžios_laikas = time.time()
klasteriai = klasterizuoti_kliutis(kliūčių_taškai)
klasterizavimo_laikas = time.time() - pradžios_laikas
print(f"Kliūčių klasterizavimas: {klasterizavimo_laikas:.3f} sekundžių")
print(f"Aptiktų klasterių skaičius: {len(klasteriai)}")

# %% [markdown]
# ## 4. Rezultatų Vizualizacija
#
# Vizualizuosime apdorojimo rezultatus naudodami matplotlib.

# %%
# Vizualizuojame rezultatus
def vizualizuoti_tasku_debesi(original_cloud=None, ground_points=None, obstacle_points=None, clusters=None):
    """Vizualizuoti taškų debesį ir apdorojimo rezultatus naudojant matplotlib."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Nustatome ašių ribas
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    
    # Nustatome ašių pavadinimus
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Nustatome pavadinimą
    ax.set_title('LiDAR Taškų Debesis ir Aptiktos Kliūtys')
    
    # Atrenkame taškus vizualizacijai (sumažiname taškų skaičių, kad vizualizacija būtų greitesnė)
    max_points = 5000
    
    # Vizualizuojame žemės taškus
    if ground_points is not None and len(ground_points) > 0:
        # Atrenkame taškus
        if len(ground_points) > max_points:
            indices = np.random.choice(len(ground_points), max_points, replace=False)
            ground_sample = ground_points[indices]
        else:
            ground_sample = ground_points
        
        # Vizualizuojame
        ax.scatter(
            ground_sample[:, 0], ground_sample[:, 1], ground_sample[:, 2],
            c='green', marker='.', s=1, alpha=0.5, label='Žemės taškai'
        )
    
    # Vizualizuojame kliūčių taškus
    if obstacle_points is not None and len(obstacle_points) > 0:
        # Atrenkame taškus
        if len(obstacle_points) > max_points:
            indices = np.random.choice(len(obstacle_points), max_points, replace=False)
            obstacle_sample = obstacle_points[indices]
        else:
            obstacle_sample = obstacle_points
        
        # Vizualizuojame
        ax.scatter(
            obstacle_sample[:, 0], obstacle_sample[:, 1], obstacle_sample[:, 2],
            c='red', marker='.', s=1, alpha=0.5, label='Kliūčių taškai'
        )
    
    # Vizualizuojame klasterius
    if clusters is not None and len(clusters) > 0:
        for i, cluster in enumerate(clusters):
            # Gauname klasterio spalvą
            color = cluster['color']
            
            # Vizualizuojame klasterio taškus
            cluster_points = cluster['points']
            if len(cluster_points) > 100:
                indices = np.random.choice(len(cluster_points), 100, replace=False)
                cluster_sample = cluster_points[indices]
            else:
                cluster_sample = cluster_points
            
            ax.scatter(
                cluster_sample[:, 0], cluster_sample[:, 1], cluster_sample[:, 2],
                c=[color], marker='.', s=3, alpha=0.8, label=f'Klasteris {i+1}'
            )
            
            # Vizualizuojame klasterio dėžę
            bbox = cluster['bbox']
            center = bbox['center']
            size = bbox['size']
            
            # Apskaičiuojame dėžės kampus
            min_bound = center - size / 2
            max_bound = center + size / 2
            
            # Sukuriame dėžės briaunas
            x = [min_bound[0], max_bound[0]]
            y = [min_bound[1], max_bound[1]]
            z = [min_bound[2], max_bound[2]]
            
            # Vizualizuojame dėžę
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        ax.plot(
                            [x[i], x[i]], [y[j], y[j]], [z[k], z[1-k]],
                            c=color, linewidth=1
                        )
                        ax.plot(
                            [x[i], x[i]], [y[j], y[1-j]], [z[k], z[k]],
                            c=color, linewidth=1
                        )
                        ax.plot(
                            [x[i], x[1-i]], [y[j], y[j]], [z[k], z[k]],
                            c=color, linewidth=1
                        )
    
    # Pridedame legendą
    ax.legend()
    
    # Rodome grafiką
    plt.tight_layout()
    plt.show()

# Vizualizuojame rezultatus
vizualizuoti_tasku_debesi(
    original_cloud=tasku_debesis,
    ground_points=žemės_taškai,
    obstacle_points=kliūčių_taškai,
    clusters=klasteriai
)

# %% [markdown]
# ## 5. Klasterių Analizė
#
# Analizuosime aptiktus klasterius ir jų savybes.

# %%
# Analizuojame klasterius
if len(klasteriai) > 0:
    print("Klasterių analizė:")
    for i, klasteris in enumerate(klasteriai):
        print(f"\nKlasteris {i+1}:")
        print(f"  Taškų skaičius: {klasteris['size']}")
        print(f"  Centroidas: {klasteris['centroid']}")
        print(f"  Dėžės centras: {klasteris['bbox']['center']}")
        print(f"  Dėžės dydis: {klasteris['bbox']['size']}")
        print(f"  Dėžės tipas: {klasteris['bbox']['type']}")
else:
    print("Nerasta jokių klasterių.")

# %% [markdown]
# ## 6. Išvados
#
# Šiame notebook'e pademonstravome supaprastintą LiDAR kliūčių aptikimo sistemą. Sistema apdoroja taškų debesį, segmentuoja žemės plokštumą ir klasterizuoja kliūtis.
#
# Pagrindiniai sistemos privalumai:
# - Paprastas ir lengvai suprantamas kodas
# - Nereikalauja sudėtingų bibliotekų (tik numpy ir matplotlib)
# - Gali būti naudojamas kaip pradinis taškas sudėtingesnėms sistemoms
#
# Tolimesni patobulinimai galėtų apimti:
# - Sudėtingesnių segmentavimo algoritmų naudojimą
# - Efektyvesnių klasterizavimo metodų įgyvendinimą
# - Objektų sekimo per laiką pridėjimą
# - Kliūčių klasifikavimą 