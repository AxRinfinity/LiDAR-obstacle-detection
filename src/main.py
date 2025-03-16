#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pagrindinis LiDAR kliūčių aptikimo programos įėjimo taškas.
"""

import os
import argparse
import time
import yaml
import numpy as np
# import open3d as o3d
from pathlib import Path

from preprocessing.point_cloud_processor import PointCloudProcessor
from segmentation.ground_plane_segmenter import GroundPlaneSegmenter
from clustering.obstacle_clusterer import ObstacleClusterer
from tracking.obstacle_tracker import ObstacleTracker
from visualization.visualizer import Visualizer

def apdoroti_argumentus():
    """Apdoroti komandinės eilutės argumentus."""
    parser = argparse.ArgumentParser(description='LiDAR Kliūčių Aptikimas')
    parser.add_argument('--config', type=str, default='config/pipeline_config.yaml',
                        help='Kelias iki konfigūracijos failo')
    parser.add_argument('--data_dir', type=str, default='data/kitti/training/velodyne',
                        help='Katalogas su taškų debesies duomenimis')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Katalogas rezultatų išsaugojimui')
    parser.add_argument('--visualize', action='store_true',
                        help='Vizualizuoti rezultatus')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Pradinis taškų debesies sekos indeksas')
    parser.add_argument('--end_idx', type=int, default=None,
                        help='Galutinis taškų debesies sekos indeksas')
    return parser.parse_args()

def ikelti_konfig(config_path):
    """Įkelti konfigūraciją iš YAML failo."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def apdoroti_tasku_debesi(tasku_debesies_kelias, vamzdyno_komponentai, config, vizualizuoti=False):
    """Apdoroti vieną taškų debesies failą per aptikimo vamzdyną."""
    # Įkelti taškų debesį
    tasku_debesis = np.fromfile(tasku_debesies_kelias, dtype=np.float32).reshape(-1, 4)
    
    # Išskleisti komponentus
    procesorius, segmentuotojas, klasterizatorius, sekėjas, vizualizatorius = vamzdyno_komponentai
    
    # Pirminis apdorojimas
    pradžios_laikas = time.time()
    filtruotas_debesis = procesorius.process(tasku_debesis)
    
    # Žemės plokštumos segmentavimas
    žemės_taškai, kliūčių_taškai = segmentuotojas.segment(filtruotas_debesis)
    
    # Kliūčių klasterizavimas
    klasteriai = klasterizatorius.cluster(kliūčių_taškai)
    
    # Kliūčių sekimas
    sekamos_kliūtys = sekėjas.update(klasteriai)
    
    # Apskaičiuoti apdorojimo laiką
    apdorojimo_laikas = time.time() - pradžios_laikas
    print(f"Apdorota {tasku_debesies_kelias.name} per {apdorojimo_laikas:.3f} sekundžių")
    
    # Vizualizacija
    if vizualizuoti:
        vizualizatorius.visualize(
            original_cloud=tasku_debesis,
            ground_points=žemės_taškai,
            obstacle_points=kliūčių_taškai,
            clusters=klasteriai,
            tracked_obstacles=sekamos_kliūtys
        )
    
    return sekamos_kliūtys, apdorojimo_laikas

def pagrindinis():
    """Pagrindinė funkcija LiDAR kliūčių aptikimo vamzdynui paleisti."""
    # Apdoroti argumentus
    args = apdoroti_argumentus()
    
    # Sukurti išvesties katalogą, jei jo nėra
    išvesties_katalogas = Path(args.output_dir)
    išvesties_katalogas.mkdir(parents=True, exist_ok=True)
    
    # Įkelti konfigūraciją
    config = ikelti_konfig(args.config)
    
    # Inicializuoti vamzdyno komponentus
    procesorius = PointCloudProcessor(config['preprocessing'])
    segmentuotojas = GroundPlaneSegmenter(config['segmentation'])
    klasterizatorius = ObstacleClusterer(config['clustering'])
    sekėjas = ObstacleTracker(config['tracking'])
    vizualizatorius = Visualizer(config['visualization'])
    
    vamzdyno_komponentai = (procesorius, segmentuotojas, klasterizatorius, sekėjas, vizualizatorius)
    
    # Gauti taškų debesies failus
    duomenų_katalogas = Path(args.data_dir)
    tasku_debesies_failai = sorted(list(duomenų_katalogas.glob('*.bin')))
    
    # Apdoroti taškų debesis
    pradžios_idx = args.start_idx
    pabaigos_idx = args.end_idx if args.end_idx is not None else len(tasku_debesies_failai)
    
    for idx in range(pradžios_idx, min(pabaigos_idx, len(tasku_debesies_failai))):
        tasku_debesies_kelias = tasku_debesies_failai[idx]
        sekamos_kliūtys, _ = apdoroti_tasku_debesi(
            tasku_debesies_kelias, 
            vamzdyno_komponentai, 
            config, 
            args.visualize
        )
        
        # Išsaugoti rezultatus
        # TODO: Įgyvendinti rezultatų išsaugojimą

if __name__ == '__main__':
    pagrindinis()