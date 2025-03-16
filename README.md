# 🚗 LiDAR Obstacle Detection / LiDAR Kliūčių Aptikimas

Šis projektas įgyvendina **kliūčių aptikimo sistemą** naudojant **KITTI duomenų rinkinį** ir **taškų debesies apdorojimo** technikas. Sistema gali aptikti ir sekti objektus (automobilius, pėsčiuosius ir kt.) iš LiDAR taškų debesies duomenų.

This project implements an **obstacle detection system** using the **KITTI dataset** and **point cloud processing** techniques. The system can detect and track objects (cars, pedestrians, etc.) from LiDAR point cloud data.

## 📌 Projekto apžvalga / Project Overview

**SVARBU**: Sistema sukurta veikti **be Open3D bibliotekos**, naudojant tik standartines Python bibliotekas kaip NumPy, scikit-learn ir matplotlib. Tai leidžia sistemai veikti platesnėje aplinkoje, tačiau reikalauja sudėtingesnio kodo.

**IMPORTANT**: The system is designed to work **without the Open3D library**, using only standard Python libraries such as NumPy, scikit-learn, and matplotlib. This allows the system to run in a wider range of environments but requires more complex code.

Sistema susideda iš kelių pagrindinių komponentų:

1. **Taškų debesies apdorojimas** - filtravimas, vokselizacija ir triukšmo šalinimas
2. **Žemės plokštumos segmentavimas** - atskiria žemės taškus nuo kliūčių
3. **Kliūčių klasterizavimas** - grupuoja taškus į atskirus objektus
4. **Kliūčių sekimas** - seka objektus per laiką naudojant Kalmano filtrą
5. **Vizualizacija** - rezultatų atvaizdavimas naudojant Matplotlib

The system consists of several main components:

1. **Point Cloud Processing** - filtering, voxelization, and noise removal
2. **Ground Plane Segmentation** - separates ground points from obstacles
3. **Obstacle Clustering** - groups points into distinct objects
4. **Obstacle Tracking** - tracks objects over time using a Kalman filter
5. **Visualization** - displays results using Matplotlib

## 💻 Minimalūs sistemos reikalavimai / Minimum System Requirements

Dėl intensyvaus taškų debesies apdorojimo, sistemai reikalingi šie minimalūs reikalavimai:

Due to intensive point cloud processing, the system requires these minimum specifications:

- **Procesorius / CPU**: 4 branduoliai, 2.5 GHz (rekomenduojama 8 branduolių, 3.0+ GHz)
- **Atmintis / RAM**: 8 GB (rekomenduojama 16+ GB)
- **Diskas / Storage**: 10 GB laisvos vietos (KITTI duomenų rinkiniui reikia papildomai ~30 GB)
- **Operacinė sistema / OS**: Windows 10/11, macOS 10.14+, Linux (Ubuntu 18.04+)
- **Python**: 3.7 arba naujesnė versija

## 🔧 Diegimas / Installation

### Priklausomybės / Dependencies

```bash
pip install numpy matplotlib scikit-learn pyyaml
```

### KITTI duomenų rinkinys / KITTI Dataset

1. Atsisiųskite KITTI duomenų rinkinį iš [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_object.php)
2. Išskleiskite duomenis į `data/kitti/` katalogą pagal instrukcijas `data/README.md` faile

1. Download the KITTI dataset from [KITTI Vision Benchmark Suite](http://www.cvlibs.net/datasets/kitti/eval_object.php)
2. Extract the data to the `data/kitti/` directory following the instructions in the `data/README.md` file

## 🚀 Naudojimas / Usage

### Komandinė eilutė / Command Line

```bash
# Paleisti sistemą
python src/main.py --config config/pipeline_config.yaml --data_dir data/kitti/training/velodyne --visualize

# Paleisti demonstracinį Jupyter notebook
jupyter notebook notebooks/lidar_obstacle_detection_demo.py
```

### Programinis naudojimas / Programmatic Usage

```python
from preprocessing.point_cloud_processor import PointCloudProcessor
from segmentation.ground_plane_segmenter import GroundPlaneSegmenter
from clustering.obstacle_clusterer import ObstacleClusterer
from tracking.obstacle_tracker import ObstacleTracker
from visualization.visualizer import Visualizer

# Inicializuoti komponentus / Initialize components
procesorius = PointCloudProcessor(config['preprocessing'])
segmentuotojas = GroundPlaneSegmenter(config['segmentation'])
klasterizatorius = ObstacleClusterer(config['clustering'])
sekėjas = ObstacleTracker(config['tracking'])
vizualizatorius = Visualizer(config['visualization'])

# Apdoroti taškų debesį / Process point cloud
filtruotas_debesis = procesorius.process(tasku_debesis)
žemės_taškai, kliūčių_taškai = segmentuotojas.segment(filtruotas_debesis)
klasteriai = klasterizatorius.cluster(kliūčių_taškai)
sekamos_kliūtys = sekėjas.update(klasteriai)

# Vizualizuoti rezultatus / Visualize results
vizualizatorius.visualize(
    original_cloud=tasku_debesis,
    ground_points=žemės_taškai,
    obstacle_points=kliūčių_taškai,
    clusters=klasteriai,
    tracked_obstacles=sekamos_kliūtys
)
```

## 📂 Projekto struktūra / Project Structure

```
LiDAR-obstacle-detection/
├── config/                  # Konfigūracijos failai / Configuration files
│   └── pipeline_config.yaml # Pagrindinė konfigūracija / Main configuration
├── data/                    # Duomenų katalogas / Data directory
│   └── kitti/               # KITTI duomenų rinkinys / KITTI dataset
├── notebooks/               # Jupyter notebooks
│   └── lidar_obstacle_detection_demo.py  # Demonstracinis notebook / Demo notebook
├── src/                     # Išeities kodas / Source code
│   ├── clustering/          # Klasterizavimo moduliai / Clustering modules
│   ├── preprocessing/       # Pirminio apdorojimo moduliai / Preprocessing modules
│   ├── segmentation/        # Segmentavimo moduliai / Segmentation modules
│   ├── tracking/            # Sekimo moduliai / Tracking modules
│   ├── visualization/       # Vizualizacijos moduliai / Visualization modules
│   ├── example.py           # Pavyzdinis skriptas / Example script
│   └── main.py              # Pagrindinis įėjimo taškas / Main entry point
└── README.md                # Projekto aprašymas / Project description
```

## 🔍 Komponentų aprašymas / Component Description

### PointCloudProcessor

Atlieka pirminį taškų debesies apdorojimą:
- Vokselių tinklelio sumažinimas (Voxel grid downsampling) - įgyvendintas naudojant NumPy vietoj Open3D
- Regiono apkarpymas (ROI cropping) - naudojamos NumPy maskas
- Triukšmo šalinimas (Outlier removal) - naudojamas scikit-learn NearestNeighbors

Performs initial point cloud processing:
- Voxel grid downsampling - implemented using NumPy instead of Open3D
- Region of interest (ROI) cropping - using NumPy masks
- Outlier removal - using scikit-learn NearestNeighbors

### GroundPlaneSegmenter

Segmentuoja žemės plokštumą nuo kliūčių:
- RANSAC algoritmas plokštumos aptikimui - naudojamas scikit-learn RANSACRegressor vietoj Open3D
- Normalės tikrinimas plokštumos validavimui
- Žemės ir kliūčių taškų atskyrimas

Segments the ground plane from obstacles:
- RANSAC algorithm for plane detection - using scikit-learn RANSACRegressor instead of Open3D
- Normal checking for plane validation
- Separation of ground and obstacle points

### ObstacleClusterer

Klasterizuoja kliūčių taškus į atskirus objektus:
- DBSCAN klasterizavimo algoritmas - naudojamas scikit-learn
- Klasterių filtravimas pagal dydį
- Ribojančių dėžių (bounding boxes) generavimas - įgyvendintas naudojant NumPy vietoj Open3D

Clusters obstacle points into distinct objects:
- DBSCAN clustering algorithm - using scikit-learn
- Cluster filtering based on size
- Bounding box generation - implemented using NumPy instead of Open3D

### ObstacleTracker

Seka kliūtis per laiką:
- Kalmano filtras būsenai įvertinti
- Duomenų asociacija tarp aptikimų ir sekimų
- Sekimų inicializavimas ir valdymas

Tracks obstacles over time:
- Kalman filter for state estimation
- Data association between detections and tracks
- Track initialization and management

### Visualizer

Vizualizuoja rezultatus:
- Taškų debesies vizualizacija - naudojamas matplotlib 3D vietoj Open3D
- Žemės plokštumos vizualizacija
- Kliūčių klasterių vizualizacija
- Sekamų kliūčių vizualizacija

Visualizes results:
- Point cloud visualization - using matplotlib 3D instead of Open3D
- Ground plane visualization
- Obstacle cluster visualization
- Tracked obstacle visualization

## 🔄 Techniniai iššūkiai be Open3D / Technical Challenges without Open3D

Open3D biblioteka yra specializuota 3D duomenų apdorojimui ir vizualizacijai, todėl jos nenaudojimas sukėlė šiuos techninius iššūkius:

The Open3D library is specialized for 3D data processing and visualization, so not using it created these technical challenges:

1. **Vokselių tinklelio sumažinimas (Voxel Grid Downsampling)** - Teko įgyvendinti vokselizacijos algoritmą nuo nulio, naudojant NumPy. Tai reikalauja efektyvaus taškų grupavimo į vokselius ir centroidų skaičiavimo.

2. **RANSAC plokštumos aptikimas** - Open3D turi optimizuotą RANSAC plokštumos aptikimo algoritmą. Vietoj jo naudojame scikit-learn RANSACRegressor, kuris nėra specializuotas 3D plokštumoms, todėl reikėjo papildomos logikos plokštumos modelio konvertavimui.

3. **Ribojančių dėžių (Bounding Boxes) skaičiavimas** - Open3D palaiko orientuotas ribojančias dėžes (OBB), kurios geriau apgaubia objektus. Be Open3D teko įgyvendinti paprastesnes ašims sulygiuotas dėžes arba sudėtingą OBB skaičiavimą naudojant NumPy.

4. **3D vizualizacija** - Open3D turi interaktyvią 3D vizualizaciją su kamera kontrole. Vietoj jos naudojame matplotlib, kuris turi ribotas 3D galimybes ir yra mažiau efektyvus dideliems taškų debesiams.

5. **Efektyvumas** - Open3D yra optimizuotas C++ bibliotekos pagrindu, todėl yra daug greitesnis nei grynas Python. Mūsų sprendimas yra lėtesnis, ypač dirbant su dideliais taškų debesimis.

1. **Voxel Grid Downsampling** - Had to implement the voxelization algorithm from scratch using NumPy. This requires efficient grouping of points into voxels and centroid calculation.

2. **RANSAC Plane Detection** - Open3D has an optimized RANSAC plane detection algorithm. Instead, we use scikit-learn's RANSACRegressor, which is not specialized for 3D planes, requiring additional logic for plane model conversion.

3. **Bounding Box Calculation** - Open3D supports oriented bounding boxes (OBB), which better enclose objects. Without Open3D, we had to implement simpler axis-aligned boxes or complex OBB calculation using NumPy.

4. **3D Visualization** - Open3D has interactive 3D visualization with camera control. Instead, we use matplotlib, which has limited 3D capabilities and is less efficient for large point clouds.

5. **Performance** - Open3D is optimized with a C++ backend, making it much faster than pure Python. Our solution is slower, especially when working with large point clouds.

## 📊 Rezultatai / Results

Sistema gali aptikti ir sekti įvairias kliūtis iš LiDAR taškų debesies duomenų. Rezultatai apima:
- Žemės plokštumos segmentavimą
- Kliūčių klasterizavimą
- Kliūčių sekimą per laiką
- 3D vizualizaciją

The system can detect and track various obstacles from LiDAR point cloud data. Results include:
- Ground plane segmentation
- Obstacle clustering
- Obstacle tracking over time
- 3D visualization

## 📝 Licencija / License

Šis projektas yra platinamas pagal MIT licenciją. Žr. `LICENSE` failą daugiau informacijos.

This project is distributed under the MIT License. See the `LICENSE` file for more information.

## 🤝 Prisidėjimas / Contributing

Prisidėjimai yra laukiami! Prašome sukurti "pull request" arba atidaryti "issue" diskusijai.

Contributions are welcome! Please create a pull request or open an issue for discussion.


