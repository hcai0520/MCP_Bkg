# MCP_Bkg
Code for millicharged particle (mCP) background analysis in 2x2 detecor
## Overview

This repository contains analysis code for studying detector backgrounds, including:

- Hot region identification
- Bad pixel filtering
- Repeated pixel removal
- DBSCAN clustering
- Cluster classification (track / shower)
- Event visualization

## Structure
src/
  dbscan.py
  build_clusters.py
  classify_clusters.py
  remove_near_points.py
  Find_bad_region.py
  plot.py

py/
  cluster.py

Notebooks/
  prelimenary_result.ipynb
  detector_energy_geant4.ipynb
  Nu_calib.ipynb
  geant4_mcp_analysis.ipynb


