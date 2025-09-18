# Dataset Description – Artifact Restoration GAN

This folder contains a **subset** of the dataset used for the **GAN-based Artifact Restoration** project.  
Due to storage constraints, only a reduced version is uploaded here.

---

##  Folder Structure

- **complete/**  
  Contains **50 artifact images** collected from publicly available sources:
  - The Metropolitan Museum of Art (Met Museum)
  - Cleveland Museum of Art

- **damaged/**  
  Contains the **damaged counterparts of the 50 subset images**.  
  Each image in `complete/` has a corresponding damaged version in this folder.

---

## Notes
- The full dataset consists of **1051 complete images** and **1051 paired damaged images**.  
- The subset provided here (50 pairs) is intended for:
  - Demonstrating dataset organization  
  - Running quick experiments  
  - Ensuring reproducibility of results in limited-resource environments  

- All damaged images were generated manually to simulate real-world degradation.  

---

## ⚠ License & Usage
- Source images are publicly available from the Met Museum and Cleveland Museum websites.  
- Damaged versions were created solely for **research and academic purposes**.  
- Redistribution of the full dataset is restricted; only the subset is made available here.
