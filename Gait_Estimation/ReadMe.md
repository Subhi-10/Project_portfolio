# Dataset Description – Gait Estimation Project

The dataset used in this project is **private and confidential**, as it was obtained during a research collaboration with the **Ministry of Defence**.  
Due to security and data-sharing restrictions, the raw data **cannot be uploaded** to this repository.

---

## Dataset Overview

- The dataset consists of **Inertial Measurement Unit (IMU) recordings** of human gait.  
- Sensors were placed on **lower body joints** (e.g., thigh, shin) to capture acceleration and angular velocity data.  
- From this data, **joint angles, stride length, and walking speed** were derived for deep learning–based gait estimation.  

---

##  Notes
- Instead of raw data, this repository contains:
  - **Pre-processing scripts** (for segmentation, joint angle generation, stride length estimation, etc.)  
  - **Model training scripts** (CNN, RNN, LSTM architectures)  
  - **Result visualizations and performance metrics**  

- The dataset follows a **time-series format** where each entry corresponds to a fixed-length gait window.  
- Labels such as **knee joint angles, stride length, and walking speed** were generated through preprocessing pipelines (included in this repo).

---

## License & Usage
- The raw dataset is **not publicly available**.  
- Researchers interested in replicating this work can run the provided code on their own gait IMU datasets.  
- Sharing of the original dataset is strictly restricted under project confidentiality agreements.

