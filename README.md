# SST-HID: Scale-aware and Shift-invariant Teeth Feature Matching for Human Identification

Code and evaluation data for the paper:

> Md. Ferdous, Saifuddin Mahmud, **Tahsin Ahmed Chowdhury**, Md. Tahmid Hasan, Shaikh al Harun.
> **SST-HID: Scale-aware and shift-invariant teeth feature matching for human identification using OBB and SSIM.**
> *Systems and Soft Computing*, 7 (2025) 200402. Elsevier, open access (CC BY 4.0).
> DOI: [10.1016/j.sasc.2025.200402](https://doi.org/10.1016/j.sasc.2025.200402)

The original repository is maintained by the corresponding author at [Md-Ferdous/SST-HID](https://github.com/Md-Ferdous/SST-HID). This copy is kept by co-author Tahsin Ahmed Chowdhury.

---

## What the method does

SST-HID identifies a person by matching **panoramic dental radiographs**, for example an antemortem record against a postmortem image. Teeth survive fire, decomposition and deliberate facial alteration far better than fingerprints or faces, so dental records are often the only usable biometric in disaster-victim identification and criminal investigations.

The pipeline has four steps:

1. **Tooth detection with oriented boxes.** A YOLOv8-OBB detector locates 16 tooth classes. Oriented bounding boxes (OBBs) follow each tooth's own angle, which horizontal boxes cannot.
2. **Scale correction with SSIM.** Radiographs taken at different times or on different machines differ in zoom. The query image is rescaled over a zoom range (0.8–1.2), and the scale with the highest Structural Similarity Index (SSIM) to the reference is kept.
3. **Shift correction.** The gap between the upper and lower jaw changes between scans. A shift vector, estimated with a statistical mean, re-aligns the two jaws.
4. **Matching with Rotated IoU.** Teeth are matched by the Rotated Intersection over Union (RIoU) of their oriented boxes. Matching is geometric throughout; no learned embedding is used.

![Oriented tooth boxes from two radiographs of the same person, before (left) and after (right) shift correction](Human%20Identification/Pipeline%20Images/matching%20image.jpg)

*Oriented tooth boxes from two radiographs (red and blue), upper and lower jaw, before (left) and after (right) shift correction.*

---

## Results reported in the paper

**Tooth detection (all 16 classes)**

| Detector | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| YOLOv8-OBB (ours) | 0.910 | 0.906 | 0.930 | 0.713 |
| YOLOv8-HBB (baseline) | 0.897 | 0.901 | 0.916 | 0.674 |

**Identification on IdC** (50 individuals, up to six teeth removed from each query to simulate extraction, trauma or surgery)

| RIoU threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.50 | 0.127 | 0.98 | 0.225 |
| 0.70 | 0.98 | 0.98 | 0.98 |
| 0.75 – 0.85 | 1.00 | 0.98 | 0.99 |
| 0.90 | 1.00 | 0.96 | 0.98 |
| 0.95 | 1.00 | 0.68 | 0.81 |

Low thresholds admit many false matches; the method is most reliable between 0.70 and 0.90.

**DoCM** (30 edited images of 10 individuals at different degrees of change): all 30 images are correctly identified with no false positives at RIoU thresholds 0.65–0.90.

**Scale and shift ablation** (10 radiographs with zoom-in, zoom-out and shifted versions): without SSIM and shift, 3 of 10 are identified; with SSIM only, 9 of 10; with SSIM and shift, all 10.

---

## Repository contents

```
Human Identification/
├── YOLOv8_obb.ipynb                     Train the YOLOv8-OBB tooth detector
├── YOLOV8_HBB.ipynb                     Train the horizontal-box (HBB) baseline
├── best.pt                              Trained YOLOv8-OBB weights (20 MB)
├── Scaling-SSIM.ipynb                   SSIM-based scale correction (zoom search)
├── HID_updated all in one.ipynb         Full identification pipeline: detection, SSIM, shift, RIoU, precision/recall
├── updated_HID(raw).ipynb               Development version of the identification pipeline
├── Human_ID(Benchmarking).ipynb         Benchmarking of the identification pipeline
├── dental_tesing.ipynb                  One-to-one matching and testing experiments
├── teeth-remove.py                      Interactive tool to remove teeth from a radiograph (simulated tooth loss)
├── IdC- (Identification Checker Dataset)/   50 reference radiographs, one per individual (640 × 640)
├── edited(IdC)/                         The same 50 radiographs with up to six teeth removed
├── DoCM Checker/                        10 reference radiographs for DoCM
├── DoCM( Degree of Change Measurement) dataset/   30 edited images, three per individual
└── Pipeline Images/                     Figures illustrating each pipeline step
modified/
├── Reference/  Zoom in/  Zoom out/  shifted/   10 radiographs each, used for the scale and shift ablation
└── *.zip                                Zipped copies of the four folders above
```

### Tooth classes

Each label combines jaw side (L/R), tooth type (C canine, I incisor, PM premolar, M molar) and jaw (U upper, D lower), giving 16 classes:
`LCD LCU LID LIU LMD LMU LPMD LPMU RCD RCU RID RIU RMD RMU RPMD RPMU`.

---

## Data

- **Detector training data:** 600 manually annotated panoramic radiographs (80/10/10 train/validation/test split), available at [doi:10.6084/m9.figshare.28856039.v1](https://doi.org/10.6084/m9.figshare.28856039.v1). The images were compiled from two public datasets: Iyer, *Dental panoramic scan* (figshare, [doi:10.6084/m9.figshare.22580263.v2](https://doi.org/10.6084/m9.figshare.22580263.v2)) and Brahmi, Jdey and Drira, *Panoramic dental X-ray dataset* (Mendeley Data, [doi:10.17632/73n3kz2k4k.3](https://doi.org/10.17632/73n3kz2k4k.3)).
- **Evaluation sets:** IdC, DoCM and the scale/shift ablation images are included in this repository.

---

## Running the code

The notebooks were written for **Google Colab** and read data from Google Drive.

1. Open a notebook in Colab and update the Drive paths to wherever you store the data.
2. Install the dependencies used across the notebooks:
   ```bash
   pip install ultralytics shapely scikit-image opencv-python roboflow
   ```
3. The training notebooks download the dataset through Roboflow. Add **your own** Roboflow API key as a Colab secret named `ROBOFLOW_API_KEY` (key icon in the Colab sidebar); the notebooks read it from there.
4. To run identification without retraining, load the provided `best.pt` weights.

---

## Citation

```bibtex
@article{ferdous2025ssthid,
  title   = {{SST-HID}: Scale-aware and shift-invariant teeth feature matching for human identification using {OBB} and {SSIM}},
  author  = {Ferdous, Md. and Mahmud, Saifuddin and Chowdhury, Tahsin Ahmed and Hasan, Md. Tahmid and Harun, Shaikh al},
  journal = {Systems and Soft Computing},
  volume  = {7},
  pages   = {200402},
  year    = {2025},
  publisher = {Elsevier},
  doi     = {10.1016/j.sasc.2025.200402}
}
```

## Licence and contact

The paper is published under CC BY 4.0. No licence has been set for the code in this repository, so please contact the authors before reusing it.

- Md. Ferdous (corresponding author): md.ferdous.kuet@gmail.com
- Tahsin Ahmed Chowdhury: tahsinchowdhury14@gmail.com
