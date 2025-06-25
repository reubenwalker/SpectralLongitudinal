# SpectralLongitudinal
# Longitudinal Spectral Development in Classical Singers

This repository contains the code, data, and supplementary materials for the study:

> **“Retrospective Longitudinal Analysis of Spectral Features Reveals Divergent Vocal Development Patterns for Treble and Non-Treble Singers”**

---

## 📄 Abstract

This study investigates the **longitudinal spectral development** of male and female classical singers throughout conservatory training. While classical singing techniques share commonalities across voice types, physiological differences have led to gender-specific pedagogy. Previous acoustic research has explored differences in resonance strategies between genders and voice types; however, **little is known about how these spectral characteristics develop during vocal training**.

In this **retrospective longitudinal study**, recordings from **117 classical voice students** at the Hochschule für Musik Carl Maria von Weber Dresden (2008–2018) were analyzed across their four-year bachelor studies. 

Voice types were grouped as:
- **Treble voices**: sopranos, mezzo-sopranos, altos, and countertenors  
- **Non-treble voices**: tenors, baritones, and basses

Spectral measures were derived from three vocal exercises using **Long-Term Average Spectrum (LTAS)**, and analyzed using **linear mixed-effects models** to assess changes over time by voice group.

**Key findings**:
- Treble singers increased relative acoustic energy in the **fundamental frequency (f₀)** range.
- Non-treble singers increased acoustic energy **above 1000 Hz**.
- Female singers exhibited **increased vocal periodicity** over time, suggesting reduced breathiness.

---

## 📁 Repository Contents

| File/Folder | Description |
|-------------|-------------|
| `StatAnalysis_JASA_20250622.Rmd` | R Markdown file containing data processing, statistical analysis, and visualizations |
| `StatAnalysis_JASA_20250622.pdf` | Rendered PDF of the R Markdown document |
| `Klang2.csv` | High Sustained Phonation Data |
| `Klang1.csv` | Medium Sustained Phonation Data |
| `Klang6.csv` | Repertoire Sample Data |
| `JASA_scripts_Final.py` | Python script for calculating LTAS-based spectral measures |
| `Audio Samples/` | Folder containing four representative audio samples used in the study |
| `.gitignore` | Git ignore file (optional, may exclude audio or other large files) |
| `README.md` | This file |

---

## How to Run the Code

### R Analysis

1. Open `analysis.Rmd` in RStudio.
2. Ensure the working directory is set automatically (e.g., via `rstudioapi` if applicable).
3. Ensure all required R packages are installed (`lme4`, `ggplot2`, etc.).
4. Knit the document to PDF or HTML.

### Python Script

To run the LTAS analysis separately:
Navigate to folder with audio files with naming convention in the form:
audioID&YYYY_MM_DD&test1.wav
Where 
test1: Medium Sustained Phonation
test2: Sustained triad 
test4: Messa di voce
test6: Avezzo a vivere 

```bash
python JASA_scripts_Final.py
