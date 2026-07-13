# Tensile Processing

Simple scripts to process tensile test CSV files.

## 1) Setup after download

From the project root:

### Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Input/output folders

- Put your input files in `data/raw_input/`
- Outputs are written to:
  - `data/processed_force_disp/`
  - `data/processed_stress_strain/`

## 3) Run each script

Run these from the project root.

Use `python` or `py` on Windows.

### Force-displacement from combined CSV files

```bash
python src/process_force_disp.py
```

### Force-displacement from separate force/dist time files (with interpolation)

```bash
python src/process_interpolate_force_disp.py
```

### Stress-strain from combined CSV files

```bash
python src/process_stress_strain.py
```

## Notes

- These scripts use constants near the top of each file (paths, thresholds, geometry).
- Edit those constants if you need different input locations or parameters before running.
