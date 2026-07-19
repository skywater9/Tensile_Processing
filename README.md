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

### Force-displacement from separate force/dist time files (with interpolation)

```bash
python src/process_interpolate_force_disp.py
```

Optionally process only specific experiment folders under `data/raw_input`:

```bash
# One folder
python src/process_interpolate_force_disp.py test-al

# Multiple folders
python src/process_interpolate_force_disp.py test-al test-2 test-3
```

The interpolated force-displacement output keeps the continuous region at or
above 30 N that contains peak force. Change or disable the cutoff with:

```bash
python src/process_interpolate_force_disp.py --force-cutoff-n 50
python src/process_interpolate_force_disp.py --force-cutoff-n 0
```

### Stress-strain from processed force-displacement files

```bash
python src/process_stress_strain.py
```

`process_stress_strain.py` also supports folder-based experiment input and optional filtering:

```bash
# One experiment folder
python src/process_stress_strain.py test-al

# Multiple experiment folders
python src/process_stress_strain.py test-al test-2 test-3
```

## Notes

- These scripts use constants near the top of each file (paths, thresholds, geometry).
- Edit those constants if you need different input locations or parameters before running.
