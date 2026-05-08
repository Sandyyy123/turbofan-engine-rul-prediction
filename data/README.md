# Data: NASA C-MAPSS Turbofan Engine Degradation Simulation

## Source
- Original: NASA Prognostics Center of Excellence (PCoE) Data Repository.
- Mirror used here: Kaggle - `behrad3d/nasa-cmaps`.
- Reference: Saxena, Goebel, Simon, Eklund (2008) "Damage propagation modeling for aircraft engine run-to-failure simulation", Int. Conf. on Prognostics and Health Management. DOI: 10.1109/PHM.2008.4711414.

## Size
- ~25 MB total. Safe to download into this folder.

## Download command
Make sure Kaggle CLI is installed and credentials are at `~/.kaggle/kaggle.json`:

```bash
cd /root/AI/liora_projects/11_nasa_turbofan/data
kaggle datasets download -d behrad3d/nasa-cmaps
unzip nasa-cmaps.zip
```

After unzip you should see a `CMaps/` (or similar) subfolder with these files:
- `train_FD001.txt` ... `train_FD004.txt` (run-to-failure training trajectories)
- `test_FD001.txt` ... `test_FD004.txt` (truncated test trajectories)
- `RUL_FD001.txt` ... `RUL_FD004.txt` (ground-truth remaining cycles for the test units)
- `readme.txt` (NASA-provided dataset description)

## File format
Each `train_FDxxx.txt` and `test_FDxxx.txt` is whitespace-separated with columns:

| col | name              | description                                |
|-----|-------------------|--------------------------------------------|
| 1   | unit_number       | engine ID                                  |
| 2   | time_in_cycles    | cycle counter (one row = one flight cycle) |
| 3-5 | op_setting_1..3   | operational settings                       |
| 6-26| sensor_1..21      | 21 noisy sensor measurements               |

Each `RUL_FDxxx.txt` is a single column of remaining cycles for the test units (one value per unit, in unit order).

## Subset characteristics (Saxena 2008)

| subset | train units | test units | op conditions | fault modes |
|--------|-------------|------------|---------------|-------------|
| FD001  | 100         | 100        | 1             | 1 (HPC)     |
| FD002  | 260         | 259        | 6             | 1 (HPC)     |
| FD003  | 100         | 100        | 1             | 2 (HPC, Fan)|
| FD004  | 248         | 249        | 6             | 2 (HPC, Fan)|

## Phase 1 use
Phase 1 scaffolds code only. **No execution.** The download is documented but the actual file fetch can be deferred to Phase 2 in the main session if Kaggle auth is unavailable inside the agent sandbox.

## Phase 2 plan
- Use FD001 as primary subset (cleanest, most-cited in literature).
- Validate on FD002-FD004 to test generalisation under varying conditions.
- Apply piece-wise linear RUL cap at 125 cycles (Heimes 2008, Zheng 2017).
- Keep raw `.txt` files in `data/`. Generated tensors and processed parquet go to `.tmp/` (not committed).
