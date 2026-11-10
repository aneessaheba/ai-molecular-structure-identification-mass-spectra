# Data

Data is **not committed** to git. Download it with:

```bash
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra -p data/raw
unzip data/raw/*.zip -d data/raw
```

- `raw/`: original competition files, never modified
- `processed/`: outputs of preprocessing scripts in `src/data/`
- `external/`: external datasets (e.g., public spectral libraries). **Check the competition rules on external data before using any.** Record the source here.

## File Descriptions
_TODO: describe each competition file once downloaded._
