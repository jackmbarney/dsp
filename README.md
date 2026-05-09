# dsp
Place to learn about and practice digital signal processing

## Setup

### Windows (quick)

Run the setup script from the repo root:

```bat
repo_setup\setup.bat
```

Then activate the environment:

```bat
.venv\Scripts\activate
```

### Linux / macOS (quick)

```bash
bash repo_setup/setup.sh
source .venv/bin/activate
```

### Manual (any platform)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt

# One-time: register the nbstripout git filter so notebook outputs
# are stripped from commits automatically.
nbstripout --install
```

## Notes

- `nbstripout --install` writes filter config into `.git/config`, so it's per-clone — you'll re-run it each time you clone the repo. The setup scripts handle this for you.
