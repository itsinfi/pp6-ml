# Install dependencies
1. run `conda env create -f environment.yml`
2. run `conda run -n pp6 pip install -e .`

# Run project
run `conda run -n pp6 <executable>`
note: executables can be found in pyproject.toml (e.g. `run_count_presets`)

# Update dependencies
1. run `conda install -n pp6 <dependency1> <dependency2> ...`
2. run `conda env export -n pp6 -f environment.yml`