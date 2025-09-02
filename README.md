# Install dependencies
1. install miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install
2. run `conda env create -f environment.yml`
3. run `conda run -n pp6 pip install -e .`
4. intall diva (179€ if you want to do so legally) https://u-he.com/products/diva/
5. copy .env.example as .env and specify your diva preset folder (on windows: 'C:\Users\your-name\Documents\u-he\Diva.data\Presets\Diva')

# Run project
run `conda run -n pp6 --live-stream <executable>`
note: executables can be found in [pyproject.toml](https://github.com/itsinfi/pp6-ml/blob/master/pyproject.toml) (e.g. `run_count_presets`)

# Update dependencies
1. run `conda install -n pp6 <dependency1> <dependency2> ...` or `conda run -n pp6 pip install <dependency1> <dependency2> ...`
2. run `conda env export -n pp6 -f environment.yml`

# Create your own dataset
TODO: describe