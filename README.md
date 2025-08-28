# Install dependencies
1. install Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install
2. run `conda env create -f environment.yml`
3. run `conda run -n pp6 pip install -e .`
4. intall diva (179€ if you want to do so legally) https://u-he.com/products/diva/
5. copy .env.example as .env and specify your diva preset folder (on windows: 'C:\Users\your-name\Documents\u-he\Diva.data\Presets\Diva')
6. install presets for generating dataset (see 'data/sources.txt') and put them inside the preset folder

# Run project
run `conda run -n pp6 <executable>`
note: executables can be found in pyproject.toml (e.g. `run_count_presets`)

# Update dependencies
1. run `conda install -n pp6 <dependency1> <dependency2> ...`
2. run `conda env export -n pp6 -f environment.yml`