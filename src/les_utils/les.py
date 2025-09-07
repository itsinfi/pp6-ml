import dawdreamer as daw
import laion_clap as lc
import numpy as np
from clap_utils import init_clap
from dawdreamer_utils import init_dawdreamer, convert_parameters_description
import pandas as pd
from typing import Dict
from diva_utils import array_to_patch
import jax
import jax.numpy as jnp
from evosax.algorithms import LearnedES
from .create_embeddings_les import create_embeddings_les
from config import DIVA_PRESET_DIR

clap = init_clap()
engine, diva = init_dawdreamer(sample_rate=16000)

def calculate_cosine_fitnesses(audio_embeds: jnp.ndarray, goal_embed: jnp.ndarray):
    dot = jnp.dot(audio_embeds, goal_embed)
    norm = (jnp.linalg.norm(audio_embeds, axis=1)) * (jnp.linalg.norm(goal_embed))
    norm = jnp.where(norm == 0, 1e-8, norm) # prevent division by zero resulting in nan values and invalid optimizer states
    fitnesses = -dot / norm
    return fitnesses

def les(
    goal_embed: np.ndarray,
    row: Dict,
    clap: lc.CLAP_Module,
    engine: daw.RenderEngine,
    diva: daw.PluginProcessor,
    params = None,
    population_size: int = 10,
    iterations: int = 5,
    param_dim: int = 24,
):
    """
    algorithmic optimizer (learned es) for clap text embeddings and audio embeddings based on work from:
    - Lange, R. et al. (2023a): https://arxiv.org/abs/2211.11260
    - Lange, R. (2025): https://pypi.org/project/evosax/
    - Cherep, M. et al. (2024). https://arxiv.org/abs/2406.00294
    """

    # convert goal embed to jnp
    goal_embed = jnp.array(goal_embed, dtype=jnp.float32)

    # initialize les
    key = jax.random.key(0)
    initial_sol = jax.random.uniform(key, shape=(param_dim,), minval=0.0, maxval=1.0)
    print(initial_sol)
    les = LearnedES(population_size, solution=initial_sol)

    # initialize parameters
    if params is None:
        params = les.default_params
    mean = jnp.zeros(param_dim)
    
    # initialize state
    key = jax.random.key(0)
    state = les.init(key, mean, params)

    # read preset file
    with open(f"{DIVA_PRESET_DIR}{row['meta_location']}", mode='r', encoding='utf-8') as f:
        preset = f.readlines()

    # read param description
    param_desc = convert_parameters_description(diva.get_parameters_description())

    for i in range(1, iterations + 1):
        print(f"iteration {i}/{iterations} for {row['meta_location']}")

        # generate canidates
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = les.ask(key_ask, state, params)

        # reshape with sigmoid function to value range
        population = jax.nn.sigmoid(population)

        # reshape to dataframe
        df_canidates = pd.DataFrame([{
            **array_to_patch(canidate),
            'meta_location': row['meta_location'],
        } for canidate in jnp.array(population)])
        
        # synthesize audio + get audio embedding
        audio_embeds = []
        for _, canidate in df_canidates.iterrows():
            audio_embeds.append(create_embeddings_les(canidate, preset, engine, diva, clap, param_desc))
        audio_embeds = jnp.array(audio_embeds)

        # compute fitnesses (negative cosine similarity between audio and text embeddings)
        fitnesses = jnp.array(calculate_cosine_fitnesses(audio_embeds, goal_embed), dtype=jnp.float32)
        print('fitnesses', fitnesses)

        # update optimizer state
        state, _ = les.tell(key_tell, population, fitnesses, state, params)
    
    # return optimal patch
    print(f'best fitness: {state.best_fitness}')
    print(f'{state.best_solution}')
    return np.array(state.best_solution, dtype=np.float32)