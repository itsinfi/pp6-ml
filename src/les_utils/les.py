import dawdreamer as daw
import laion_clap as lc
import numpy as np
from clap_utils import init_clap
from dawdreamer_utils import init_dawdreamer
import pandas as pd
from typing import Dict
from diva_utils import array_to_patch
import json
import jax
import jax.numpy as jnp
from evosax.algorithms import LearnedES
from .create_embeddings_les import create_embeddings_les

clap = init_clap()
engine, diva = init_dawdreamer()

def les(
    text_embed: np.ndarray,
    row: Dict,
    clap: lc.CLAP_Module,
    engine: daw.RenderEngine,
    diva: daw.PluginProcessor,
    population_size: int = 25,
    iterations: int = 50,
    param_dim: int = 24
):
    """
    algorithmic optimizer (learned es) for clap text embeddings and audio embeddings based on work from:
    - Lange, R. et al. (2023a): https://arxiv.org/abs/2211.11260
    - Lange, R. (2025): https://pypi.org/project/evosax/
    - Cherep, M. et al. (2024). https://arxiv.org/abs/2406.00294
    """

    # initialize les
    les = LearnedES(population_size, solution=jnp.zeros(param_dim))

    # initialize parameters
    params = les.default_params
    mean = jnp.zeros(param_dim)
    
    # initialize state
    key = jax.random.key(0)
    state = les.init(key, mean, params)

    for i in range(1, iterations + 1):
        print(f'iteration {i}/{iterations}')

        # generate canidates
        key, key_ask, key_tell = jax.random.split(key, 3)
        population, state = les.ask(key_ask, state, params)

        # reshape with sigmoid function to value range
        population = jax.nn.sigmoid(population)

        # reshape to dataframe
        df_canidates = pd.DataFrame([{
            **array_to_patch(canidate),
            'meta_name': row['meta_name'], 
            'meta_location': row['meta_location'],
            'tags_categories': row['tags_categories'],
            'tags_features': row['tags_features'],
            'tags_character': row['tags_character'],
        } for canidate in np.array(population)])
        
        # synthesize audio + get audio embedding
        audio_embeds = []
        for _, canidate in df_canidates.iterrows():
            canidate_with_embeds = create_embeddings_les(canidate, engine, diva, clap)
            audio_embeds.append(np.array(
                json.loads(canidate_with_embeds['embeddings_audio']),
                dtype=np.float32,
            ))

        # compute fitness (negative cosine similarity between audio and text embeddings)
        fitness = jnp.array(
            -np.dot(audio_embeds, text_embed) / (
                np.linalg.norm(audio_embeds, axis=1) * 
                np.linalg.norm(text_embed) + 1e-8
            ),
            dtype=jnp.float32,
        )

        # update optimizer state
        state, _ = les.tell(key_tell, population, fitness, state, params)
    
    # return optimal patch
    print(f'best fitness: {state.best_fitness}')
    print(f'{state.best_solution}')
    return state.best_solution