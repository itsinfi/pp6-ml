from __future__ import annotations
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from evosax.algorithms import LearnedES, CMA_ES
from diva_utils import array_to_patch, DIVA_ENV_TO_INDEX_MAP, DIVA_MAP
from dawdreamer_utils import init_dawdreamer, read_diva_value, convert_parameters_description
from config import NOTE_PITCH, NOTE_DURATION, NOTE_VELOCITY, DIVA_PRESET_DIR, RENDER_DURATION
import dawdreamer as daw
import numpy as np
from typing import List, Tuple, Any
import pandas as pd
from clap_utils import init_clap

def train_les(
    embeds_train: np.ndarray[np.ndarray[np.float32]],
    df_train: pd.DataFrame,
    meta_iter: int = 2,
    les_pop: int = 2,
    les_iter: int = 2,
    param_dim: int = 24,
):
    # filter input data
    rows_idxs_np = np.array(df_train.index.to_numpy())

    mask = ~(np.all(embeds_train == 0, axis=1))

    goal_embeds = jnp.array(embeds_train[mask], dtype=jnp.float32)
    row_idxs = rows_idxs_np[mask]

    meta_pop = goal_embeds.shape[0]

    # init les
    key = jax.random.PRNGKey(42)
    init_sol = jax.random.uniform(key, shape=(param_dim,), minval=0.0, maxval=1.0)
    les = LearnedES(population_size=les_pop, solution=init_sol)
    init_params = les.default_params

    # flatten les params
    meta_sol, unravel_fn = ravel_pytree(init_params)
    meta_dim = meta_sol.shape[0]
    print(f"Meta search dimension (flattened LES net params): {meta_dim}")

    # init cma-es
    key, key_cma = jax.random.split(key)
    cma_es = CMA_ES(population_size=meta_pop, solution=meta_sol)
    meta_params = cma_es.default_params
    meta_state = cma_es.init(key_cma, meta_sol, meta_params)

    # init clap
    clap = init_clap()

    # init dawdreamer
    _, diva = init_dawdreamer()

    # init parameter description
    param_desc = convert_parameters_description(diva.get_parameters_description())

    def calc_fit(cands: jnp.ndarray, goal_embed: np.ndarray, base_patch: List[Tuple[str, Any]]):
        embeds = []
        for cand in cands:
            cand_patch = array_to_patch(cand)

            # iterate through the envelope params from the dataframe and replace params in patch
            for key, val in cand_patch.items():

                if key in DIVA_ENV_TO_INDEX_MAP:
                    idx = DIVA_ENV_TO_INDEX_MAP[key]

                    for i, (k, _) in enumerate(base_patch):
                        if k == idx:
                            base_patch[i] = (k, val)
                            break

            # init dawdreamer and base patch (needs to be initialized again because of an internal dawdreamer bug)
            engine, diva = init_dawdreamer(sample_rate=16000)

            # change preset
            diva.set_patch(base_patch)

            # load diva into engine graph
            engine.load_graph([(diva, [])])

            # load midi config
            diva.add_midi_note(note=NOTE_PITCH, velocity=NOTE_VELOCITY, start_time=0.5, duration=NOTE_DURATION) # do not set start_time to zero, there will be no audible audio signal!

            # render audio
            engine.render(RENDER_DURATION)
            
            # get audio, make audio signal mono, convert it to ndarray and expand its dimensionality to 2 for clap
            audio_np = np.expand_dims(np.asarray(np.mean(engine.get_audio(), axis=0), dtype=np.float32), axis=0)

            # generate audio embedding
            embeds.append(np.asarray(clap.get_audio_embedding_from_data(audio_np, use_tensor=False)[0], dtype= np.float32))
            
        # calculate negative cosine similarity
        embeds = jnp.array(embeds)
        dot = jnp.dot(jnp.array(embeds), goal_embed)
        norm = (jnp.linalg.norm(embeds, axis=1)) * (jnp.linalg.norm(goal_embed))
        norm = jnp.where(norm == 0, 1e-8, norm) # prevent division by zero resulting in nan values and invalid optimizer states
        return jnp.array(-dot / norm, dtype=jnp.float32)


    def read_patch(diva: daw.PluginProcessor, preset: str):
        patch = []
        for key, val in (param for param in diva.get_patch()):
            if key in DIVA_MAP:
                patch.append((key, read_diva_value(lines=preset, index=key, group=DIVA_MAP[key]['group'], key=DIVA_MAP[key]['key'], param_desc=param_desc)))
                continue
            patch.append((key, val))
        return patch

    def eval_meta_batch(flat_cands: jnp.ndarray, seeds: jnp.ndarray, row_idxs: np.ndarray, goal_embeds: jnp.ndarray):
        best_fitnesses = []
        for flat_jnp, seed, row_idx, goal_embed in zip(flat_cands, seeds, row_idxs, goal_embeds):
            # init dawdreamer and base patch
            _, diva = init_dawdreamer(sample_rate=16000)

            # read preset
            meta_location = df_train.loc[row_idx]['meta_location']
            with open(f"{DIVA_PRESET_DIR}{meta_location}", mode='r', encoding='utf-8') as f:
                preset = f.readlines()
            
            # prepare patch and render engine
            base_patch = read_patch(diva, preset)

            # reconstruct pytree of LES network params
            les_init_params = unravel_fn(flat_jnp)

            # init local les
            key_local = jax.random.PRNGKey(seed)
            key_local, key_les_init = jax.random.split(key_local)
            les = LearnedES(population_size=les_pop, solution=init_sol)
            mean = jnp.zeros(param_dim)
            les_state = les.init(key_les_init, mean, les_init_params)

            # ask, calculate fitness and tell les in a loop
            key_local_inner = key_local
            for j in range(les_iter):
                print(f'inner loop: {j}/{meta_iter} for {meta_location}')
                key_local_inner, key_les_ask, key_les_tell = jax.random.split(key_local_inner, 3)
                population, les_state = les.ask(key_les_ask, les_state, les_init_params)
                fitness = calc_fit(population, goal_embed, base_patch)
                print('fitness', fitness)
                les_state, _ = les.tell(key_les_tell, population, fitness, les_state, les_init_params)

            best_fitnesses.append(les_state.best_fitness)
        print('best_fitnesses', best_fitnesses)
        return jnp.array(best_fitnesses, dtype=jnp.float32)

    for i in range(meta_iter):
        print(f'outer loop: {i}/{meta_iter}')

        key, key_meta_ask, key_meta_tell = jax.random.split(key, 3)
        cand_flat, meta_state = cma_es.ask(key_meta_ask, meta_state, meta_params)

        # convert to jax numpy and clip extreme values
        cand_flat_jnp = jnp.clip(jnp.array(cand_flat), -5.0, 5.0)

        # evaluate candiates and collect fitnesses
        seeds = jnp.arange(cand_flat_jnp.shape[0]) + i * 1000 * 42
                
        fitnesses = eval_meta_batch(cand_flat_jnp, seeds, row_idxs, goal_embeds)

        # tell cma-es fitnesses
        meta_state, _ = cma_es.tell(key_meta_tell, cand_flat, fitnesses, meta_state, meta_params)

        # logging
        best_idx = int(jnp.argmin(fitnesses))
        print(f"[meta gen {i:03d}] best candidate fitness = {float(fitnesses[best_idx]):.6f}")

    # save best les params
    meta_mean = meta_state.mean
    best_meta_flat = jnp.clip(meta_mean, -5.0, 5.0)
    best_meta_pytree = unravel_fn(best_meta_flat)

    print("meta optimization finished.")
    return best_meta_pytree