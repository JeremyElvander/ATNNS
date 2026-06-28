
import time
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
import os
import subprocess
from system import ATNNS, ATNNS_GPU
import logging
tf.get_logger().setLevel(logging.ERROR)

base_path = Path().resolve()

df = pd.read_parquet(base_path / 'data' / 'eaim_example_data.parquet')

actual_outputs = df[['phase', 'amm_nit', 'amm_chl', 'water_content']].copy()
inputs = df.drop(['water_content', 'p_H2O', 'p_HNO3', 'p_HCl', 'p_NH3', 'p_H2SO4', 'amm_nit', 'amm_chl', 'phase'], axis=1)

actual_outputs.reset_index(drop=True, inplace=True)
inputs.reset_index(drop=True, inplace=True)

EAIM_DIR = base_path / 'eaim'

def time_eaim(inputs, repeats, actual_outputs=None):
    def _read_eaim_header():
        data_path = EAIM_DIR / 'eaim.dat'
        with open(data_path, 'r') as f:
            content = f.read()
        cutoff = content.find('####')
        cutoff = content.find('\n', cutoff) + 1
        return content[:cutoff]
    
    def write_eaim_dat(inputs):
        header = _read_eaim_header()
        lines = []
        for _, row in inputs.iterrows():
            line = (
                f"{row['TEMP']}D0 1.D0 1.D0 1 1 {row['RH']}D0 0.0 "
                    f"{row['NH4+']} {row['NA+']} "
                    f"{row['SO42-']} {row['NO3-']} {row['CL-']} "
                    f"0.0, 0.0, 0.0, 4, 4, 4, 4, 3, 0"
                    )
            lines.append(line)
        
        data_path = EAIM_DIR / 'eaim.dat'
        with open(data_path, 'w') as f:
            f.write(header)
            f.write('\n'.join(lines))
            f.write('\n===========\n')
    
    def parse_output():
        import re

        # Parse eaim.r2a for water content and liquid-phase partial pressures.
        # Read line-by-line to avoid pandas choking on the ragged solid compound
        # name columns at the end of each row.
        r2a_path = EAIM_DIR / 'eaim.r2a'
        with open(r2a_path) as f:
            header = f.readline().split()
            rows = [line.split()[:30] for line in f if len(line.split()) >= 30]
        r2a = pd.DataFrame(rows, columns=header[:30])
        for col in ['n_H2O(aq)', 'n_HNO3(g)', 'n_HCl(g)', 'n_NH3(g)']:
            r2a[col] = pd.to_numeric(r2a[col])

        water_content = r2a['n_H2O(aq)'].values.copy()
        phase = (r2a['n_H2O(aq)'].values == 0).astype(int)
        amm_nit = (r2a['n_HNO3(g)'] * r2a['n_NH3(g)']).values.copy()
        amm_chl = (r2a['n_HCl(g)'] * r2a['n_NH3(g)']).values.copy()

        # Parse eaim.rs1 for solid-phase partial pressure products.
        # The subroutine already writes this file alongside r2a — no second
        # E-AIM run needed. For solid rows, look for the
        # "Calculated partial pressures (over solids)" section and extract
        # NH4NO3 and NH4Cl products by compound name.
        rs1_path = EAIM_DIR / 'eaim.rs1'
        with open(rs1_path) as f:
            content = f.read()

        problems = re.split(r'Problem no\.\s+(\d+)', content)
        for i in range(1, len(problems) - 1, 2):
            prob_idx = int(problems[i]) - 1
            if prob_idx >= len(r2a) or phase[prob_idx] != 1:
                continue
            section = problems[i + 1]
            if 'Calculated partial pressures (over solids)' not in section:
                continue
            m = re.search(r'=\s*([\d.E+\-]+),\s*NH4NO3', section)
            if m:
                amm_nit[prob_idx] = float(m.group(1))
            m = re.search(r'=\s*([\d.E+\-]+),\s*NH4Cl', section)
            if m:
                amm_chl[prob_idx] = float(m.group(1))

        return pd.DataFrame({
            'phase':         phase,
            'amm_nit':       amm_nit,
            'amm_chl':       amm_chl,
            'water_content': water_content,
        })

    print('Starting Timing...')
    write_eaim_dat(inputs)

    start = time.perf_counter()
    subprocess.run(['./eaimIntel.exe'], cwd=EAIM_DIR, capture_output=True)
    cold_time = time.perf_counter() - start
    print(f'Cold time: {cold_time}')

    print('Timing Warm Runs:')
    warm_times = []
    for i in range(repeats):
        print(f'\t\rRun {i+1}', end='', flush=True)
        write_eaim_dat(inputs)
        start = time.perf_counter()
        subprocess.run(['./eaimIntel.exe'], cwd=EAIM_DIR, capture_output=True)
        _ = parse_output()
        warm_times.append(time.perf_counter()-start)

    return warm_times


def time_gpu_model(inputs, repeats, actual_outputs=None):
    print('Starting Timing...')
    input_tensor = tf.convert_to_tensor(inputs, dtype=tf.float32)

    print('Timing Initialization:')
    init_times = []
    for i in range(repeats):
        print(f'\t\rRun {i+1}', end='', flush=True)
        tf.keras.backend.clear_session()
        start = time.perf_counter()
        solver = ATNNS_GPU()
        init_times.append(time.perf_counter()-start)
         
    print('\nTiming JIT Compilation:')
    jit_times = []
    for i in range(repeats):
        print(f'\t\rRun {i+1}', end='', flush=True)
        solver = ATNNS_GPU()
        start = time.perf_counter()
        result = solver.gpu_prediction(input_tensor)
        #GPU Sync
        _ = result.numpy()
        jit_times.append(time.perf_counter()-start)
    
    solver = ATNNS_GPU()
    _ = solver.gpu_prediction(input_tensor).numpy()

    print('\nTiming Warm Runs:')
    warm_times = []
    for i in range(repeats):
        print(f'\t\rRun {i+1}', end='', flush=True)
        start = time.perf_counter()
        result = solver.gpu_prediction(input_tensor)
        _ = result.numpy()
        warm_times.append(time.perf_counter()-start)

    return init_times, jit_times, warm_times

def time_sequential_model(inputs, repeats, actual_outputs=None):
    print('Starting Timing...')

    print('Timing Initialization:')
    init_times = []
    for i in range(repeats):
        print(f'\tRun {i+1}', end='', flush=True)
        # Clear Keras state for same reason as GPU init loop
        tf.keras.backend.clear_session()
        start = time.perf_counter()
        solver = ATNNS()  # trivial — no __init__ work
        init_times.append(time.perf_counter() - start)

    print('\nTiming First Call (cold disk load):')
    first_call_times = []
    for i in range(repeats):
        print(f'\tRun {i+1}', end='', flush=True)
        tf.keras.backend.clear_session()  # prevent Keras caching across iterations
        solver = ATNNS()
        start = time.perf_counter()
        result = solver.prediction(inputs)
        _ = result.values  # materialize DataFrame, no GPU sync needed
        first_call_times.append(time.perf_counter() - start)

    # Warm the OS file cache with one throwaway call
    solver = ATNNS()
    _ = solver.prediction(inputs).values

    print('\nTiming Warm Runs (OS cache hot, no Python state):')
    warm_times = []
    for i in range(repeats):
        print(f'\tRun {i+1}', end='', flush=True)
        # Note: solver is reinstantiated each call to reflect real ATNNS behavior —
        # there is no persistent warm state, unlike ATNNS_GPU
        solver = ATNNS()
        start = time.perf_counter()
        result = solver.prediction(inputs)
        _ = result.values
        warm_times.append(time.perf_counter() - start)

    return init_times, first_call_times, warm_times


INPUT_SIZES = [1, 10, 100, 1000, 10000, 100000, 1000000]
REPEATS = 30

records = []

for n in INPUT_SIZES:
    sample = inputs.sample(n=n, random_state=42).reset_index(drop=True)
    print(f'\n{"="*60}')
    print(f'  n = {n}')
    print(f'{"="*60}')

    # ── EAIM ─────────────────────────────────────────────────────
    print('\n[EAIM]')
    eaim_warm = time_eaim(sample, REPEATS)
    print()
    for rep, t in enumerate(eaim_warm):
        records.append({
            'model':    'eaim',
            'phase':    'warm',
            'n_inputs': n,
            'repeat':   rep,
            'time_s':   t,
        })

    # ── CPU (sequential) ─────────────────────────────────────────
    print('\n[CPU]')
    cpu_init, cpu_first, cpu_warm = time_sequential_model(sample, REPEATS)
    print()
    for phase, times in [('init', cpu_init), ('first_call', cpu_first), ('warm', cpu_warm)]:
        for rep, t in enumerate(times):
            records.append({
                'model':    'cpu',
                'phase':    phase,
                'n_inputs': n,
                'repeat':   rep,
                'time_s':   t,
            })

    # ── GPU ───────────────────────────────────────────────────────
    print('\n[GPU]')
    gpu_init, gpu_jit, gpu_warm = time_gpu_model(sample, REPEATS)
    print()
    for phase, times in [('init', gpu_init), ('jit', gpu_jit), ('warm', gpu_warm)]:
        for rep, t in enumerate(times):
            records.append({
                'model':    'gpu',
                'phase':    phase,
                'n_inputs': n,
                'repeat':   rep,
                'time_s':   t,
            })

# ── write results ─────────────────────────────────────────────────────────────

results_df = pd.DataFrame(records)
out_path = base_path / 'timing_results.csv'
results_df.to_csv(out_path, index=False)
print(f'\nResults written to {out_path}')
print(results_df.groupby(['model', 'phase', 'n_inputs'])['time_s'].mean().round(4).to_string())
    

