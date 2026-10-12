"""Final independent matrix audit and explicit insertion-boundary counts.

Historical columns are immutable. Auxiliary counts come from programmed XML
times and SUMO's step rounding, not inferred losses or changed simulations.
"""
import argparse
import csv
import json
import math
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

from tools import run_matched_experiment as original
from tools.run_matched_parallel import retry_publication


def boundary_count(departures, step, end):
    return sum(math.ceil(depart / step) * step >= end for depart in departures)


def insertion_counts(row, departures):
    future = boundary_count(departures, row['step_s'], row['actual_end_s'])
    return {'pending_not_yet_attempted': future,
            'pending_insertion_total': row['pending_insertion'] + future,
            'programmed_balance_after_boundary': row['programmed_balance_residual'] - future,
            'pending_balance_after_boundary': row['pending_balance_residual'] - future}


def audit(output):
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if original.snapshot() != manifest['resource_sha256']:
        raise ValueError('Frozen simulation resources changed')
    db = sqlite3.connect(output / 'runs.sqlite')
    try:
        expected = {(level, seed, mode) for level, settings in original.LEVELS.items()
                    for seed in settings['seeds'] for mode in original.MODES}
        entries = db.execute('SELECT level,seed,mode,status,row_json FROM runs WHERE experiment_id=?',
                             (manifest['experiment_id'],)).fetchall()
        if {(e[0], e[1], e[2]) for e in entries} != expected or any(e[3] != 'done' for e in entries):
            raise ValueError('Expected exactly the 240 completed matrix keys')
        with (output / 'demand_manifest.csv').open(encoding='utf-8', newline='') as handle:
            demands = {(d['traffic_level'], int(d['seed'])): d for d in csv.DictReader(handle)}
        if len(demands) != 60:
            raise ValueError('Expected 60 frozen demands')
        departures = {}
        for key, demand in demands.items():
            route = output / demand['route_path']
            vehicles = list(ET.parse(route).iter('vehicle'))
            departures[key] = [float(v.get('depart')) for v in vehicles]
            if original.file_hash(route) != demand['route_sha256'] or len(vehicles) != int(demand['scheduled_vehicles']):
                raise ValueError(f'Changed demand or programmed count: {key}')
        rows, groups = [], {}
        metadata = {'exporter_sha256': original.file_hash(__file__),
                    'command': 'python -m tools.finish_matched_experiment',
                    'definition': 'ceil(XML depart / step) * step >= actual_end_s; historical insertion queue plus these not-yet-attempted vehicles',
                    'sources': ['https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_26_0/src/microsim/MSVehicleContainer.cpp',
                                'https://raw.githubusercontent.com/eclipse-sumo/sumo/v1_26_0/src/microsim/MSNet.cpp']}
        for level, seed, mode, _, row_json in entries:
            row = json.loads(row_json)
            original.check_row(row)
            key = (level, seed)
            if (row['traffic_level'], row['seed'], row['mode']) != (level, seed, mode):
                raise ValueError('Result identity does not match its database key')
            if row['SUMO_seed'] != 42 or row['step_s'] != 1 or row['horizon_s'] != 3600 or not row['deterministic'] or row['yellow_s'] != 4:
                raise ValueError('Wrong simulation protocol')
            synced = mode in ('plain_offset', 'v3')
            if row['offsets'] != list(original.OFFSETS if synced else (0, 0, 0, 0)):
                raise ValueError('Wrong offsets')
            j0_bounds = (15, 15) if mode.startswith('plain') else (5, 45)
            receiver_bounds = (5, 45) if mode == 'v3' else (15, 15)
            if (row['J0_green_min_s'], row['J0_green_max_s']) != j0_bounds or (row['receiver_green_min_s'], row['receiver_green_max_s']) != receiver_bounds:
                raise ValueError('Wrong green duration limits')
            if mode == 'v3' and (row['controller_class'] != 'PreannouncedWindow' or row['delays_s'] != {'J2': 12, 'J10': 22, 'J16': 32}):
                raise ValueError('V3 original corridor delays not preserved')
            if not mode.startswith('plain') and row['checkpoint'] != original.CHECKPOINT.relative_to(original.ROOT).as_posix():
                raise ValueError('Wrong PPO checkpoint')
            if row['scheduled_vehicles'] != len(departures[key]) or row['route_sha256'] != demands[key]['route_sha256']:
                raise ValueError('Wrong programmed vehicles or matched demand hash')
            signature = tuple(row[k] for k in ('route_sha256', 'cohort_J2_sha256', 'cohort_full_sha256', 'cohort_strict_sha256'))
            if key in groups and groups[key] != signature:
                raise ValueError('Unmatched demands/cohorts')
            groups[key] = signature
            row.update(insertion_counts(row, departures[key]))
            rows.append(row)
        # These fields augment counts only; all original metrics and evidence
        # stay intact. Persist them so the original --stage export reproduces CSV.
        with db:
            for row in rows:
                db.execute('UPDATE runs SET row_json=? WHERE experiment_id=? AND level=? AND seed=? AND mode=?',
                           (json.dumps(row, ensure_ascii=False, allow_nan=False), manifest['experiment_id'],
                            row['traffic_level'], row['seed'], row['mode']))
            db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('final_export_enrichment', json.dumps(metadata)))
        preflight = json.loads(db.execute("SELECT value FROM metadata WHERE key='preflight'").fetchone()[0])
        atomic_json, atomic_csv = original.atomic_json, original.atomic_csv
        original.atomic_json = lambda *a: retry_publication(atomic_json, *a)
        original.atomic_csv = lambda *a: retry_publication(atomic_csv, *a)
        if not original.export(output, db, manifest, preflight):
            raise ValueError('Final export unexpectedly incomplete')
        validation = json.loads((output / 'validation.json').read_text(encoding='utf-8'))
        validation['independent_final_audit'] = {
            'matrix_keys_exact': True, 'demand_hashes_and_programmed_counts_exact': True,
            'scheduled_cohorts_matched': True, 'preserved_historical_metrics': True,
            'insertion_boundary_enrichment': metadata,
            'teleport_runs': sum(r['teleports'] > 0 for r in rows),
            'teleport_vehicle_ids_across_runs': sum(r['teleports'] for r in rows),
            'unexplained_balance_runs': [{'run_id': r['run_id'], 'programmed_residual': r['programmed_balance_after_boundary'],
                                         'pending_residual': r['pending_balance_after_boundary'], 'inserted_residual': r['inserted_balance_residual'],
                                         'teleports': r['teleports']}
                                        for r in rows if any(r[k] for k in ('programmed_balance_after_boundary', 'pending_balance_after_boundary', 'inserted_balance_residual'))],
            'rows_with_missing_metrics': sum(bool(r['missing_metrics']) for r in rows),
            'min_active_pending': min(r['pending_active'] for r in rows),
            'max_active_pending': max(r['pending_active'] for r in rows)}
        original.atomic_json(output / 'validation.json', validation)
        print(json.dumps({'rows': len(rows), **validation['independent_final_audit']}, ensure_ascii=False), flush=True)
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=original.DEFAULT_OUTPUT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert boundary_count([0, 3599, 3599.5], 1, 3600) == 1
        row = {'step_s': 1, 'actual_end_s': 3600, 'pending_insertion': 1236,
               'programmed_balance_residual': 1, 'pending_balance_residual': 1}
        assert insertion_counts(row, [3599.5]) == {'pending_not_yet_attempted': 1,
            'pending_insertion_total': 1237, 'programmed_balance_after_boundary': 0,
            'pending_balance_after_boundary': 0}
        assert boundary_count([], 1, 3600) == 0
        print('PASS: final-step rounding and historical insertion queue retained')
        return
    audit(args.output_dir.resolve())


if __name__ == '__main__':
    main()
