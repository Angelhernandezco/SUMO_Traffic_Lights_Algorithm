"""Execution-only parallel coordinator for an already frozen matched experiment.

Simulation workers and historical metrics remain byte-for-byte unchanged.
One owner schedules work and publishes SQLite/CSV; threads only launch isolated
worker processes. Benchmark repetitions never replace completed experiment rows.
"""
import argparse
import concurrent.futures as cf
import ctypes
import csv
import json
import os
import time
import traceback
from pathlib import Path

from tools import run_matched_experiment as original
from tests.validate_sync import compare


def retry_publication(publish, *args):
    """Windows readers/antivirus can briefly deny replace on an existing file."""
    for attempt in range(8):
        try:
            return publish(*args)
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.05 * (attempt + 1))


def memory_available_gib():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ("total", "available", "page_total", "page_available", "virtual_total",
             "virtual_available", "extended_available")]
    value = Memory()
    value.length = ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):
        raise ctypes.WinError()
    return value.available / 1024**3


def timed_worker(route, seed, mode):
    started = time.monotonic()
    return original.execute_worker(route, seed, mode), time.monotonic() - started


def run_wave(jobs, workers, work=timed_worker, sample=memory_available_gib):
    """Bound concurrency; keep publication and exceptions on the owner thread."""
    started = time.monotonic()
    minimum = sample()
    results = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(work, *job): job for job in jobs}
        while pending:
            done, _ = cf.wait(pending, timeout=2, return_when=cf.FIRST_COMPLETED)
            minimum = min(minimum, sample())
            for future in done:
                job = pending.pop(future)
                results.append((job, future.result()))
    return results, time.monotonic() - started, minimum


def benchmark(output, db, manifest, workers):
    saved = db.execute("SELECT row_json,evidence_json FROM runs WHERE status='done' ORDER BY level DESC,seed,CASE mode WHEN 'plain' THEN 0 WHEN 'plain_offset' THEN 1 WHEN 'only_ppo' THEN 2 ELSE 3 END LIMIT ?", (workers,)).fetchall()
    if len(saved) != workers:
        raise ValueError("Not enough completed runs for a paired concurrency benchmark")
    expected = {}
    jobs = []
    serial_s = 0
    for row_json, evidence_json in saved:
        row, evidence = json.loads(row_json), json.loads(evidence_json)
        job = (output / row['route_path'], row['seed'], row['mode'])
        jobs.append(job)
        expected[job] = evidence
        serial_s += row['elapsed_wall_s']
    print(f"BENCHMARK START workers={workers} repeated_completed_runs={workers}", flush=True)
    results, elapsed, available = run_wave(jobs, workers)
    for job, (payload, duration) in results:
        compare(payload['summary'], expected[job]['summary'])
        compare(payload['canonical_traces'], expected[job]['canonical_traces'])
        compare(payload['historical_row'], expected[job]['historical_row'])
        original.check_row(payload['row'])
        print(f"BENCHMARK PASS workers={workers} seed={job[1]} mode={job[2]} elapsed={duration:.3f}s", flush=True)
    if original.snapshot() != manifest['resource_sha256']:
        raise ValueError("Frozen resources changed during benchmark")
    result = {'workers': workers, 'repeated_runs': workers, 'wall_s': elapsed,
              'serial_reference_wall_s': serial_s, 'relative_throughput': serial_s / elapsed,
              'runs_per_hour': workers * 3600 / elapsed, 'min_available_ram_gib': available,
              'all_summaries_and_seven_signatures_equal': True,
              'repetitions_excluded_from_main_csv': True,
              'jobs': [{'seed': j[1], 'mode': j[2], 'wall_s': duration}
                       for j, (_, duration) in results]}
    print(f"BENCHMARK DONE {json.dumps(result)}", flush=True)
    return result


def enrich(payload, duration, demand, output, manifest, workers, coordinator_hash):
    row = payload.pop('row')
    if row['route_sha256'] != demand['route_sha256']:
        raise ValueError('Worker used a different demand')
    original.check_row(row)
    for key in ('generation_period_s', 'nominal_attempts_per_hour', 'attempted_trips',
                'generated_trips', 'generation_losses', 'routing_losses'):
        row[key] = float(demand[key]) if key in ('generation_period_s', 'nominal_attempts_per_hour') else int(demand[key])
    identity = manifest['experiment_id']
    route = output / demand['route_path']
    row.update(experiment_id=identity, run_id=f"{identity}:{demand['traffic_level']}:{demand['seed']}:{row['mode']}",
               traffic_level=demand['traffic_level'], demand_id=demand['demand_id'], route_path=demand['route_path'],
               route_semantic_sha256=demand['route_semantic_sha256'],
               code_sha256=original.digest(manifest['resource_sha256']), environment=manifest['environment'],
               run_command=[original.sys.executable, '-m', 'tools.run_matched_parallel', '--workers', str(workers), '--output-dir', str(output)],
               sumo_command_template=[original.sumo_binary(), '--net-file', '<temporary network with recorded offsets>',
                                      '--route-files', str(route), '--begin', '0', '--end', '3600', '--step-length', '1',
                                      '--seed', '42', '--no-step-log', 'true', '--no-warnings', 'true'],
               elapsed_wall_s=round(duration, 3))
    # Execution metadata is separate from the frozen simulation protocol.
    payload['execution'] = {'parallel_workers': workers, 'coordinator_sha256': coordinator_hash}
    return row, payload


def execute_pending(output, db, manifest, demands, preflight, workers, coordinator_hash):
    identity = manifest['experiment_id']
    jobs = [(d, mode) for d in demands for mode in original.MODES]
    started = time.monotonic()
    runtime = {'workers': workers, 'coordinator_sha256': coordinator_hash,
               'min_available_ram_gib': memory_available_gib(), 'completed_this_session': 0,
               'started_UTC': original.utc()}
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {}
        while jobs or pending:
            # A future change of concurrency can drain workers without losing work.
            if (output / 'parallel_stop.request').exists():
                jobs.clear()
            while jobs and len(pending) < workers:
                demand, mode = jobs.pop(0)
                level, seed = demand['traffic_level'], int(demand['seed'])
                if not original.begin_attempt(db, identity, level, seed, mode):
                    continue
                route = output / demand['route_path']
                if original.snapshot() != manifest['resource_sha256'] or original.file_hash(route) != demand['route_sha256']:
                    raise ValueError('Resources or demand changed before simulation')
                future = pool.submit(timed_worker, route, seed, mode)
                pending[future] = (demand, mode)
                print(f"START parallel={workers} {level} seed={seed} mode={mode} {original.utc()}", flush=True)
            if not pending:
                break
            done, _ = cf.wait(pending, timeout=5, return_when=cf.FIRST_COMPLETED)
            available = memory_available_gib()
            runtime.update(available_ram_gib=available,
                           min_available_ram_gib=min(runtime['min_available_ram_gib'], available),
                           active_workers=len(pending), updated_UTC=original.utc(),
                           elapsed_wall_s=time.monotonic() - started)
            for future in done:
                demand, mode = pending.pop(future)
                level, seed = demand['traffic_level'], int(demand['seed'])
                try:
                    payload, duration = future.result()
                    row, evidence = enrich(payload, duration, demand, output, manifest, workers, coordinator_hash)
                    original.finish_attempt(db, identity, level, seed, mode, row, evidence)
                    runtime['completed_this_session'] += 1
                    print(f"DONE parallel={workers} {level} seed={seed} mode={mode} elapsed={duration:.3f}s", flush=True)
                except Exception:
                    error = traceback.format_exc()
                    original.finish_attempt(db, identity, level, seed, mode, error=error)
                    jobs.insert(0, (demand, mode))  # Existing two-attempt limit applies.
                    print(f"FAILED {level} seed={seed} mode={mode}\n{error}", flush=True)
                original.export(output, db, manifest, preflight)
            try:
                original.atomic_json(output / 'parallel_runtime.json', runtime)
            except PermissionError:
                # Optional monitoring must never discard completed worker results.
                pass
    return original.export(output, db, manifest, preflight)


def self_test():
    import threading
    active, peak = 0, 0
    lock = threading.Lock()
    def work(value):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(.03)
        with lock:
            active -= 1
        return value * 2
    for workers in (4, 10):
        peak = 0
        count = workers * 2 + 1
        results, _, _ = run_wave([(n,) for n in range(count)], workers, work, lambda: 4)
        assert peak == workers
        assert sorted(value for _, value in results) == list(range(0, count * 2, 2))
    try:
        run_wave([(0,)], 4, lambda _: (_ for _ in ()).throw(ValueError('expected')), lambda: 4)
    except ValueError:
        pass
    else:
        raise AssertionError('Worker failure swallowed')
    attempts = []
    def transient_reader():
        attempts.append(1)
        if len(attempts) < 3:
            raise PermissionError('reader sharing violation')
        return 'published'
    assert retry_publication(transient_reader) == 'published'
    assert len(attempts) == 3
    print('PASS: concurrency limit, every job exactly once, worker exception propagated, Windows publication retries')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, choices=range(1, 11), default=4)
    parser.add_argument('--autotune', action='store_true', help='Benchmark four, conditionally five, then resume')
    parser.add_argument('--trial-six', action='store_true', help='Benchmark six against the saved five-worker test, then resume with six or five')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=original.DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.trial_six and args.autotune:
        parser.error('Choose --trial-six or --autotune')
    if args.self_test:
        self_test()
        return
    # Parent publication only. Isolated simulation workers import the original
    # module in a fresh process, with unchanged controllers and measurements.
    atomic_json, atomic_csv = original.atomic_json, original.atomic_csv
    original.atomic_json = lambda *a: retry_publication(atomic_json, *a)
    original.atomic_csv = lambda *a: retry_publication(atomic_csv, *a)
    output = args.output_dir.resolve()
    if (output / 'parallel_stop.request').exists():
        raise ValueError('Remove parallel_stop.request after the previous owner has drained')
    lock = output / 'parallel.lock'
    # The sequential owner must have stopped before entering this coordinator.
    # A stale lock after interruption requires verifying the recorded PID is gone.
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    db = None
    try:
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        core = original.manifest_core()
        if original.digest(core) != manifest['experiment_id']:
            raise ValueError('Frozen experiment code/protocol changed')
        db = original.open_state(output / 'runs.sqlite', manifest)
        preflight = json.loads(db.execute("SELECT value FROM metadata WHERE key='preflight'").fetchone()[0])
        with (output / 'demand_manifest.csv').open(encoding='utf-8', newline='') as handle:
            demands = list(csv.DictReader(handle))
        coordinator_hash = original.file_hash(__file__)
        report_path = output / 'parallel_execution.json'
        previous = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
        history = previous.get('benchmarks', [])
        for trial in history:
            trial.setdefault('coordinator_sha256', previous.get('coordinator_sha256'))
        report = {'experiment_id': manifest['experiment_id'], 'coordinator_sha256': coordinator_hash,
                  'started_UTC': original.utc(), 'benchmarks': [], 'selected_workers': args.workers,
                  'note': 'Historical sequential timing is an estimate under a different machine load; throughput, not single-run latency, selects concurrency.'}
        report['benchmarks'] = history
        if args.trial_six and not any(b['workers'] == 5 for b in history):
            raise ValueError('Trial six requires a saved five-worker benchmark')
        original.atomic_json(report_path, report)
        if args.autotune:
            four = benchmark(output, db, manifest, 4)
            report['benchmarks'].append(four)
            report['selected_workers'] = 4 if four['relative_throughput'] >= 1.05 and four['min_available_ram_gib'] >= 1 else 1
            original.atomic_json(report_path, report)
            if report['selected_workers'] == 4 and four['min_available_ram_gib'] >= 1.5:
                five = benchmark(output, db, manifest, 5)
                report['benchmarks'].append(five)
                if five['relative_throughput'] >= four['relative_throughput'] * 1.05 and five['min_available_ram_gib'] >= 1:
                    report['selected_workers'] = 5
            original.atomic_json(report_path, report)
        if args.trial_six:
            five = next(b for b in reversed(history) if b['workers'] == 5)
            six = benchmark(output, db, manifest, 6)
            six['coordinator_sha256'] = coordinator_hash
            report['benchmarks'].append(six)
            report['selected_workers'] = 6 if six['relative_throughput'] >= five['relative_throughput'] * 1.05 and six['min_available_ram_gib'] >= 1 else 5
            original.atomic_json(report_path, report)
        print(f"SELECT workers={report['selected_workers']}", flush=True)
        complete = execute_pending(output, db, manifest, demands, preflight, report['selected_workers'], coordinator_hash)
        report.update(finished_UTC=original.utc(), complete=complete)
        original.atomic_json(report_path, report)
        print('COMPLETE: 240 rows' if complete else 'INCOMPLETE: inspect validation.json', flush=True)
    finally:
        if db is not None:
            db.close()
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
