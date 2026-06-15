# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.9",
# ]
# [tool.marimo.runtime]
# auto_instantiate = false
# on_cell_change = "lazy"
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.9",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
async def _():
    import micropip
    await micropip.install([
        "matplotlib", "numpy", "scipy", "pandas", "scikit-learn", "asciitree"
    ])


    # pyiodide workarounds

    ## numba doesn't work so wrap with identity decorator matching jit interface
    import pathlib
    import sysconfig

    site_packages = pathlib.Path(sysconfig.get_paths()["purelib"])
    numba_dir = site_packages / "numba"
    numba_dir.mkdir(exist_ok=True)
    (numba_dir / "__init__.py").write_text("""
    def jit(fn=None, *args, **kwargs):
        if fn is None:
            return lambda f: f
        return fn

    njit = jit
    prange = range
    """)
    dist_info = site_packages / "numba-99.0.0.dist-info"
    dist_info.mkdir(exist_ok=True)
    (dist_info / "METADATA").write_text("Name: numba\nVersion: 99.0.0\n")


    # SharedMemory immediately fails on import and is used a lot in SI
    import itertools
    import sys
    import types

    class FakeSharedMemory:
        _counter = itertools.count()
        _buffers = {}

        def __init__(self, name=None, create=False, size=0):
            if create:
                self.name = name or f"fake_shm_{next(self._counter)}"
                self._buffers[self.name] = bytearray(size)
            else:
                self.name = name

            self.buf = memoryview(self._buffers[self.name])

        def close(self):
            pass

        def unlink(self):
            self._buffers.pop(self.name, None)

    shared_memory = types.ModuleType("multiprocessing.shared_memory")
    shared_memory.SharedMemory = FakeSharedMemory
    sys.modules["multiprocessing.shared_memory"] = shared_memory



    # requests e.g. when doing synthetic data gen have to go through pyiodide
    # TODO: si.download_dataset doesn't work 
    import io
    import urllib.request
    from pyodide.http import open_url

    class Response(io.BytesIO):
        def getcode(self):
            return 200

        def info(self):
            return {}

    def urlopen(url, *args, **kwargs):
        if hasattr(url, "full_url"):
            url = url.full_url
        return Response(open_url(str(url)).read().encode())

    urllib.request.urlopen = urlopen



    await micropip.install(["spikeinterface==0.104.5", "pynapple"])


    # ProcessPoolExec.. imports multiprocessing 
    import concurrent.futures
    import sys
    import types

    process = types.ModuleType("concurrent.futures.process")
    process.ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor
    sys.modules["concurrent.futures.process"] = process
    concurrent.futures.ProcessPoolExecutor = concurrent.futures.ThreadPoolExecutor


    # Avoid creating new threads (not possible to spawn w pyiodide)
    import spikeinterface.full as si
    si.set_global_job_kwargs(n_jobs=1)


    # minor: detecting sorters relies on subprocess calls.
    # not sure if all the sorters here work so should change the behaviour of this..
    import spikeinterface.sorters.sorterlist as sorterlist
    import spikeinterface.sorters as ss
    import spikeinterface.full as si

    def installed_sorters():
        out = []
        for sorter in sorterlist.sorter_full_list:
            try:
                if sorter.is_installed():
                    out.append(sorter.sorter_name)
            except Exception as e:
                pass  # can make specific for [Errno 138] emscripten does not support processes.
        return sorted(out)

    sorterlist.installed_sorters = installed_sorters
    ss.installed_sorters = installed_sorters
    si.installed_sorters = installed_sorters
    return si, ss


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Spikeinterface
    """)
    return


@app.cell
def _(si):
    import matplotlib.pyplot as plt

    _, recording, gt_sorting, extra_infos = si.generate_drifting_recording(
        probe_name="Neuropixels1-128",  # or use probe= instead to avoid the http call to probe library (https://probeinterface.readthedocs.io/en/main/)
        num_units=10,
        duration=10.0,
        seed=2205,
        extra_outputs=True,
        generate_displacement_vector_kwargs=dict(
            displacement_sampling_frequency=5.0,
            drift_start_um=[0, 20],
            drift_stop_um=[0, -20],
            drift_step_um=1,
            motion_list=[
                dict(
                    drift_mode="zigzag",
                    non_rigid_gradient=None,
                    t_start_drift=0.0,
                    t_end_drift=None,
                    period_s=10.0,
                ),
            ],
        ),
    )

    si.plot_traces(recording, channel_ids=recording.channel_ids[10:15], time_range=(0, 5.0))

    plt.show()
    return gt_sorting, plt, recording


@app.cell
def _(recording, ss):
    sorting = ss.run_sorter(
        "simple",
        recording,
        remove_existing_folder=True,
        clusterer="affinity_propagation",  # default is hdbscan which is difficult to install
    )

    print(sorting)
    return (sorting,)


@app.cell
def _(recording, si, sorting):
    sorting_analyzer = si.create_sorting_analyzer(
        sorting=sorting,
        recording=recording,
        format="memory",
    )

    sorting_analyzer.compute(["random_spikes", "waveforms", "templates"])
    return (sorting_analyzer,)


@app.cell
def _(plt, sorting_analyzer):
    import spikeinterface.widgets as sw

    sw.plot_unit_waveforms(sorting_analyzer, unit_ids=[0])
    plt.show()
    return (sw,)


@app.cell
def _(gt_sorting, plt, sorting, sw):
    import spikeinterface.comparison as sc

    comp = sc.compare_sorter_to_ground_truth(gt_sorting, sorting)


    sw.plot_confusion_matrix(comp)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # pynapple
    """)
    return


@app.cell
def _():
    import pynapple as nap
    import numpy as np

    return nap, np


@app.cell
def _(nap, np):
    tsd = nap.Tsd(t=np.arange(100), d=np.random.rand(100))

    print(tsd)
    return


@app.cell
def _(nap, np):
    tsdframe = nap.TsdFrame(
        t=np.arange(100), d=np.random.rand(100, 3), columns=["a", "b", "c"]
    )

    print(tsdframe)
    return


@app.cell
def _(nap, np):
    tsdtensor = nap.TsdTensor(
        t=np.arange(100), d=np.random.rand(100, 3, 4)
    )

    print(tsdtensor)
    return


@app.cell
def _(nap, np):
    # Feature
    T = 500
    dt_feature = 0.02
    times_feature = np.arange(0, T, dt_feature)
    feature = nap.Tsd(
        t=times_feature, d=np.pi + np.pi * np.cos(2 * np.pi * times_feature / 10)
    )

    # Spikes
    N = 6
    max_rate = 20
    dt_spikes = 0.002
    feature_interp = feature.interpolate(nap.Ts(np.arange(0, T, dt_spikes)))
    centers = np.linspace(0, 2 * np.pi, N, endpoint=False)
    rates = max_rate * np.exp(
        -10 * (np.sin((feature_interp.d[:, np.newaxis] - centers) / 2)) ** 2
    )
    tsgroup_1d = nap.TsGroup(
        {
            i + 1: nap.Ts(
                feature_interp.t[np.random.poisson(rates[:, i] * dt_spikes) > 0]
            )
            for i in range(N)
        },
    )
    return feature, tsgroup_1d


@app.cell
def _(feature, nap, np, tsgroup_1d):
    tuning_curves_1d = nap.compute_tuning_curves(
        data=tsgroup_1d,
        features=feature,
        bins=120, 
        range=(0, 2*np.pi),
        feature_names=["feature"]
        )
    tuning_curves_1d
    return (tuning_curves_1d,)


@app.cell
def _(plt, tuning_curves_1d):
    tuning_curves_1d.name = "firing rate"
    tuning_curves_1d.attrs["unit"] = "Hz"
    tuning_curves_1d.coords["feature"].attrs["unit"] = "rad"
    tuning_curves_1d.plot.line(x="feature", add_legend=False);
    plt.show()
    return


if __name__ == "__main__":
    app.run()
