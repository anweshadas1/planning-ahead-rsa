# Planning Ahead with RSA

Implementation for the AAMAS-2026 paper **"Planning Ahead with RSA:
Efficient Signalling in Dynamic Environments by Projecting User Awareness
across Future Timesteps"** by Anwesha Das, John Duff, Jörg Hoffmann and Vera
Demberg (Saarland University / UCLA).

Paper: https://dl.acm.org/doi/10.65109/BRGK2754

The code was previously hosted on OSF (https://doi.org/10.17605/OSF.IO/TPCE6).

An assistive agent watching a fast-changing environment has to decide not only
*what* to tell its user but *when*, and *how specifically*, given that human
attention is a zero-sum resource. This repository implements that decision as
Bayesian reference resolution in the Rational Speech Act framework, extended
with temporal belief tracking, user-specific priors and finite-horizon
planning, and evaluates it on 800 simulated drone-monitoring trials.

## The four model variants

The models share one pragmatic core and differ along exactly two axes:
whether the speaker plans over a horizon or acts greedily, and whether it
reasons about a specific user's beliefs or assumes uniform priors.

| Key in the code | Planning | User priors | Name in the paper |
|---|---|---|---|
| `full` | yes | yes | d-RSA + Priors + Planning |
| `uniform` | yes | no | d-RSA + Planning |
| `myopic` | no | yes | d-RSA + Priors |
| `baseline` | no | no | Dynamic RSA (d-RSA) |

The short keys are what appear in filenames, in the `model_name` column of the
comparison tables, and in the R analysis. They are kept so that result files
and notebooks stay interoperable with the artifacts released alongside the
paper; the CLI prints the paper name next to the key on every run.

## Setup

Python 3.9 or later, plus R if you want to regenerate the figures and the
mixed-effect models.

    git clone https://github.com/anweshadas1/planning-ahead-rsa.git
    cd planning-ahead-rsa
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt

Or with conda/mamba:

    mamba env create -f environment.yml
    mamba activate rsa-planning

**NumPy must stay below 2.0.** The pipeline writes belief dictionaries into its
result files and reads them back with `ast.literal_eval`. Under NumPy 2 a
scalar reprs as `np.float64(0.5)` instead of `0.5`, which that round-trip
cannot parse, and trajectory reconstruction fails. The constraint is pinned in
`requirements.txt`. The paper's runs used Python 3.9.21 with NumPy 1.x.

For the analysis notebooks you need R (4.4.3 was used) with `tidyverse`,
`lme4` (1.1.35.5), `lmerTest` and `emmeans`.

## Running the pipeline

Everything goes through one entry point:

    python run_pipeline.py --help

Check your install on one generated scenario with the cheapest variant:

    python run_pipeline.py scenarios --per-type 1 --seed 1 --version demo
    python run_pipeline.py simulate --model baseline --limit 1 --processes 1 --test-cases data/scenarios/dynamic_test_cases_vdemo.csv
    python run_pipeline.py summarize --model baseline --test-cases data/scenarios/dynamic_test_cases_vdemo.csv

The stages, in order:

    scenarios    generate the Drone World scenario set and its test cases
    precompute   enumerate legal utterance sequences up to the horizon
    simulate     run one or all four variants over the scenarios
    summarize    reconstruct rank-1 trajectories, extract per-run metrics
    compare      build the wide and long tables the R analysis reads

`simulate` builds the sequence tree automatically if it is not there yet.
Runs are resumable: a scenario whose result file already exists is skipped
unless you pass `--overwrite`, so an interrupted sweep can simply be restarted.

To run the whole sweep over a scenario set of your own:

    python run_pipeline.py scenarios --per-type 50 --seed 1 --version mine
    python run_pipeline.py all --test-cases data/scenarios/dynamic_test_cases_vmine.csv

The notebooks in `analysis/` are the analysis as it was run for the paper, kept
here as a record of method rather than as a step in the pipeline. `figures.Rmd`
produces the four figures (2a `reward_by_num_crit`, 2b `reward_by_dispersion`,
3a `specificity_by_user_aware`, 3b `delay_by_num_crit`) and `statistics.Rmd`
fits the linear mixed-effect regressions behind every p-value and confidence
interval in the Results section. Both read a long comparison table at
`output/comparisons/comparison_long.csv`, so they run against output you have
generated yourself; knitting them in a fresh clone will stop at that check.

### This is a cluster-scale run

`python run_pipeline.py all` is 800 scenarios x 4 variants = 3,200 runs. The
two planning variants search a tree of 94,391 legal utterance sequences per
scenario at H=7; a single such run takes minutes and holds a large belief
table in memory. The paper's runs used roughly 40 concurrent HTCondor jobs on
a pool of Linux machines, each allocated at least 30 CPU cores and 640 GB RAM.
No GPU is used.

`cluster/run_job.sh` and `cluster/run_job.sub` are a starting point for that.
Set `RSA_PLANNING_OUTPUT` to put results on scratch instead of in the
repository; several jobs can share one output directory, since each skips
scenarios that are already done.

The two greedy variants (`myopic`, `baseline`) are far cheaper and will run on
a workstation.

## Running your own scenarios

To try the models on new trials rather than the paper's, generate a small set
and point the pipeline at it.

### 1. Make the scenarios

    python run_pipeline.py scenarios --per-type 2 --seed 42 --version mytest

`--per-type` is the number of scenarios drawn for each of the 2, 3 and 4
critical-property conditions, and every scenario becomes four test cases, one
per user awareness level (10/25/50/80%). So `--per-type 2` gives 3 x 2 = 6
scenarios and 24 runnable test cases. The paper's 800 come from
`--per-type 25`.

Two safeguards worth knowing about. `--version` names the output files, and
the command refuses to overwrite an existing set, so a set already on disk
cannot be clobbered by accident. `--start-id` (default 101) sets where scenario
numbering begins, which keeps new ids clear of the 1-75 range the paper's set
occupied; without it the generator would restart at 51 and collide. `--seed`
makes the draw reproducible, which matters because the awareness centres are
otherwise drawn unseeded.

This writes `data/scenarios/dynamic_scenarios_vmytest.csv` and
`data/scenarios/dynamic_test_cases_vmytest.csv`, and prints the `--test-cases`
argument to use next.

### 2. Run the models

Pass your test-case file to every later stage. `--model all` runs the four
variants in the right order (`full` first, since the others record its reward
for the head-to-head comparison):

    python run_pipeline.py simulate  --model all --test-cases data/scenarios/dynamic_test_cases_vmytest.csv --processes 8
    python run_pipeline.py summarize --model all --test-cases data/scenarios/dynamic_test_cases_vmytest.csv
    python run_pipeline.py compare

To run one variant at a time — useful when you want the cheap ones first, or
want to send only the expensive ones to a cluster:

    python run_pipeline.py simulate --model baseline --test-cases <your csv>   # d-RSA
    python run_pipeline.py simulate --model myopic   --test-cases <your csv>   # + Priors
    python run_pipeline.py simulate --model uniform  --test-cases <your csv>   # + Planning
    python run_pipeline.py simulate --model full     --test-cases <your csv>   # + Priors + Planning

Results land in `output/runs/<model>/`, summaries in `output/summaries/`, and
the comparison tables in `output/comparisons/`. `compare` requires all four
summaries and tells you which are missing if you run it early.

Runs are resumable: a scenario whose result file already exists is skipped, so
you can interrupt a sweep and restart it, or add scenarios to an existing set
and re-run to fill only the gaps. Use `--overwrite` to force recomputation.

### Runtime

The greedy variants (`baseline`, `myopic`) take seconds per scenario. The
planning variants (`full`, `uniform`) search ~94k sequences per scenario at
H=7 and take several minutes each, so more than a handful of scenarios is a
cluster job.

`--horizon` runs a smaller H into `output/runs/<model>_H<n>/`, but
`summarize` only works at H=7.

## Repository layout

    run_pipeline.py            single entry point for every stage
    rsa_planning/
      paths.py                 repository-relative paths, model aliases, H
      properties.py            the 6 drone attributes and their thresholds
      rsa.py                   literal speaker, pragmatic listener, matrix helpers
      sequences.py             utterance legality and the pruned search tree
      worldgen.py              world states, criticality, initial user beliefs
      scenarios.py             scenario and test-case generation
      simulation.py            the shared RSA core and the two search strategies
      precompute.py            caching of the legal-sequence tree
      runner.py                dispatch of a variant over a scenario set
      trajectories.py          rank-1 trajectory reconstruction
      summarize.py             per-run metric extraction
      compare.py               the wide and long cross-model tables
    analysis/
      figures.Rmd              the paper's figures, as source
      statistics.Rmd           the mixed-effect regressions, as source
    data/
      lexicon_tables/          utterance-to-property semantics
      utterance_mapping/       utterance durations, costs, property specs
    cluster/                   HTCondor submit files
    output/                    everything the pipeline generates (git-ignored)

`data/precomputed_4D6P/` is generated rather than shipped: the sequence tree
depends only on the utterance map, so it is deterministic and rebuilt on
demand in about a minute.

## Notes on the paper's configuration

**The scenario set is not distributed.** This repository ships the model and
the pipeline, not the 800 trials behind the reported results. A generated set
will not reproduce those trials in any case: the generator draws per-property
awareness centres from `random.uniform()`, which the research code left
unseeded, so the original draw cannot be recovered. `--seed` makes a new set
reproducible, not an old one. What the pipeline reproduces is the method and
the model comparison, over whatever scenarios you generate.

**Planning horizon.** H is 7 throughout, as reported in the paper, and is set
in one place (`DEFAULT_HORIZON` in `rsa_planning/paths.py`), overridable with
`--horizon`. In the research code it was hard-coded inside the simulation
class and had drifted across commits; note that trajectory reconstruction
assumes a 7-timestep horizon, so other values need that code adjusted too.

**Run order.** The three non-full variants record the full model's reward for
the same utterance sequence, for the head-to-head comparison. Run `full`
first, or use `--model all`, which already orders them correctly.

## Citing

    @inproceedings{das2026planning,
      title     = {Planning Ahead with {RSA}: Efficient Signalling in Dynamic
                   Environments by Projecting User Awareness across Future Timesteps},
      author    = {Das, Anwesha and Duff, John and Hoffmann, J{\"o}rg and Demberg, Vera},
      booktitle = {Proceedings of the 25th International Conference on Autonomous
                   Agents and Multiagent Systems (AAMAS 2026)},
      year      = {2026},
      doi       = {10.65109/BRGK2754}
    }

## License

MIT — see [LICENSE](LICENSE).
