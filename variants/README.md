# variants/

Learner variants selectable from the command line:

    python main.py -m ij.3 -V <variant>

`python main.py --help` lists the names. The registry is `__init__.py`; adding a
variant means adding a file to one of the folders below and one line there.

| folder         | approach                                                                 |
|----------------|--------------------------------------------------------------------------|
| `model_based/` | builds an explicit MDP estimate and runs bounded value iteration on it    |
| `model_free/`  | the TDQL ladder of PAC stages                                            |

Both families take the same constructor arguments and expose `learn(analysis_dir)`
and `print_summary(...)`, which is the whole interface `main.py` uses.

## Copies vs. re-exports

Files that record a past state of the code are **copies** and are meant to stay
frozen: `model_free/original.py`, `model_free/cumulative.py`.

Files that name the learner the repo currently runs are **re-exports** of the
live module: `model_free/current.py`, `model_based/current.py`. They are not
copies on purpose -- a copy would drift from production the first time either
was edited, and the variant would quietly stop running the thing it names.
