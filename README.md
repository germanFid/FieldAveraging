# Averaging Fields of Physical Quantities

A Python module to operate with 2D/3D physical fields

**before running project**, install prequesites:

```bash
python -m pip install -r requirements.txt
```

## Use cases

To use user interface run

```bash
python ./main.py # for shell Interface (non-interactive, use --help)
python ./mainUI.py # for GUI
```

Also, you can use it like Python module:

```python
import averager
import structures

data = structures.StreamData("myfile.csv", 10, 10) # example of initializing StreamData class
```

for other examples of use, you can refer to `documentation.ipynb`
