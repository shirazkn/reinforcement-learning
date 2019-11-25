See notebook file for an overview.

***
#### Troubleshooting :

On MacOSX Mojave and later, PyGame opens up as a blank window. Due to an
apparent conflict with the dark mode overlay.
 
Can be solved by reinstalling Python using the official MacOSx64
installer (as opposed to Homebrew). I had to re-link my ipython to the
new python installation using,
```
cd /usr/local/opt 
rm python 
ln -s python3 python
```
***

How to run Jupyter Notebook in a virtual environment ... handy if you
don't want *gym_maze* sitting in your packages forever!

- Activate your venv
- Install `ipykernel`
- `python -m ipykernel install --name NAME`
- Open Jupyter notebook and select kernel NAME

My *autonomous-atc* repo has some instructions on managing virtualenvs.