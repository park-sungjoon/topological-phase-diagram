# topological-phase-diagram

Codes used in "Unsupervised learning of topological phase diagram using topological data analysis" by S. Park, Y. Hwang, and B.-J. Yang.
There, we studied three models: SSH model, three-band model, and QWZ model.
The codes for producing the phase diagram for each of the three models are independent of each other (this results in some overlaps between codes).

## Python setup
To setup virtual environment (my python version is 3.7.6):
```bash
virtualenv phase_diagram
source phase_diagram/bin/activate
pip install -r requirements.txt
```

# Usage
The files "(model_name)_deformation_dirichlet.py" and "(model_name)_phase_diagram_multiprocessing.py" are the main files, where model_name can be ssh, tb, or qwz.
Running "(model_name)_deformation_dirichlet.py" will produce the persistence diagrams and the embeddings of the wavefunctions in Euclidean space before the deformation and after the deformation, in both the topologically trivial phase and the topologically nontrivial phase.
Running "(model_name)_phase_diagram_multiprocessing.py" will  produce the phase diagram for the model.
