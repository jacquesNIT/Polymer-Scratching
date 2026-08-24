This project is meant as a complete pipeline to polymer scratching simulation. This file below explains how to use it.


Parameter definition:

Base.py contains all the simulations parameters : substrate, indeter, mesh, models, solver etc..
Those parameters remained largely unchanged throughout the entire project, with the sole exception of polymer_default.

polymer_default serves as the base for modifying parameters, if you want to change the mesh size, scratching time etc.. Here is the place to do it.

Families.py is an extension of polymer_default that directly creates the polymer types wanted with the assigned models, any parameter modified here will override polymer_default.


Running:

Current running parameters are meant for cluster run only.
submit.sh allows to choose the node type as well as the number of CPUs and memory used.

Multiple studies were implemented, those can be found in run_parameter_study.py. 
To launch those studies on the cluster, launch_cluster_jobs is used, you can directly select the type of study and family you want to start a job on.