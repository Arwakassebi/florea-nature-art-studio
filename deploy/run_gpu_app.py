"""Lancement de la version COMPLÈTE (GPU) sur une VM GPU : python deploy/run_gpu_app.py

Interface accessible par tunnel SSH :  ssh -L 7860:localhost:7860 azureuser@<IP>  puis http://localhost:7860
"""
import os

from florea import pipeline as P, ui

P.init(save_dir=os.path.expanduser("~/florea_gallery"))
ui.launch(share=False)
