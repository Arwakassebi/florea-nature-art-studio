"""Gestionnaire de modèles GPU : chargement à la demande + éviction LRU (le moins récemment utilisé).

Sans dépendance GPU : les fonctions de chargement et de libération sont fournies par l'appelant, donc testable partout.
"""
import time
from collections import OrderedDict


class ModelManager:
    def __init__(self, budget_gb, free_fn=lambda: None, verbose=True):
        self.budget, self.free_fn, self.verbose = float(budget_gb), free_fn, verbose
        self.specs = {}                      # nom -> (loader, taille estimée en Go)
        self.loaded = OrderedDict()          # nom -> objet (le plus ancien en premier)
        self.sizes = {}
        self.loads = {}                      # nom -> nombre de chargements (utile pour les tests et le suivi)

    def register(self, name, loader, size_gb):
        self.specs[name] = (loader, float(size_gb))

    def used_gb(self):
        return sum(self.sizes[n] for n in self.loaded)

    def get(self, name):
        if name in self.loaded:
            self.loaded.move_to_end(name)
            return self.loaded[name]
        loader, size = self.specs[name]
        while self.loaded and self.used_gb() + size > self.budget:
            victim = next(iter(self.loaded))            # le plus ancien
            if self.verbose:
                print(f"[mémoire] libère « {victim} » ({self.sizes[victim]:.1f} Go) pour charger « {name} »")
            self.unload(victim)
        t0 = time.time()
        self.loaded[name] = loader()
        self.sizes[name] = size
        self.loads[name] = self.loads.get(name, 0) + 1
        if self.verbose:
            print(f"[mémoire] « {name} » chargé en {time.time() - t0:.0f}s ({self.used_gb():.1f}/{self.budget:.1f} Go estimés)")
        return self.loaded[name]

    def unload(self, name):
        if name in self.loaded:
            del self.loaded[name]
            self.free_fn()

    def unload_all(self):
        for n in list(self.loaded):
            self.unload(n)

    def status(self):
        return ", ".join(f"{n} ({self.sizes[n]:.1f} Go)" for n in self.loaded) or "aucun"
