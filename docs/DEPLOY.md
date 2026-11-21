# Déployer FLOREA Lite sur Azure (offre étudiante)

Ce guide met en ligne la **version Lite** (CPU) sur une VM Linux **B1s** (1 Go de RAM), avec HTTPS automatique.
La version complète (génération d'image) demande un GPU : voir la dernière section.

## Ce que ça coûte (à vérifier dans *Cost Management*)

- La VM B1s : couverte par les 750 h/mois de l'offre gratuite (≈ 744 h dans un mois de 31 jours).
- **Ne sont pas forcément couverts** : l'IP publique Standard et le disque (30 Go). Comptez quelques dollars par mois et surveillez *Cost analysis* la première semaine.
- Créez un **budget avec alerte** avant de commencer (*Cost Management → Budgets*).

## Étape 1 : publier le code Lite sur GitHub

```bash
git add src/florea/lite.py tests/test_lite.py requirements-lite.txt deploy docs Dockerfile .dockerignore README.md
git commit -m "Add FLOREA Lite (CPU demo) and Azure deployment scripts"
git push
```

## Étape 2 : créer la VM (depuis votre ordinateur)

Prérequis : [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli) puis `az login`.

```bash
bash deploy/azure_create_vm.sh
```

Le script vérifie que `Standard_B1s` existe dans la région, crée la VM, ouvre les ports 80 et 443, et affiche un **nom DNS** du type `florea-ab12cd.northeurope.cloudapp.azure.com`. Notez-le.

Si la taille n'est pas disponible : `LOCATION=westeurope bash deploy/azure_create_vm.sh`, ou `SIZE=Standard_B2ats_v2 bash deploy/azure_create_vm.sh`.

## Étape 3 : installer sur la VM

```bash
ssh azureuser@<nom-DNS>
git clone https://github.com/Arwakassebi/florea-nature-art-studio.git
cd florea-nature-art-studio
bash deploy/setup_vm.sh <nom-DNS>
```

Le script : ajoute 1 Go de swap, installe Python et Caddy, installe les dépendances Lite (sans PyTorch), crée le service `florea-lite` (redémarrage automatique, limité à 700 Mo) et configure Caddy pour obtenir le certificat HTTPS.

## Étape 4 : vérifier

Attendez ~20 s (le GCN s'entraîne au démarrage), puis ouvrez `https://<nom-DNS>`.

```bash
sudo systemctl status florea-lite          # doit être « active (running) »
sudo journalctl -u florea-lite -f          # logs de l'application
sudo journalctl -u caddy -f                # logs HTTPS
free -m                                    # mémoire
```

## Étape 5 : mettre à jour après un `git push`

```bash
ssh azureuser@<nom-DNS>
cd florea-nature-art-studio && bash deploy/update.sh
```

## Étape 6 : l'ajouter au README et au profil

Remplacez dans `README.md` l'adresse de la section « Démo en ligne » par votre nom DNS, puis ajoutez le lien dans le champ *Website* du dépôt GitHub et sur LinkedIn.

## Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `https://…` ne répond pas | Ports 80/443 fermés, ou DNS pas encore actif | `az vm open-port` (déjà fait par le script) ; attendre 2 min ; `sudo journalctl -u caddy -f` |
| Erreur de certificat | Let's Encrypt ne joint pas la VM | Vérifier que le port 80 est ouvert et que le nom DNS passé à `setup_vm.sh` est le bon |
| Service qui redémarre en boucle | Mémoire saturée | `free -m` ; vérifier le swap (`swapon --show`) ; consulter `journalctl -u florea-lite` |
| `SkuNotAvailable` à la création | Taille non proposée dans la région | Autre région ou `SIZE=Standard_B2ats_v2` |
| Recherche peu pertinente en anglais ou français libre | Embeddings lexicaux (limite connue de Lite) | Normal ; la version complète utilise MiniLM |

## Supprimer tout (arrête toute facturation)

```bash
az group delete -n florea-rg --yes
```

## Variante Docker

```bash
docker build -t florea-lite .
docker run -p 7860:7860 florea-lite        # http://localhost:7860
```

Le même `Dockerfile` convient à tout hébergeur de conteneurs (non testé ici : vérifiez le build chez vous).

## Version complète (GPU)

Elle nécessite une VM GPU (famille `NCASv3_T4`) : demandez d'abord le quota dans *Abonnements → Utilisation + quotas*. Sur la VM :

```bash
pip install torch && pip install -r requirements.txt && pip install -e . --no-deps
python deploy/run_gpu_app.py
ssh -L 7860:localhost:7860 azureuser@<IP>      # depuis votre ordinateur, puis http://localhost:7860
```

Une VM T4 coûte environ 0,5 $/h : **désallouez-la** (`az vm deallocate -g <rg> -n <vm>`) dès que vous avez fini.


## Démo en ligne — FLOREA Lite

FLOREA Lite est une version allégée du projet, conçue pour fonctionner sur CPU avec une faible consommation de mémoire et pouvant être déployée sur une VM Azure.

### Fonctionnalités
- Analyse de scènes naturelles.
- Recherche dans la base de connaissances.
- Graph-RAG avec chemins explicables.
- Classement des styles artistiques par GCN.
- Génération de prompts artistiques.

Cette version ne réalise pas directement la génération des poèmes et des images, contrairement à la version complète.

### Installation locale

```bash
pip install -r requirements-lite.txt
pip install -e . --no-deps
python -m florea.lite
```

### Déploiement Azure

Consultez le guide : [docs/DEPLOY.md](docs/DEPLOY.md).