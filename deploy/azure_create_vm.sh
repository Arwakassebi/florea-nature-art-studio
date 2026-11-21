#!/usr/bin/env bash
# Crée sur Azure : un groupe de ressources, une VM Linux B1s, une IP publique avec nom DNS, les ports 80/443.
# Usage (depuis votre ordinateur, après `az login`) :  bash deploy/azure_create_vm.sh
# Variables facultatives : RG, LOCATION, VM, SIZE, DNS_LABEL
set -euo pipefail

RG="${RG:-florea-rg}"
LOCATION="${LOCATION:-northeurope}"
VM="${VM:-florea-lite}"
SIZE="${SIZE:-Standard_B1s}"
DNS_LABEL="${DNS_LABEL:-florea-$(head -c3 /dev/urandom | od -An -tx1 | tr -d ' \n')}"

echo ">> Taille demandée : $SIZE en $LOCATION (vérification de disponibilité)"
if ! az vm list-skus -l "$LOCATION" --size "$SIZE" --resource-type virtualMachines -o tsv 2>/dev/null | grep -q .; then
  echo "!! $SIZE n'est pas proposée dans $LOCATION. Essayez : LOCATION=westeurope bash $0  ou  SIZE=Standard_B2ats_v2 bash $0"
  exit 1
fi

az group create -n "$RG" -l "$LOCATION" -o none
az vm create -g "$RG" -n "$VM" \
  --image Ubuntu2204 --size "$SIZE" \
  --os-disk-size-gb 30 --storage-sku StandardSSD_LRS \
  --admin-username azureuser --generate-ssh-keys \
  --public-ip-sku Standard --public-ip-address-dns-name "$DNS_LABEL" \
  -o none
az vm open-port -g "$RG" -n "$VM" --port 80  --priority 900 -o none
az vm open-port -g "$RG" -n "$VM" --port 443 --priority 901 -o none

FQDN="$DNS_LABEL.$LOCATION.cloudapp.azure.com"
IP=$(az vm show -d -g "$RG" -n "$VM" --query publicIps -o tsv)
cat <<MSG

✅ VM créée.
   Nom DNS : $FQDN
   IP      : $IP

Étape suivante :
   ssh azureuser@$FQDN
   git clone <URL de votre dépôt> && cd florea-nature-art-studio
   bash deploy/setup_vm.sh $FQDN

Pour tout supprimer plus tard :  az group delete -n $RG --yes
MSG
