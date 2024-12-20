# This dictionary defines how Raven annotations will be transformed for model training
# If a species that was annotated is missing, please edit this file.

groups = {
    "Apodemus": "target_stm",
    "Apodemus_sylvaticus": "target_stm",
    "Apodemus_flavicollis": "target_stm",
    "Mus_musculus": "target_stm",
    "Micromys_minutus": "target_stm",
    "Micromys-minutus": "target_stm",
    "Rattus_norvegicus": "target_stm",
    "Arvicola_amphibius": "target_stm",
    "Microtus_agrestis": "target_stm",
    "Myodes_glareolus": "target_stm",
    "Sorex_araneus": "target_stm",
    "Sorex_minutus": "target_stm",
    "Neomys_fodiens": "target_stm",
    "Arvicolinae": "target_stm",
    "Rattus": "target_stm",
    "Soricidae": "target_stm",
    "Crocidura_russula": "target_stm",
    "Rattus_rattus": "target_stm",
    "Neomys_fodiens": "target_stm",
    "targetMammal": "target_stm",
    "Eptesicus_serotinus": "bats",
    "Myotis_daubentonii": "bats",
    "Myotis_myotis": "bats",
    "Myotis_mystacinus": "bats",
    "Nyctalus_leisleri": "bats",
    "Nyctalus_noctula": "bats",
    "Plecotus_auritus": "bats",
    "Plecotus_austriacus": "bats",
    "Pipistrellus_nathusii": "bats",
    "Pipistrellus_pipistrellus": "bats",
    "Pipistrellus_pipistrelus": "bats",
    "Pipistrellus_pygmaeus": "bats",
    "bat": "bats",
    "noise": "Noise",
}

classes_grp = ("target_stm", "bats", "Noise")
