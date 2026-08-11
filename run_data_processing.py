"""Extract scratch features from the simulation file(s) in a folder.

This data was simulated with a different constitutive model (Drucker-Prager
plasticity for a glassy polymer, ``family = glassy_pc``) than the rest of the
project (power-law hardening metals with ``E, A, B, n, mu`` parameters). The
main pipeline (``data_processing_main``) fits tensile parameters and builds
Buckingham-Pi dimensionless groups around that metal model, so it does not
apply here. This script instead runs only the topography feature-extraction
stage directly on each ``*_Results.csv`` in the target folder and writes the
resulting physical (dimensional) features to ``features.csv``.

Adaptation a l'arborescence ScratchSimulation (aucune logique de calcul
modifiee) :
  * les imports ``ml_scratch.*`` pointent vers le package local
    ``ScratchFeatures`` ;
  * le dossier de donnees n'est plus celui du script mais un argument de ligne
    de commande, de sorte que le script puisse rester dans le depot pendant que
    les CSV sont ailleurs (rapatries par WinSCP, par exemple) ;
  * backend matplotlib non interactif, pour tourner sans DISPLAY (cluster) ;
  * creation du dossier de sortie des figures s'il n'existe pas.

Usage :
    python run_data_processing.py <dossier_des_csv> [<dossier_de_sortie>]
    python run_data_processing.py            # -> dossier courant
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas de DISPLAY sur le cluster

import pandas as pd

# Le package est resolu qu'il soit importe depuis la racine du depot ou que ce
# script soit lance depuis un autre repertoire de travail.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ScratchFeatures.constants as C
from ScratchFeatures.scratch_simulation_helpers import simulation_main
from ScratchFeatures.scratch_feature_extraction_helpers import (
    feature_extraction_pipeline,
)


def main(data_dir=None, out_dir=None):
    data_dir = Path(data_dir) if data_dir else Path.cwd()
    out_dir = Path(out_dir) if out_dir else data_dir
    os.makedirs(str(out_dir), exist_ok=True)

    sim_files = sorted(data_dir.glob("*_Results.csv"))
    if not sim_files:
        print(f"No *_Results.csv found in {data_dir}")
        return

    all_features = []
    for sim_file in sim_files:
        X, Y, Z, F_n, F_t, parameters = simulation_main(
            simID=sim_file.name,
            path=str(data_dir),
            z_values=[C.scratch_length],
        )

        features = feature_extraction_pipeline(
            X=X,
            Y=Y,
            Z=Z,
            F_n=F_n,
            F_t=F_t,
            parameters=parameters,
            z_value=C.scratch_length,
            plot=True,
            save_dir=f"{out_dir}/",
            get_additional_features=True,
            get_yz_profile_features=True,
            get_volume_features=True,
        )
        features["sim_id"] = sim_file.stem.removesuffix("_Results")
        all_features.append(features)

    features_df = pd.DataFrame(all_features)
    features_df.to_csv(out_dir / "features.csv", index=False)
    print(f"Wrote {len(features_df)} row(s) to {out_dir / 'features.csv'}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else None,
        sys.argv[2] if len(sys.argv) > 2 else None,
    )