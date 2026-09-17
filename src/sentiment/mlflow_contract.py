"""
mlflow_contract.py
====================

Encapsula el logging a MLflow segun el contrato EXACTO del Anexo A
(A.1/A.2) de la guia del laboratorio -- ya confirmado leyendo el PDF
real, no inferido.

Experiment de MLflow: nombre exacto ``nlp-lab2-sentiment140`` (A.1).

Tipos de run (tag reservado ``lab_run_type``, valores exactos):
``protocol``, ``experiment``, ``final`` (A.1). Un run sin ese tag es
"exploratorio" y el evaluador lo ignora.

Run de protocolo (A.2) -- exactamente uno para todo el equipo:
    - tag: lab_run_type=protocol
    - params EXACTOS: dataset_id, dataset_revision, sampling_strategy,
      sample_size, random_seed, cv_strategy, cv_folds, cv_shuffle
      (delegados a ``sentiment.data.protocol_params()``).
    - artifacts: protocol/partitions.csv, protocol/members.csv.

Run experimental (A.2) -- T0, B0 y cada comparacion/ablacion:
    - tags: lab_run_type=experiment, lab_protocol_run_id, lab_stage,
      lab_experiment_id, lab_member_id, lab_configuration_id,
      notebook_arn. (``lab_experiment_id`` esta confirmado como tag
      obligatorio por la Seccion 3 -- "los codigos P_*, R_* y C_*
      identifican las comparaciones y son valores del tag propio
      lab_experiment_id de cada run" -- y por la tabla completa de
      A.4, aunque el bullet de "Tags" de A.2 no lo vuelve a listar
      explicitamente; se incluye igual porque sin el, el evaluador no
      podria distinguir p. ej. P_STOPWORDS de P_LEMMA.)
    - metricas: macro_f1_fold_0, macro_f1_fold_1, macro_f1_fold_2,
      macro_f1_mean, macro_f1_std (ddof=0) -- exactamente lo que
      devuelve ``sentiment.cv.run_cv()``.
    - artifacts: run/configuration.json,
      provenance/sagemaker-resource-metadata.json.
    - si es una ABLACION, ademas: tag lab_ablation_parent_run_id, param
      ablation_reverted_decision (uno de los 6 valores exactos de A.4,
      ver ``sentiment.config.AblatableDecision``) y metrica
      macro_f1_delta = macro_f1_mean(parent) - macro_f1_mean(ablation).

Run final (A.2) -- reentrenamiento con todo train + evaluacion en test:
    - tags: lab_run_type=final, lab_protocol_run_id,
      lab_selected_experiment_run_id, lab_configuration_id,
      lab_member_id, notebook_arn.
    - param: training_size=1360000.
    - metrica: test_macro_f1.
    - artifacts: run/configuration.json,
      provenance/sagemaker-resource-metadata.json,
      reports/error_analysis.csv, reports/error_analysis.md.

Procedencia (A.2): dentro de una SageMaker Notebook Instance, AWS
expone ``/opt/ml/metadata/resource-metadata.json``; se sube SIN editar
como ``provenance/sagemaker-resource-metadata.json``, y su campo
``ResourceArn`` debe coincidir EXACTO con el tag ``notebook_arn`` del
mismo run. Por eso ``read_notebook_arn()`` lo lee de ahi en vez de que
cada integrante lo transcriba a mano.

Nota sobre tipos y MLflow: los params de MLflow se guardan como texto.
La guia fija el valor EXACTO ``cv_shuffle=true`` en minuscula, pero
``str(True)`` en Python da ``'True'`` (mayuscula) -- por eso
``build_protocol_params_dict()`` normaliza booleanos a ``"true"``/``"false"``
antes de devolver el dict que se pasa a ``mlflow.log_params``.

Separacion pura vs. MLflow real: las funciones ``build_*`` solo
arman/validan dicts (tags/params/metrics) -- no tocan MLflow y se
prueban aqui sin tenerlo instalado. Las funciones ``log_*`` si llaman a
la libreria ``mlflow`` (import perezoso) y necesitan un run activo y un
tracking server real -- no se pueden probar en este sandbox (no hay
``mlflow`` instalado ni servidor, ni ``/opt/ml/metadata/resource-metadata.json``
porque esto no es una SageMaker Notebook Instance).
"""

from __future__ import annotations

import json as _json
import os
import tempfile
import typing
from typing import Any, Optional

from sentiment.config import AblatableDecision, PipelineConfig
from sentiment.data import protocol_params

# ---------------------------------------------------------------------------
# Constantes confirmadas contra el Anexo A
# ---------------------------------------------------------------------------

MLFLOW_EXPERIMENT_NAME = "nlp-lab2-sentiment140"  # A.1

RUN_TYPE_PROTOCOL = "protocol"
RUN_TYPE_EXPERIMENT = "experiment"
RUN_TYPE_FINAL = "final"

STAGE_REFERENCE = "reference"  # T0
STAGE_BASELINE = "baseline"  # B0
STAGE_PREPROCESSING = "preprocessing"
STAGE_REPRESENTATION = "representation"
STAGE_CLASSIFIER = "classifier"
STAGE_ABLATION = "ablation"

# lab_experiment_id -> lab_stage, los 15 codigos fijos de la tabla de A.4
# (EXTRA no esta aqui: su stage depende de que compare el equipo -- pasar
# stage= explicito al usar EXTRA).
EXPERIMENT_ID_STAGES: dict[str, str] = {
    "T0": STAGE_REFERENCE,
    "B0": STAGE_BASELINE,
    "P_STOPWORDS": STAGE_PREPROCESSING,
    "P_STOPWORDS_NEGATION": STAGE_PREPROCESSING,
    "P_LEMMA": STAGE_PREPROCESSING,
    "P_ELONGATION": STAGE_PREPROCESSING,
    "P_EMOJI": STAGE_PREPROCESSING,
    "R_BOW": STAGE_REPRESENTATION,
    "R_TFIDF_UNI": STAGE_REPRESENTATION,
    "R_TFIDF_UNI_BI": STAGE_REPRESENTATION,
    "R_SPACY": STAGE_REPRESENTATION,
    "C_LOGREG": STAGE_CLASSIFIER,
    "C_LINEAR_SVM": STAGE_CLASSIFIER,
    "C_SGD": STAGE_CLASSIFIER,
    "ABLATION": STAGE_ABLATION,
}

# Ruta EXACTA del artifact de configuracion dentro del run (A.3).
CONFIGURATION_ARTIFACT_DIR = "run"
CONFIGURATION_ARTIFACT_FILENAME = "configuration.json"
CONFIGURATION_ARTIFACT_PATH = f"{CONFIGURATION_ARTIFACT_DIR}/{CONFIGURATION_ARTIFACT_FILENAME}"

# Procedencia (A.2).
SAGEMAKER_RESOURCE_METADATA_SRC = "/opt/ml/metadata/resource-metadata.json"
PROVENANCE_ARTIFACT_DIR = "provenance"
PROVENANCE_ARTIFACT_FILENAME = "sagemaker-resource-metadata.json"

# Reports del run final (A.2/A.6).
ERROR_ANALYSIS_ARTIFACT_DIR = "reports"
ERROR_ANALYSIS_CSV_FILENAME = "error_analysis.csv"
ERROR_ANALYSIS_MD_FILENAME = "error_analysis.md"

# training_size del run final (A.2) -- filas de train completo (Seccion 2),
# igual al valor que sentiment.data.load_train_test() valida.
FINAL_TRAINING_SIZE = 1_360_000

_REQUIRED_CV_METRIC_SUFFIXES = ("macro_f1_mean", "macro_f1_std")
_VALID_ABLATABLE_DECISIONS = typing.get_args(AblatableDecision)


# ---------------------------------------------------------------------------
# Builders puros -- arman/validan dicts, sin tocar MLflow
# ---------------------------------------------------------------------------


def build_protocol_run_tags() -> dict[str, str]:
    """Tags EXACTOS del run de protocolo (A.2): solo lab_run_type."""
    return {"lab_run_type": RUN_TYPE_PROTOCOL}


def build_protocol_params_dict() -> dict[str, Any]:
    """Params EXACTOS del run de protocolo (A.2), delegando en
    ``sentiment.data.protocol_params()``. Normaliza booleanos a
    ``"true"``/``"false"`` en minuscula (ver nota del modulo sobre
    ``cv_shuffle``)."""
    raw = dict(protocol_params())
    return {k: (str(v).lower() if isinstance(v, bool) else v) for k, v in raw.items()}


def build_experimental_run_tags(
    *,
    protocol_run_id: str,
    experiment_id: str,
    member_id: str,
    configuration_id: str,
    notebook_arn: str,
    stage: Optional[str] = None,
    ablation_parent_run_id: Optional[str] = None,
) -> dict[str, str]:
    """Tags EXACTOS de un run experimental (A.2). ``stage`` se infiere
    de ``EXPERIMENT_ID_STAGES[experiment_id]`` si no se pasa; para
    ``experiment_id="EXTRA"`` hay que pasarlo explicito (A.4: EXTRA
    puede ser preprocessing/representation/classifier/ablation segun lo
    que compare el equipo)."""
    if stage is None:
        try:
            stage = EXPERIMENT_ID_STAGES[experiment_id]
        except KeyError:
            raise ValueError(
                f"No se puede inferir lab_stage para lab_experiment_id={experiment_id!r}. "
                "Es valido solo para los 15 codigos fijos de A.4 -- para 'EXTRA' (u otro "
                "codigo propio) pasa stage= explicito."
            ) from None

    tags = {
        "lab_run_type": RUN_TYPE_EXPERIMENT,
        "lab_protocol_run_id": protocol_run_id,
        "lab_stage": stage,
        "lab_experiment_id": experiment_id,
        "lab_member_id": member_id,
        "lab_configuration_id": configuration_id,
        "notebook_arn": notebook_arn,
    }
    if ablation_parent_run_id is not None:
        tags["lab_ablation_parent_run_id"] = ablation_parent_run_id
    return tags


def build_final_run_tags(
    *,
    protocol_run_id: str,
    selected_experiment_run_id: str,
    configuration_id: str,
    member_id: str,
    notebook_arn: str,
) -> dict[str, str]:
    """Tags EXACTOS del run final (A.2)."""
    return {
        "lab_run_type": RUN_TYPE_FINAL,
        "lab_protocol_run_id": protocol_run_id,
        "lab_selected_experiment_run_id": selected_experiment_run_id,
        "lab_configuration_id": configuration_id,
        "lab_member_id": member_id,
        "notebook_arn": notebook_arn,
    }


def build_final_run_params() -> dict[str, Any]:
    """Param EXACTO del run final (A.2): training_size=1360000."""
    return {"training_size": FINAL_TRAINING_SIZE}


def build_test_macro_f1_metric(test_macro_f1: float) -> dict[str, float]:
    """Metrica EXACTA del run final (A.2): test_macro_f1."""
    return {"test_macro_f1": float(test_macro_f1)}


def compute_macro_f1_delta(parent_macro_f1_mean: float, ablation_macro_f1_mean: float) -> float:
    """macro_f1_delta = macro_f1_mean(parent) - macro_f1_mean(ablation) (A.2)."""
    return float(parent_macro_f1_mean - ablation_macro_f1_mean)


def build_ablation_extra(
    *,
    ablation_parent_run_id: str,
    reverted_decision: str,
    macro_f1_delta: float,
) -> tuple[dict[str, str], dict[str, str], dict[str, float]]:
    """Devuelve ``(tags_extra, params_extra, metrics_extra)`` para una
    ablacion (A.2): tag ``lab_ablation_parent_run_id``, param
    ``ablation_reverted_decision`` (validado contra los 6 valores
    exactos de ``sentiment.config.AblatableDecision``, A.4) y metrica
    ``macro_f1_delta``."""
    if reverted_decision not in _VALID_ABLATABLE_DECISIONS:
        raise ValueError(
            f"reverted_decision debe ser uno de {_VALID_ABLATABLE_DECISIONS}, "
            f"recibio {reverted_decision!r}"
        )
    return (
        {"lab_ablation_parent_run_id": ablation_parent_run_id},
        {"ablation_reverted_decision": reverted_decision},
        {"macro_f1_delta": float(macro_f1_delta)},
    )


def build_configuration_artifact(config: PipelineConfig) -> tuple[str, str]:
    """``(nombre_de_archivo, contenido_json)`` para el artifact de
    configuracion. La ruta final (``run/configuration.json``) la fija
    ``artifact_path=CONFIGURATION_ARTIFACT_DIR`` al subirlo."""
    config.validate()
    return CONFIGURATION_ARTIFACT_FILENAME, config.to_json()


def build_cv_metrics_dict(cv_metrics: dict[str, float]) -> dict[str, float]:
    """Valida que ``cv_metrics`` (tal como lo devuelve
    ``sentiment.cv.run_cv``) tenga exactamente las claves esperadas
    antes de loguearlas."""
    fold_keys = {k for k in cv_metrics if k.startswith("macro_f1_fold_")}
    missing = set(_REQUIRED_CV_METRIC_SUFFIXES) - set(cv_metrics)
    if missing:
        raise ValueError(f"Faltan metricas obligatorias en cv_metrics: {sorted(missing)}")
    if not fold_keys:
        raise ValueError("cv_metrics no tiene ninguna clave 'macro_f1_fold_N'")
    extra = set(cv_metrics) - fold_keys - set(_REQUIRED_CV_METRIC_SUFFIXES)
    if extra:
        raise ValueError(f"cv_metrics tiene claves inesperadas: {sorted(extra)}")
    return dict(cv_metrics)


def build_flat_config_params(config: PipelineConfig) -> dict[str, Any]:
    """EXTRA opcional (NO exigido por A.2 -- A.1 permite params/tags
    adicionales): aplana preprocessing/representation/classifier como
    params sueltos, para filtrar/comparar runs en la UI de MLflow sin
    abrir cada configuration.json."""
    config.validate()
    flat: dict[str, Any] = {"classifier_type": config.classifier.type}
    if config.preprocessing is not None:
        for k, v in config.preprocessing.to_dict().items():
            if k in ("resources", "additional", "negators"):
                continue
            flat[f"preprocessing_{k}"] = v
    if config.representation is not None:
        for k, v in config.representation.to_dict().items():
            if k == "parameters":
                continue
            flat[f"representation_{k}"] = v
    return {k: v for k, v in flat.items() if v is not None}


def read_notebook_arn(*, source_path: str = SAGEMAKER_RESOURCE_METADATA_SRC) -> str:
    """Lee ``ResourceArn`` de ``resource-metadata.json`` -- el mismo
    valor que debe usarse en el tag ``notebook_arn`` (A.2), para no
    transcribirlo a mano y arriesgar un typo que invalide el run."""
    with open(source_path, encoding="utf-8") as f:
        data = _json.load(f)
    try:
        return data["ResourceArn"]
    except KeyError:
        raise KeyError(
            f"{source_path} no tiene el campo 'ResourceArn' esperado. Campos "
            f"presentes: {sorted(data)}"
        ) from None


# ---------------------------------------------------------------------------
# Logging real a MLflow (requiere 'mlflow' instalado + tracking server)
# ---------------------------------------------------------------------------


def log_configuration_artifact(config: PipelineConfig, *, tmp_dir: Optional[str] = None) -> None:
    """Sube ``run/configuration.json`` al run ACTIVO de mlflow (dentro
    de un ``with mlflow.start_run():``)."""
    import mlflow  # type: ignore

    filename, content = build_configuration_artifact(config)
    with tempfile.TemporaryDirectory(dir=tmp_dir) as d:
        path = os.path.join(d, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        mlflow.log_artifact(path, artifact_path=CONFIGURATION_ARTIFACT_DIR)


def log_provenance_artifact(*, source_path: str = SAGEMAKER_RESOURCE_METADATA_SRC) -> None:
    """Sube una copia SIN EDITAR de ``resource-metadata.json`` (que AWS
    expone dentro de cualquier SageMaker Notebook Instance real -- Seccion 6:
    "No se aceptarán runs presentados desde SageMaker Studio o Google
    Colab") como ``provenance/sagemaker-resource-metadata.json``."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(
            f"No se encontro {source_path}. Este archivo solo existe dentro de "
            "una SageMaker Notebook Instance real."
        )
    import mlflow  # type: ignore

    mlflow.log_artifact(source_path, artifact_path=PROVENANCE_ARTIFACT_DIR)


def log_error_analysis_artifacts(*, csv_path: str, md_path: str) -> None:
    """Sube ``reports/error_analysis.csv`` y ``reports/error_analysis.md``
    (A.2/A.6) al run FINAL activo."""
    import mlflow  # type: ignore

    mlflow.log_artifact(csv_path, artifact_path=ERROR_ANALYSIS_ARTIFACT_DIR)
    mlflow.log_artifact(md_path, artifact_path=ERROR_ANALYSIS_ARTIFACT_DIR)


def log_protocol_run(
    *,
    run_name: str = "protocol",
    partitions_csv_path: str,
    members_csv_path: str,
) -> str:
    """Abre y cierra el run de PROTOCOLO (Seccion 2 -- una sola vez por
    equipo). Devuelve el ``run_id`` (se lo pasas a los runs
    experimentales/final como ``lab_protocol_run_id``)."""
    import mlflow  # type: ignore

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(build_protocol_run_tags())
        mlflow.log_params(build_protocol_params_dict())
        mlflow.log_artifact(partitions_csv_path, artifact_path="protocol")
        mlflow.log_artifact(members_csv_path, artifact_path="protocol")
        return run.info.run_id


def log_experimental_run(
    *,
    run_name: str,
    config: PipelineConfig,
    cv_metrics: dict[str, float],
    protocol_run_id: str,
    experiment_id: str,
    member_id: str,
    configuration_id: str,
    notebook_arn: Optional[str] = None,
    stage: Optional[str] = None,
    ablation_parent_run_id: Optional[str] = None,
    ablation_reverted_decision: Optional[str] = None,
    ablation_macro_f1_delta: Optional[float] = None,
    resource_metadata_path: str = SAGEMAKER_RESOURCE_METADATA_SRC,
    log_flat_params: bool = False,
) -> str:
    """Abre y cierra un run experimental completo (T0, B0, P_*, R_*,
    C_*, o una ablacion) con todo el contrato de A.2: tags, metricas de
    CV, ``run/configuration.json`` y
    ``provenance/sagemaker-resource-metadata.json``.

    Si ``notebook_arn`` no se pasa, se lee de ``resource_metadata_path``
    con ``read_notebook_arn()``. Para una ablacion, pasa
    ``ablation_parent_run_id`` + ``ablation_reverted_decision`` +
    ``ablation_macro_f1_delta`` (los tres juntos)."""
    import mlflow  # type: ignore

    config.validate()
    build_cv_metrics_dict(cv_metrics)

    if notebook_arn is None:
        notebook_arn = read_notebook_arn(source_path=resource_metadata_path)

    is_ablation = ablation_parent_run_id is not None
    tags = build_experimental_run_tags(
        protocol_run_id=protocol_run_id,
        experiment_id=experiment_id,
        member_id=member_id,
        configuration_id=configuration_id,
        notebook_arn=notebook_arn,
        stage=stage,
        ablation_parent_run_id=ablation_parent_run_id,
    )

    params: dict[str, Any] = {}
    metrics = dict(cv_metrics)
    if is_ablation:
        if ablation_reverted_decision is None or ablation_macro_f1_delta is None:
            raise ValueError(
                "Una ablacion (ablation_parent_run_id dado) requiere tambien "
                "ablation_reverted_decision y ablation_macro_f1_delta"
            )
        _, ablation_params, ablation_metrics = build_ablation_extra(
            ablation_parent_run_id=ablation_parent_run_id,
            reverted_decision=ablation_reverted_decision,
            macro_f1_delta=ablation_macro_f1_delta,
        )
        params.update(ablation_params)
        metrics.update(ablation_metrics)
    if log_flat_params:
        params.update(build_flat_config_params(config))

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        log_configuration_artifact(config)
        log_provenance_artifact(source_path=resource_metadata_path)
        return run.info.run_id


def log_final_run(
    *,
    run_name: str = "final",
    config: PipelineConfig,
    protocol_run_id: str,
    selected_experiment_run_id: str,
    configuration_id: str,
    member_id: str,
    test_macro_f1: float,
    error_analysis_csv_path: str,
    error_analysis_md_path: str,
    notebook_arn: Optional[str] = None,
    resource_metadata_path: str = SAGEMAKER_RESOURCE_METADATA_SRC,
) -> str:
    """Abre y cierra el run FINAL (Seccion 5 -- reentrenamiento con todo
    train, evaluacion en test una sola vez, registro del modelo)."""
    import mlflow  # type: ignore

    config.validate()
    if notebook_arn is None:
        notebook_arn = read_notebook_arn(source_path=resource_metadata_path)

    tags = build_final_run_tags(
        protocol_run_id=protocol_run_id,
        selected_experiment_run_id=selected_experiment_run_id,
        configuration_id=configuration_id,
        member_id=member_id,
        notebook_arn=notebook_arn,
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(tags)
        mlflow.log_params(build_final_run_params())
        mlflow.log_metrics(build_test_macro_f1_metric(test_macro_f1))
        log_configuration_artifact(config)
        log_provenance_artifact(source_path=resource_metadata_path)
        log_error_analysis_artifacts(
            csv_path=error_analysis_csv_path, md_path=error_analysis_md_path
        )
        return run.info.run_id


if __name__ == "__main__":
    from sentiment.config import b0_config, t0_config

    b0 = b0_config(
        rep_library="scikit-learn",
        rep_version="1.5.2",
        clf_library="scikit-learn",
        clf_version="1.5.2",
    )
    b0.validate()

    # -- tags -------------------------------------------------------------------
    assert build_protocol_run_tags() == {"lab_run_type": "protocol"}
    print("OK: build_protocol_run_tags")

    exp_tags = build_experimental_run_tags(
        protocol_run_id="RUN_PROTOCOL",
        experiment_id="T0",
        member_id="E01",
        configuration_id="CFG_T0",
        notebook_arn="arn:aws:sagemaker:us-east-1:123:notebook-instance/N01",
    )
    assert exp_tags["lab_stage"] == "reference"  # inferido de EXPERIMENT_ID_STAGES["T0"]
    assert exp_tags["lab_run_type"] == "experiment"
    assert set(exp_tags) == {
        "lab_run_type", "lab_protocol_run_id", "lab_stage", "lab_experiment_id",
        "lab_member_id", "lab_configuration_id", "notebook_arn",
    }
    print("OK: build_experimental_run_tags infiere lab_stage para T0")

    assert EXPERIMENT_ID_STAGES["R_TFIDF_UNI_BI"] == "representation"
    assert EXPERIMENT_ID_STAGES["P_LEMMA"] == "preprocessing"
    assert len(EXPERIMENT_ID_STAGES) == 15  # los 15 codigos fijos de A.4 (sin EXTRA)
    print("OK: EXPERIMENT_ID_STAGES tiene los 15 codigos fijos de A.4")

    try:
        build_experimental_run_tags(
            protocol_run_id="R", experiment_id="EXTRA", member_id="E01",
            configuration_id="CFG_X", notebook_arn="arn:...",
        )
        raise AssertionError("se esperaba ValueError (EXTRA sin stage explicito)")
    except ValueError:
        print("OK: build_experimental_run_tags exige stage= explicito para EXTRA")

    extra_tags = build_experimental_run_tags(
        protocol_run_id="R", experiment_id="EXTRA", member_id="E01",
        configuration_id="CFG_X", notebook_arn="arn:...", stage="classifier",
    )
    assert extra_tags["lab_stage"] == "classifier"
    print("OK: build_experimental_run_tags acepta stage= explicito para EXTRA")

    ablation_tags = build_experimental_run_tags(
        protocol_run_id="R", experiment_id="ABLATION", member_id="E01",
        configuration_id="CFG_Y", notebook_arn="arn:...",
        ablation_parent_run_id="RUN_CANDIDATE",
    )
    assert ablation_tags["lab_ablation_parent_run_id"] == "RUN_CANDIDATE"
    assert ablation_tags["lab_stage"] == "ablation"
    print("OK: build_experimental_run_tags agrega lab_ablation_parent_run_id")

    final_tags = build_final_run_tags(
        protocol_run_id="RUN_PROTOCOL", selected_experiment_run_id="RUN_CANDIDATE",
        configuration_id="CFG_FINAL", member_id="E01", notebook_arn="arn:...",
    )
    assert set(final_tags) == {
        "lab_run_type", "lab_protocol_run_id", "lab_selected_experiment_run_id",
        "lab_configuration_id", "lab_member_id", "notebook_arn",
    }
    assert final_tags["lab_run_type"] == "final"
    print("OK: build_final_run_tags")

    # -- params -------------------------------------------------------------------
    protocol_params_dict = build_protocol_params_dict()
    assert protocol_params_dict["cv_shuffle"] == "true"  # minuscula, no "True"
    assert protocol_params_dict["sample_size"] == 200_000
    assert protocol_params_dict["dataset_revision"] == "b6037e127257d95b9b23d31f78b264b9ebe697fd"
    print("OK: build_protocol_params_dict normaliza cv_shuffle a 'true' (minuscula)")

    assert build_final_run_params() == {"training_size": 1_360_000}
    print("OK: build_final_run_params")

    assert build_test_macro_f1_metric(0.8123) == {"test_macro_f1": 0.8123}
    print("OK: build_test_macro_f1_metric")

    assert abs(compute_macro_f1_delta(0.80, 0.75) - 0.05) < 1e-9
    print("OK: compute_macro_f1_delta")

    tags_extra, params_extra, metrics_extra = build_ablation_extra(
        ablation_parent_run_id="RUN_CANDIDATE",
        reverted_decision="preprocessing.stopwords",
        macro_f1_delta=0.05,
    )
    assert params_extra == {"ablation_reverted_decision": "preprocessing.stopwords"}
    assert metrics_extra == {"macro_f1_delta": 0.05}
    print("OK: build_ablation_extra con decision valida")

    try:
        build_ablation_extra(
            ablation_parent_run_id="R", reverted_decision="not_a_real_decision",
            macro_f1_delta=0.0,
        )
        raise AssertionError("se esperaba ValueError")
    except ValueError:
        print("OK: build_ablation_extra rechaza reverted_decision invalida")

    # -- configuration.json / cv_metrics -------------------------------------------
    filename, content = build_configuration_artifact(b0)
    assert filename == "configuration.json"
    assert PipelineConfig.from_json(content).is_equivalent_to(b0)
    assert CONFIGURATION_ARTIFACT_PATH == "run/configuration.json"
    print("OK: build_configuration_artifact / CONFIGURATION_ARTIFACT_PATH")

    good_metrics = {
        "macro_f1_fold_0": 0.81, "macro_f1_fold_1": 0.79, "macro_f1_fold_2": 0.80,
        "macro_f1_mean": 0.80, "macro_f1_std": 0.0082,
    }
    assert build_cv_metrics_dict(good_metrics) == good_metrics
    try:
        build_cv_metrics_dict({**good_metrics, "accuracy": 0.9})
        raise AssertionError("se esperaba ValueError")
    except ValueError:
        pass
    print("OK: build_cv_metrics_dict")

    # -- flat params opcionales -------------------------------------------------
    flat_b0 = build_flat_config_params(b0)
    assert flat_b0["classifier_type"] == "logistic_regression"
    t0 = t0_config(library="scikit-learn", library_version="1.5.2")
    t0.validate()
    assert build_flat_config_params(t0) == {"classifier_type": "most_frequent"}
    print("OK: build_flat_config_params (EXTRA opcional, no exigido por A.2)")

    # -- read_notebook_arn (sin SageMaker real: probamos con un archivo temporal) --
    with tempfile.TemporaryDirectory() as d:
        fake_path = os.path.join(d, "resource-metadata.json")
        with open(fake_path, "w", encoding="utf-8") as f:
            _json.dump(
                {"ResourceArn": "arn:aws:sagemaker:us-east-1:123:notebook-instance/N01"}, f
            )
        assert read_notebook_arn(source_path=fake_path) == (
            "arn:aws:sagemaker:us-east-1:123:notebook-instance/N01"
        )
        print("OK: read_notebook_arn lee ResourceArn de un resource-metadata.json valido")

        bad_path = os.path.join(d, "sin_arn.json")
        with open(bad_path, "w", encoding="utf-8") as f:
            _json.dump({"algo_mas": 1}, f)
        try:
            read_notebook_arn(source_path=bad_path)
            raise AssertionError("se esperaba KeyError")
        except KeyError:
            print("OK: read_notebook_arn exige el campo ResourceArn")

        try:
            log_provenance_artifact(source_path=os.path.join(d, "no_existe.json"))
            raise AssertionError("se esperaba FileNotFoundError")
        except FileNotFoundError:
            print(
                "OK: log_provenance_artifact detecta que no estamos en una SageMaker "
                "Notebook Instance real (sin necesitar mlflow instalado)"
            )

    # -- las funciones log_* que SI llegan a MLflow requieren el paquete real ------
    try:
        import mlflow  # noqa: F401  type: ignore

        raise AssertionError("se esperaba que 'mlflow' NO estuviera instalado en este sandbox")
    except ImportError:
        print(
            "OK (esperado): 'mlflow' no esta instalado aqui -- log_protocol_run / "
            "log_experimental_run / log_final_run se prueban en SageMaker, no en este sandbox"
        )

    print("\nTodas las pruebas pasaron.")