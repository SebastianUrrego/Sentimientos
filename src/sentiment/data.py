"""
data.py
=======

Maneja el dataset, la muestra y los folds del laboratorio (Seccion 2 y A.2).

Diseno: separamos la logica "pura" (muestreo estratificado, construccion de
folds, lectura/escritura de protocol/*.csv) de la parte que depende de la
libreria ``datasets`` de Hugging Face. La parte pura no necesita internet y
esta cubierta por pruebas en este mismo archivo (bloque ``if __name__``); la
parte que sí depende de HF (``load_train_test``) es deliberadamente delgada
para que sea facil de verificar a mano en la SageMaker Notebook Instance.

IMPORTANTE -- esto se corre UNA sola vez por equipo (quien haga el run de
protocolo), no una vez por integrante: la muestra y los folds se fijan al
principio y no pueden cambiar durante el laboratorio (Seccion 2).

Uso tipico (en la SageMaker Notebook Instance, con `pip install datasets`):

    from sentiment.data import (
        load_train_test, build_protocol_sample_and_folds,
        save_partitions_csv, save_members_csv,
    )

    train_ds, test_ds = load_train_test()
    partitions_df = build_protocol_sample_and_folds(train_ds)
    save_partitions_csv(partitions_df, "protocol/partitions.csv")
    save_members_csv(
        [("E01", "arn:aws:sagemaker:...:notebook-instance/NOTEBOOK_E01"),
         ("E02", "arn:aws:sagemaker:...:notebook-instance/NOTEBOOK_E02")],
        "protocol/members.csv",
    )

    # Mas adelante, para entrenar/evaluar el fold 0:
    from sentiment.data import load_partitions_csv, get_fold_indices
    partitions_df = load_partitions_csv("protocol/partitions.csv")
    train_idx, val_idx = get_fold_indices(partitions_df, fold=0)
    fold_train_ds = train_ds.select(train_idx)
    fold_val_ds = train_ds.select(val_idx)
"""

from __future__ import annotations

import os
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# ---------------------------------------------------------------------------
# Constantes del protocolo (Seccion 2 y A.2 -- NO cambiar)
# ---------------------------------------------------------------------------

DATASET_ID = "adilbekovich/Sentiment140Twitter"
DATASET_REVISION = "b6037e127257d95b9b23d31f78b264b9ebe697fd"

SAMPLE_SIZE = 200_000
RANDOM_SEED = 42
CV_FOLDS = 3
CV_STRATEGY = "StratifiedKFold"
CV_SHUFFLE = True
SAMPLING_STRATEGY = "stratified"

LABEL_NEGATIVE = 0
LABEL_POSITIVE = 1

# Candidatos de nombres de columna a autodetectar en el dataset de HF.
# AJUSTA esto (o pasa text_column/label_column explicitos) despues de mirar
# `train_ds.column_names` una vez lo cargues de verdad en SageMaker.
_TEXT_COLUMN_CANDIDATES = ("text", "tweet", "content", "sentence")
_LABEL_COLUMN_CANDIDATES = ("label", "sentiment", "target", "polarity")


# ---------------------------------------------------------------------------
# Carga del dataset (depende de `datasets`; requiere internet)
# ---------------------------------------------------------------------------


def load_train_test(revision: str = DATASET_REVISION):
    """Carga train/test del dataset fijando la revision exacta (Seccion 2).

    Devuelve (train_dataset, test_dataset), ambos objetos `datasets.Dataset`.
    Requiere ``pip install datasets`` y acceso a internet (correr esto desde
    la SageMaker Notebook Instance, no desde este sandbox).
    """
    from datasets import load_dataset  # import perezoso: no es dependencia dura del modulo

    ds = load_dataset(DATASET_ID, revision=revision)
    train_ds = ds["train"]
    test_ds = ds["test"]

    if len(train_ds) != 1_360_000:
        raise RuntimeError(
            f"Se esperaban 1.360.000 registros en train para la revision "
            f"{revision}, se encontraron {len(train_ds)}. Verifica que la "
            f"revision fijada es correcta."
        )
    if len(test_ds) != 240_000:
        raise RuntimeError(
            f"Se esperaban 240.000 registros en test para la revision "
            f"{revision}, se encontraron {len(test_ds)}. Verifica que la "
            f"revision fijada es correcta."
        )
    return train_ds, test_ds


def resolve_columns(
    column_names: Sequence[str],
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
) -> tuple[str, str]:
    """Autodetecta (o valida) el nombre de la columna de texto y de label."""

    def _resolve(explicit: Optional[str], candidates: Sequence[str], kind: str) -> str:
        if explicit is not None:
            if explicit not in column_names:
                raise ValueError(
                    f"La columna {kind} {explicit!r} no existe. "
                    f"Columnas disponibles: {list(column_names)}"
                )
            return explicit
        for cand in candidates:
            if cand in column_names:
                return cand
        raise ValueError(
            f"No pude autodetectar la columna de {kind}. "
            f"Columnas disponibles: {list(column_names)}. "
            f"Pasa el nombre explicito con {kind}_column=..."
        )

    text_col = _resolve(text_column, _TEXT_COLUMN_CANDIDATES, "text")
    label_col = _resolve(label_column, _LABEL_COLUMN_CANDIDATES, "label")
    return text_col, label_col


# ---------------------------------------------------------------------------
# Logica pura: muestreo estratificado + folds (sin dependencia de HF)
# ---------------------------------------------------------------------------


def stratified_sample_indices(
    labels: Sequence[int],
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Selecciona ``sample_size`` posiciones (0-based) de ``labels`` de forma
    estratificada, con semilla fija. Devuelve un array ORDENADO ascendente
    de indices (el orden ascendente es solo para que quede determinista y
    facil de auditar; no afecta la estratificacion).
    """
    labels = np.asarray(labels)
    n = len(labels)
    if sample_size >= n:
        raise ValueError(f"sample_size ({sample_size}) debe ser menor que n ({n})")

    all_positions = np.arange(n)
    sample_idx, _ = train_test_split(
        all_positions,
        train_size=sample_size,
        stratify=labels,
        random_state=seed,
    )
    return np.sort(sample_idx)


def build_folds(
    sample_indices: np.ndarray,
    labels_at_sample: Sequence[int],
    n_splits: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Construye la asignacion de folds sobre la muestra ya seleccionada.

    ``sample_indices``: posiciones originales en train (0-based), tal como
        las devuelve ``stratified_sample_indices``.
    ``labels_at_sample``: las labels correspondientes a esas posiciones, EN
        EL MISMO ORDEN que ``sample_indices``.

    Devuelve un DataFrame con columnas exactas ``index,fold`` (A.2),
    ordenado ascendente por ``index``, sin duplicados.
    """
    sample_indices = np.asarray(sample_indices)
    labels_at_sample = np.asarray(labels_at_sample)
    if len(sample_indices) != len(labels_at_sample):
        raise ValueError("sample_indices y labels_at_sample deben tener el mismo largo")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=CV_SHUFFLE, random_state=seed)

    fold_of: dict[int, int] = {}
    # skf.split trabaja sobre posiciones DENTRO de la muestra (0..len-1);
    # las traducimos de vuelta a los indices originales de train.
    for fold_number, (_train_pos, val_pos) in enumerate(
        skf.split(np.zeros(len(sample_indices)), labels_at_sample)
    ):
        for pos in val_pos:
            fold_of[int(sample_indices[pos])] = fold_number

    df = pd.DataFrame(
        {"index": list(fold_of.keys()), "fold": list(fold_of.values())}
    )
    df = df.sort_values("index", kind="mergesort").reset_index(drop=True)

    # Sanity checks duros -- si algo de esto falla, el protocolo esta mal.
    if df["index"].duplicated().any():
        raise RuntimeError("Indices duplicados en partitions_df -- esto no deberia pasar")
    if set(df["fold"].unique()) - set(range(n_splits)):
        raise RuntimeError("Se generaron valores de fold fuera de rango")
    if len(df) != len(sample_indices):
        raise RuntimeError("El numero de filas de partitions_df no coincide con la muestra")

    return df


def build_protocol_sample_and_folds(
    train_dataset,
    *,
    text_column: Optional[str] = None,
    label_column: Optional[str] = None,
    sample_size: int = SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
    n_splits: int = CV_FOLDS,
) -> pd.DataFrame:
    """Orquesta todo: resuelve columnas, saca la muestra estratificada del
    ``train_dataset`` de HF, construye los folds, y devuelve el
    ``partitions_df`` listo para ``save_partitions_csv``.
    """
    _text_col, label_col = resolve_columns(
        train_dataset.column_names, text_column, label_column
    )
    all_labels = np.asarray(train_dataset[label_col])

    sample_idx = stratified_sample_indices(all_labels, sample_size=sample_size, seed=seed)
    labels_at_sample = all_labels[sample_idx]

    return build_folds(sample_idx, labels_at_sample, n_splits=n_splits, seed=seed)


# ---------------------------------------------------------------------------
# Lectura / escritura de protocol/*.csv (A.2)
# ---------------------------------------------------------------------------


def save_partitions_csv(partitions_df: pd.DataFrame, path: str) -> None:
    """Escribe protocol/partitions.csv con encabezado exacto ``index,fold``,
    UTF-8, ordenado ascendente por index, sin indices repetidos (A.2)."""
    df = partitions_df[["index", "fold"]].sort_values("index", kind="mergesort")
    if df["index"].duplicated().any():
        raise ValueError("partitions_df tiene indices repetidos, no se puede guardar")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, columns=["index", "fold"], encoding="utf-8")


def load_partitions_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    expected = {"index", "fold"}
    if set(df.columns) != expected:
        raise ValueError(f"{path} debe tener encabezado exacto index,fold; tiene {list(df.columns)}")
    return df.sort_values("index", kind="mergesort").reset_index(drop=True)


def get_fold_indices(partitions_df: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (train_idx, val_idx) -- posiciones originales en train --
    para un fold dado. val_idx = fold=k (validacion); train_idx = los otros
    dos folds (entrenamiento), tal como define la Seccion 2."""
    val_mask = partitions_df["fold"] == fold
    val_idx = partitions_df.loc[val_mask, "index"].to_numpy()
    train_idx = partitions_df.loc[~val_mask, "index"].to_numpy()
    return np.sort(train_idx), np.sort(val_idx)


def save_members_csv(members: list[tuple[str, str]], path: str) -> None:
    """Escribe protocol/members.csv con encabezado exacto
    ``member_id,notebook_arn``, una fila por integrante, orden lexicografico
    ascendente por member_id (A.2)."""
    df = pd.DataFrame(members, columns=["member_id", "notebook_arn"])
    df = df.sort_values("member_id", kind="mergesort")
    if df["member_id"].duplicated().any():
        raise ValueError("member_id repetido en members.csv")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False, columns=["member_id", "notebook_arn"], encoding="utf-8")


# ---------------------------------------------------------------------------
# Params exactos del run de protocolo (A.2) -- listos para mlflow.log_param
# ---------------------------------------------------------------------------


def protocol_params() -> dict[str, Any]:
    """Los params EXACTOS que el run de protocolo debe registrar (A.2)."""
    return {
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "sampling_strategy": SAMPLING_STRATEGY,
        "sample_size": SAMPLE_SIZE,
        "random_seed": RANDOM_SEED,
        "cv_strategy": CV_STRATEGY,
        "cv_folds": CV_FOLDS,
        "cv_shuffle": CV_SHUFFLE,
    }


if __name__ == "__main__":
    # Auto-chequeo con datos SINTETICOS (no requiere internet ni `datasets`).
    # Simula una version pequena de train (60,000 filas, ~50/50 balanceado)
    # para validar que el muestreo, los folds y los CSV funcionan bien antes
    # de correrlo contra el dataset real en SageMaker.
    rng = np.random.default_rng(0)
    n_fake_train = 60_000
    fake_labels = rng.integers(0, 2, size=n_fake_train)  # 0/1 balanceado-ish

    print(f"Simulando train con {n_fake_train} filas (label 1: {fake_labels.mean():.3f})")

    sample_size = 6_000  # analogo pequeno de 200_000
    sample_idx = stratified_sample_indices(fake_labels, sample_size=sample_size, seed=RANDOM_SEED)
    print(f"Muestra: {len(sample_idx)} filas, indices en [{sample_idx.min()}, {sample_idx.max()}]")
    assert len(sample_idx) == sample_size
    assert len(set(sample_idx.tolist())) == sample_size, "hay indices repetidos en la muestra"

    labels_at_sample = fake_labels[sample_idx]
    frac_pos_full = fake_labels.mean()
    frac_pos_sample = labels_at_sample.mean()
    print(f"Proporcion de positivos -- train: {frac_pos_full:.4f} / muestra: {frac_pos_sample:.4f}")
    assert abs(frac_pos_full - frac_pos_sample) < 0.01, "la estratificacion no preservo las proporciones"

    partitions_df = build_folds(sample_idx, labels_at_sample, n_splits=CV_FOLDS, seed=RANDOM_SEED)
    print("\npartitions_df:")
    print(partitions_df.head())
    print("...")
    print(f"Total filas: {len(partitions_df)}")
    print("Conteo por fold:", partitions_df["fold"].value_counts().sort_index().to_dict())

    assert len(partitions_df) == sample_size
    assert list(partitions_df.columns) == ["index", "fold"]
    assert partitions_df["index"].is_monotonic_increasing
    assert not partitions_df["index"].duplicated().any()
    assert set(partitions_df["fold"].unique()) == set(range(CV_FOLDS))

    # Reproducibilidad: correr de nuevo con la misma semilla debe dar lo mismo
    sample_idx_2 = stratified_sample_indices(fake_labels, sample_size=sample_size, seed=RANDOM_SEED)
    assert np.array_equal(sample_idx, sample_idx_2), "la muestra no es reproducible con la misma semilla"
    partitions_df_2 = build_folds(sample_idx_2, fake_labels[sample_idx_2], n_splits=CV_FOLDS, seed=RANDOM_SEED)
    assert partitions_df.equals(partitions_df_2), "los folds no son reproducibles con la misma semilla"
    print("\nOK: muestra y folds son reproducibles con seed=42")

    # Round-trip de los CSV
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        part_path = os.path.join(tmp, "protocol", "partitions.csv")
        save_partitions_csv(partitions_df, part_path)
        with open(part_path, encoding="utf-8") as f:
            header = f.readline().strip()
        assert header == "index,fold", f"encabezado incorrecto: {header!r}"

        loaded = load_partitions_csv(part_path)
        assert loaded.equals(partitions_df.reset_index(drop=True))
        print("OK: partitions.csv round-trip correcto, encabezado exacto 'index,fold'")

        train_idx, val_idx = get_fold_indices(loaded, fold=0)
        assert set(train_idx) | set(val_idx) == set(loaded["index"])
        assert set(train_idx) & set(val_idx) == set()
        assert len(val_idx) == (loaded["fold"] == 0).sum()
        print(f"OK: fold 0 -> train={len(train_idx)}, val={len(val_idx)} (sin solapamiento)")

        members_path = os.path.join(tmp, "protocol", "members.csv")
        save_members_csv(
            [("E02", "arn:aws:sagemaker:us-east-1:xxx:notebook-instance/N02"),
             ("E01", "arn:aws:sagemaker:us-east-1:xxx:notebook-instance/N01")],
            members_path,
        )
        members_df = pd.read_csv(members_path)
        assert list(members_df.columns) == ["member_id", "notebook_arn"]
        assert members_df["member_id"].tolist() == ["E01", "E02"], "no quedo en orden lexicografico"
        print("OK: members.csv con encabezado exacto y orden lexicografico por member_id")

    print("\nprotocol_params():", protocol_params())
    print("\nTodas las pruebas pasaron.")
