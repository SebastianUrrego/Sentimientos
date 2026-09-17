"""
cv.py
=====

Corre validacion cruzada de ``n_splits`` folds (protocolo fijo en
``data.py``: ``StratifiedKFold`` con 3 folds) para UNA configuracion
completa (``PipelineConfig`` = preprocessing + representation +
classifier), usando las particiones ya calculadas (``partitions_df``,
ver ``data.py``: columnas exactas ``index,fold``).

Metricas EXACTAS que produce (Seccion 2 / A.2):

    macro_f1_fold_0, macro_f1_fold_1, macro_f1_fold_2,
    macro_f1_mean, macro_f1_std

``macro_f1_std`` se calcula con ``ddof=0`` (desviacion poblacional --
que es ademas el default de ``numpy.std``).

Diseno: ``run_fold`` corre UN fold completo (preprocesar -> representar
-> entrenar -> predecir -> macro-F1); ``run_cv`` itera ``run_fold``
sobre los folds de ``partitions_df`` y agrega las metricas. Ambas
funciones reciben los textos/labels ya en memoria y los recursos
externos (stopword_set, lemmatize_fn, emoji_converter, nlp) ya cargados
-- no dependen de internet, por eso se prueban aqui con datos
sinteticos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import numpy as np
from sklearn.metrics import f1_score

from sentiment.classifiers import build_classifier
from sentiment.config import PipelineConfig
from sentiment.data import CV_FOLDS, get_fold_indices
from sentiment.preprocessing import preprocess_text
from sentiment.representation import build_representation


@dataclass
class FoldResult:
    fold: int
    macro_f1: float
    n_train: int
    n_val: int


def _apply_preprocessing(
    texts: Sequence[str],
    config: PipelineConfig,
    *,
    stopword_set: Optional[set[str]],
    lemmatize_fn: Optional[Callable[[list[str]], list[str]]],
    emoji_converter: Optional[Callable[[str], str]],
) -> list[str]:
    if config.preprocessing is None:
        return list(texts)
    return [
        preprocess_text(
            t,
            config.preprocessing,
            stopword_set=stopword_set,
            lemmatize_fn=lemmatize_fn,
            emoji_converter=emoji_converter,
        )
        for t in texts
    ]


def run_fold(
    config: PipelineConfig,
    train_texts: Sequence[str],
    train_labels: Sequence[int],
    val_texts: Sequence[str],
    val_labels: Sequence[int],
    *,
    fold: int,
    stopword_set: Optional[set[str]] = None,
    lemmatize_fn: Optional[Callable[[list[str]], list[str]]] = None,
    emoji_converter: Optional[Callable[[str], str]] = None,
    nlp: Any = None,
) -> FoldResult:
    """Corre un fold completo: preprocesar -> representar -> entrenar ->
    predecir -> macro-F1. ``config`` debe ser una ``PipelineConfig`` ya
    valida (T0: ``representation is None``; en cualquier otro caso,
    ``representation`` y ``preprocessing`` no son None -- lo exige
    ``config.validate()``, ver config.py)."""
    is_t0 = config.representation is None

    if is_t0:
        X_train = np.zeros((len(train_texts), 1))
        X_val = np.zeros((len(val_texts), 1))
    else:
        train_proc = _apply_preprocessing(
            train_texts,
            config,
            stopword_set=stopword_set,
            lemmatize_fn=lemmatize_fn,
            emoji_converter=emoji_converter,
        )
        val_proc = _apply_preprocessing(
            val_texts,
            config,
            stopword_set=stopword_set,
            lemmatize_fn=lemmatize_fn,
            emoji_converter=emoji_converter,
        )
        rep = build_representation(config.representation, nlp=nlp)
        X_train = rep.fit_transform(train_proc)
        X_val = rep.transform(val_proc)

    clf = build_classifier(config.classifier)
    clf.fit(X_train, train_labels)
    preds = clf.predict(X_val)

    macro_f1 = f1_score(val_labels, preds, average="macro")
    return FoldResult(
        fold=fold, macro_f1=float(macro_f1), n_train=len(train_texts), n_val=len(val_texts)
    )


def run_cv(
    config: PipelineConfig,
    texts: Sequence[str],
    labels: Sequence[int],
    partitions_df,
    *,
    n_splits: int = CV_FOLDS,
    stopword_set: Optional[set[str]] = None,
    lemmatize_fn: Optional[Callable[[list[str]], list[str]]] = None,
    emoji_converter: Optional[Callable[[str], str]] = None,
    nlp: Any = None,
) -> dict[str, float]:
    """Corre los ``n_splits`` folds definidos en ``partitions_df`` para
    ``config`` y devuelve el dict de metricas EXACTO que exige el
    protocolo: ``macro_f1_fold_0..k-1``, ``macro_f1_mean``,
    ``macro_f1_std``.

    ``texts`` / ``labels`` deben ser indexables por las posiciones
    originales que trae ``partitions_df["index"]`` (una lista o
    ``np.ndarray`` alineada con el train dataset completo, o el propio
    ``datasets.Dataset`` de HF, que soporta indexado por entero)."""
    config.validate()

    fold_f1s: list[float] = []
    metrics: dict[str, float] = {}
    for fold in range(n_splits):
        train_idx, val_idx = get_fold_indices(partitions_df, fold)
        train_texts = [texts[int(i)] for i in train_idx]
        train_labels = [labels[int(i)] for i in train_idx]
        val_texts = [texts[int(i)] for i in val_idx]
        val_labels = [labels[int(i)] for i in val_idx]

        result = run_fold(
            config,
            train_texts,
            train_labels,
            val_texts,
            val_labels,
            fold=fold,
            stopword_set=stopword_set,
            lemmatize_fn=lemmatize_fn,
            emoji_converter=emoji_converter,
            nlp=nlp,
        )
        metrics[f"macro_f1_fold_{fold}"] = result.macro_f1
        fold_f1s.append(result.macro_f1)

    metrics["macro_f1_mean"] = float(np.mean(fold_f1s))
    metrics["macro_f1_std"] = float(np.std(fold_f1s, ddof=0))
    return metrics


if __name__ == "__main__":
    from sentiment.config import b0_config, t0_config
    from sentiment.data import RANDOM_SEED, build_folds

    rng = np.random.default_rng(0)
    n = 300
    pos_words = ["amazing", "great", "love", "good", "awesome"]
    neg_words = ["terrible", "awful", "bad", "hate", "horrible"]
    filler = ["the", "movie", "today", "really", "so", "this", "was"]

    texts: list[str] = []
    labels: list[int] = []
    for _ in range(n):
        is_pos = rng.random() < 0.6  # levemente desbalanceado, como Sentiment140 real
        core = rng.choice(pos_words if is_pos else neg_words, size=2).tolist()
        noise = rng.choice(filler, size=3).tolist()
        words = core + noise
        rng.shuffle(words)
        texts.append(" ".join(words))
        labels.append(1 if is_pos else 0)

    # folds sobre TODA la muestra sintetica (sample_indices = 0..n-1)
    partitions_df = build_folds(np.arange(n), np.array(labels), n_splits=CV_FOLDS, seed=RANDOM_SEED)

    b0 = b0_config(
        rep_library="scikit-learn",
        rep_version="1.5.2",
        clf_library="scikit-learn",
        clf_version="1.5.2",
    )
    b0.validate()
    metrics_b0 = run_cv(b0, texts, labels, partitions_df)
    print("B0 metrics:", metrics_b0)
    assert set(metrics_b0) == {
        "macro_f1_fold_0",
        "macro_f1_fold_1",
        "macro_f1_fold_2",
        "macro_f1_mean",
        "macro_f1_std",
    }
    assert all(0.0 <= v <= 1.0 for v in metrics_b0.values())
    print("OK: run_cv(B0) produce las 5 claves de metrica exactas, todas en [0,1]")

    t0 = t0_config(library="scikit-learn", library_version="1.5.2")
    t0.validate()
    metrics_t0 = run_cv(t0, texts, labels, partitions_df)
    print("T0 metrics:", metrics_t0)
    assert metrics_b0["macro_f1_mean"] > metrics_t0["macro_f1_mean"], (
        "B0 (BoW + LogReg) deberia superar claramente a T0 (most_frequent) "
        "en este ejemplo sintetico, que es facilmente separable"
    )
    print(f"OK: B0 ({metrics_b0['macro_f1_mean']:.3f}) supera a T0 ({metrics_t0['macro_f1_mean']:.3f})")

    metrics_b0_again = run_cv(b0, texts, labels, partitions_df)
    assert metrics_b0 == metrics_b0_again
    print("OK: run_cv es reproducible (mismo config + mismos datos -> mismas metricas)")

    fold_values = [metrics_b0[f"macro_f1_fold_{k}"] for k in range(CV_FOLDS)]
    assert abs(metrics_b0["macro_f1_std"] - float(np.std(fold_values, ddof=0))) < 1e-12
    print("OK: macro_f1_std usa ddof=0 (poblacional)")

    print("\nTodas las pruebas pasaron.")
