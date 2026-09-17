"""
representation.py
==================

Construye la representacion vectorial del texto segun
``RepresentationConfig`` (ver ``config.py`` / Anexo A.3): ``bow``,
``tfidf`` o ``spacy_embedding``.

Todas las implementaciones comparten la misma interfaz minima (estilo
scikit-learn): ``fit(texts)``, ``transform(texts)``, ``fit_transform(texts)``,
donde ``texts`` es una lista de strings ya preprocesados (ver
``preprocessing.py``) y el resultado es una matriz de forma
``(n_samples, n_features)`` (dispersa para bow/tfidf, densa -- ``np.ndarray``
-- para spacy_embedding).

Separacion pura vs. recursos externos
--------------------------------------
- ``bow`` / ``tfidf`` se implementan con scikit-learn
  (``CountVectorizer`` / ``TfidfVectorizer``), que ya esta instalado y no
  requiere red -- 100% testeable en este sandbox.
- ``spacy_embedding`` SI requiere un modelo de spaCy descargado **con
  vectores reales** (ojo: ``en_core_web_sm`` no trae vectores de
  palabras utiles -- hay que usar un modelo ``_md`` o ``_lg``, confirmar
  contra la guia cual exige). La carga del modelo esta aislada en
  ``load_spacy_nlp_for_embeddings`` (requiere internet, correr en
  SageMaker); la logica de pooling (``mean_pool``) es pura y se prueba
  aqui con vectores y un ``nlp`` de juguete inyectados.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sentiment.config import RepresentationConfig

# ---------------------------------------------------------------------------
# Interfaz comun
# ---------------------------------------------------------------------------


class Representation(Protocol):
    """Interfaz minima que exponen todas las representaciones de este
    modulo (documentacion, no se hace cumplir en tiempo de ejecucion)."""

    def fit(self, texts: list[str]) -> "Representation": ...

    def transform(self, texts: list[str]) -> Any: ...

    def fit_transform(self, texts: list[str]) -> Any: ...


# ---------------------------------------------------------------------------
# bow / tfidf (scikit-learn) -- puro, sin recursos externos
# ---------------------------------------------------------------------------


def build_bow_tfidf_vectorizer(config: RepresentationConfig):
    """Construye un ``CountVectorizer`` (bow) o ``TfidfVectorizer``
    (tfidf) de scikit-learn a partir del config, con ``ngram_range`` y
    cualquier hiperparametro adicional en ``config.parameters`` (p. ej.
    ``max_features``, ``min_df``) pasado tal cual como kwargs."""
    if config.type not in ("bow", "tfidf"):
        raise ValueError(
            f"build_bow_tfidf_vectorizer solo acepta type 'bow'/'tfidf', recibio {config.type!r}"
        )
    cls = CountVectorizer if config.type == "bow" else TfidfVectorizer
    kwargs = dict(config.parameters)
    kwargs["ngram_range"] = tuple(config.ngram_range)
    return cls(**kwargs)


class SklearnRepresentation:
    """Envuelve un vectorizador de scikit-learn ya construido (bow o
    tfidf) con la interfaz comun fit/transform/fit_transform."""

    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def fit(self, texts: list[str]) -> "SklearnRepresentation":
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts: list[str]):
        return self.vectorizer.transform(texts)

    def fit_transform(self, texts: list[str]):
        return self.vectorizer.fit_transform(texts)

    @property
    def vocabulary_size(self) -> int:
        return len(self.vectorizer.vocabulary_)


# ---------------------------------------------------------------------------
# spacy_embedding -- pooling puro + loader de recursos externos
# ---------------------------------------------------------------------------


def mean_pool(token_vectors: np.ndarray, *, dim: Optional[int] = None) -> np.ndarray:
    """Promedia vectores de tokens (forma ``(n_tokens, dim)``) en un solo
    vector de documento (forma ``(dim,)``). Si ``token_vectors`` esta
    vacio (documento sin tokens con vector, p. ej. tras remover
    stopwords), devuelve un vector de ceros -- se necesita ``dim`` en ese
    caso porque un array vacio de forma ``(0,)`` no trae esa informacion."""
    if token_vectors.size == 0:
        if dim is None:
            raise ValueError("Se requiere 'dim' para el vector cero cuando no hay tokens")
        return np.zeros(dim)
    return token_vectors.mean(axis=0)


_DOCUMENT_VECTOR_METHODS: dict[str, Callable[..., np.ndarray]] = {
    "mean_pooling": mean_pool,
}


def get_document_vector_method(name: str) -> Callable[..., np.ndarray]:
    """Resuelve el nombre de ``parameters.document_vector_method`` (A.3)
    a la funcion de pooling correspondiente."""
    try:
        return _DOCUMENT_VECTOR_METHODS[name]
    except KeyError:
        raise ValueError(
            f"document_vector_method desconocido: {name!r}. "
            f"Metodos disponibles: {sorted(_DOCUMENT_VECTOR_METHODS)}"
        ) from None


class SpacyEmbeddingRepresentation:
    """Representa cada texto como el pooling de los vectores de palabra
    de spaCy (``nlp``). No hay nada que "ajustar" al corpus -- ``fit`` es
    un no-op -- por eso ``fit_transform`` == ``transform``."""

    def __init__(self, nlp, document_vector_method: str):
        self.nlp = nlp
        self.document_vector_method = document_vector_method
        self._pool_fn = get_document_vector_method(document_vector_method)
        self._dim = getattr(getattr(nlp, "vocab", None), "vectors_length", None) or 300

    def fit(self, texts: list[str]) -> "SpacyEmbeddingRepresentation":
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        rows = []
        for doc in self.nlp.pipe(texts):
            token_vectors = np.array(
                [tok.vector for tok in doc if not getattr(tok, "is_space", False)],
                dtype=float,
            )
            rows.append(self._pool_fn(token_vectors, dim=self._dim))
        return np.vstack(rows) if rows else np.zeros((0, self._dim))

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        return self.fit(texts).transform(texts)


def load_spacy_nlp_for_embeddings(model: str = "en_core_web_md"):
    """Carga un modelo de spaCy con vectores reales para usar con
    ``spacy_embedding``. Requiere ``pip install spacy`` y
    ``python -m spacy download <model>`` -- correr en una instancia con
    internet (SageMaker), no en este sandbox.

    OJO: ``en_core_web_sm`` NO trae vectores de palabras reales
    (``vectors_length == 0``) -- para ``spacy_embedding`` se necesita un
    modelo ``_md`` o ``_lg`` (confirmar el exacto contra la guia del
    laboratorio)."""
    import spacy  # type: ignore

    try:
        nlp = spacy.load(model, disable=["parser", "ner", "lemmatizer"])
    except OSError as exc:
        raise OSError(
            f"El modelo de spaCy {model!r} no esta instalado. Instalalo con "
            f"'python -m spacy download {model}' en una instancia con internet."
        ) from exc

    if getattr(nlp.vocab, "vectors_length", 0) == 0:
        raise ValueError(
            f"El modelo {model!r} no tiene vectores de palabras reales "
            "(vectors_length=0) -- no sirve para representation.type="
            "'spacy_embedding'. Usa un modelo con vectores (en_core_web_md, "
            "en_core_web_lg, etc.)."
        )
    return nlp


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_representation(config: RepresentationConfig, *, nlp=None) -> Representation:
    """Construye la representacion correspondiente a ``config`` (ya
    validado). Para ``spacy_embedding`` hay que pasar ``nlp`` (un modelo
    de spaCy con vectores, real o de prueba) -- no se carga aqui adentro
    para mantener esta funcion testeable sin internet."""
    config.validate()
    if config.type in ("bow", "tfidf"):
        return SklearnRepresentation(build_bow_tfidf_vectorizer(config))
    if config.type == "spacy_embedding":
        if nlp is None:
            raise ValueError(
                "Se requiere 'nlp' (modelo de spaCy con vectores) para "
                "representation.type='spacy_embedding'"
            )
        method = config.parameters.get("document_vector_method")
        return SpacyEmbeddingRepresentation(nlp, method)
    raise ValueError(f"representation.type desconocido: {config.type!r}")


if __name__ == "__main__":
    from sentiment.config import RepresentationConfig as RC

    corpus = [
        "el gato duerme en el sofa",
        "el perro corre en el parque",
        "el gato y el perro juegan",
    ]

    # -- bow uni vs uni+bi ---------------------------------------------------
    bow_uni_cfg = RC(
        type="bow", ngram_range=[1, 1], library="scikit-learn", library_version="1.5.2"
    )
    bow_uni_cfg.validate()
    bow_uni = SklearnRepresentation(build_bow_tfidf_vectorizer(bow_uni_cfg))
    X_uni = bow_uni.fit_transform(corpus)
    assert X_uni.shape[0] == 3
    assert all(" " not in t for t in bow_uni.vectorizer.get_feature_names_out())
    print(f"OK: bow uni -> {X_uni.shape}, vocab={bow_uni.vocabulary_size}")

    bow_bi_cfg = RC(
        type="bow", ngram_range=[1, 2], library="scikit-learn", library_version="1.5.2"
    )
    bow_bi_cfg.validate()
    bow_bi = SklearnRepresentation(build_bow_tfidf_vectorizer(bow_bi_cfg))
    X_bi = bow_bi.fit_transform(corpus)
    assert bow_bi.vocabulary_size > bow_uni.vocabulary_size
    assert any(" " in t for t in bow_bi.vectorizer.get_feature_names_out())
    print(f"OK: bow uni+bi -> {X_bi.shape}, vocab={bow_bi.vocabulary_size} (> uni)")

    # -- tfidf ----------------------------------------------------------------
    tfidf_cfg = RC(
        type="tfidf", ngram_range=[1, 1], library="scikit-learn", library_version="1.5.2"
    )
    tfidf_cfg.validate()
    tfidf_rep = build_representation(tfidf_cfg)
    assert isinstance(tfidf_rep, SklearnRepresentation)
    X_tfidf = tfidf_rep.fit_transform(corpus)
    row_norms = np.asarray(X_tfidf.multiply(X_tfidf).sum(axis=1)).ravel()
    assert np.allclose(row_norms, 1.0, atol=1e-6)  # tfidf normaliza L2 por defecto
    print(f"OK: tfidf -> {X_tfidf.shape}, filas normalizadas L2")

    # fit separado de transform (como se usaria en CV: fit en train, transform en val)
    tfidf_rep2 = build_representation(tfidf_cfg)
    tfidf_rep2.fit(corpus[:2])
    X_val = tfidf_rep2.transform(corpus[2:])
    assert X_val.shape[0] == 1
    print("OK: fit(train) + transform(val) por separado")

    # -- parametros extra pasan directo al vectorizador -----------------------
    bow_minmax_cfg = RC(
        type="bow",
        ngram_range=[1, 1],
        library="scikit-learn",
        library_version="1.5.2",
        parameters={"max_features": 5},
    )
    bow_minmax_cfg.validate()
    bow_minmax = SklearnRepresentation(build_bow_tfidf_vectorizer(bow_minmax_cfg))
    X_minmax = bow_minmax.fit_transform(corpus)
    assert X_minmax.shape[1] == 5
    print("OK: parameters (max_features=5) llega al vectorizador de scikit-learn")

    # -- mean_pool puro ---------------------------------------------------------
    toy_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert np.allclose(mean_pool(toy_vectors), [1 / 3, 1 / 3, 1 / 3])
    assert np.allclose(mean_pool(np.empty((0, 3)), dim=3), [0.0, 0.0, 0.0])
    print("OK: mean_pool (promedio y caso sin tokens)")

    try:
        get_document_vector_method("no_existe")
        raise AssertionError("se esperaba ValueError")
    except ValueError:
        print("OK: get_document_vector_method rechaza metodos desconocidos")

    # -- spacy_embedding con un 'nlp' de juguete (sin spaCy real) -------------
    class _FakeToken:
        def __init__(self, vector, is_space=False):
            self.vector = np.array(vector, dtype=float)
            self.is_space = is_space

    class _FakeDoc:
        def __init__(self, tokens):
            self._tokens = tokens

        def __iter__(self):
            return iter(self._tokens)

    class _FakeVocab:
        vectors_length = 3

    class _FakeNlp:
        def __init__(self):
            self.vocab = _FakeVocab()

        def pipe(self, texts):
            for t in texts:
                if t == "":
                    yield _FakeDoc([])
                else:
                    yield _FakeDoc([_FakeToken([len(w), 1.0, 0.0]) for w in t.split()])

    fake_nlp = _FakeNlp()
    spacy_cfg = RC(
        type="spacy_embedding",
        ngram_range=[],
        library="spacy",
        library_version="3.7.0",
        spacy_model="en_core_web_md",
        spacy_model_version="3.7.0",
        parameters={"document_vector_method": "mean_pooling"},
    )
    spacy_cfg.validate()
    spacy_rep = build_representation(spacy_cfg, nlp=fake_nlp)
    assert isinstance(spacy_rep, SpacyEmbeddingRepresentation)
    X_spacy = spacy_rep.fit_transform(["hi there", ""])
    assert X_spacy.shape == (2, 3)
    assert np.allclose(X_spacy[0], [3.5, 1.0, 0.0])  # mean("hi"=2, "there"=5) en la 1ra dim
    assert np.allclose(X_spacy[1], [0.0, 0.0, 0.0])  # texto vacio -> vector cero
    print("OK: spacy_embedding con nlp de juguete (fit_transform, pooling, texto vacio)")

    try:
        build_representation(spacy_cfg)
        raise AssertionError("se esperaba ValueError sin nlp")
    except ValueError:
        print("OK: build_representation exige 'nlp' para spacy_embedding")

    print("\nTodas las pruebas pasaron.")
