"""
classifiers.py
===============

Factory de clasificadores segun ``ClassifierConfig`` (ver ``config.py`` /
Anexo A.3): ``logistic_regression``, ``linear_svm``, ``sgd`` y
``most_frequent`` (la referencia trivial T0, Seccion 2).

Todo esto usa scikit-learn, que ya esta instalado y no requiere red --
100% testeable en este sandbox con datos sinteticos.

Reproducibilidad
-----------------
El protocolo (A.2) exige resultados reproducibles con semilla fija. Por
default, ``build_classifier`` inyecta ``random_state=RANDOM_SEED`` (el
mismo 42 usado para el muestreo/folds en ``data.py``) en cualquier
clasificador que lo soporte, PERO nunca pisa un ``random_state`` que ya
venga explicito en ``config.parameters`` -- ese siempre gana, para que
lo que quede logueado en ``configuration.json`` (Anexo A.3) sea
exactamente lo que se uso.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC

from sentiment.config import ClassifierConfig
from sentiment.data import RANDOM_SEED

_CLASSIFIER_BUILDERS: dict[str, Callable[..., Any]] = {
    "logistic_regression": LogisticRegression,
    "linear_svm": LinearSVC,
    "sgd": SGDClassifier,
    "most_frequent": DummyClassifier,
}

# Los cuatro tipos soportan random_state en scikit-learn.
_SUPPORTS_RANDOM_STATE = frozenset(_CLASSIFIER_BUILDERS)


def build_classifier(config: ClassifierConfig, *, random_state: Optional[int] = RANDOM_SEED):
    """Construye el estimador de scikit-learn correspondiente a
    ``config`` (ya validado), con la interfaz habitual
    ``fit(X, y)`` / ``predict(X)``.

    ``config.parameters`` se pasa tal cual como kwargs del constructor
    (p. ej. ``{"C": 0.5}`` para logistic_regression). Si no trae
    ``random_state`` explicito, se usa el parametro ``random_state`` de
    esta funcion (por defecto ``RANDOM_SEED``); pasar ``random_state=None``
    aqui desactiva esa inyeccion automatica."""
    config.validate()
    try:
        cls = _CLASSIFIER_BUILDERS[config.type]
    except KeyError:
        raise ValueError(f"classifier.type desconocido: {config.type!r}") from None

    kwargs = dict(config.parameters)

    if config.type == "most_frequent":
        # T0 (Seccion 2): predice siempre la clase mas frecuente. El
        # default de DummyClassifier desde sklearn 0.24 es strategy="prior",
        # que no es lo mismo (predict_proba distinto) -- lo fijamos.
        kwargs.setdefault("strategy", "most_frequent")

    if (
        random_state is not None
        and "random_state" not in kwargs
        and config.type in _SUPPORTS_RANDOM_STATE
    ):
        kwargs["random_state"] = random_state

    return cls(**kwargs)


if __name__ == "__main__":
    import numpy as np

    from sentiment.config import ClassifierConfig as CC

    # -- datos sinteticos linealmente separables en 1D -----------------------
    X_sep = np.array([[i] for i in range(10)])
    y_sep = np.array([0] * 5 + [1] * 5)

    # -- datos desbalanceados para probar most_frequent -----------------------
    X_imb = np.array([[i] for i in range(8)])
    y_imb = np.array([0] * 6 + [1] * 2)  # mayoria = 0

    for clf_type, library in [
        ("logistic_regression", "scikit-learn"),
        ("linear_svm", "scikit-learn"),
        ("sgd", "scikit-learn"),
    ]:
        cfg = CC(type=clf_type, library=library, library_version="1.5.2")
        cfg.validate()
        clf = build_classifier(cfg)
        clf.fit(X_sep, y_sep)
        preds = clf.predict(X_sep)
        assert np.array_equal(preds, y_sep), f"{clf_type} no separo datos triviales: {preds}"
        assert clf.random_state == RANDOM_SEED, f"{clf_type} no recibio random_state por defecto"
        print(f"OK: {clf_type} separa datos linealmente separables y usa random_state={RANDOM_SEED}")

    # -- most_frequent (T0) ----------------------------------------------------
    t0_cfg = CC(type="most_frequent", library="scikit-learn", library_version="1.5.2")
    t0_cfg.validate()
    t0_clf = build_classifier(t0_cfg)
    t0_clf.fit(X_imb, y_imb)
    t0_preds = t0_clf.predict(X_imb)
    assert np.all(t0_preds == 0), f"most_frequent deberia predecir siempre la clase mayoritaria: {t0_preds}"
    assert t0_clf.strategy == "most_frequent"
    print("OK: most_frequent (T0) predice siempre la clase mas frecuente")

    # -- random_state explicito en parameters gana sobre el default -----------
    cfg_fixed_seed = CC(
        type="logistic_regression",
        library="scikit-learn",
        library_version="1.5.2",
        parameters={"random_state": 7},
    )
    cfg_fixed_seed.validate()
    clf_fixed = build_classifier(cfg_fixed_seed)  # build_classifier() default random_state=RANDOM_SEED=42
    assert clf_fixed.random_state == 7, "config.parameters.random_state debe ganarle al default"
    print("OK: random_state explicito en config.parameters no se pisa con el default")

    # -- reproducibilidad: mismo config -> mismas predicciones ----------------
    cfg_repro = CC(type="sgd", library="scikit-learn", library_version="1.5.2")
    cfg_repro.validate()
    clf_a = build_classifier(cfg_repro)
    clf_a.fit(X_sep, y_sep)
    clf_b = build_classifier(cfg_repro)
    clf_b.fit(X_sep, y_sep)
    assert np.array_equal(clf_a.predict(X_sep), clf_b.predict(X_sep))
    print("OK: dos clasificadores 'sgd' con el mismo config son reproducibles")

    # -- parametros extra llegan al estimador ----------------------------------
    cfg_c = CC(
        type="logistic_regression",
        library="scikit-learn",
        library_version="1.5.2",
        parameters={"C": 0.5, "max_iter": 500},
    )
    cfg_c.validate()
    clf_c = build_classifier(cfg_c)
    assert clf_c.C == 0.5 and clf_c.max_iter == 500
    print("OK: parameters (C, max_iter) llegan al estimador de scikit-learn")

    # -- tipo desconocido -------------------------------------------------------
    try:
        bad_cfg = CC(type="not_a_real_classifier", library="scikit-learn", library_version="1.5.2")
        build_classifier(bad_cfg)
        raise AssertionError("se esperaba ValueError")
    except ValueError:
        print("OK: build_classifier rechaza classifier.type desconocido")

    print("\nTodas las pruebas pasaron.")
