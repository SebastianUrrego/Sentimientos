"""
config.py
=========

Representa el esquema EXACTO de ``run/configuration.json`` definido en el
Anexo A.3 de la guia del laboratorio (Analisis de sentimientos - PLN).

Este modulo es la "fuente de verdad" del pipeline: en vez de que cada
integrante arme a mano el diccionario que se sube a MLflow, todos usan
estas mismas clases. Eso evita el error mas caro del laboratorio: que dos
runs con la misma configuracion efectiva terminen con JSON ligeramente
distintos (un campo de mas, un valor con otra capitalizacion, etc.) y el
evaluador automatico los trate como configuraciones diferentes.

Uso tipico:

    from sentiment.config import b0_config, t0_config, PipelineConfig

    b0 = b0_config(rep_library="scikit-learn", rep_version="1.5.2",
                    clf_library="scikit-learn", clf_version="1.5.2")
    b0.validate()
    print(b0.to_json())

    # P_STOPWORDS: "Igual a B0, salvo stopwords=remove" (A.4)
    p_stopwords = b0.copy()
    p_stopwords.preprocessing.stopwords = "remove"
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# ---------------------------------------------------------------------------
# Vocabulario controlado (A.3 "Valores controlados")
# ---------------------------------------------------------------------------

UrlMentionValue = str  # "keep" | "drop" | "token:<valor>" -- validado en validate()
WhitespaceValue = Literal["keep", "normalize"]
StopwordsValue = Literal["keep", "remove", "remove_preserve_negation"]
ElongationValue = Literal["keep", "normalize"]
EmojiValue = Literal["keep", "text"]
RepresentationType = Literal["bow", "tfidf", "spacy_embedding"]
ClassifierType = Literal["logistic_regression", "linear_svm", "sgd", "most_frequent"]

# Las seis decisiones ablacionables definidas en A.4 (valores EXACTOS)
AblatableDecision = Literal[
    "preprocessing.stopwords",
    "preprocessing.lemmatize",
    "preprocessing.elongation",
    "preprocessing.emoji",
    "representation",
    "classifier",
]

_VALID_URL_MENTION_PREFIXES = ("keep", "drop")


def _is_valid_url_mention(value: str) -> bool:
    return value in _VALID_URL_MENTION_PREFIXES or value.startswith("token:")


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    lowercase: bool
    url: UrlMentionValue
    mention: UrlMentionValue
    whitespace: WhitespaceValue
    stopwords: StopwordsValue
    negators: list[str] = field(default_factory=list)
    lemmatize: bool = False
    elongation: ElongationValue = "keep"
    elongation_spec: Optional[str] = None
    emoji: EmojiValue = "keep"
    emoji_spec: Optional[str] = None
    resources: dict[str, Any] = field(default_factory=dict)
    additional: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lowercase": self.lowercase,
            "url": self.url,
            "mention": self.mention,
            "whitespace": self.whitespace,
            "stopwords": self.stopwords,
            "negators": list(self.negators),
            "lemmatize": self.lemmatize,
            "elongation": self.elongation,
            "elongation_spec": self.elongation_spec,
            "emoji": self.emoji,
            "emoji_spec": self.emoji_spec,
            "resources": dict(self.resources),
            "additional": dict(self.additional),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PreprocessingConfig":
        return PreprocessingConfig(
            lowercase=d["lowercase"],
            url=d["url"],
            mention=d["mention"],
            whitespace=d["whitespace"],
            stopwords=d["stopwords"],
            negators=list(d.get("negators", [])),
            lemmatize=d.get("lemmatize", False),
            elongation=d.get("elongation", "keep"),
            elongation_spec=d.get("elongation_spec"),
            emoji=d.get("emoji", "keep"),
            emoji_spec=d.get("emoji_spec"),
            resources=dict(d.get("resources", {})),
            additional=dict(d.get("additional", {})),
        )

    def validate(self) -> None:
        errors: list[str] = []
        if not _is_valid_url_mention(self.url):
            errors.append(f"preprocessing.url invalido: {self.url!r}")
        if not _is_valid_url_mention(self.mention):
            errors.append(f"preprocessing.mention invalido: {self.mention!r}")
        if self.whitespace not in ("keep", "normalize"):
            errors.append(f"preprocessing.whitespace invalido: {self.whitespace!r}")
        if self.stopwords not in ("keep", "remove", "remove_preserve_negation"):
            errors.append(f"preprocessing.stopwords invalido: {self.stopwords!r}")
        if self.stopwords == "remove_preserve_negation" and not self.negators:
            errors.append(
                "negators debe ser una lista NO vacia cuando "
                "stopwords=remove_preserve_negation (A.3)"
            )
        if self.stopwords != "remove_preserve_negation" and self.negators:
            errors.append(
                "negators debe ser [] salvo con stopwords=remove_preserve_negation (A.3)"
            )
        if self.elongation not in ("keep", "normalize"):
            errors.append(f"preprocessing.elongation invalido: {self.elongation!r}")
        if self.elongation == "normalize" and self.elongation_spec is None:
            errors.append("elongation_spec debe ser no nulo cuando elongation=normalize (A.3)")
        if self.elongation != "normalize" and self.elongation_spec is not None:
            errors.append("elongation_spec debe ser null salvo con elongation=normalize (A.3)")
        if self.emoji not in ("keep", "text"):
            errors.append(f"preprocessing.emoji invalido: {self.emoji!r}")
        if self.emoji == "text" and self.emoji_spec is None:
            errors.append("emoji_spec debe ser no nulo cuando emoji=text (A.3)")
        if self.emoji != "text" and self.emoji_spec is not None:
            errors.append("emoji_spec debe ser null salvo con emoji=text (A.3)")
        if errors:
            raise ValueError("PreprocessingConfig invalido:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# representation
# ---------------------------------------------------------------------------


@dataclass
class RepresentationConfig:
    type: RepresentationType
    ngram_range: list[int]
    library: str
    library_version: str
    spacy_model: Optional[str] = None
    spacy_model_version: Optional[str] = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "ngram_range": list(self.ngram_range),
            "library": self.library,
            "library_version": self.library_version,
            "spacy_model": self.spacy_model,
            "spacy_model_version": self.spacy_model_version,
            "parameters": dict(self.parameters),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "RepresentationConfig":
        return RepresentationConfig(
            type=d["type"],
            ngram_range=list(d["ngram_range"]),
            library=d["library"],
            library_version=d["library_version"],
            spacy_model=d.get("spacy_model"),
            spacy_model_version=d.get("spacy_model_version"),
            parameters=dict(d.get("parameters", {})),
        )

    def validate(self) -> None:
        errors: list[str] = []
        if self.type not in ("bow", "tfidf", "spacy_embedding"):
            errors.append(f"representation.type invalido: {self.type!r}")
        if self.type in ("bow", "tfidf"):
            if len(self.ngram_range) != 2:
                errors.append("ngram_range debe tener 2 enteros para bow/tfidf (A.3)")
            if self.spacy_model is not None or self.spacy_model_version is not None:
                errors.append("spacy_model/spacy_model_version deben ser null para bow/tfidf")
        if self.type == "spacy_embedding":
            if self.ngram_range != []:
                errors.append("ngram_range debe ser [] para spacy_embedding (A.3)")
            method = self.parameters.get("document_vector_method")
            if not isinstance(method, str) or not method.strip():
                errors.append(
                    "parameters.document_vector_method debe ser un string no vacio "
                    "cuando type=spacy_embedding (A.3)"
                )
        if errors:
            raise ValueError("RepresentationConfig invalido:\n  - " + "\n  - ".join(errors))


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------


@dataclass
class ClassifierConfig:
    type: ClassifierType
    library: str
    library_version: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "library": self.library,
            "library_version": self.library_version,
            "parameters": dict(self.parameters),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "ClassifierConfig":
        return ClassifierConfig(
            type=d["type"],
            library=d["library"],
            library_version=d["library_version"],
            parameters=dict(d.get("parameters", {})),
        )

    def validate(self) -> None:
        if self.type not in ("logistic_regression", "linear_svm", "sgd", "most_frequent"):
            raise ValueError(f"classifier.type invalido: {self.type!r}")


# ---------------------------------------------------------------------------
# PipelineConfig -- el documento completo
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    preprocessing: Optional[PreprocessingConfig]
    representation: Optional[RepresentationConfig]
    classifier: ClassifierConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocessing": self.preprocessing.to_dict() if self.preprocessing else None,
            "representation": self.representation.to_dict() if self.representation else None,
            "classifier": self.classifier.to_dict(),
        }

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "PipelineConfig":
        return PipelineConfig(
            preprocessing=(
                PreprocessingConfig.from_dict(d["preprocessing"])
                if d.get("preprocessing") is not None
                else None
            ),
            representation=(
                RepresentationConfig.from_dict(d["representation"])
                if d.get("representation") is not None
                else None
            ),
            classifier=ClassifierConfig.from_dict(d["classifier"]),
        )

    @staticmethod
    def from_json(s: str) -> "PipelineConfig":
        return PipelineConfig.from_dict(json.loads(s))

    def copy(self) -> "PipelineConfig":
        return copy.deepcopy(self)

    def validate(self) -> None:
        errors: list[str] = []
        is_t0 = self.classifier.type == "most_frequent"
        if is_t0:
            if self.preprocessing is not None or self.representation is not None:
                errors.append(
                    "Para T0 (classifier.type=most_frequent), preprocessing y "
                    "representation deben ser null (A.3)"
                )
        else:
            if self.preprocessing is None or self.representation is None:
                errors.append(
                    "preprocessing y representation solo pueden ser null para T0"
                )
        if self.preprocessing is not None:
            self.preprocessing.validate()
        if self.representation is not None:
            self.representation.validate()
        self.classifier.validate()
        if errors:
            raise ValueError("PipelineConfig invalido:\n  - " + "\n  - ".join(errors))

    # -- equivalencia -------------------------------------------------------

    def canonical_json(self, *, ignore_resources: bool = True) -> str:
        """JSON con claves ordenadas (para hashear/comparar). El orden de
        listas SI importa para la equivalencia (A.3), por eso no se tocan.

        Por defecto ignora ``preprocessing.resources`` (lo pisa a ``{}``
        antes de serializar), porque el Anexo A.3 dice explicitamente:
        "la igualdad se evaluara sobre todos los campos excepto
        preprocessing.resources, que reflejara los recursos realmente
        utilizados y podra variar sin convertirse en variable
        experimental". Pasa ``ignore_resources=False`` si en cambio
        necesitas comparar el JSON byte-a-byte incluyendo ese campo."""
        d = self.to_dict()
        if ignore_resources and d.get("preprocessing") is not None:
            d = copy.deepcopy(d)
            d["preprocessing"]["resources"] = {}
        return json.dumps(d, sort_keys=True, ensure_ascii=False)

    def is_equivalent_to(self, other: "PipelineConfig", *, ignore_resources: bool = True) -> bool:
        """True si ambos configs son la 'misma configuracion efectiva' segun
        A.3: mismo contenido JSON salvo el orden de claves de objetos, e
        ignorando ``preprocessing.resources`` por defecto (ver
        ``canonical_json``) -- dos runs con distinto ``resources`` pero
        por lo demas identicos deben compartir el mismo
        ``lab_configuration_id`` segun la guia."""
        return self.canonical_json(ignore_resources=ignore_resources) == other.canonical_json(
            ignore_resources=ignore_resources
        )


# ---------------------------------------------------------------------------
# Builders canonicos (A.3: "Forma canonica de B0" / "Forma canonica de T0")
# ---------------------------------------------------------------------------


def t0_config(*, library: str, library_version: str) -> PipelineConfig:
    """Referencia trivial T0: predice la clase mas frecuente (Seccion 2)."""
    return PipelineConfig(
        preprocessing=None,
        representation=None,
        classifier=ClassifierConfig(
            type="most_frequent",
            library=library,
            library_version=library_version,
            parameters={},
        ),
    )


def b0_config(
    *,
    rep_library: str,
    rep_version: str,
    clf_library: str,
    clf_version: str,
) -> PipelineConfig:
    """Baseline B0 (Seccion 2 y A.3): minusculas, URL/mencion a tokens,
    normaliza espacios, BoW unigramas, Logistic Regression, hiperparametros
    por defecto."""
    return PipelineConfig(
        preprocessing=PreprocessingConfig(
            lowercase=True,
            url="token:url",
            mention="token:user",
            whitespace="normalize",
            stopwords="keep",
            negators=[],
            lemmatize=False,
            elongation="keep",
            elongation_spec=None,
            emoji="keep",
            emoji_spec=None,
            resources={},
            additional={},
        ),
        representation=RepresentationConfig(
            type="bow",
            ngram_range=[1, 1],
            library=rep_library,
            library_version=rep_version,
            spacy_model=None,
            spacy_model_version=None,
            parameters={},
        ),
        classifier=ClassifierConfig(
            type="logistic_regression",
            library=clf_library,
            library_version=clf_version,
            parameters={},
        ),
    )


# ---------------------------------------------------------------------------
# Ablacion (A.4: "Revertir una decision significa devolver a B0 ...")
# ---------------------------------------------------------------------------


def revert_decision(
    candidate: PipelineConfig,
    decision: AblatableDecision,
    b0: PipelineConfig,
) -> PipelineConfig:
    """Devuelve una COPIA del candidato con una sola decision revertida a B0,
    dejando las demas sin cambio -- tal como exige la Seccion 4 / A.4.

    ``decision`` debe ser uno de los seis valores exactos de A.4.
    """
    result = candidate.copy()

    if decision == "preprocessing.stopwords":
        assert result.preprocessing is not None
        result.preprocessing.stopwords = "keep"
        result.preprocessing.negators = []
    elif decision == "preprocessing.lemmatize":
        assert result.preprocessing is not None
        result.preprocessing.lemmatize = False
    elif decision == "preprocessing.elongation":
        assert result.preprocessing is not None
        result.preprocessing.elongation = "keep"
        result.preprocessing.elongation_spec = None
    elif decision == "preprocessing.emoji":
        assert result.preprocessing is not None
        result.preprocessing.emoji = "keep"
        result.preprocessing.emoji_spec = None
    elif decision == "representation":
        result.representation = RepresentationConfig(
            type="bow",
            ngram_range=[1, 1],
            library=b0.representation.library,
            library_version=b0.representation.library_version,
            spacy_model=None,
            spacy_model_version=None,
            parameters={},
        )
    elif decision == "classifier":
        result.classifier = ClassifierConfig(
            type="logistic_regression",
            library=b0.classifier.library,
            library_version=b0.classifier.library_version,
            parameters={},
        )
    else:  # pragma: no cover - Literal ya restringe esto estaticamente
        raise ValueError(f"Decision ablacionable desconocida: {decision!r}")

    return result


def differs_from_b0(candidate: PipelineConfig, b0: PipelineConfig) -> list[AblatableDecision]:
    """Lista las decisiones ablacionables (de las 6 de A.4) en las que
    ``candidate`` difiere de B0. Util para decidir cuantas ablaciones exige
    la Seccion 4 (0 si no difiere en ninguna, 1 si difiere en una, 2+ si
    difiere en dos o mas)."""
    diffs: list[AblatableDecision] = []
    cp, bp = candidate.preprocessing, b0.preprocessing
    assert cp is not None and bp is not None

    if (cp.stopwords, cp.negators) != (bp.stopwords, bp.negators):
        diffs.append("preprocessing.stopwords")
    if cp.lemmatize != bp.lemmatize:
        diffs.append("preprocessing.lemmatize")
    if (cp.elongation, cp.elongation_spec) != (bp.elongation, bp.elongation_spec):
        diffs.append("preprocessing.elongation")
    if (cp.emoji, cp.emoji_spec) != (bp.emoji, bp.emoji_spec):
        diffs.append("preprocessing.emoji")

    reverted_rep = revert_decision(candidate, "representation", b0).representation
    if not _representation_equal(candidate.representation, reverted_rep):
        diffs.append("representation")

    reverted_clf = revert_decision(candidate, "classifier", b0).classifier
    if candidate.classifier.to_dict() != reverted_clf.to_dict():
        diffs.append("classifier")

    return diffs


def _representation_equal(a: RepresentationConfig, b: RepresentationConfig) -> bool:
    return a.to_dict() == b.to_dict()


if __name__ == "__main__":
    # Auto-chequeo rapido: reproduce la "Forma canonica de B0" y de T0 del
    # Anexo A.3 y valida que pasan las reglas de esquema.
    b0 = b0_config(
        rep_library="scikit-learn",
        rep_version="1.5.2",
        clf_library="scikit-learn",
        clf_version="1.5.2",
    )
    b0.validate()
    print("=== B0 ===")
    print(b0.to_json())

    t0 = t0_config(library="scikit-learn", library_version="1.5.2")
    t0.validate()
    print("\n=== T0 ===")
    print(t0.to_json())

    # Ejemplo: P_STOPWORDS = "Igual a B0, salvo stopwords=remove" (A.4)
    p_stopwords = b0.copy()
    p_stopwords.preprocessing.stopwords = "remove"
    p_stopwords.validate()
    print("\n=== P_STOPWORDS ===")
    print(p_stopwords.to_json())

    # Ejemplo de ablacion: revertir stopwords de un candidato hipotetico
    candidate = b0.copy()
    candidate.preprocessing.stopwords = "remove_preserve_negation"
    candidate.preprocessing.negators = ["no", "not", "never"]
    candidate.validate()
    print("\n=== Candidato hipotetico (stopwords=remove_preserve_negation) ===")
    print(candidate.to_json())
    print("\nDifiere de B0 en:", differs_from_b0(candidate, b0))

    ablated = revert_decision(candidate, "preprocessing.stopwords", b0)
    ablated.validate()
    print("\n=== Ablacion de esa decision (deberia volver a igualar B0) ===")
    print("Es equivalente a B0:", ablated.is_equivalent_to(b0))

    # A.3: la igualdad ignora preprocessing.resources por defecto -- dos
    # configs con distinto 'resources' pero por lo demas identicos deben
    # seguir siendo equivalentes (mismo lab_configuration_id).
    b0_with_resources = b0.copy()
    b0_with_resources.preprocessing.resources = {"nltk_version": "3.8"}
    assert b0_with_resources.is_equivalent_to(b0), (
        "por defecto, 'resources' distinto no deberia romper la equivalencia (A.3)"
    )
    assert not b0_with_resources.is_equivalent_to(b0, ignore_resources=False), (
        "con ignore_resources=False, 'resources' distinto SI deberia romper la equivalencia"
    )
    print(
        "\nOK: is_equivalent_to ignora preprocessing.resources por defecto (A.3), "
        "y ignore_resources=False lo tiene en cuenta cuando se necesita"
    )