"""
preprocessing.py
=================

Transformaciones de texto controladas por ``PreprocessingConfig`` (ver
``config.py``, que implementa el esquema exacto del Anexo A.3).

Cada decision del config se traduce en un paso independiente y
composable, para poder activar/desactivar exactamente una decision a la
vez durante las comparaciones por etapas (Seccion 3) y las ablaciones
(Seccion 4 / A.4).

Orden de aplicacion dentro de ``preprocess_text`` (fijo y documentado --
si la guia del laboratorio especifica un orden distinto, avisar para
ajustarlo; los pasos son independientes entre si salvo por esto):

    1. lowercase
    2. url        (keep | drop | token:<valor>)
    3. mention     (keep | drop | token:<valor>)
    4. elongation  (keep | normalize)
    5. emoji       (keep | text)
    6. stopwords   (keep | remove | remove_preserve_negation)
    7. lemmatize
    8. whitespace  (keep | normalize)

Nota sobre "whitespace" combinado con stopwords/lemmatize: esos dos
pasos requieren tokenizar (separar por espacios) y luego reconstruir el
texto con ``" ".join(tokens)``, lo que de por si deja un solo espacio
entre tokens. Es decir, si se activa stopwords!="keep" o lemmatize=True,
el resultado queda con espaciado normalizado sin importar el valor de
``whitespace`` -- esto es una simplificacion intencional (en la practica
B0 nunca combina esto: B0 tiene stopwords="keep", lemmatize=False).

Separacion pura vs. recursos externos
--------------------------------------
Igual que en ``data.py``: la logica que solo necesita texto y estructuras
en memoria (regex de url/mention, normalizacion de elongacion, remocion
de stopwords, lematizacion, union de tokens) es "pura" y se prueba aqui
mismo sin internet, usando recursos de juguete inyectados por parametro.

Cargar los recursos REALES (lista de stopwords de NLTK, modelo de spaCy,
libreria `emoji`) si requiere red/paquetes que no estan disponibles en
este sandbox de desarrollo -- por eso esta aislado en tres funciones
"loader" (``load_nltk_stopwords``, ``load_spacy_lemmatizer``,
``load_emoji_converter``) que solo se deben ejecutar donde haya internet
(la instancia de SageMaker).
"""

from __future__ import annotations

import re
from typing import Callable, Literal, Optional

from sentiment.config import PreprocessingConfig

# ---------------------------------------------------------------------------
# Regex / constantes
# ---------------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_WHITESPACE_RE = re.compile(r"\s+")
_ELONGATION_RE = re.compile(r"(.)\1{2,}")

DEFAULT_MAX_REPEAT = 2

UrlMentionKind = Literal["url", "mention"]


# ---------------------------------------------------------------------------
# Pasos puros (sin recursos externos)
# ---------------------------------------------------------------------------


def apply_url_mention(text: str, *, kind: UrlMentionKind, mode: str) -> str:
    """Aplica keep/drop/token:<valor> sobre URLs o menciones (@usuario)."""
    pattern = _URL_RE if kind == "url" else _MENTION_RE
    if mode == "keep":
        return text
    if mode == "drop":
        return pattern.sub("", text)
    if mode.startswith("token:"):
        token_value = mode.split(":", 1)[1]
        return pattern.sub(token_value, text)
    raise ValueError(f"Valor de {kind!r} invalido: {mode!r}")


def normalize_whitespace(text: str) -> str:
    """Colapsa cualquier corrida de espacios/tabs/saltos de linea a un solo
    espacio y recorta los extremos."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def _parse_max_repeat(spec: Optional[str], default: int = DEFAULT_MAX_REPEAT) -> int:
    if not spec:
        return default
    if spec.startswith("max_repeat:"):
        try:
            return max(1, int(spec.split(":", 1)[1]))
        except ValueError:
            return default
    return default


def normalize_elongation(text: str, spec: Optional[str]) -> str:
    """Colapsa caracteres repetidos 3+ veces seguidas (p. ej. 'sooooo') a
    ``max_repeat`` copias (por defecto 2, para no perder del todo la señal
    de enfasis). ``spec`` sigue el formato ``"max_repeat:<n>"``; si es None
    o no matchea ese formato, se usa el default."""
    max_repeat = _parse_max_repeat(spec)

    def _collapse(m: "re.Match[str]") -> str:
        return m.group(1) * max_repeat

    return _ELONGATION_RE.sub(_collapse, text)


def tokenize(text: str) -> list[str]:
    """Tokenizador simple por espacios en blanco. Suficiente aqui porque
    url/mention ya se reemplazaron por un solo token antes de llegar a este
    paso."""
    return text.split()


def remove_stopwords(
    tokens: list[str],
    stopword_set: set[str],
    mode: str,
    negators: list[str],
) -> list[str]:
    """Remueve stopwords de ``tokens`` segun ``mode``:

    - "keep": no hace nada (devuelve una copia de ``tokens``).
    - "remove": quita cualquier token (case-insensitive) presente en
      ``stopword_set``.
    - "remove_preserve_negation": igual que "remove", pero nunca quita un
      token que este en ``negators`` (case-insensitive), aunque tambien
      sea una stopword -- para no perder la negacion (A.3).
    """
    if mode == "keep":
        return list(tokens)
    if mode not in ("remove", "remove_preserve_negation"):
        raise ValueError(f"preprocessing.stopwords invalido: {mode!r}")

    stop_lower = {w.lower() for w in stopword_set}
    negator_lower = {n.lower() for n in negators}

    result: list[str] = []
    for tok in tokens:
        low = tok.lower()
        is_stop = low in stop_lower
        if is_stop and mode == "remove_preserve_negation" and low in negator_lower:
            is_stop = False
        if not is_stop:
            result.append(tok)
    return result


def lemmatize_tokens(
    tokens: list[str],
    lemmatize_fn: Callable[[list[str]], list[str]],
) -> list[str]:
    """Aplica ``lemmatize_fn`` (que recibe y devuelve una lista de tokens,
    en el mismo orden y cantidad) sobre ``tokens``. Trabajar por lista
    completa -- en vez de token por token -- es lo que permite que la
    implementacion real con spaCy corra el pipeline una sola vez por
    tweet en vez de una vez por palabra."""
    if not tokens:
        return []
    lemmas = lemmatize_fn(tokens)
    if len(lemmas) != len(tokens):
        raise ValueError(
            "lemmatize_fn debe devolver la misma cantidad de tokens que "
            f"recibe (recibio {len(tokens)}, devolvio {len(lemmas)})"
        )
    return lemmas


def emoji_to_text(text: str, *, converter: Optional[Callable[[str], str]] = None) -> str:
    """Convierte emojis a texto descriptivo. Si no se inyecta un
    ``converter`` propio (util para pruebas sin internet), usa la libreria
    `emoji` real via ``load_emoji_converter()``."""
    fn = converter if converter is not None else load_emoji_converter()
    return fn(text)


# ---------------------------------------------------------------------------
# Orquestador
# ---------------------------------------------------------------------------


def preprocess_text(
    text: str,
    config: PreprocessingConfig,
    *,
    stopword_set: Optional[set[str]] = None,
    lemmatize_fn: Optional[Callable[[list[str]], list[str]]] = None,
    emoji_converter: Optional[Callable[[str], str]] = None,
) -> str:
    """Aplica el pipeline completo de preprocesamiento descrito arriba,
    segun lo que indique ``config`` (una ``PreprocessingConfig`` ya
    validada). Lanza ``ValueError`` si el config pide un paso que
    necesita un recurso externo (``stopword_set`` / ``lemmatize_fn``) y
    no se lo paso."""
    if config.lowercase:
        text = text.lower()

    text = apply_url_mention(text, kind="url", mode=config.url)
    text = apply_url_mention(text, kind="mention", mode=config.mention)

    if config.elongation == "normalize":
        text = normalize_elongation(text, config.elongation_spec)

    if config.emoji == "text":
        text = emoji_to_text(text, converter=emoji_converter)

    needs_tokens = config.stopwords != "keep" or config.lemmatize
    if needs_tokens:
        tokens = tokenize(text)
        if config.stopwords != "keep":
            if stopword_set is None:
                raise ValueError(
                    "Se requiere stopword_set cuando preprocessing.stopwords "
                    f"!= 'keep' (config pide {config.stopwords!r})"
                )
            tokens = remove_stopwords(tokens, stopword_set, config.stopwords, config.negators)
        if config.lemmatize:
            if lemmatize_fn is None:
                raise ValueError(
                    "Se requiere lemmatize_fn cuando preprocessing.lemmatize=True"
                )
            tokens = lemmatize_tokens(tokens, lemmatize_fn)
        text = " ".join(tokens)
    elif config.whitespace == "normalize":
        text = normalize_whitespace(text)

    return text


# ---------------------------------------------------------------------------
# Loaders de recursos reales (requieren internet / paquetes adicionales)
# ---------------------------------------------------------------------------


def load_nltk_stopwords(language: str = "english") -> set[str]:
    """Carga la lista de stopwords de NLTK para ``language``. Requiere
    ``pip install nltk`` y, la primera vez, ``nltk.download('stopwords')``
    (necesita internet) -- correr esto en la instancia de SageMaker, no
    en un entorno sin red."""
    import nltk  # type: ignore

    try:
        from nltk.corpus import stopwords  # type: ignore

        return set(stopwords.words(language))
    except LookupError:
        nltk.download("stopwords")
        from nltk.corpus import stopwords  # type: ignore

        return set(stopwords.words(language))


def load_spacy_lemmatizer(model: str = "en_core_web_sm") -> Callable[[list[str]], list[str]]:
    """Carga un modelo de spaCy y devuelve una funcion
    ``list[str] -> list[str]`` que lematiza una lista de tokens ya
    tokenizados (sin volver a tokenizar el texto). Requiere
    ``pip install spacy`` y ``python -m spacy download <model>`` --
    correr esto en la instancia de SageMaker, no en un entorno sin red."""
    import spacy  # type: ignore

    try:
        nlp = spacy.load(model, disable=["parser", "ner"])
    except OSError as exc:
        raise OSError(
            f"El modelo de spaCy {model!r} no esta instalado. Instalalo con "
            f"'python -m spacy download {model}' en una instancia con internet."
        ) from exc

    def _lemmatize_fn(tokens: list[str]) -> list[str]:
        doc = spacy.tokens.Doc(nlp.vocab, words=tokens)
        for _, proc in nlp.pipeline:
            doc = proc(doc)
        return [t.lemma_ for t in doc]

    return _lemmatize_fn


def load_emoji_converter() -> Callable[[str], str]:
    """Devuelve una funcion que convierte emojis a texto descriptivo,
    usando la libreria `emoji` (``pip install emoji``). Ej.: '😀' ->
    ' grinning face '."""
    import emoji  # type: ignore

    def _convert(text: str) -> str:
        demojized = emoji.demojize(text, delimiters=(" ", " "))
        return demojized.replace("_", " ")

    return _convert


if __name__ == "__main__":
    from sentiment.config import b0_config

    # -- pasos puros, uno por uno -------------------------------------------
    assert apply_url_mention("mira esto http://t.co/abc ya", kind="url", mode="token:url") == (
        "mira esto url ya"
    )
    assert apply_url_mention("hola @juan como vas", kind="mention", mode="drop") == "hola  como vas"
    assert apply_url_mention("sin cambios @juan", kind="mention", mode="keep") == "sin cambios @juan"
    print("OK: apply_url_mention (keep/drop/token)")

    assert normalize_whitespace("  hola   mundo\t\n bien  ") == "hola mundo bien"
    print("OK: normalize_whitespace")

    assert normalize_elongation("soooo buenooo", "max_repeat:2") == "soo buenoo"
    assert normalize_elongation("hello", "max_repeat:2") == "hello"  # 'll' y 'o' no llegan a 3
    assert normalize_elongation("holaaaa", None) == "holaa"  # default max_repeat=2
    print("OK: normalize_elongation")

    assert tokenize("hola  mundo   feliz") == ["hola", "mundo", "feliz"]
    print("OK: tokenize")

    toy_stopwords = {"the", "is", "a", "not", "no"}
    toy_negators = ["not", "no"]
    tokens_in = ["this", "movie", "is", "not", "a", "masterpiece"]
    assert remove_stopwords(tokens_in, toy_stopwords, "keep", []) == tokens_in
    assert remove_stopwords(tokens_in, toy_stopwords, "remove", []) == [
        "this",
        "movie",
        "masterpiece",
    ]
    assert remove_stopwords(tokens_in, toy_stopwords, "remove_preserve_negation", toy_negators) == [
        "this",
        "movie",
        "not",
        "masterpiece",
    ]
    print("OK: remove_stopwords (keep/remove/remove_preserve_negation)")

    def toy_lemmatize_fn(tokens: list[str]) -> list[str]:
        # juguete: quita una 's' final si la palabra termina en ella (no es
        # linguisticamente correcto, solo prueba el cableado de la funcion)
        return [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in tokens]

    assert lemmatize_tokens(["movies", "cats", "is"], toy_lemmatize_fn) == ["movie", "cat", "is"]
    print("OK: lemmatize_tokens")

    def toy_emoji_converter(text: str) -> str:
        return text.replace("🙂", " smile ").replace("😡", " angry ")

    assert emoji_to_text("great! 🙂", converter=toy_emoji_converter) == "great!  smile "
    print("OK: emoji_to_text (con converter inyectado)")

    try:
        emoji_to_text("sin converter inyectado 🙂")
        raise AssertionError("se esperaba ImportError (paquete 'emoji' no instalado aqui)")
    except ImportError:
        print("OK: emoji_to_text sin converter intenta cargar la libreria real ('emoji' -- "
              "no disponible en este sandbox sin internet, como se esperaba)")

    # -- preprocess_text end-to-end ------------------------------------------
    b0 = b0_config(
        rep_library="scikit-learn",
        rep_version="1.5.2",
        clf_library="scikit-learn",
        clf_version="1.5.2",
    )
    raw = "  Check THIS out http://t.co/xyz  @amazing_user !!  sooo goooood  "
    out_b0 = preprocess_text(raw, b0.preprocessing)
    print("\nB0 (lowercase + url/mention token + whitespace normalize):")
    print(repr(out_b0))
    assert out_b0 == "check this out url user !! sooo goooood"

    p_stopwords = b0.copy()
    p_stopwords.preprocessing.stopwords = "remove"
    p_stopwords.validate()
    out_p = preprocess_text(raw, p_stopwords.preprocessing, stopword_set={"this", "out"})
    print("\nP_STOPWORDS (con stopword_set de juguete):")
    print(repr(out_p))

    try:
        preprocess_text(raw, p_stopwords.preprocessing)
        raise AssertionError("se esperaba ValueError sin stopword_set")
    except ValueError:
        print("OK: preprocess_text exige stopword_set cuando stopwords != 'keep'")

    p_elongation = b0.copy()
    p_elongation.preprocessing.elongation = "normalize"
    p_elongation.preprocessing.elongation_spec = "max_repeat:2"
    p_elongation.validate()
    out_e = preprocess_text(raw, p_elongation.preprocessing)
    print("\nP_ELONGATION (sooo -> soo, goooood -> good con max_repeat:2):")
    print(repr(out_e))
    assert "sooo" not in out_e and "goooood" not in out_e

    p_emoji = b0.copy()
    p_emoji.preprocessing.emoji = "text"
    p_emoji.preprocessing.emoji_spec = "emoji.demojize"
    p_emoji.validate()
    out_em = preprocess_text(
        "me encanta esto 🙂 url @amazing_user",
        p_emoji.preprocessing,
        emoji_converter=toy_emoji_converter,
    )
    print("\nP_EMOJI (con converter inyectado):")
    print(repr(out_em))
    assert "smile" in out_em

    print("\nTodas las pruebas pasaron.")
