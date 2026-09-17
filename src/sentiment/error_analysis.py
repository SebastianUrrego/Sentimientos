"""
error_analysis.py
===================

Construye los artefactos de analisis de errores del run FINAL (Seccion 5
y Anexo A.6): ``reports/error_analysis.csv`` y ``reports/error_analysis.md``.

Que exige la guia (confirmado contra el PDF real):

    - Seleccionar aleatoriamente al menos 20 errores del modelo final
      (texto de ``test`` mal clasificado), con semilla 42, incluyendo
      errores de ambas clases cuando existan.
    - Clasificar cada caso en una de las 10 categorias EXACTAS de A.6:
      negation, intensification, contrast, mixed, emoji, elongation,
      informal, hashtag, sarcasm, other.
    - ``reports/error_analysis.csv``: UTF-8, encabezado exacto
      ``index,text,true_label,predicted_label,category``; ``index`` es
      la posicion ORIGINAL en test; comas/comillas/saltos de linea
      codificados como CSV valido; labels exclusivamente
      ``negative``/``positive``.
    - ``reports/error_analysis.md``: resume la frecuencia por categoria
      e interpreta los dos patrones mas frecuentes (o la unica
      categoria, si solo aparece una). "No se exige plantilla
      adicional" -- el formato del .md es libre.

Por que este modulo NO decide la categoria de cada error por si solo
--------------------------------------------------------------------
La Seccion 5 pide clasificar "según el fenómeno predominante" y luego
"interpretar" los patrones -- eso es, a proposito, un juicio humano
sobre ejemplos concretos, no algo que un heuristico de texto pueda
hacer de forma confiable (sarcasmo y "mixed" en particular casi siempre
necesitan leer la frase). Por eso ``suggest_category()`` es solo un
PUNTO DE PARTIDA (senales lexicas simples: hashtags, emojis,
alargamientos, negadores, contraste, intensificadores, jerga informal)
-- el equipo debe revisar/corregir cada sugerencia a mano antes de
guardar el CSV, y escribir su propia interpretacion en el .md (las
frases con "(AJUSTAR: ...)" que genera ``build_error_analysis_md`` son
un borrador para reemplazar, no el analisis final).

Lo que SI es mecanico y esta probado aqui con datos sinteticos: el
muestreo reproducible con semilla fija incluyendo ambas clases, la
validacion y escritura del CSV con el esquema exacto, y el resumen de
frecuencias del .md.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Optional, Sequence

from sentiment.data import LABEL_NEGATIVE, LABEL_POSITIVE, RANDOM_SEED

MIN_ERROR_SAMPLE_SIZE = 20  # Seccion 5: "al menos 20 errores"

# Las 10 categorias EXACTAS de A.6.
CATEGORIES: tuple[str, ...] = (
    "negation",
    "intensification",
    "contrast",
    "mixed",
    "emoji",
    "elongation",
    "informal",
    "hashtag",
    "sarcasm",
    "other",
)

VALID_LABELS = ("negative", "positive")

CSV_HEADER = ["index", "text", "true_label", "predicted_label", "category"]

# Ayuda breve para redactar la interpretacion humana en el .md -- NO es
# una definicion normativa del laboratorio, es solo un borrador a
# reemplazar con lo que el equipo observe en sus propios ejemplos.
CATEGORY_HINTS: dict[str, str] = {
    "negation": "la negacion (not, no, never, n't) invierte el sentido esperado y el modelo no la capta bien.",
    "intensification": "adverbios/mayusculas/signos que intensifican el sentimiento (really, so, !!!, MAYUSCULAS) confunden al modelo.",
    "contrast": "conectores de contraste (but, however, although) cambian el sentimiento a mitad de frase.",
    "mixed": "el texto mezcla sentimiento positivo y negativo genuinamente, sin un fenomeno unico dominante.",
    "emoji": "el sentimiento depende de un emoji/emoticono que el preprocesamiento no interpreto bien.",
    "elongation": "alargamientos de caracteres (soooo, nooo) cambian la intensidad percibida y no se normalizaron bien.",
    "informal": "jerga, abreviaturas o errores ortograficos alejan el texto del vocabulario aprendido.",
    "hashtag": "el sentimiento esta codificado en un hashtag (#loveit, #fail) mas que en el texto plano.",
    "sarcasm": "sarcasmo o ironia: el sentido literal es opuesto al sentimiento real -- dificil de detectar sin contexto.",
    "other": "no encaja claramente en las categorias anteriores.",
}


def _label_name(label: int) -> str:
    if label == LABEL_POSITIVE:
        return "positive"
    if label == LABEL_NEGATIVE:
        return "negative"
    raise ValueError(f"Label desconocida: {label!r} (se esperaba {LABEL_NEGATIVE} o {LABEL_POSITIVE})")


# ---------------------------------------------------------------------------
# Muestreo reproducible de errores
# ---------------------------------------------------------------------------


def sample_errors(
    indices: Sequence[int],
    texts: Sequence[str],
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    *,
    n: int = MIN_ERROR_SAMPLE_SIZE,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Encuentra los casos con ``true_labels[i] != pred_labels[i]`` y
    selecciona una muestra reproducible de tamano ``n`` (Seccion 5),
    garantizando al menos un error de cada clase verdadera cuando ambas
    estan presentes entre los errores.

    ``indices`` debe ser la posicion ORIGINAL en ``test`` de cada fila
    (A.6) -- no la posicion dentro de este batch. Devuelve una lista de
    dicts con ``index``, ``text``, ``true_label``, ``predicted_label``
    (ya como ``"negative"``/``"positive"``) y ``category=None`` (a
    llenar por el equipo, ver ``suggest_category``)."""
    import numpy as np

    if not (len(indices) == len(texts) == len(true_labels) == len(pred_labels)):
        raise ValueError("indices, texts, true_labels y pred_labels deben tener el mismo largo")

    error_positions = [i for i in range(len(indices)) if true_labels[i] != pred_labels[i]]
    if not error_positions:
        return []

    rng = np.random.default_rng(seed)
    n = min(n, len(error_positions))

    by_class: dict[int, list[int]] = {}
    for pos in error_positions:
        by_class.setdefault(true_labels[pos], []).append(pos)

    chosen: list[int] = []
    if len(by_class) > 1:
        for cls in sorted(by_class):
            pool = np.array(by_class[cls])
            chosen.extend(rng.choice(pool, size=1, replace=False).tolist())

    remaining_pool = np.array([p for p in error_positions if p not in chosen])
    remaining_needed = n - len(chosen)
    if remaining_needed > 0 and len(remaining_pool) > 0:
        extra = rng.choice(remaining_pool, size=min(remaining_needed, len(remaining_pool)), replace=False)
        chosen.extend(extra.tolist())

    chosen = sorted(set(chosen))
    return [
        {
            "index": int(indices[pos]),
            "text": texts[pos],
            "true_label": _label_name(true_labels[pos]),
            "predicted_label": _label_name(pred_labels[pos]),
            "category": None,
        }
        for pos in chosen
    ]


# ---------------------------------------------------------------------------
# Heuristica de arranque para sugerir categoria (NO reemplaza juicio humano)
# ---------------------------------------------------------------------------

_HASHTAG_RE = re.compile(r"#\w+")
_EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
_ELONGATION_RE = re.compile(r"(.)\1{2,}")
_NEGATION_RE = re.compile(r"\b(not|no|never|n't|nobody|nothing|none)\b")
_CONTRAST_RE = re.compile(r"\b(but|however|although|yet|still)\b")
_INTENSIFIER_RE = re.compile(r"!{2,}|\b(really|so|very|extremely)\b")
_INFORMAL_MARKERS = {"u", "ur", "lol", "omg", "gonna", "wanna", "gotta", "kinda", "lmao", "btw"}


def suggest_category(text: str) -> str:
    """Sugiere una de las 10 categorias de A.6 a partir de senales
    lexicas simples. Es un PUNTO DE PARTIDA para llenar el CSV mas
    rapido -- revisar y corregir a mano antes de guardarlo. No intenta
    detectar 'sarcasm' ni 'mixed' (casi siempre requieren leer la
    frase); esos casos caen en 'other' hasta que el equipo los
    reclasifique."""
    lower = text.lower()
    if _HASHTAG_RE.search(text):
        return "hashtag"
    if _EMOJI_RE.search(text):
        return "emoji"
    if _ELONGATION_RE.search(text):
        return "elongation"
    if _NEGATION_RE.search(lower):
        return "negation"
    if _CONTRAST_RE.search(lower):
        return "contrast"
    if _INTENSIFIER_RE.search(lower) or text.isupper():
        return "intensification"
    if any(w in _INFORMAL_MARKERS for w in lower.split()):
        return "informal"
    return "other"


# ---------------------------------------------------------------------------
# reports/error_analysis.csv
# ---------------------------------------------------------------------------


def write_error_analysis_csv(records: Sequence[dict], path: str) -> None:
    """Escribe ``reports/error_analysis.csv`` con el encabezado EXACTO
    (A.6), validando antes de escribir que cada registro tenga
    ``true_label``/``predicted_label`` en {"negative","positive"} y
    ``category`` en ``CATEGORIES`` (nunca ``None``). Usa el modulo
    ``csv`` estandar (``QUOTE_MINIMAL``) para que comas, comillas y
    saltos de linea en ``text`` queden codificados como CSV valido."""
    for r in records:
        if r.get("true_label") not in VALID_LABELS or r.get("predicted_label") not in VALID_LABELS:
            raise ValueError(
                f"true_label/predicted_label deben ser {VALID_LABELS}, "
                f"registro index={r.get('index')}: {r!r}"
            )
        if r.get("category") not in CATEGORIES:
            raise ValueError(
                f"category invalida o sin asignar en el registro index={r.get('index')}: "
                f"{r.get('category')!r}. Debe ser una de {CATEGORIES}"
            )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        writer.writeheader()
        for r in sorted(records, key=lambda r: r["index"]):
            writer.writerow({k: r[k] for k in CSV_HEADER})


def read_error_analysis_csv(path: str) -> list[dict]:
    """Lee ``error_analysis.csv``, validando el encabezado exacto."""
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != CSV_HEADER:
            raise ValueError(
                f"{path} debe tener encabezado exacto {CSV_HEADER}, tiene {reader.fieldnames}"
            )
        return [{**row, "index": int(row["index"])} for row in reader]


# ---------------------------------------------------------------------------
# reports/error_analysis.md
# ---------------------------------------------------------------------------


def summarize_categories(records: Sequence[dict]) -> dict[str, int]:
    """Frecuencia por categoria (solo las categorias con al menos un
    caso), en el orden fijo de ``CATEGORIES``."""
    counts = {c: 0 for c in CATEGORIES}
    for r in records:
        counts[r["category"]] += 1
    return {k: v for k, v in counts.items() if v > 0}


def top_categories(counts: dict[str, int], n: int = 2) -> list[tuple[str, int]]:
    """Las ``n`` categorias mas frecuentes (desempate alfabetico)."""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]


def build_error_analysis_md(records: Sequence[dict], *, title: str = "Análisis de errores") -> str:
    """Arma el contenido de ``error_analysis.md``: tabla de frecuencia
    por categoria + un BORRADOR de interpretacion de los dos patrones
    mas frecuentes (o de la unica categoria, si solo aparece una --
    Seccion 5). Las frases marcadas ``(AJUSTAR: ...)`` deben
    reemplazarse por la interpretacion real del equipo basada en los
    ejemplos concretos del CSV -- lo que genera esta funcion es
    estructura y un punto de partida, no el analisis final."""
    if not records:
        raise ValueError("No hay errores para reportar")

    counts = summarize_categories(records)
    total = len(records)
    ranked = top_categories(counts, n=len(counts))

    lines = [f"# {title}", "", f"Total de errores analizados: {total}", ""]
    lines.append("| Categoría | Frecuencia | % |")
    lines.append("|---|---|---|")
    for cat, count in ranked:
        pct = 100 * count / total
        lines.append(f"| {cat} | {count} | {pct:.1f}% |")
    lines.append("")

    if len(ranked) == 1:
        cat, count = ranked[0]
        lines.append(f"## Interpretación: {cat}")
        lines.append("")
        lines.append(
            f"Todos los errores analizados ({count}/{total}) caen en la categoría "
            f"**{cat}**. {CATEGORY_HINTS.get(cat, '')} "
            "(AJUSTAR: reemplazar por la interpretación real del equipo con base en los "
            "ejemplos concretos del CSV)."
        )
    else:
        lines.append("## Interpretación de los dos patrones más frecuentes")
        lines.append("")
        for cat, count in ranked[:2]:
            pct = 100 * count / total
            lines.append(
                f"**{cat}** ({count}/{total}, {pct:.1f}%): {CATEGORY_HINTS.get(cat, '')} "
                "(AJUSTAR: reemplazar por la interpretación real del equipo con base en los "
                "ejemplos concretos del CSV)."
            )
            lines.append("")

    return "\n".join(lines)


def write_error_analysis_md(
    records: Sequence[dict], path: str, *, title: str = "Análisis de errores"
) -> None:
    content = build_error_analysis_md(records, title=title)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    import tempfile

    import numpy as np

    rng = np.random.default_rng(1)
    n = 300
    indices = list(range(1000, 1000 + n))  # simula posiciones originales en test

    templates_pos = ["great movie", "loved it", "amazing service", "so good", "not bad at all"]
    templates_neg = ["terrible movie", "hated it", "awful service", "so bad", "not good at all"]

    texts: list[str] = []
    true_labels: list[int] = []
    for _ in range(n):
        is_pos = rng.random() < 0.5
        texts.append(str(rng.choice(templates_pos if is_pos else templates_neg)))
        true_labels.append(LABEL_POSITIVE if is_pos else LABEL_NEGATIVE)

    flip_mask = rng.random(n) < 0.15
    pred_labels = [
        (LABEL_POSITIVE + LABEL_NEGATIVE - t) if flip else t
        for t, flip in zip(true_labels, flip_mask)
    ]
    n_errors = int(flip_mask.sum())
    print(f"Errores simulados: {n_errors} de {n}")
    assert n_errors >= MIN_ERROR_SAMPLE_SIZE, "ajusta la simulacion para tener suficientes errores de prueba"

    sample1 = sample_errors(indices, texts, true_labels, pred_labels, n=20, seed=RANDOM_SEED)
    sample2 = sample_errors(indices, texts, true_labels, pred_labels, n=20, seed=RANDOM_SEED)
    assert sample1 == sample2
    assert len(sample1) >= MIN_ERROR_SAMPLE_SIZE
    print(f"OK: sample_errors reproducible con seed={RANDOM_SEED}, {len(sample1)} registros")

    labels_in_sample = {r["true_label"] for r in sample1}
    assert labels_in_sample == {"negative", "positive"}
    print("OK: sample_errors incluye errores de ambas clases cuando existen")

    assert suggest_category("this is not good at all") == "negation"
    assert suggest_category("soooo tired of this") == "elongation"
    assert suggest_category("great day #blessed") == "hashtag"
    assert suggest_category("loved it \U0001F600") == "emoji"
    assert suggest_category("good but slow") == "contrast"
    assert suggest_category("REALLY GOOD MOVIE") == "intensification"
    assert suggest_category("gonna love this") == "informal"
    assert suggest_category("just an ordinary sentence") == "other"
    print("OK: suggest_category detecta las senales lexicas basicas (heuristica de arranque)")

    for r in sample1:
        r["category"] = suggest_category(r["text"])

    with tempfile.TemporaryDirectory() as d:
        csv_path = os.path.join(d, "reports", "error_analysis.csv")
        write_error_analysis_csv(sample1, csv_path)
        with open(csv_path, encoding="utf-8") as f:
            header = f.readline().strip()
        assert header == "index,text,true_label,predicted_label,category"
        print("OK: write_error_analysis_csv encabezado exacto")

        loaded = read_error_analysis_csv(csv_path)
        assert len(loaded) == len(sample1)
        assert {r["index"] for r in loaded} == {r["index"] for r in sample1}
        print("OK: error_analysis.csv round-trip correcto")

        tricky = [
            {
                "index": 9999,
                "text": 'a "quoted", tricky, text',
                "true_label": "negative",
                "predicted_label": "positive",
                "category": "other",
            }
        ]
        tricky_path = os.path.join(d, "tricky.csv")
        write_error_analysis_csv(tricky, tricky_path)
        loaded_tricky = read_error_analysis_csv(tricky_path)
        assert loaded_tricky[0]["text"] == 'a "quoted", tricky, text'
        print("OK: comas/comillas en 'text' se escapan y se leen de vuelta correctamente")

        try:
            write_error_analysis_csv(
                [{"index": 1, "text": "x", "true_label": "negative", "predicted_label": "positive", "category": None}],
                os.path.join(d, "bad.csv"),
            )
            raise AssertionError("se esperaba ValueError")
        except ValueError:
            print("OK: write_error_analysis_csv exige category valida")

        try:
            write_error_analysis_csv(
                [{"index": 1, "text": "x", "true_label": "neg", "predicted_label": "positive", "category": "other"}],
                os.path.join(d, "bad2.csv"),
            )
            raise AssertionError("se esperaba ValueError")
        except ValueError:
            print("OK: write_error_analysis_csv exige true_label/predicted_label exactos")

        md_path = os.path.join(d, "reports", "error_analysis.md")
        write_error_analysis_md(sample1, md_path)
        with open(md_path, encoding="utf-8") as f:
            md_content = f.read()
        counts = summarize_categories(sample1)
        assert all(cat in md_content for cat in counts)
        print("OK: error_analysis.md incluye todas las categorias presentes en la muestra")

    single_cat_records = [
        {
            "index": i, "text": f"example {i}",
            "true_label": "negative", "predicted_label": "positive", "category": "sarcasm",
        }
        for i in range(5)
    ]
    single_md = build_error_analysis_md(single_cat_records)
    assert "sarcasm" in single_md and "Interpretación: sarcasm" in single_md
    print("OK: build_error_analysis_md maneja el caso de una sola categoria presente")

    print("\nTodas las pruebas pasaron.")
