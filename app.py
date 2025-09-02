
import streamlit as st
import pandas as pd
from pathlib import Path
import os, re, json, html, glob
from typing import Optional, List, Tuple, Dict, Set, Any

# =========================
# CONFIG GLOBALE
# =========================
st.set_page_config(page_title="Clinical Trial Matching App", layout="wide")
DATA_DIR = Path(os.environ.get("DATA_DIR", Path(__file__).parent)).resolve()
CASES_CLEAN_DIR = DATA_DIR / "cases_clean"
CRITERIA_MATCHES_DIR = DATA_DIR / "criteria_matches"

# =========================
# NORMALISATION METHODE
# =========================
def normalize_method(method: str) -> str:
    """
    Retourne 'APS1' ou 'APS3' quelle que soit la variante saisie ('APS-1', 'aps 1', 'APS-3 GPT4', etc.)
    """
    s = re.sub(r'[^a-z0-9]+', '', str(method).lower())
    if 'aps1' in s:
        return 'APS1'
    if 'aps3' in s or 'gpt4' in s or 'aps4' in s:
        return 'APS3'
    return 'APS3'

# =========================
# I/O helpers
# =========================
def load_csv(filename, **kwargs):
    """Charge un CSV depuis DATA_DIR avec erreurs claires."""
    path = DATA_DIR / filename
    try:
        return pd.read_csv(path, encoding=kwargs.pop("encoding", "utf-8-sig"), **kwargs)
    except FileNotFoundError:
        st.error(f"❌ Fichier introuvable : {path}")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Erreur en lisant {path.name} : {e}")
        st.stop()

def load_table_any(basename: str) -> pd.DataFrame:
    """Charge basename.csv sinon basename.tsv depuis DATA_DIR."""
    for ext in (".csv", ".tsv"):
        path = DATA_DIR / f"{basename}{ext}"
        if path.exists():
            if ext == ".csv":
                return pd.read_csv(path, encoding="utf-8-sig")
            else:
                return pd.read_csv(path, encoding="utf-8-sig", sep="\t")
    st.error(f"❌ Fichier introuvable : {basename}.csv/tsv dans {DATA_DIR}")
    st.stop()

def _collect_criterion_entity_status_from_entity_match(match_json: Dict[str, Any]) -> Dict[str, str]:
    """
    Construit {criterion_entity_id: 'pos'|'neg'|'unk'} à partir de logical_entities d'UN critère.
    - On lit d'abord le statut au niveau des members (members[*].entity_match) + leur 'id'
    - On retombe sur le statut groupe si besoin
    - On gère aussi les variantes d'ID (criterion_entity_ids, query_ids, ...), list/dict/str
    Priorité de résolution : pos > neg > unk.
    """
    out: Dict[str, str] = {}
    priority = {"pos": 3, "neg": 2, "unk": 1, "": 0}

    def apply_one(_id: Any, em_val: Any):
        if _id is None:
            return
        status = _normalize_entity_match(em_val)  # 'pos'|'neg'|'unk'
        sid = str(_id)
        prev = out.get(sid, "")
        if priority.get(status, 0) > priority.get(prev, 0):
            out[sid] = status

    def apply_many(ids: Any, em_val: Any):
        if ids is None:
            return
        if isinstance(ids, (str, int)):
            apply_one(ids, em_val)
        elif isinstance(ids, list):
            for x in ids:
                if isinstance(x, dict):
                    apply_one(x.get("id"), em_val)
                else:
                    apply_one(x, em_val)

    if not match_json:
        return out

    for group in match_json.get("logical_entities", []):
        g_em = (group.get("entity_match") or group.get("group_match")
                or group.get("match_type") or group.get("status") or group.get("match"))

        # 1) d'abord les members : leur 'id' est l'ID d'entité de CRITÈRE
        for m in group.get("members", []):
            m_em = (m.get("entity_match") or m.get("match_type") or
                    m.get("status") or m.get("match") or g_em)
            apply_one(m.get("id"), m_em)  # << clé manquante avant

            # variantes éventuelles (parfois présentes)
            for key in (
                "criterion_entity_ids", "criterion_ids", "criteria_entity_ids", "criteria_ids",
                "crit_entity_ids", "crit_ids", "query_entity_ids", "query_ids",
                "criterion_entity_id", "criterion_id", "query_entity_id", "query_id",
                "criterion_entities"
            ):
                apply_many(m.get(key), m_em)

        # 2) au niveau groupe : si des listes d'IDs existent
        for key in (
            "criterion_entity_ids", "criterion_ids", "criteria_entity_ids", "criteria_ids",
            "crit_entity_ids", "crit_ids", "query_entity_ids", "query_ids",
            "criterion_entity_id", "criterion_id", "query_entity_id", "query_id",
            "criterion_entities"
        ):
            apply_many(group.get(key), g_em)

    return out



def build_highlighted_criterion_html(crit_text: str, match_json: Dict[str, Any]) -> str:
    """
    Surligne les ENTITÉS DU CRITÈRE avec code couleur :
      vert = 'pos' (full/partial/match), rouge = 'neg' (no match), gris = 'unk' sinon.
    Les offsets viennent de match_json['entities'] et sont relatifs à match_json['text'].
    """
    if not match_json:
        return html.escape(crit_text)

    src = match_json.get("text", "") or crit_text
    # si le texte CSV diffère trop du texte source utilisé pour les offsets, on rend sur 'src'
    def _norm(s: str) -> str: return re.sub(r"\s+", " ", str(s)).strip().lower()
    text = src if _norm(src) != _norm(crit_text) else crit_text

    status_map = _collect_criterion_entity_status_from_entity_match(match_json)

    spans: List[Tuple[int, int, str]] = []  # (start, end, status)
    for ent in match_json.get("entities", []):
        crit_ent_id = str(ent.get("id") or ent.get("entity_id") or ent.get("uid") or "")
        status = status_map.get(crit_ent_id, "unk")

        # offsets robustes
        pairs: List[Tuple[int, int]] = []
        off = ent.get("offset")
        offs = ent.get("offsets")
        if isinstance(off, list) and len(off) == 2 and all(isinstance(x, int) for x in off):
            pairs.append((off[0], off[1]))
        if isinstance(offs, list):
            for o in offs:
                if isinstance(o, list) and len(o) == 2 and all(isinstance(x, int) for x in o):
                    pairs.append((o[0], o[1]))

        for s, e in pairs:
            # NB: offsets définis vs 'src' ; si on rend sur 'crit_text' mais longueurs diffèrent,
            # on a mis 'text' = 'src' ci-dessus pour rester cohérent.
            if 0 <= s < e <= len(src):
                spans.append((s, e, status))

    if not spans:
        return html.escape(text)

    # non-chevauchement avec priorité pos>neg>unk puis longueur
    priority = {"pos": 3, "neg": 2, "unk": 1}
    spans.sort(key=lambda t: (priority.get(t[2], 0), (t[1]-t[0])), reverse=True)

    chosen: List[Tuple[int, int, str]] = []
    def overlaps(a, b): return not (a[1] <= b[0] or b[1] <= a[0])
    for s, e, stt in spans:
        if all(not overlaps((s, e), (cs, ce)) for cs, ce, _ in chosen):
            chosen.append((s, e, stt))
    chosen.sort(key=lambda t: t[0])

    out, cur = [], 0
    for s, e, stt in chosen:
        s = max(0, min(s, len(text))); e = max(0, min(e, len(text)))
        if cur < s:
            out.append(html.escape(text[cur:s]))
        klass = "ent-patient-pos" if stt == "pos" else ("ent-patient-neg" if stt == "neg" else "ent-patient-unk")
        out.append(f'<span class="{klass}">{html.escape(text[s:e])}</span>')
        cur = e
    if cur < len(text):
        out.append(html.escape(text[cur:]))

    return "".join(out)



# =========================
# Patient id helpers
# =========================
def _patient_slug_from_id(patient_id: str) -> str:
    """'patient 1'/'patient_0001' -> 'patient_0001'."""
    digits = "".join(ch for ch in str(patient_id) if ch.isdigit())
    if digits:
        return f"patient_{int(digits):04d}"
    return str(patient_id).strip().replace(" ", "_")

def _patient_simple_from_id(patient_id: str) -> str:
    """'patient 1'/'patient_0001' -> 'patient 1' (pour tables 'patient X')."""
    digits = "".join(ch for ch in str(patient_id) if ch.isdigit())
    if digits:
        return f"patient {int(digits)}"
    s = str(patient_id).strip().replace("_", " ")
    return s if s.startswith("patient ") else f"patient {s}"

# =========================
# Chargement des données de base
# =========================
criteria_df       = load_csv("criteria.csv")
eligibility_df    = load_csv("eligibility_results_openAI_interface.csv")  # détail APS-3
patients_df       = load_csv("patients.csv")
clinical_trials   = load_csv("clinicaltrials.csv")

# Clean nct_id dans criteria_df
if "nct_id" in criteria_df.columns:
    criteria_df["nct_id"] = criteria_df["nct_id"].apply(
        lambda x: str(x).strip("[]").replace("'", "").strip()
    )

# =========================
# Styles UI
# =========================
st.title("🩺 Clinical Trial Matching App")

st.markdown("""
<style>
/* --- Surbrillance d'entités (par label pour critères) --- */
.ent { padding:0 2px; border-radius:4px; }
.ent-default { background:#FFF7CC; color:#6B5900; }
.ent-Condition  { background:#E6F2FF; color:#0B3D91; }
.ent-Procedure  { background:#FFF4E5; color:#8A4B00; }
.ent-Drug       { background:#FDE7E9; color:#A80000; }
.ent-Measurement{ background:#EAF7F0; color:#0B6B00; }
.ent-Value      { background:#F3E8FF; color:#5B2E91; }
.ent-Unit       { background:#F0F4F8; color:#243B53; }
.ent-Temporal   { background:#E8FBFF; color:#035388; }
.ent-Duration   { background:#E8FBFF; color:#035388; }
.ent-Age        { background:#FFEFE6; color:#8A2D00; }
.ent-Sex        { background:#E8E8FD; color:#2D2DB3; }
.ent-Comparator { background:#EFEFEF; color:#333333; }
.ent-Qualifier  { background:#EFEFEF; color:#333333; }
.ent-LabTest    { background:#E6FFF5; color:#006B5B; }
.ent-Biomarker  { background:#E6FFF5; color:#006B5B; }
.ent-Stage      { background:#FFF0F7; color:#93104D; }
.ent-ECOG       { background:#FFF0F7; color:#93104D; }
.ent-Metastasis { background:#FFE3EC; color:#7A0036; }
.ent-Gene       { background:#FFF7E6; color:#664200; }
.ent-Variant    { background:#FFF7E6; color:#664200; }
.ent-Device     { background:#EAF3FF; color:#003E92; }
.ent-Imaging    { background:#EAF3FF; color:#003E92; }
.ent-Score      { background:#F0F7FF; color:#1F3B77; }

/* --- Patient context: couleurs vert/rouge/gris --- */
.ent-patient-pos { background:#DFF6DD; color:#0B6B00; }  /* full/partial match */
.ent-patient-neg { background:#FDE7E9; color:#A80000; }  /* no_match */
.ent-patient-unk { background:#C9CED6; color:#1F2937; }  /* gris plus foncé */

/* --- Mise en page plus contenue --- */
.block-container {
  max-width: 1200px;
  padding-left: 2rem;
  padding-right: 2rem;
  margin-left: auto;
  margin-right: auto;
}

/* --- Cards liste essais --- */
.trial-card {
  background:#f7f7f8;
  border:1px solid #e5e5e5;
  border-radius:14px;
  padding:16px 20px;
  margin:16px auto;
  max-width: 95%;
}
.trial-card-header {
  display:flex; align-items:center; justify-content:space-between; gap:12px;
}
.trial-card-title { font-weight:600; line-height:1.35; }

/* --- Badge ELIGIBLE/NON ELIGIBLE --- */
.badge {font-weight:700; padding:2px 10px; border-radius:8px; display:inline-block;}
.badge-eligible { background:#DFF6DD; color:#0B6B00; }
.badge-not      { background:#FDE7E9; color:#A80000; }

/* --- Boutons ("See More" / "Back") : gris par défaut, orange au survol --- */
.btn {
  display:inline-block;
  padding:8px 24px;
  min-width:120px;
  text-align:center;
  border-radius:8px;
  border:1px solid #ccc;
  background:#f9f9f9;
  color:#555 !important;
  font-weight:600;
  text-decoration:none !important;
  cursor:pointer;
  transition:all .2s ease;
}
.btn:hover {
  border-color:#ff8c00;
  color:#ff8c00 !important;
  background:#fff;
  text-decoration:none !important;
}
.btn:focus {
  outline:2px solid #ff8c00;
  outline-offset:2px;
}

/* --- Lignes critères (vue détail) --- */
.crit-section { margin-top: 8px; }
.crit-row {
  display:flex; align-items:flex-start; justify-content:space-between;
  gap:12px; padding:8px 12px; border-bottom:1px solid #eee;
}
.crit-text { flex:1; }

/* --- Petits badges TRUE/FALSE/UNKNOWN --- */
.small-badge {
  font-weight:700; padding:2px 10px; border-radius:8px; display:inline-block;
  min-width:100px; text-align:center;
}
.badge-true    { background:#DFF6DD; color:#0B6B00; }
.badge-false   { background:#FDE7E9; color:#A80000; }
.badge-unknown { background:#ECEFF3; color:#3b3b3b; }
</style>
""", unsafe_allow_html=True)

# =========================
# UI helpers
# =========================
def eligibility_badge(label: str) -> str:
    up = "ELIGIBLE" if str(label).lower().startswith("eligible") else "NON ELIGIBLE"
    klass = "badge-eligible" if up == "ELIGIBLE" else "badge-not"
    return f'<span class="badge {klass}">{up}</span>'

def truth_badge(status) -> str:
    if status is True:
        label, klass = "TRUE", "badge-true"
    elif status is False:
        label, klass = "FALSE", "badge-false"
    else:
        label, klass = "UNKNOWN", "badge-unknown"
    return f'<span class="small-badge {klass}">{label}</span>'

# =========================
# criteria_matches helpers
# =========================
@st.cache_data(show_spinner=False)
def index_matches_for_patient(patient_slug: str) -> Dict[str, Any]:
    """
    Index {crit_text_strict: json_obj} pour fichiers:
      critere_*__vs__{patient_slug}__matched*.json
      critere_*_vs_{patient_slug}_matched*.json
    """
    index: Dict[str, Any] = {}
    if not CRITERIA_MATCHES_DIR.exists():
        return index
    patterns: List[str] = [
        f"critere_*__vs__{patient_slug}__matched*.json",
        f"critere_*_vs_{patient_slug}_matched*.json",
    ]
    files: List[str] = []
    for pat in patterns:
        files.extend(glob.glob(str(CRITERIA_MATCHES_DIR / pat)))
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            crit_text = str(data.get("text", "")).strip()
            if crit_text:
                index.setdefault(crit_text, data)
        except Exception:
            continue
    return index

def find_match_json_for_criterion(crit_text: str, patient_id: str) -> Optional[Dict[str, Any]]:
    patient_slug = _patient_slug_from_id(patient_id)
    idx = index_matches_for_patient(patient_slug)
    j = idx.get(str(crit_text).strip())
    if j:
        return j
    # fallback normalisé
    crit_norm = " ".join(str(crit_text).split())
    for k, v in idx.items():
        if " ".join(k.split()) == crit_norm:
            return v
    return None

# =========================
# Surlignage des CRITERES (via entities de criteria_matches)
# =========================
ENTITY_LABEL_CLASS: Dict[str, str] = {
    "Condition":   "ent-Condition",
    "Procedure":   "ent-Procedure",
    "Drug":        "ent-Drug",
    "Measurement": "ent-Measurement",
    "Value":       "ent-Value",
    "Unit":        "ent-Unit",
    "Temporal":    "ent-Temporal",
    "Duration":    "ent-Duration",
    "Age":         "ent-Age",
    "Sex":         "ent-Sex",
    "Comparator":  "ent-Comparator",
    "Qualifier":   "ent-Qualifier",
    "LabTest":     "ent-LabTest",
    "Biomarker":   "ent-Biomarker",
    "Stage":       "ent-Stage",
    "ECOG":        "ent-ECOG",
    "Metastasis":  "ent-Metastasis",
    "Gene":        "ent-Gene",
    "Variant":     "ent-Variant",
    "Device":      "ent-Device",
    "Imaging":     "ent-Imaging",
    "Score":       "ent-Score",
}
def label_to_class(label: str) -> str:
    return ENTITY_LABEL_CLASS.get(str(label), "ent-default")

def build_highlighted_html_from_entities(crit_text: str, match_json: Dict[str, Any]) -> str:
    """
    Surbrille les entités du critère à partir de match_json['entities'] (offset/offsets).
    """
    if not match_json:
        return html.escape(crit_text)
    src = match_json.get("text", "")
    text = crit_text
    spans: List[Tuple[int, int, str]] = []
    for ent in match_json.get("entities", []):
        label = ent.get("label", "")
        # offsets robustes
        pairs: List[Tuple[int, int]] = []
        off = ent.get("offset")
        offs = ent.get("offsets")
        if isinstance(off, list) and len(off) == 2 and all(isinstance(x, int) for x in off):
            pairs.append((off[0], off[1]))
        if isinstance(offs, list):
            for o in offs:
                if isinstance(o, list) and len(o) == 2 and all(isinstance(x, int) for x in o):
                    pairs.append((o[0], o[1]))
        for s, e in pairs:
            if 0 <= s < e <= len(src):
                spans.append((s, e, label))
    if not spans:
        return html.escape(text)
    # choisir spans non chevauchants (longueur d'abord)
    spans.sort(key=lambda t: (t[1]-t[0]), reverse=True)
    chosen: List[Tuple[int, int, str]] = []
    def overlaps(a,b): return not (a[1] <= b[0] or b[1] <= a[0])
    for s, e, label in spans:
        if all(not overlaps((s, e), (cs, ce)) for cs, ce, _ in chosen):
            chosen.append((s, e, label))
    chosen.sort(key=lambda t: t[0])
    out: List[str] = []
    cur = 0
    for s, e, label in chosen:
        s = max(0, min(s, len(text))); e = max(0, min(e, len(text)))
        if cur < s:
            out.append(html.escape(text[cur:s]))
        out.append(f'<span class="ent {label_to_class(label)}">{html.escape(text[s:e])}</span>')
        cur = e
    if cur < len(text):
        out.append(html.escape(text[cur:]))
    return "".join(out)

# =========================
# PATIENT CONTEXT — lecture cases_clean + mapping couleurs via entity_match
# =========================
def _iter_case_entities(case_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("entities", "Entities", "annotations", "mentions"):
        if isinstance(case_json.get(key), list):
            return case_json[key]
    return []

def _pairs_from_any(obj) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    if obj is None:
        return pairs
    if isinstance(obj, list) and len(obj) == 2 and all(isinstance(x, int) for x in obj):
        return [(obj[0], obj[1])]
    if isinstance(obj, list):
        for it in obj:
            pairs.extend(_pairs_from_any(it))
        return pairs
    if isinstance(obj, dict):
        for ks in (("start", "end"), ("begin", "end"), ("from", "to"), ("start_char", "end_char")):
            if ks[0] in obj and ks[1] in obj:
                s, e = obj[ks[0]], obj[ks[1]]
                if isinstance(s, int) and isinstance(e, int):
                    pairs.append((s, e))
                    return pairs
        return pairs
    if isinstance(obj, str):
        m = re.match(r"^\s*(\d+)\s*[-:]\s*(\d+)\s*$", obj)
        if m:
            pairs.append((int(m.group(1)), int(m.group(2))))
        return pairs
    return pairs

def _iter_entity_offsets(ent: Dict[str, Any]) -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for key in ("offset", "offsets", "spans", "locations", "loc"):
        if key in ent:
            pairs.extend(_pairs_from_any(ent[key]))
    if not pairs:
        if isinstance(ent.get("start"), int) and isinstance(ent.get("end"), int):
            pairs.append((ent["start"], ent["end"]))
        elif isinstance(ent.get("begin"), int) and isinstance(ent.get("end"), int):
            pairs.append((ent["begin"], ent["end"]))
    return pairs

def _candidate_patient_texts(case_json: Dict[str, Any]) -> Dict[str, str]:
    cand: Dict[str, str] = {}
    for key in ("text", "Text", "raw_text", "clean_text", "case_text", "document", "source_text", "content", "fulltext", "full_text"):
        val = case_json.get(key)
        if isinstance(val, str) and val:
            cand[key] = val
    return cand

def _norm(s: str) -> str:
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def _select_best_text_field(case_json: Dict[str, Any], ents: List[Dict[str, Any]]) -> Tuple[str, str]:
    cands = _candidate_patient_texts(case_json)
    if not cands:
        return ("", "")
    best_key, best_txt, best_score = "", "", -1
    for key, txt in cands.items():
        score = 0
        for ent in ents:
            ent_txt = ent.get("text") or ent.get("mention") or ""
            if not ent_txt:
                continue
            for s, e in _iter_entity_offsets(ent):
                if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(txt):
                    if _norm(txt[s:e]) == _norm(ent_txt):
                        score += 1
                        break
        if score > best_score:
            best_key, best_txt, best_score = key, txt, score
    if not best_key:
        if "text" in cands: return ("text", cands["text"])
        k0 = next(iter(cands.keys()))
        return (k0, cands[k0])
    return (best_key, best_txt)

def _normalize_entity_match(val: Any) -> str:
    """
    Normalise 'entity_match' d'une entité critère -> 'pos'|'neg'|'unk'
    """
    if val is None:
        return "unk"
    if isinstance(val, bool):
        return "pos" if val else "neg"
    if isinstance(val, (int, float)):
        return "pos" if val > 0 else "neg"
    s = str(val).strip().lower()
    if "full" in s or "partial" in s:
        return "pos"
    if "no_match" in s or "no match" in s or s == "nomatch":
        return "neg"
    if s in ("match", "matched", "true", "yes", "1", "positive", "pos", "ok", "present"):
        return "pos"
    if s in ("false", "no", "0", "negative", "neg", "absent"):
        return "neg"
    return "unk"

def _collect_patient_entity_status_from_entity_match(patient_id: str,
                                                     criteria_texts: List[str]) -> Dict[str, str]:
    """
    Construit { patient_entity_id : 'pos'|'neg'|'unk' } à partir de logical_entities
    des fichiers criteria_matches de TOUS les critères de l'essai (agrégation).
    En cas de conflit: pos > neg > unk.
    """
    out: Dict[str, str] = {}
    priority = {"pos": 3, "neg": 2, "unk": 1, "": 0}

    def apply(ids: Any, entity_match_value: Any):
        if not isinstance(ids, list) or not ids:
            return
        status = _normalize_entity_match(entity_match_value)
        for _id in ids:
            sid = str(_id)
            prev = out.get(sid, "")
            if priority.get(status, 0) > priority.get(prev, 0):
                out[sid] = status

    for crit in criteria_texts:
        mj = find_match_json_for_criterion(crit, patient_id)
        if not mj:
            continue
        for group in mj.get("logical_entities", []):
            g_em = group.get("entity_match")
            if isinstance(group.get("matching_ids"), list):
                apply(group.get("matching_ids"), g_em)
            for m in group.get("members", []):
                em = m.get("entity_match", g_em)
                for key in ("matching_ids", "patient_entity_ids", "case_entity_ids", "case_ids"):
                    ids = m.get(key)
                    if isinstance(ids, list) and ids:
                        apply(ids, em)
                single = m.get("patient_entity_id") or m.get("case_entity_id")
                if single is not None:
                    apply([single], em)

    return out

def _find_patient_case_file(patient_id: str) -> Optional[str]:
    """Essaie plusieurs motifs pour trouver le fichier JSON du patient dans cases_clean."""
    slug   = _patient_slug_from_id(patient_id)    # ex: patient_0001
    simple = _patient_simple_from_id(patient_id)  # ex: patient 1
    candidates: List[Path] = [CASES_CLEAN_DIR / f"{slug}.json"]
    candidates += [Path(p) for p in glob.glob(str(CASES_CLEAN_DIR / f"*{slug}*.json"))]
    candidates += [Path(p) for p in glob.glob(str(CASES_CLEAN_DIR / f"*{simple}*.json"))]
    for p in candidates:
        if p.exists():
            return str(p)
    return None

def _load_patient_case(patient_id: str) -> Optional[Dict[str, Any]]:
    fp = _find_patient_case_file(patient_id)
    if not fp:
        return None
    try:
        with open(fp, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _extract_spans_from_patient_case(case_json: Dict[str, Any],
                                     id_status: Dict[str, str],
                                     include_others: bool = True) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Construit les spans du texte patient.
    - id_status: {entity_id: 'pos'|'neg'|'unk'} pour IDs référencés (colorés)
    - include_others=True: ajoute toutes les autres entités en 'unk' (gris).
    """
    if not case_json:
        return "", []
    ents = _iter_case_entities(case_json)
    _field, text = _select_best_text_field(case_json, ents)
    if not text:
        text = case_json.get("text") or case_json.get("Text") or ""
    spans: List[Tuple[int, int, str]] = []
    for ent in ents:
        ent_id = str(ent.get("id") or ent.get("ent_id") or ent.get("uid") or ent.get("entity_id") or "")
        status = id_status.get(ent_id)
        if status is None:
            if not include_others:
                continue
            status = "unk"  # entité non mentionnée -> gris
        for s, e in _iter_entity_offsets(ent):
            if isinstance(s, int) and isinstance(e, int) and 0 <= s < e <= len(text):
                spans.append((s, e, status))
    return text, spans

def _render_highlighted_patient_text(text: str,
                                     spans: List[Tuple[int, int, str]]) -> str:
    """
    Coloration: pos -> vert, neg -> rouge, unk -> gris.
    Priorité: pos > neg > gris, puis longueur décroissante, puis ordre.
    """
    if not text or not spans:
        return html.escape(text)
    priority = {"pos": 3, "neg": 2, "unk": 1}
    spans.sort(key=lambda t: (priority.get(t[2], 0), (t[1]-t[0])), reverse=True)
    chosen: List[Tuple[int, int, str]] = []
    def overlaps(a,b): return not (a[1] <= b[0] or b[1] <= a[0])
    for s, e, status in spans:
        if all(not overlaps((s, e), (cs, ce)) for cs, ce, _ in chosen):
            chosen.append((s, e, status))
    chosen.sort(key=lambda t: t[0])
    out: List[str] = []
    cur = 0
    for s, e, status in chosen:
        s = max(0, min(s, len(text))); e = max(0, min(e, len(text)))
        if cur < s:
            out.append(html.escape(text[cur:s]))
        cls = ("ent-patient-pos" if status == "pos"
               else "ent-patient-neg" if status == "neg"
               else "ent-patient-unk")
        out.append(f'<span class="{cls}">{html.escape(text[s:e])}</span>')
        cur = e
    if cur < len(text):
        out.append(html.escape(text[cur:]))
    return "".join(out)

# =========================
# LISTING essais pour un patient/méthode
# =========================
def build_scores_for_patient(method: str, selected_patient_row: pd.Series) -> Tuple[pd.DataFrame, str, str]:
    """
    Retourne (df trié, score_col, title_key) avec colonnes: nct_id, score_col, eligible, BriefTitle
    """
    mode = normalize_method(method)
    patient_id    = selected_patient_row["Id streamlit"]
    patient_slug  = _patient_slug_from_id(patient_id)    # 'patient_0001'
    patient_label = _patient_simple_from_id(patient_id)  # 'patient 1'

    if mode == "APS1":
        scores_aps1 = load_table_any("trial_eligibility_scores_APS1")
        col = patient_slug  # 'patient_000X'
        if "nct_id" not in scores_aps1.columns or col not in scores_aps1.columns:
            st.error(f"❌ Colonnes manquantes dans trial_eligibility_scores_APS1 (nct_id, {col}).")
            st.stop()
        base = scores_aps1[["nct_id", col]].copy()
        base["eligible"] = base[col].apply(lambda x: "Eligible" if pd.to_numeric(x, errors="coerce") > 0 else "Not eligible")
        score_col = col
    else:
        # APS-3 GPT4 — listing basé sur 'patient X_match'
        scores_gpt4 = load_table_any("trial_eligibility_gpt4")
        col = f"{patient_label}_match"          # 'patient 1_match'
        alt_col = col.replace(" ", "_")         # fallback 'patient_1_match'
        if "nct_id" not in scores_gpt4.columns:
            st.error("❌ Colonne 'nct_id' manquante dans trial_eligibility_gpt4.")
            st.stop()
        if col not in scores_gpt4.columns and alt_col in scores_gpt4.columns:
            col = alt_col
        if col not in scores_gpt4.columns:
            st.error(f"❌ Colonne '{col}' absente dans trial_eligibility_gpt4.")
            st.stop()
        base = scores_gpt4[["nct_id", col]].copy()
        base["eligible"] = base[col].apply(lambda x: "Eligible" if pd.to_numeric(x, errors="coerce") > 0 else "Not eligible")
        score_col = col

    title_key_local = "BriefTitle" if "BriefTitle" in clinical_trials.columns else None
    trials_id_col   = "NCTId" if "NCTId" in clinical_trials.columns else None

    base = base.sort_values(by=score_col, ascending=False)
    if trials_id_col:
        if title_key_local:
            base = base.merge(
                clinical_trials[[trials_id_col, title_key_local]],
                left_on="nct_id", right_on=trials_id_col, how="left"
            )
            base[title_key_local] = base[title_key_local].fillna("Title not found")
        else:
            base = base.merge(
                clinical_trials[[trials_id_col]],
                left_on="nct_id", right_on=trials_id_col, how="left"
            )
            base["BriefTitle"] = "Title not found"
            title_key_local = "BriefTitle"
    else:
        base["BriefTitle"] = "Title not found"
        title_key_local = "BriefTitle"

    return base.reset_index(drop=True), score_col, title_key_local

# =========================
# DETAIL (critères + patient context)
# =========================
def render_criteria_detail(method: str, selected_patient_row: pd.Series, nct_id: str) -> None:
    mode = normalize_method(method)

    # Sécurité colonnes
    if "nct_id" not in criteria_df.columns or "critere_type" not in criteria_df.columns or "critere" not in criteria_df.columns:
        st.error("❌ Colonnes attendues manquantes dans criteria.csv (attendu: 'nct_id', 'critere_type', 'critere').")
        st.stop()

    # Critères de l’essai
    trial_criteria = criteria_df[criteria_df["nct_id"] == nct_id]
    if trial_criteria.empty:
        st.info("Aucun critère trouvé pour cet essai.")
        return

    patient_id_for_match = selected_patient_row["Id streamlit"]

    # ---------- PATIENT CONTEXT ----------
    if mode == "APS1":
        # agrège les matching_ids de TOUS les critères de CET essai
        crit_texts_for_trial = [str(x).strip() for x in trial_criteria["critere"].astype(str).tolist()]
        id_status = _collect_patient_entity_status_from_entity_match(patient_id_for_match, crit_texts_for_trial)
        case_json = _load_patient_case(patient_id_for_match)
        patient_text, patient_spans = _extract_spans_from_patient_case(case_json, id_status, include_others=True)

        st.markdown("### Patient context (matched entities)")
        if patient_text:
            patient_html = _render_highlighted_patient_text(patient_text, patient_spans)
            st.markdown(
                f'<div style="padding:12px 14px;border:1px solid #eee;border-radius:10px;background:#fafafa;">{patient_html}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("Texte du patient non trouvé ou aucun ID correspondant dans cases_clean.")
        st.markdown("---")
    else:
        # APS-3 GPT4 : aucun surlignage du patient
        case_json = _load_patient_case(patient_id_for_match)
        raw_text = ""
        if case_json:
            raw_text = (case_json.get("text") or case_json.get("Text")
                        or case_json.get("raw_text") or case_json.get("clean_text") or "")
        if not raw_text:
            raw_text = str(selected_patient_row.get("Text", "")) if pd.notna(selected_patient_row.get("Text", "")) else ""
        st.markdown("### Patient context")
        if raw_text:
            st.markdown(
                f'<div style="padding:12px 14px;border:1px solid #eee;border-radius:10px;background:#fafafa;">{html.escape(raw_text)}</div>',
                unsafe_allow_html=True
            )
        else:
            st.info("Texte du patient indisponible.")
        st.markdown("---")

    # ---------- CALCUL DES STATUTS PAR CRITÈRE ----------
    results: Dict[str, List[Any]] = {"inclusion": [], "exclusion": []}
    patient_slug  = _patient_slug_from_id(selected_patient_row["Id streamlit"])   # ex: patient_0001
    patient_label = _patient_simple_from_id(selected_patient_row["Id streamlit"]) # ex: patient 1

    if mode == "APS1":
        criteria_patient_df = load_csv("criteria_patient_matrix.csv")
        if "criterion_text" not in criteria_patient_df.columns:
            st.error("Colonne 'criterion_text' absente dans criteria_patient_matrix.csv")
            st.stop()

    for crit_type in ["inclusion", "exclusion"]:
        crits = trial_criteria[trial_criteria["critere_type"] == crit_type]
        for _, crit in crits.iterrows():
            crit_text = str(crit["critere"]).strip()

            if mode == "APS1":
                rows = criteria_patient_df.loc[
                    criteria_patient_df["criterion_text"].astype(str).str.strip() == crit_text
                ]
                if not rows.empty:
                    val = rows.iloc[0].get(patient_slug, None)  # 'patient_000X'
                    if isinstance(val, str):
                        v = val.lower().strip()
                        if v in ("true", "1", "yes"):
                            status = True
                        elif v in ("false", "0", "no"):
                            status = False
                        else:
                            status = None
                    else:
                        status = bool(val) if pd.notna(val) else None
                else:
                    status = None
                # tuple à 2 éléments pour APS-1
                results[crit_type].append((crit_text, status))

            else:
                # APS-3 GPT4 -> eligibility_results_openAI_interface : 'patient X_match' / 'patient X_reason'
                match_col_space  = f"{patient_label}_match"   # 'patient 1_match'
                reason_col_space = f"{patient_label}_reason"  # 'patient 1_reason'
                match_col  = match_col_space  if match_col_space  in eligibility_df.columns else match_col_space.replace(" ", "_")
                reason_col = reason_col_space if reason_col_space in eligibility_df.columns else reason_col_space.replace(" ", "_")

                if match_col not in eligibility_df.columns:
                    st.error(f"Colonne '{match_col_space}' absente (et fallback '{match_col_space.replace(' ', '_')}' aussi) dans eligibility_results_openAI_interface.csv")
                    st.stop()

                row_ = eligibility_df.loc[eligibility_df["critere"].astype(str).str.strip() == crit_text]
                if not row_.empty:
                    score = pd.to_numeric(row_.iloc[0][match_col], errors="coerce")
                    status = True if pd.notna(score) and score > 0 else (False if pd.notna(score) else None)
                    explanation = (
                        row_.iloc[0][reason_col] if reason_col in eligibility_df.columns and pd.notna(row_.iloc[0][reason_col])
                        else "No explanation available"
                    )
                else:
                    status, explanation = None, "No explanation available"
                # tuple à 3 éléments pour APS-3
                results[crit_type].append((crit_text, status, explanation))

    # ---------- RENDU DES CRITÈRES ----------
    def render_section(title: str, items: List[Any]):
        st.markdown(title)
        st.markdown('<div class="crit-section">', unsafe_allow_html=True)
        for item in items:
            if mode == "APS1":
                # tolérant 2/3 éléments
                crit = item[0]
                status = item[1] if len(item) > 1 else None
                badge_html = truth_badge(status)
                # APS-1 : surlignage des critères
                match_json = find_match_json_for_criterion(crit, patient_id_for_match)
                crit_html  = build_highlighted_criterion_html(crit, match_json)  # couleur selon entity_match

                st.markdown(
                    f'''
                    <div class="crit-row">
                      <div class="crit-text">{crit_html}</div>
                      <div>{badge_html}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
            else:
                # tolérant 2/3 éléments
                crit = item[0]
                status = item[1] if len(item) > 1 else None
                explanation = item[2] if len(item) > 2 else "No explanation available"
                badge_html = truth_badge(status)
                # APS-3 : PAS de surlignage des critères
                crit_html = html.escape(crit)
                st.markdown(
                    f'''
                    <div class="crit-row">
                      <div class="crit-text">{crit_html}</div>
                      <div>{badge_html}</div>
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
                with st.expander("Explanation", expanded=False):
                    st.write(explanation)
        st.markdown('</div>', unsafe_allow_html=True)

    render_section("### Inclusion Criteria", results["inclusion"])
    render_section("### Exclusion Criteria", results["exclusion"])

# =========================
# ROUTER via query params
# =========================
params = st.query_params

def get_patient_options(df: pd.DataFrame) -> List[str]:
    return (df["Id streamlit"] + " - " + df["pathologie"]).tolist()

def get_patient_row_from_label(df: pd.DataFrame, label: str) -> pd.Series:
    return df.loc[(df["Id streamlit"] + " - " + df["pathologie"]) == label].iloc[0]

# =========================
# VUES
# =========================
view = params.get("view", "home")

if view == "home":
    st.subheader("Select patient and method")

    if "Id streamlit" not in patients_df.columns or "pathologie" not in patients_df.columns:
        st.error("❌ Colonnes attendues manquantes dans patients.csv (attendu : 'Id streamlit', 'pathologie').")
        st.stop()

    # Sélection patient
    patient_label = st.selectbox("Patient", options=get_patient_options(patients_df), index=0)

    # Ligne du patient
    try:
        selected_patient_row = get_patient_row_from_label(patients_df, patient_label)
    except Exception:
        st.error("Patient sélectionné introuvable dans patients.csv.")
        st.stop()

    # Texte clinique brut (page d'accueil)
    with st.expander("📋 Patient Report", expanded=False):
        st.markdown(f"**Patient ID :** {selected_patient_row['Id streamlit']}")
        st.markdown(f"**Pathology :** {selected_patient_row['pathologie']}")
        if "Text" in selected_patient_row and pd.notna(selected_patient_row["Text"]):
            st.write(selected_patient_row["Text"])
        else:
            st.info("ℹ️ Colonne 'Text' absente ou vide pour ce patient.")

    # Choix méthode
    method = st.radio("Choose the approach:", ["APS-3 GPT4", "APS-1"])

    # Bouton -> redirection vers la LISTE
    if st.button("Run"):
        st.query_params.update({
            "view": "list",
            "patient": selected_patient_row["Id streamlit"],
            "method": method
        })
        st.rerun()

elif view == "list":
    patient_id = params.get("patient")
    method = params.get("method", "APS-3 GPT4")
    if not patient_id:
        st.warning("Aucun patient sélectionné. Retour à l'accueil.")
        st.markdown('<a class="btn" href="./?view=home" target="_self">Go to home</a>', unsafe_allow_html=True)
        st.stop()

    try:
        selected_patient_row = patients_df.loc[patients_df["Id streamlit"] == patient_id].iloc[0]
    except Exception:
        st.error(f"Patient '{patient_id}' introuvable.")
        st.stop()

    st.subheader(f"Clinical Trials for {selected_patient_row['Id streamlit']} — {method}")

    # Bouton retour en haut
    st.markdown(
        '<div style="margin: 8px 0 16px 0;"><a class="btn" href="./?view=home" target="_self">← Back to home</a></div>',
        unsafe_allow_html=True
    )

    df_display, score_col, title_key = build_scores_for_patient(method, selected_patient_row)

    for _, row in df_display.iterrows():
        status_badge = eligibility_badge(row["eligible"])
        st.markdown(
            f"""
            <div class="trial-card">
              <div class="trial-card-header">
                <div class="trial-card-title">
                  {row[title_key]} ({row["nct_id"]}) &nbsp; {status_badge}
                </div>
                <div>
                  <a class="btn" href="./?view=detail&patient={selected_patient_row['Id streamlit']}&method={method}&trial={row['nct_id']}" target="_self">See More</a>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Bouton retour en bas (optionnel)
    st.markdown('<a class="btn" href="./?view=home" target="_self">Back to home</a>', unsafe_allow_html=True)

elif view == "detail":
    patient_id = params.get("patient")
    method     = params.get("method", "APS-3 GPT4")
    trial_id   = params.get("trial")
    if not (patient_id and trial_id):
        st.warning("Informations insuffisantes pour afficher le détail.")
        st.markdown('<a class="btn" href="./?view=home" target="_self">Back to home</a>', unsafe_allow_html=True)
        st.stop()

    try:
        selected_patient_row = patients_df.loc[patients_df["Id streamlit"] == patient_id].iloc[0]
    except Exception:
        st.error(f"Patient '{patient_id}' introuvable.")
        st.stop()

    df_display, score_col, title_key = build_scores_for_patient(method, selected_patient_row)
    trial_row = df_display.loc[df_display["nct_id"] == trial_id]
    if trial_row.empty:
        st.error(f"Essai {trial_id} introuvable pour ce patient.")
        st.stop()
    trial_title    = trial_row.iloc[0][title_key]
    eligible_label = trial_row.iloc[0]["eligible"]

    st.subheader(f"🔎 {trial_title} ({trial_id}) — {method}")
    st.markdown(f"**Eligibility:** {eligibility_badge(eligible_label)}", unsafe_allow_html=True)
    st.markdown(f"**Patient ID:** {selected_patient_row['Id streamlit']}")
    st.markdown(f"**Pathology:** {selected_patient_row['pathologie']}")
    st.markdown("**Overall Status:** Ongoing")
    st.markdown("---")

    # Détails critères pour ce couple (patient, essai)
    render_criteria_detail(method, selected_patient_row, trial_id)

    st.markdown(
        f'<a class="btn" href="./?view=list&patient={selected_patient_row["Id streamlit"]}&method={method}" target="_self">Back to list</a>',
        unsafe_allow_html=True
    )

else:
    st.markdown('<a class="btn" href="./?view=home" target="_self">Back to home</a>', unsafe_allow_html=True)
