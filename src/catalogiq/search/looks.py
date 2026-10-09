"""Shop the look: complementary items from OTHER categories, chosen by simple explainable rules.

Given one item (a catalog id, or a photo plus an attribute predictor), recommend items from the
roles that complete an outfit (a top -> bottoms, footwear, accessories, bags). Every candidate gets
a score that is a weighted sum of five easy-to-read components, and each recommendation carries
its reasons.

Hard rules (a candidate failing one is never shown):
  1. different role from the query and listed as a complement of it (``role_complements``);
  2. gender-compatible: same gender or Unisex (adult Unisex queries accept Men, Women and Unisex);
  3. shares a style family with the query (casual, formal, sporty, ethnic, party;
     ``article_styles``), e.g. formal trousers are never paired with sports shoes;
  4. its catalog label is confirmed by the Phase 2 classifier (optional ``verified`` mask), so
     plainly mislabelled catalog photos never appear in a look;
  5. not the query item itself and at most one item per near-duplicate group.

Soft score components (each in 0..1; weights in ``configs/search.yaml``):
  * pairing 1 if the two article types share a style family, scaled up to 1 when they share several;
  * usage   1 if same usage (Casual, Formal...), 0.5 if either is unknown/"Other", else 0;
  * season  1 same season, 0.5 adjacent season or unknown, else 0;
  * colour  1 if either colour is a neutral (black, white, grey, beige, brown) or the pair is in
            the curated pairing list, 0.7 for the same colour (tonal), 0.3 if a "Multi" print is
            involved, 0.5 if unknown, else 0.2;
  * style   CLIP image similarity between the query and the candidate, min-max scaled within the
            remaining candidates of that role, so it is a *relative* fit among sensible options
            (captures the overall "vibe" that the rules above cannot).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from catalogiq.config import get_search_config
from catalogiq.search.engine import SearchEngine
from catalogiq.search.index import l2_normalise

UNKNOWN = {"Other", "NA", "nan", "None", ""}
ADULT_GENDERS = {"Men", "Women", "Unisex"}

AttributeFn = Callable[[Any], dict[str, str]]


@dataclass
class Look:
    """Result of ``shop_the_look``: the query's attributes and the recommended items."""

    query: dict[str, Any]
    items: pd.DataFrame


def attributes_from_predictions(predictions: dict[str, tuple[str, float, bool]]) -> dict[str, str]:
    """Turn ``catalogiq.vision.predict.predict`` output into plain ``{attribute: label}``."""
    return {attr: label for attr, (label, _conf, _review) in predictions.items()}


def _is_unknown(value: Any) -> bool:
    return value is None or str(value) in UNKNOWN


class ShopTheLook:
    """Rule-based outfit completion on top of a :class:`SearchEngine`.

    Args:
        engine: Search engine (provides the catalog, embeddings and optionally an encoder).
        attribute_fn: Optional ``image -> {attribute: label}`` used for photo queries (typically the
            Phase 2 classifier). Not needed for catalog-id queries.
        verified: Optional boolean array aligned with ``engine.catalog``: True where the catalog
            label is confirmed by the classifier (see ``search.labelcheck``). Used when
            ``shop_the_look.require_label_verified`` is true.
    """

    def __init__(
        self,
        engine: SearchEngine,
        attribute_fn: AttributeFn | None = None,
        verified: np.ndarray | None = None,
    ) -> None:
        cfg = get_search_config()["shop_the_look"]
        self.cfg = cfg
        self.engine = engine
        self.attribute_fn = attribute_fn
        self.verified = verified
        self.article_styles: dict[str, set[str]] = {}
        for style, articles in cfg["article_styles"].items():
            for article in articles:
                self.article_styles.setdefault(article, set()).add(style)
        self.article_to_role = {
            a: role for role, articles in cfg["article_roles"].items() for a in articles
        }
        self.neutral = set(cfg["neutral_colours"])
        self.pairs = {frozenset(p) for p in cfg["colour_pairs"]}
        self.adjacent = {frozenset(p) for p in cfg["adjacent_seasons"]}
        cat = engine.catalog
        self.roles = cat["articleType"].map(self.article_to_role)

    # ---------------------------------------------------------------- score components
    def shared_styles(self, query_article: str, candidate_article: str) -> set[str]:
        """Style families both article types belong to."""
        return self.article_styles.get(query_article, set()) & self.article_styles.get(
            candidate_article, set()
        )

    def pairing_score(self, query_article: str, candidate_articles: pd.Series) -> pd.Series:
        """1.0 if the types share at least two style families, 0.7 for one, 0 for none.

        An article type that is not in the style table scores 0.5 against everything (unknown).
        """
        if query_article not in self.article_styles:
            return pd.Series(0.5, index=candidate_articles.index)
        lookup = {
            a: {0: 0.0, 1: 0.7}.get(len(self.shared_styles(query_article, a)), 1.0)
            for a in candidate_articles.astype(str).unique()
        }
        return candidate_articles.astype(str).map(lookup)

    def colour_score(self, query_colour: str, candidate_colours: pd.Series) -> pd.Series:
        """Colour compatibility of each candidate with the query colour (see module docstring)."""

        def one(c: str) -> float:
            if _is_unknown(query_colour) or _is_unknown(c):
                return 0.5
            if query_colour in self.neutral or c in self.neutral:
                return 1.0
            if frozenset((query_colour, c)) in self.pairs:
                return 1.0
            if query_colour == c:
                return 0.7
            if "Multi" in (query_colour, c):
                return 0.3
            return 0.2

        lookup = {c: one(c) for c in candidate_colours.astype(str).unique()}
        return candidate_colours.astype(str).map(lookup)

    def season_score(self, query_season: str, candidate_seasons: pd.Series) -> pd.Series:
        """1 for the same season, 0.5 for an adjacent or unknown season, 0 otherwise."""

        def one(s: str) -> float:
            if _is_unknown(query_season) or _is_unknown(s):
                return 0.5
            if s == query_season:
                return 1.0
            return 0.5 if frozenset((query_season, s)) in self.adjacent else 0.0

        lookup = {s: one(s) for s in candidate_seasons.astype(str).unique()}
        return candidate_seasons.astype(str).map(lookup)

    def usage_score(self, query_usage: str, candidate_usages: pd.Series) -> pd.Series:
        """1 for the same usage, 0.5 if either is unknown, else 0."""

        def one(u: str) -> float:
            if _is_unknown(query_usage) or _is_unknown(u):
                return 0.5
            return 1.0 if u == query_usage else 0.0

        lookup = {u: one(u) for u in candidate_usages.astype(str).unique()}
        return candidate_usages.astype(str).map(lookup)

    @staticmethod
    def allowed_genders(gender: str) -> set[str]:
        """Genders a recommended item may have for a query of the given gender."""
        if gender == "Unisex":
            return set(ADULT_GENDERS)
        if _is_unknown(gender):
            return set(ADULT_GENDERS) | {"Boys", "Girls"}
        return {gender, "Unisex"}

    # ---------------------------------------------------------------- main entry point
    def _query_from(self, item: Any) -> tuple[dict[str, Any], np.ndarray, int | None]:
        if isinstance(item, (int, np.integer)):
            pos = self.engine.id_to_pos[int(item)]
            row = self.engine.catalog.iloc[pos]
            attrs = {
                k: row.get(k) for k in ("articleType", "baseColour", "gender", "usage", "season")
            }
            attrs["id"] = int(item)
            return attrs, self.engine.emb[pos], int(item)
        if self.attribute_fn is None or self.engine.encoder is None:
            raise RuntimeError("photo queries need an attribute_fn and an encoder")
        attrs = dict(self.attribute_fn(item))
        vec = l2_normalise(self.engine.encoder.encode_image(item).reshape(1, -1))[0]
        return attrs, vec, None

    def recommend(self, item: Any, per_category: int | None = None) -> Look:
        """Recommend complementary items for a catalog id or a photo.

        Args:
            item: Catalog id (``int``) or an image (needs ``attribute_fn`` and an encoder).
            per_category: Items per complementary role (default ``shop_the_look.per_category``).

        Returns:
            ``Look`` with the query attributes (including its ``role``) and a DataFrame of
            recommendations with per-component scores and human-readable ``reasons``. The
            DataFrame is empty if the query's article type is not part of any outfit role.
        """
        n_per = per_category or self.cfg["per_category"]
        weights = self.cfg["weights"]
        query, qvec, query_id = self._query_from(item)
        role = self.article_to_role.get(str(query.get("articleType")))
        query["role"] = role
        if role is None:
            return Look(query, pd.DataFrame())

        cat = self.engine.catalog
        allowed = self.allowed_genders(str(query.get("gender")))
        base = cat["gender"].astype(str).isin(allowed) & self.roles.isin(
            self.cfg["role_complements"][role]
        )
        if query_id is not None:
            base &= cat["id"] != query_id
        if self.cfg.get("require_label_verified") and self.verified is not None:
            base &= pd.Series(self.verified, index=cat.index)
        query_article = str(query.get("articleType"))
        if query_article in self.article_styles:  # hard rule 3: must share a style family
            base &= cat["articleType"].map(
                lambda a: bool(self.shared_styles(query_article, str(a)))
            )
        pool_all = cat[base].copy()
        pool_all["role"] = self.roles[base]
        pool_all["_pos"] = np.where(base)[0]
        sims = self.engine.emb[pool_all["_pos"].to_numpy()] @ qvec

        picked: list[pd.DataFrame] = []
        for target_role in self.cfg["role_complements"][role]:
            mask = (pool_all["role"] == target_role).to_numpy()
            if not mask.any():
                continue
            pool = pool_all[mask].copy()
            s = sims[mask]
            style = (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else np.ones_like(s)
            pool["pairing_score"] = self.pairing_score(
                query_article, pool["articleType"]
            ).to_numpy()
            pool["usage_score"] = self.usage_score(
                str(query.get("usage")), pool["usage"]
            ).to_numpy()
            pool["season_score"] = self.season_score(
                str(query.get("season")), pool["season"]
            ).to_numpy()
            pool["colour_score"] = self.colour_score(
                str(query.get("baseColour")), pool["baseColour"]
            ).to_numpy()
            pool["style_score"] = style
            pool["score"] = (
                weights["pairing"] * pool["pairing_score"]
                + weights["usage"] * pool["usage_score"]
                + weights["season"] * pool["season_score"]
                + weights["colour"] * pool["colour_score"]
                + weights["style"] * pool["style_score"]
            )
            pool = pool.sort_values(["score", "id"], ascending=[False, True])
            diverse = target_role in self.cfg["diverse_roles"]
            chosen, seen_types, seen_groups = [], set(), set()
            for _, r in pool.iterrows():
                group = r.get("dup_group")
                if group in seen_groups or (diverse and r["articleType"] in seen_types):
                    continue
                chosen.append(r)
                seen_groups.add(group)
                seen_types.add(r["articleType"])
                if len(chosen) == n_per:
                    break
            if chosen:
                picked.append(pd.DataFrame(chosen))
        if not picked:
            return Look(query, pd.DataFrame())
        out = pd.concat(picked, ignore_index=True)
        out["rank_in_role"] = out.groupby("role").cumcount() + 1
        out["reasons"] = [self._reasons(query, r) for _, r in out.iterrows()]
        cols = [
            "role",
            "rank_in_role",
            "id",
            "score",
            "productDisplayName",
            "articleType",
            "baseColour",
            "gender",
            "usage",
            "season",
            "pairing_score",
            "usage_score",
            "season_score",
            "colour_score",
            "style_score",
            "reasons",
        ]
        return Look(query, out.reindex(columns=cols).reset_index(drop=True))

    def _reasons(self, query: dict[str, Any], r: pd.Series) -> str:
        parts = []
        shared = sorted(self.shared_styles(str(query.get("articleType")), str(r["articleType"])))
        if shared:
            parts.append("both " + "/".join(shared) + " style")
        if r["usage_score"] == 1.0:
            parts.append(f"same usage ({r['usage']})")
        elif r["usage_score"] == 0.0:
            parts.append(f"different usage ({query.get('usage')} vs {r['usage']})")
        if r["season_score"] == 1.0:
            parts.append(f"same season ({r['season']})")
        elif r["season_score"] == 0.0:
            parts.append(f"clashing season ({query.get('season')} vs {r['season']})")
        if r["colour_score"] >= 1.0:
            parts.append(f"colour works: {query.get('baseColour')} + {r['baseColour']}")
        elif r["colour_score"] == 0.7:
            parts.append(f"tonal match ({r['baseColour']})")
        elif r["colour_score"] <= 0.3:
            parts.append(f"colour clash risk: {query.get('baseColour')} + {r['baseColour']}")
        parts.append(f"visual style fit {r['style_score']:.2f}")
        return "; ".join(parts)


def shop_the_look(
    engine: SearchEngine,
    item: Any,
    attribute_fn: AttributeFn | None = None,
    per_category: int | None = None,
) -> Look:
    """Convenience wrapper: build a :class:`ShopTheLook` and recommend for one item."""
    return ShopTheLook(engine, attribute_fn).recommend(item, per_category)
