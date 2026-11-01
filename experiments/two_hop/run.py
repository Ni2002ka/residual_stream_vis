"""Two-hop recall: "Fact: the capital of the state containing Dallas is" → Austin.

Lindsey et al. (2025, "On the Biology of a Large Language Model") find Claude 3.5
Haiku routes this through Texas: Dallas → Texas → say Austin, plus a direct
Dallas → Austin shortcut, and swapping in California gives Sacramento. Here:

* **readout**: at the city token and the last token, each layer's rank of the
  state ("Texas"), the capital ("Austin") and a control state and capital from
  another item, under the logit lens and the J-lens. Is the state readable
  before the capital?
* **edits**, per pair (Dallas, Oakland), success = the new capital is top-1:
  - ``patch_city`` / ``patch_last``: that cell copied from the prompt naming the
    new city (the ceiling for editing that cell).
  - ``state@last`` / ``state@city``: swap the old state's direction for the new
    state's (Texas → California), with each lens's direction. At the last token
    this edits an *intermediate* concept, not the output word.
  - ``city@city``: swap the city itself (Dallas → Oakland), as edit_directions does.
  - ``random@last``: a random change as big as the J-lens state swap at the last token.

    python experiments/two_hop/run.py google/gemma-3-4b-it
    python experiments/two_hop/run.py gpt2 --pairs 3 --layers 4,8   # quick look
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

import residual_stream_vis as rsv
from residual_stream_vis.edit import _run, swap_vector

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "edit_directions"))
from run import Directions, random_unit  # noqa: E402
from diagnose import run_patches  # noqa: E402

TEMPLATES = {
    "us": "Fact: the capital of the state containing {city} is",
    "world": "Fact: the capital of the country containing {city} is",
}
# (city, state or country, capital). Items whose city or state isn't one token are dropped.
ITEMS = {
    "us": [
        ("Dallas", "Texas", "Austin"), ("Houston", "Texas", "Austin"), ("Chicago", "Illinois", "Springfield"),
        ("Seattle", "Washington", "Olympia"), ("Spokane", "Washington", "Olympia"), ("Tacoma", "Washington", "Olympia"),
        ("Miami", "Florida", "Tallahassee"), ("Orlando", "Florida", "Tallahassee"), ("Tampa", "Florida", "Tallahassee"),
        ("Jacksonville", "Florida", "Tallahassee"), ("Detroit", "Michigan", "Lansing"),
        ("Philadelphia", "Pennsylvania", "Harrisburg"), ("Pittsburgh", "Pennsylvania", "Harrisburg"),
        ("Cleveland", "Ohio", "Columbus"), ("Cincinnati", "Ohio", "Columbus"), ("Toledo", "Ohio", "Columbus"),
        ("Baltimore", "Maryland", "Annapolis"), ("Milwaukee", "Wisconsin", "Madison"),
        ("Memphis", "Tennessee", "Nashville"), ("Knoxville", "Tennessee", "Nashville"),
        ("Louisville", "Kentucky", "Frankfort"), ("Omaha", "Nebraska", "Lincoln"), ("Wichita", "Kansas", "Topeka"),
        ("Savannah", "Georgia", "Atlanta"), ("Anchorage", "Alaska", "Juneau"), ("Fairbanks", "Alaska", "Juneau"),
        ("Tucson", "Arizona", "Phoenix"), ("Fresno", "California", "Sacramento"),
        ("Oakland", "California", "Sacramento"), ("Billings", "Montana", "Helena"), ("Missoula", "Montana", "Helena"),
        ("Reno", "Nevada", "Carson City"), ("Shreveport", "Louisiana", "Baton Rouge"),
        ("Burlington", "Vermont", "Montpelier"), ("Bridgeport", "Connecticut", "Hartford"),
        ("Stamford", "Connecticut", "Hartford"), ("Worcester", "Massachusetts", "Boston"),
        ("Birmingham", "Alabama", "Montgomery"), ("Huntsville", "Alabama", "Montgomery"),
        ("Gulfport", "Mississippi", "Jackson"), ("Wilmington", "Delaware", "Dover"),
        ("Norfolk", "Virginia", "Richmond"), ("Provo", "Utah", "Salt Lake City"), ("Ogden", "Utah", "Salt Lake City"),
        ("Laramie", "Wyoming", "Cheyenne"), ("Casper", "Wyoming", "Cheyenne"), ("Davenport", "Iowa", "Des Moines"),
        ("Gary", "Indiana", "Indianapolis"), ("Hilo", "Hawaii", "Honolulu"), ("Eugene", "Oregon", "Salem"),
    ],
    "world": [
        ("Toronto", "Canada", "Ottawa"), ("Vancouver", "Canada", "Ottawa"), ("Montreal", "Canada", "Ottawa"),
        ("Sydney", "Australia", "Canberra"), ("Melbourne", "Australia", "Canberra"), ("Perth", "Australia", "Canberra"),
        ("Shanghai", "China", "Beijing"), ("Guangzhou", "China", "Beijing"), ("Shenzhen", "China", "Beijing"),
        ("Osaka", "Japan", "Tokyo"), ("Kyoto", "Japan", "Tokyo"), ("Istanbul", "Turkey", "Ankara"),
        ("Barcelona", "Spain", "Madrid"), ("Seville", "Spain", "Madrid"), ("Valencia", "Spain", "Madrid"),
        ("Milan", "Italy", "Rome"), ("Naples", "Italy", "Rome"), ("Venice", "Italy", "Rome"), ("Florence", "Italy", "Rome"),
        ("Munich", "Germany", "Berlin"), ("Hamburg", "Germany", "Berlin"), ("Frankfurt", "Germany", "Berlin"),
        ("Lyon", "France", "Paris"), ("Marseille", "France", "Paris"), ("Zurich", "Switzerland", "Bern"),
        ("Geneva", "Switzerland", "Bern"), ("Casablanca", "Morocco", "Rabat"), ("Marrakech", "Morocco", "Rabat"),
        ("Lagos", "Nigeria", "Abuja"), ("Krakow", "Poland", "Warsaw"), ("Porto", "Portugal", "Lisbon"),
        ("Glasgow", "Scotland", "Edinburgh"), ("Manchester", "England", "London"), ("Liverpool", "England", "London"),
        ("Karachi", "Pakistan", "Islamabad"), ("Lahore", "Pakistan", "Islamabad"), ("Almaty", "Kazakhstan", "Astana"),
        ("Alexandria", "Egypt", "Cairo"), ("Busan", "Korea", "Seoul"), ("Antwerp", "Belgium", "Brussels"),
        ("Rotterdam", "Netherlands", "Amsterdam"), ("Gothenburg", "Sweden", "Stockholm"), ("Bergen", "Norway", "Oslo"),
        ("Salzburg", "Austria", "Vienna"), ("Medellín", "Colombia", "Bogotá"), ("Cusco", "Peru", "Lima"),
        ("Córdoba", "Argentina", "Buenos Aires"), ("Mombasa", "Kenya", "Nairobi"), ("Thessaloniki", "Greece", "Athens"),
        ("Novosibirsk", "Russia", "Moscow"), ("Kharkiv", "Ukraine", "Kyiv"), ("Haifa", "Israel", "Jerusalem"),
        ("Chittagong", "Bangladesh", "Dhaka"),
    ],
}
STRENGTHS = [2.0, 4.0]


def one(tok, word: str) -> int | None:
    ids = tok(" " + word, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) == 1 else None


def build_items(tok, template: str) -> list[dict]:
    """Items with one-token city and state. The capital is scored by its first token, which must not be a state's."""
    out = []
    states = {s for _, s, _ in ITEMS[template]}
    state_ids = {one(tok, s) for s in states}
    for city, state, capital in ITEMS[template]:
        c, s = one(tok, city), one(tok, state)
        cap = tok(" " + capital, add_special_tokens=False)["input_ids"][0]
        if c is None or s is None or cap in state_ids:
            continue
        out.append(dict(template=template, city=city, state=state, capital=capital, city_id=c, state_id=s,
                        capital_id=cap, prompt=TEMPLATES[template].format(city=city)))
    return out


@torch.no_grad()
def ranks(model, lens, h: torch.Tensor, layer: int, ids: list[int]) -> list[int]:
    """Rank (0 = top) of each token in ``ids`` under ``lens`` at one cell."""
    logits = rsv.lens_logits(model, lens, h[None], layer, on_device=True)[0]
    return [int((logits > logits[i]).sum()) for i in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("model")
    p.add_argument("--templates", default="us,world")
    p.add_argument("--pairs", type=int, default=40, help="edit pairs per template")
    p.add_argument("--layers", default="all")
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    model, tok = rsv.load_model(a.model)
    lens = rsv.JacobianLens.from_pretrained(a.model)
    device = next(model.parameters()).device
    layers = lens.layers[:-1] if a.layers == "all" else [int(l) for l in a.layers.split(",")]
    dirs = Directions(model, lens)
    out = Path(a.out or HERE / "results" / f"{a.model.replace('/', '__')}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    f = out.open("w")
    t0 = time.time()

    for template in a.templates.split(","):
        rng = random.Random(f"{a.seed}-{template}")
        items = build_items(tok, template)
        good = []
        for it in items:
            h, probs, ids = _run(model, tok, it["prompt"])
            top = int(probs.argmax())
            it.update(h=h, ids=ids, top=tok.decode(top), correct=top == it["capital_id"],
                      p_capital=float(probs[it["capital_id"]]), city_pos=max(j for j, t in enumerate(ids) if t == it["city_id"]))
            if it["correct"]:
                good.append(it)
        f.write(json.dumps(dict(meta=True, model=a.model, template=template, n_items=len(items), n_correct=len(good),
                                wrong=[(it["city"], it["capital"], it["top"]) for it in items if not it["correct"]])) + "\n")
        print(f"[{template}] {len(items)} one-token items, {len(good)} answered correctly", flush=True)

        # Readout: ranks of this item's state and capital and of a control item's (different state and capital).
        for it in good:
            ctrl = rng.choice([o for o in items if o["state_id"] != it["state_id"] and o["capital_id"] != it["capital_id"]])
            watch = [it["state_id"], it["capital_id"], ctrl["state_id"], ctrl["capital_id"]]
            for where, pos in (("city", it["city_pos"]), ("last", len(it["ids"]) - 1)):
                for layer in lens.layers:
                    for kind, L in (("logit", None), ("jlens", lens)):
                        r = ranks(model, L, it["h"][layer, pos], layer, watch)
                        f.write(json.dumps(dict(model=a.model, type="readout", template=template, city=it["city"],
                                                where=where, layer=layer, lens=kind, state=r[0], capital=r[1],
                                                ctrl_state=r[2], ctrl_capital=r[3])) + "\n")

        pairs = [(o, n) for o in good for n in good if o["state_id"] != n["state_id"] and o["capital_id"] != n["capital_id"]]
        for i, (old, new) in enumerate(rng.sample(pairs, min(a.pairs, len(pairs)))):
            cf = old["prompt"].replace(old["city"], new["city"])
            h_cf, p_cf, ids_cf = _run(model, tok, cf)
            n, cpos = len(old["ids"]), old["city_pos"]
            assert len(ids_cf) == n, (old["city"], new["city"])
            last = n - 1
            base = dict(model=a.model, type="edit", template=template, old=old["city"], new=new["city"],
                        new_capital=new["capital"], cf_success=int(p_cf.argmax()) == new["capital_id"])
            enc = {k: v.to(device) for k, v in tok(old["prompt"], return_tensors="pt").items()}
            runs, patches = [], []
            for layer in layers:
                h = old["h"][layer]
                runs += [dict(layer=layer, edit="patch_city"), dict(layer=layer, edit="patch_last")]
                patches += [(layer, [cpos], h_cf[layer][[cpos]]), (layer, [last], h_cf[layer][[last]])]
                for s in STRENGTHS:
                    jl = None
                    for kind in ("unembed", "jlens"):
                        d = lambda t: dirs(t, layer, kind, center=False)
                        for edit, pos, o_id, n_id in (("state@last", last, old["state_id"], new["state_id"]),
                                                      ("state@city", cpos, old["state_id"], new["state_id"]),
                                                      ("city@city", cpos, old["city_id"], new["city_id"])):
                            cell = swap_vector(h[pos], d(o_id), d(n_id), s)
                            if edit == "state@last" and kind == "jlens":
                                jl = (cell - h[pos]).norm()
                            runs.append(dict(layer=layer, edit=edit, kind=kind, strength=s))
                            patches.append((layer, [pos], cell[None]))
                    runs.append(dict(layer=layer, edit="random@last", kind="random", strength=s))
                    patches.append((layer, [last], (h[last] + jl * random_unit(h.shape[-1], a.seed, template, i, layer, s))[None]))
            tops, pr = run_patches(model, enc, patches, [new["capital_id"], old["capital_id"]], a.batch)
            for r, top, (p_new, p_old) in zip(runs, tops, pr.tolist()):
                f.write(json.dumps(dict(**base, **r, success=top == new["capital_id"], kept=top == old["capital_id"],
                                        top=tok.decode(top), p_new=p_new, p_old=p_old)) + "\n")
            print(f"  [{template}] pair {i + 1}  {old['city']}→{new['city']}  {time.time() - t0:.0f}s", flush=True)
        f.flush()
    f.close()
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
