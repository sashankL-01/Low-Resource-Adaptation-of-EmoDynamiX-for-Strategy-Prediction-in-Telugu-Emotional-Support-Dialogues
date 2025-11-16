import argparse
import json
import torch

from modules.roberta.model_telugu import XLMRHeterogeneousGraphTelugu
from utils import seed_everything


STRATEGY_MAP = json.load(open("data/esconv/strategies.json", "r", encoding="utf-8"))


DEFAULT_DIALOGUE = {
    "dialog": [
        {
            "speaker": "seeker",
            "annotation": {},
            "content": "హలో"
        },
        {
            "speaker": "supporter",
            "annotation": {"strategy": "Question"},
            "content": "హలో, మీరు దేని గురించి మాట్లాడాలనుకుంటున్నారు?"
        },
        {
            "speaker": "seeker",
            "annotation": {},
            "content": "నా ప్రస్తుత ఉద్యోగాన్ని విడిచిపెట్టడం గురించి నాకు చాలా ఆందోళన ఉంది. ఇది చాలా ఒత్తిడితో కూడుకున్నది కాని బాగా చెల్లిస్తుంది"
        },
        {
            "speaker": "supporter",
            "annotation": {"strategy": "Question"},
            "content": "మీ ఉద్యోగం మీ కోసం ఒత్తిడితో కూడుకున్నది ఏమిటి?"
        },
        {
            "speaker": "seeker",
            "annotation": {"feedback": "5"},
            "content": "నేను కఠినమైన ఆర్థిక పరిస్థితులలో చాలా మందితో వ్యవహరించాలి మరియు ఇది కలత చెందుతోంది"
        },
        {
            "speaker": "supporter",
            "annotation": {"strategy": "Question"},
            "content": "మీ ఖాతాదారులకు మెరుగైన ఆర్థిక పరిస్థితులకు అనుగుణంగా మీరు సహాయం చేస్తున్నారా?"
        },
        {
            "speaker": "seeker",
            "annotation": {},
            "content": "నేను చేస్తాను, కాని తరచుగా వారు కోరుకున్నదానికి తిరిగి రావడం లేదు. భద్రతలను ఎత్తివేసినప్పుడు చాలా మంది తమ ఇంటిని కోల్పోతారు"
        },
        {
            "speaker": "supporter",
            "annotation": {"strategy": "Affirmation and Reassurance"},
            "content": "కానీ మీరు ప్రస్తుతం ఉన్నదానికంటే మంచి భవిష్యత్తును అందిస్తారు. ఇది వారు కోరుకున్నది కాకపోవచ్చు, కానీ ఇది దీర్ఘకాలంలో వారికి సహాయపడుతుంది."
        },
        {
            "speaker": "seeker",
            "annotation": {"feedback": "5"},
            "content": "ఇది నిజం కాని కొన్నిసార్లు నేను నా భావాలను మరియు ఆరోగ్యాన్ని మొదట ఉంచాలని భావిస్తున్నాను"
        },
        {
            "speaker": "supporter",
            "annotation": {"strategy": "Affirmation and Reassurance"},
            "content": "నేను దానిని అర్థం చేసుకోగలను."
        },
        {
            "speaker": "supporter",
            "annotation": {"strategy": "Question"},
            "content": "మీరు ప్రస్తుతం చేస్తున్నదానికి దగ్గరగా చెల్లించే మరో ఉద్యోగం ఉందా?"
        },
        {
            "speaker": "seeker",
            "annotation": {"feedback": "5"},
            "content": "బహుశా కాదు. నేను చాలా కాలంగా ఒకే సంస్థతో ఉన్నాను మరియు నేను ప్రతి సంవత్సరం స్థిరంగా బోనస్ పొందుతాను"
        },
        {
            "speaker": "supporter",
            "annotation": {"strategy": "Others"},
            "content": "మీ ఖాతాదారుల భయంకరమైన ఆర్థిక పరిస్థితులను మీరు ఎలా చూస్తారో రీఫ్రేమ్ చేయడం సాధ్యమేనా?"
        },
        {
            "speaker": "seeker",
            "annotation": {},
            "content": "నేను ప్రయత్నించగలను. ఇది ఎక్కువగా రోజు చివరిలో నాకు వస్తుంది"
        },
        {
            "speaker": "supporter",
            "annotation": {"strategy": "Information"},
            "content": "కొంతమంది మీరు చేసే పనిని చేయలేరు ఎందుకంటే వేరొకరికి చెడ్డ వార్తలు ఇవ్వడానికి వారికి హృదయం లేదు. వాస్తవికత ఏమిటంటే, ఎవరైనా ఆ పాత్రను నింపాలి మరియు మీరు ప్రజలకు సహాయం చేస్తారు"
        },
        {
            "speaker": "seeker",
            "annotation": {"feedback": "4"},
            "content": "అది కూడా నిజం. కొన్నిసార్లు ఇది నిజంగా నాకు అయితే నేను ఆశ్చర్యపోతున్నాను"
        }
    ]
}
# ,
#         {
#             "speaker": "supporter",
#             "annotation": {"strategy": "Self-disclosure"},
#             "content": "నేను చెడు ఆర్థిక స్థితిలో ఉన్నప్పుడు ఇంతకు ముందు సేకరణలతో వ్యవహరించాల్సి వచ్చింది. ఇతర పంక్తిలో ఉన్న వ్యక్తి నిజంగా సహాయకారిగా ఉన్నాడు. ఆమె అర్థం చేసుకుంది,"
#         },
#         {
#             "speaker": "supporter",
#             "annotation": {"strategy": "Providing Suggestions"},
#             "content": "ఇది మీ కోసం కాకపోవచ్చు. మీ స్థానాన్ని ఉంచడం యొక్క లాభాలు మరియు నష్టాల గురించి మీరు ఆలోచించాలని నేను భావిస్తున్నాను. ఇది మీ కోసం విషయాలు స్పష్టంగా చెప్పవచ్చు."
#         },
#         {
#             "speaker": "seeker",
#             "annotation": {"feedback": "5"},
#             "content": "అది నిజం. బహుశా నేను కూర్చుని దాని గురించి ఆలోచించాలి"
#         },
#         {
#             "speaker": "supporter",
#             "annotation": {"strategy": "Restatement or Paraphrasing"},
#             "content": "ఇది మీ మానసిక ఆరోగ్యాన్ని నిజంగా ప్రతికూల మార్గంలో ప్రభావితం చేస్తే నేను ఉండను. అయినప్పటికీ, మీరు జూమ్ అవుట్ చేయవలసి ఉంటుంది మరియు పెద్ద చిత్రాన్ని చూడవలసి ఉంటుంది: మీరు అవసరమైన సేవను అందిస్తారని మరియు మీరు కరుణతో చేస్తారు"
#         },
#         {
#             "speaker": "seeker",
#             "annotation": {},
#             "content": "ఇది నిజంగా పెద్ద నిర్ణయం"
#         },
#         {
#             "speaker": "seeker",
#             "annotation": {},
#             "content": "విభిన్న దృక్పథానికి ధన్యవాదాలు"
#         }

def normalize_turn(turn):
    text = turn.get("text") or turn.get("టెక్స్ట్") or turn.get("content") or ""
    speaker = turn.get("speaker") or turn.get("స్పీకర్") or ""
    annotation = turn.get("annotation") or {}
    strategy = annotation.get("strategy") or annotation.get("వ్యూహం")
    feedback = annotation.get("feedback")
    norm = {"text": text, "speaker": speaker}
    if strategy:
        norm["strategy"] = strategy
    if feedback is not None:
        norm["feedback"] = feedback
    return norm


def prepare_sample(dialogue):
    return {
        "dialog": [normalize_turn(t) for t in dialogue],
        "strategy": None,
    }


def build_args(cfg):
    class _Args:
        def __init__(self):
            self.hg_dim = cfg.hg_dim
            self.erc_temperature = cfg.erc_temperature
            self.erc_mixed = cfg.erc_mixed
            self.telugu_erc_path = cfg.telugu_erc_path
            self.parse_bs = cfg.parse_bs
            self.parse_ctx_len = cfg.parse_ctx_len
            self.context_max_len = cfg.context_max_len
            self.erc_max_len = cfg.erc_max_len
            self.exclude_others = 0
            self.feedback_threshold = 0
            self.dataset = "esconv-telugu"

    return _Args()


def load_dialogue(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "dialog" in payload:
        return payload["dialog"]
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dialogue_path", type=str, help="Path to Telugu dialogue JSON")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_step", type=int, required=True)
    parser.add_argument("--telugu_erc_path", type=str, default="telugu_erc_xlmroberta_trained_v2")
    parser.add_argument("--hg_dim", type=int, default=512)
    parser.add_argument("--erc_temperature", type=float, default=0.5)
    parser.add_argument("--erc_mixed", type=int, default=1)
    parser.add_argument("--parse_bs", type=int, default=512)
    parser.add_argument("--parse_ctx_len", type=int, default=32)
    parser.add_argument("--context_max_len", type=int, default=192)
    parser.add_argument("--erc_max_len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=114514)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_args = build_args(args)
    model = XLMRHeterogeneousGraphTelugu(model_args, lightmode=False).to(device)

    ckpt_path = f"{args.checkpoint_dir}/checkpoint-{args.checkpoint_step}.pth"
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    if args.dialogue_path:
        dialogue = load_dialogue(args.dialogue_path)
    else:
        dialogue = DEFAULT_DIALOGUE["dialog"]

    sample = prepare_sample(dialogue)
    # Adapt sample into the batch format expected by the model
    # Build a minimal pseudo-batch matching dataloader contract
    utterances = [t["text"] for t in sample["dialog"]]
    speakers = ["seeker" if t["speaker"].lower().startswith("s") else "supporter" for t in sample["dialog"]]
    dialogue_history = "</s>".join(utterances)
    speaker_turn = " ".join(speakers)
    # Strategy history: -1 for unknown supporter turns, 0 for seeker
    strategy_history = []
    for t in sample["dialog"]:
        strat = t.get("strategy")
        if strat is None:
            strategy_history.append(-1)
        else:
            strategy_history.append(STRATEGY_MAP.get(strat, 0))
    strategy_history_str = "[" + ",".join(str(x) for x in strategy_history) + "]"

    batch = {
        "dialogue_history": [dialogue_history],
        "strategy_history": [strategy_history_str],
        "speaker_turn": [speaker_turn],
        "gold_standard": [""],
        "label": torch.tensor([0]),
    }

    with torch.no_grad():
        result = model(batch)
        logits = result["logits"][0].cpu()
        pred_idx = logits.argmax().item()

    inv_strategy = {v: k for k, v in STRATEGY_MAP.items()}
    print(json.dumps(
        {
            "predicted_strategy_id": pred_idx,
            "predicted_strategy": inv_strategy.get(pred_idx, "UNKNOWN"),
            "logits": logits.tolist(),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
