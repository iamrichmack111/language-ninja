from __future__ import annotations

import csv
import random
from pathlib import Path

TARGET_PER_CLASS = 500
SEED = 42

SUBJECTS = [
    "this application", "the app", "this program", "the interface", "this tool",
    "the service", "this feature", "the update", "the workflow", "the design",
    "the product", "the experience", "the system", "this software", "the terminal interface",
]

POS_VERBS = [
    "love", "really like", "enjoy", "appreciate", "am impressed by", "am happy with",
    "am pleased with", "prefer", "recommend", "feel good about",
]
POS_ADJ = [
    "excellent", "great", "fantastic", "useful", "smooth", "pleasant", "impressive",
    "helpful", "reliable", "well designed", "easy to use", "solid", "wonderful",
    "fast and clean", "a big improvement",
]
NEG_VERBS = [
    "hate", "really dislike", "cannot stand", "am frustrated by", "am disappointed with",
    "regret using", "would not recommend", "am annoyed by", "do not enjoy", "am unhappy with",
]
NEG_ADJ = [
    "terrible", "awful", "frustrating", "broken", "confusing", "unpleasant", "unreliable",
    "poorly designed", "hard to use", "slow", "disappointing", "a mess", "annoying", "bad",
    "worse than before",
]
NEU_VERBS = [
    "opened", "started", "closed", "updated", "loaded", "saved a file", "displayed a message",
    "ran", "connected", "finished",
]
NEU_TAILS = [
    "this morning", "at nine o clock", "after lunch", "on my computer", "without an error",
    "for ten minutes", "after the update", "during the test", "in the terminal", "yesterday",
]

REQUIRED = {
    "positive": [
        "I absolutely love this application.",
        "You did well, your little business is alright.",
        "I don't dislike the new interface.",
        "I thought this was going to be terrible but it is actually pretty good.",
        "Not bad at all.",
        "This is much better now.",
        "You handled that very well.",
        "I am pleasantly surprised.",
    ],
    "negative": [
        "I absolutely hate this application.",
        "You should speak more like a human instead of a robot.",
        "This sounds robotic and unnatural.",
        "That response was not very helpful.",
        "I expected better than this.",
        "This is not what I wanted.",
        "It works, but I still dislike it.",
        "I am not happy with how this works.",
    ],
    "neutral": [
        "The application opened at nine this morning.",
        "The response contained three paragraphs.",
        "The model produced a prediction.",
        "The file was saved to disk.",
        "The application is running in the terminal.",
        "The process completed after training.",
        "This message has twelve words.",
        "The screen shows the history tab.",
    ],
}

POS_PATTERNS = [
    "I do not hate {s}.", "I don't dislike {s}.", "{S} is not bad.", "{S} is not terrible.",
    "I expected {s} to be bad, but it is actually good.",
    "{S} was frustrating at first, but now I like it.",
    "{S} has problems, but overall I am happy with it.",
    "I would not call {s} bad at all.", "{S} is better than I expected.",
    "I almost gave up on {s}, but it turned out great.",
]
NEG_PATTERNS = [
    "I do not like {s}.", "I don't enjoy {s}.", "{S} is not good.", "{S} is not useful.",
    "I expected {s} to be good, but it is actually bad.",
    "{S} looked promising, but now I hate it.",
    "{S} has a few nice ideas, but overall it is frustrating.",
    "I would not call {s} good.", "{S} is worse than I expected.",
    "I tried to like {s}, but it is terrible.",
]
NEU_PATTERNS = [
    "I used {s} today.", "{S} has a menu and several buttons.", "{S} was installed yesterday.",
    "The documentation mentions {s}.", "I opened {s} and viewed the history page.",
    "{S} produced three output values.", "The test for {s} finished at noon.",
    "{S} is version two point two.", "I clicked a button in {s}.", "{S} contains a training page.",
]

CONVERSATIONAL = {
    "positive": [
        "That was actually pretty good.", "Honestly, I like this.", "You nailed it this time.",
        "I can work with this; it is pretty good.", "This feels natural and helpful.",
        "That answer was clear and useful.", "I like how human this sounds.",
        "The response is concise but still helpful.", "This works exactly the way I hoped.",
        "The changes fixed what was bothering me.",
    ],
    "negative": [
        "You could have done much better.", "Honestly, this is frustrating.",
        "This feels awkward and clumsy.", "The response feels cold and robotic.",
        "This answer missed what I was asking.", "The interface keeps getting in my way.",
        "This is technically working but still unpleasant to use.",
        "The result is confusing and not useful.", "I keep running into the same annoying problem.",
        "This feels worse after the update.",
    ],
    "neutral": [
        "You spoke for about two minutes.", "The business opened at nine.",
        "The program returned a result.", "The file contains three columns.",
        "The button is located below the text box.", "The test ran on my computer.",
        "The model file is stored in the models directory.", "The export completed at noon.",
        "There are three sentiment labels.", "The history table contains six rows.",
    ],
}


def build_candidates() -> dict[str, set[str]]:
    pools = {"positive": set(REQUIRED["positive"]), "negative": set(REQUIRED["negative"]), "neutral": set(REQUIRED["neutral"])}
    for subject in SUBJECTS:
        S = subject.capitalize()
        for verb in POS_VERBS:
            pools["positive"].add(f"I {verb} {subject}.")
            pools["positive"].add(f"I absolutely {verb} {subject}.")
        for adjective in POS_ADJ:
            pools["positive"].add(f"{S} is {adjective}.")
            pools["positive"].add(f"Overall, {subject} is {adjective}.")
        for verb in NEG_VERBS:
            pools["negative"].add(f"I {verb} {subject}.")
            pools["negative"].add(f"I absolutely {verb} {subject}.")
        for adjective in NEG_ADJ:
            pools["negative"].add(f"{S} is {adjective}.")
            pools["negative"].add(f"Overall, {subject} is {adjective}.")
        for verb in NEU_VERBS:
            for tail in NEU_TAILS:
                pools["neutral"].add(f"{S} {verb} {tail}.")
        for pattern in POS_PATTERNS:
            pools["positive"].add(pattern.format(s=subject, S=S))
        for pattern in NEG_PATTERNS:
            pools["negative"].add(pattern.format(s=subject, S=S))
        for pattern in NEU_PATTERNS:
            pools["neutral"].add(pattern.format(s=subject, S=S))
    for label, examples in CONVERSATIONAL.items():
        pools[label].update(examples)
    return pools


def select_balanced(pools: dict[str, set[str]]) -> list[tuple[str, str]]:
    rng = random.Random(SEED)
    rows: list[tuple[str, str]] = []
    for label in ("positive", "negative", "neutral"):
        required = list(dict.fromkeys(REQUIRED[label]))
        remainder = sorted(pools[label] - set(required))
        needed = TARGET_PER_CLASS - len(required)
        if len(remainder) < needed:
            raise RuntimeError(f"Not enough unique {label} examples: {len(remainder) + len(required)}")
        chosen = required + rng.sample(remainder, needed)
        rows.extend((text, label) for text in chosen)
    rng.shuffle(rows)
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "data" / "propositions.csv"
    rows = select_balanced(build_candidates())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} examples to {output}")
    for label in ("positive", "negative", "neutral"):
        print(label, sum(1 for _, current in rows if current == label))


if __name__ == "__main__":
    main()
