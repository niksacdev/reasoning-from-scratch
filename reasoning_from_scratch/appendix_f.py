
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

from reasoning_from_scratch.ch02 import generate_text_basic_stream_cache
import re


def predict_choice(
    model, tokenizer, prompt_fmt, max_new_tokens=8
):
    generated = []
    for t in generate_text_basic_stream_cache(
        model=model,
        token_ids=prompt_fmt,
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
    ):
        generated.extend(t.squeeze(0).tolist())
    # Decode together: a token containing "Answer" is not the choice "A".
    answer = tokenizer.decode(generated)
    match = re.search(
        r"(?:^|\banswer\s*:)\s*\(?([ABCD])(?=\s|[).]|$)",
        answer, flags=re.IGNORECASE,
    )
    return match.group(1).upper() if match else None


def elo_ratings(vote_pairs, k_factor=32, initial_rating=1000):
    # Initialize all models with the same base rating
    ratings = {
        model: initial_rating
        for pair in vote_pairs
        for model in pair
    }

    # Update ratings after each match
    for winner, loser in vote_pairs:
        rating_winner, rating_loser = ratings[winner], ratings[loser]

        # Expected score for the current winner given the ratings
        expected_winner = 1.0 / (
            1.0 + 10 ** ((rating_loser - rating_winner) / 400.0)
        )

        # k_factor determines sensitivity of rating updates
        ratings[winner] = (
            rating_winner + k_factor * (1 - expected_winner)
        )
        ratings[loser] = (
            rating_loser + k_factor * (0 - (1 - expected_winner))
        )

    return ratings
