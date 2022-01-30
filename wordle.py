# %%
from cgitb import grey
import copy
import random
from dataclasses import dataclass, field
from enum import Enum
from multiprocessing.sharedctypes import Value

from tqdm import tqdm
from absl import app, flags

flags.DEFINE_integer("seed", 0, "random seed")

FLAGS = flags.FLAGS

WORD_LEN = 5


class Matching(Enum):
    EXACT = "+"
    SEMI = "?"
    NONE = "_"


class Status(Enum):
    IN_PROGRESS = "IN_PROGRESS"
    WIN = "WIN"
    LOSE = "LOSE"


def render_ascii(matchings: list[Matching]) -> str:
    return "".join(matching.value if matching else " " for matching in matchings)


def load_words_from_file(fname: str) -> frozenset[str]:
    with open(fname) as f:
        return frozenset(f.read().splitlines())


@dataclass
class Wordle:
    allowed_words: frozenset[str] = load_words_from_file("allowed_words.txt")
    answer_words: frozenset[str] = load_words_from_file("answer_words.txt")
    max_attemps: int = 6

    attemps_left: int = 0
    status: Status = Status.IN_PROGRESS
    history: list[tuple[str, list[Matching]]] = field(default_factory=list)
    answer: str | None = None

    def reset(self, answer: str | None = None) -> "Wordle":
        if answer:
            self.answer = answer
        else:
            self.answer = random.choice(list(self.answer_words))
        self.history = []
        self.status = Status.IN_PROGRESS
        self.attemps_left = self.max_attemps
        return self

    @property
    def attemps(self):
        return self.max_attemps - self.attemps_left

    def submit_guess(self, word):
        if not self._word_is_allowed(word):
            raise ValueError(f"{word} is not allowed")

        self.attemps_left -= 1
        matching_result = Wordle.match_words(self.answer, word)
        self.history.append((word, matching_result))
        self._update_status(matching_result)

        return matching_result

    def _update_status(self, matching_result):
        if all(r == Matching.EXACT for r in matching_result):
            self.status = Status.WIN
        elif self.attemps_left == 0:
            self.status = Status.LOSE
        else:
            self.status = Status.IN_PROGRESS

    def _word_is_allowed(self, word):
        return word in self.allowed_words or word in self.answer_words

    @staticmethod
    def match_words(answer, guess) -> list[Matching]:
        result: list[Matching | None] = [None] * WORD_LEN
        used = set()

        for i, guess_char in enumerate(guess):
            if answer[i] == guess_char:
                result[i] = Matching.EXACT
                used.add(i)

        for i, guess_char in enumerate(guess):
            if result[i] is None:
                for j, answer_char in enumerate(answer):
                    if guess_char == answer_char and (j not in used):
                        result[i] = Matching.SEMI
                        used.add(j)
                        break

        return [r if r else Matching.NONE for r in result]

    def get_candidates(self):
        for answer in self.answer_words:
            if self._could_answer_fit_history(answer):
                yield answer

    def _could_answer_fit_history(self, answer):
        for prev_guess, matchings in self.history:
            if Wordle.match_words(answer, prev_guess) != matchings:
                return False
        return True

    def fork(self):
        w = Wordle(allowed_words=self.allowed_words, answer_words=self.answer_words)
        w.reset(self.answer)
        w.history = copy.deepcopy(self.history)
        w.status = self.status
        w.attemps_left = self.attemps_left
        w.max_attemps = self.max_attemps
        return w


@dataclass
class TerminalGame:
    wordle: Wordle = field(default_factory=Wordle)

    def start(self):
        self.wordle.reset()
        while self.wordle.status == Status.IN_PROGRESS:
            guess = input("> ")

            if guess == "/cheat":
                self.cheat()
                continue

            try:
                self.wordle.submit_guess(guess)
            except ValueError:
                print("Not a word, try again.\n")

            self.print_history()
            print("\n")

        if self.wordle.status == Status.WIN:
            print("You won!")
        else:
            print("You lost!")

    def print_history(self):
        for guess, matchings in self.wordle.history:
            print(guess, render_ascii(matchings))

    def cheat(self):
        print(list(self.wordle.get_candidates()))


class Policy:
    def guess(self, wordle: Wordle) -> str:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class RandomPolicy(Policy):
    def guess(self, wordle: Wordle):
        return random.choice(list(wordle.get_candidates()))


class GreedyEliminationPolicy(Policy):
    def guess(self, wordle: Wordle) -> str:
        candidates = list(wordle.get_candidates())
        scores = []
        curr_best = 0
        for candidate in candidates:
            fork = wordle.fork()
            fork.submit_guess(candidate)
            score = len(candidates) - len(list(fork.get_candidates()))
            scores.append(score)
            curr_best = max(curr_best, score)
        return candidates[scores.index(curr_best)]


def benchmark_policy(wordle, policy):
    wordle = Wordle()
    stats = {
        "num_wins": 0,
        "num_loses": 0,
        "num_guesses": 0,
        "num_games": len(wordle.answer_words),
    }
    progress_bar = tqdm(wordle.answer_words)
    for answer in progress_bar:
        wordle.reset(answer)
        while wordle.status == Status.IN_PROGRESS:
            guess = policy.guess(wordle)
            wordle.submit_guess(guess)
        if wordle.status == Status.WIN:
            stats["num_wins"] += 1
            stats["num_guesses"] += wordle.attemps
        elif wordle.status == Status.LOSE:
            stats["num_loses"] += 1
            stats["num_guesses"] += wordle.attemps

        stats["average_guesses"] = round(stats["num_guesses"] / stats["num_games"], 3)
        stats["win_rate"] = round(
            stats["num_wins"] / (stats["num_wins"] + stats["num_loses"]), 3
        )
        progress_bar.set_description(str(stats))
    return stats


# %%
def main(*args):
    random.seed(FLAGS.seed)
    game = TerminalGame()
    game.start()


# %%
benchmark_policy(Wordle(), GreedyEliminationPolicy())
# %%

# %%
