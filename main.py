"""
main.py

Advent of Code 2022
"""

from __future__ import annotations

import heapq
import math
import re
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from data import (
    DAY_02_SCORES_1,
    DAY_02_SCORES_2,
    DAY_03_SCORES,
    DAY_09_HEAD_MOVES,
    DAY_09_MOVES,
    DAY_09_TAIL_MOVES,
)

ALL_PROBLEMS = []
INPUT_FOLDER = Path("inputs")


def register(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register functions."""
    ALL_PROBLEMS.append(func)
    return func


def run_problem(func: Callable[[str], tuple[Any, Any]]) -> None:
    """Run a problem on a downloaded dataset."""
    day = str(func.__doc__)[4:6]
    with open(INPUT_FOLDER / f"{day}.txt", encoding="utf-8") as file_obj:
        raw_string = file_obj.read()
    for i, result in enumerate(func(raw_string)):
        print(f"Day {day}, Part {i + 1}: {result}")


def run_all() -> None:
    """Run all registered functions."""
    for func in ALL_PROBLEMS:
        run_problem(func)


@register
def day_01(data: str) -> tuple[int, int]:
    """Day 01."""
    inventories = [sum(int(j) for j in i.split()) for i in data.split("\n\n")]
    return max(inventories), sum(heapq.nlargest(n=3, iterable=inventories))


@register
def day_02(data: str) -> tuple[int, int]:
    """Day 02."""
    plays = data.splitlines()
    return (
        sum(DAY_02_SCORES_1[play] for play in plays),
        sum(DAY_02_SCORES_2[play] for play in plays),
    )


@register
def day_03(data: str) -> tuple[int, int]:
    """Day 03."""
    lines = data.splitlines()
    common_1 = [
        set(line[: len(line) // 2]).intersection(set(line[len(line) // 2 :]))
        for line in lines
    ]
    score_1 = sum(DAY_03_SCORES[letter] for s in common_1 for letter in s)
    qty = 3
    common_2 = [
        set.intersection(*[set(lines[qty * i + n]) for n in range(qty)])
        for i in range(len(lines) // qty)
    ]
    score_2 = sum(DAY_03_SCORES[letter] for s in common_2 for letter in s)
    return score_1, score_2


@register
def day_04(data: str) -> tuple[int, int]:
    """Day 04."""
    lines = data.splitlines()
    ranges = [[int(s) for s in re.findall(r"\d+", line)] for line in lines]
    total_overlaps = sum(
        1
        for x1, y1, x2, y2 in ranges
        if (x1 <= x2 and y1 >= y2) or (x1 >= x2 and y1 <= y2)
    )
    partial_overlaps = sum(
        1
        for x1, y1, x2, y2 in ranges
        if (x2 <= x1 <= y2)
        or (x2 <= y1 <= y2)
        or (x1 <= x2 <= y1)
        or (x1 <= y2 <= y1)
    )
    return total_overlaps, partial_overlaps


@register
def day_05(data: str) -> tuple[str, str]:
    """Day 05."""
    [raw_stacks, raw_moves] = data.split("\n\n")
    stacks = defaultdict(list)
    levels = raw_stacks.splitlines()[-2::-1]
    for level in levels:
        for i, crate in enumerate(level[1::4]):
            if crate != " ":
                stacks[i + 1].append(crate)

    moves = [
        [int(s) for s in re.findall(r"\d+", line)]
        for line in raw_moves.split("\n")[:-1]
    ]

    def crate_mover_9000() -> str:
        stacks_ = deepcopy(stacks)
        for quantity, source, destination in moves:
            takeoff = stacks_[source][: -1 - quantity : -1]
            stacks_[source] = stacks_[source][:-quantity]
            stacks_[destination].extend(takeoff)
        return "".join(stacks_[i][-1] for i in range(1, 10))

    def crate_mover_9001() -> str:
        stacks_ = deepcopy(stacks)
        for quantity, source, destination in moves:
            takeoff = stacks_[source][-quantity:]
            stacks_[source] = stacks_[source][:-quantity]
            stacks_[destination].extend(takeoff)
        return "".join(stacks_[i][-1] for i in range(1, 10))

    return crate_mover_9000(), crate_mover_9001()


@register
def day_06(data: str) -> tuple[int, int]:
    """Day 06."""
    code = data.strip()

    def day_06_inner(repeat_len: int) -> int:
        for end in range(repeat_len, len(code) + 1):
            if len(set(code[end - repeat_len : end])) == repeat_len:
                break
        else:
            raise ValueError
        return end

    return day_06_inner(4), day_06_inner(14)


@register
def day_07(data: str) -> tuple[int, int]:
    """Day 07."""
    threshold = 100000
    total_space = 70000000
    required_space = 30000000

    class Directory:
        """A directory."""

        def __init__(self, name: str, parent: Directory | None = None):
            self.name = name
            self.parent = parent
            self.subdirectories: dict[str, Directory] = {}
            self.files: dict[str, int] = {}
            self.size: int = 0

    root = Directory(name="/")
    all_directories: set[Directory] = set()
    location = root

    for line in data.splitlines():
        match line.split():
            case ["$", "cd", "/"]:
                location = root
            case ["$", "cd", ".."]:
                if location.parent:
                    location.parent.size += location.size
                    location = location.parent
                else:
                    raise ValueError
            case ["$", "cd", subdirectory]:
                location = location.subdirectories[subdirectory]
            case ["$", "ls"]:
                continue
            case [size, filename] if size.isnumeric():
                location.files[filename] = int(size)
                location.size += int(size)
            case [type_, subdir_name] if type_ == "dir":
                subdir_obj = Directory(name=subdir_name, parent=location)
                all_directories.add(subdir_obj)
                location.subdirectories[subdir_name] = subdir_obj
            case _:
                raise ValueError

    while location != root:
        if location.parent:
            location.parent.size += location.size
            location = location.parent

    part1 = sum([dir.size for dir in all_directories if dir.size <= threshold])
    min_del = max(0, required_space - (total_space - root.size))
    part2 = min(dir.size for dir in all_directories if dir.size >= min_del)
    return part1, part2


@register
def day_08(data: str) -> tuple[int, int]:
    """Day 08."""
    heights = {}
    i = 0
    j = 0
    for i, row in enumerate(data.splitlines()):
        for j, height in enumerate(row):
            heights[i, j] = int(height)

    def is_visible(x: int, y: int) -> bool:
        """Check if location is visible."""
        visible = (
            heights[x, y] > max([heights[a, y] for a in range(x)], default=-1)
            or heights[x, y]
            > max([heights[a, y] for a in range(x + 1, i + 1)], default=-1)
            or heights[x, y]
            > max([heights[x, b] for b in range(y)], default=-1)
            or heights[x, y]
            > max([heights[x, b] for b in range(y + 1, j + 1)], default=-1)
        )
        return visible

    def score(x: int, y: int) -> int:
        """Get score for a location."""
        loc_scores = []
        for dx, dy in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            dir_score = 0
            while True:
                next_height = heights.get(
                    (x + (1 + dir_score) * dx, y + (1 + dir_score) * dy)
                )
                if next_height is None:
                    break
                dir_score += 1
                if next_height >= heights[x, y]:
                    break
            loc_scores.append(dir_score)
        return math.prod(loc_scores)

    qty_visible = sum([is_visible(*location) for location in heights])
    max_score = max(score(*location) for location in heights)
    return qty_visible, max_score


@register
def day_09_part_1(data: str) -> tuple[int]:
    """Day 09."""
    head = [0, 0]
    rel_tail = (0, 0)
    tail_pos = (0, 0)
    seen = set()

    for direction, distance in [line.split() for line in data.splitlines()]:
        for _ in range(int(distance)):
            head[0] += DAY_09_HEAD_MOVES[direction][0]
            head[1] += DAY_09_HEAD_MOVES[direction][1]
            rel_tail = DAY_09_TAIL_MOVES[rel_tail, direction]
            tail_pos = (head[0] + rel_tail[0], head[1] + rel_tail[1])
            seen.add(tail_pos)

    return (len(seen),)


if __name__ == "__main__":
    run_all()
