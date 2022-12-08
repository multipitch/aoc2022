"""
main.py

Advent of Code 2022
"""

from __future__ import annotations
from pathlib import Path
import heapq
import re
from collections import defaultdict
from copy import deepcopy
from typing import Any, Callable

from data import DAY_02_SCORES_1, DAY_02_SCORES_2, DAY_03_SCORES

ALL_PROBLEMS = []
INPUT_FOLDER = Path("inputs")


def register(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register functions."""
    ALL_PROBLEMS.append(func)
    return func


def run_all() -> None:
    """Run all registered functions."""
    for func in ALL_PROBLEMS:
        func()


@register
def day_01() -> None:
    """Day 01."""
    with open(INPUT_FOLDER / "01.txt", encoding="utf-8") as file_obj:
        raw = file_obj.read()
    inventories = [sum(int(j) for j in i.split()) for i in raw.split("\n\n")]
    print(f"Day 01, Part 1: {max(inventories)}")
    print(f"Day 01, Part 2: {sum(heapq.nlargest(n=3, iterable=inventories))}")


@register
def day_02() -> None:
    """Day 02."""
    with open(INPUT_FOLDER / "02.txt", encoding="utf-8") as file_obj:
        plays = file_obj.readlines()
    print(f"Day 02, Part 1: {sum(DAY_02_SCORES_1[play]for play in plays)}")
    print(f"Day 02, Part 2: {sum(DAY_02_SCORES_2[play]for play in plays)}")


@register
def day_03() -> None:
    """Day 03."""
    with open(INPUT_FOLDER / "03.txt", encoding="utf-8") as file_obj:
        data = [line.strip() for line in file_obj.readlines()]

    common_1 = [
        set(line[: len(line) // 2]).intersection(set(line[len(line) // 2 :]))
        for line in data
    ]
    score_1 = sum(DAY_03_SCORES[letter] for s in common_1 for letter in s)
    print(f"Day 03, Part 1: {score_1}")

    qty = 3
    common_2 = [
        set.intersection(*[set(data[qty * i + n]) for n in range(qty)])
        for i in range(len(data) // qty)
    ]
    score_2 = sum(DAY_03_SCORES[letter] for s in common_2 for letter in s)
    print(f"Day 03, Part 2: {score_2}")


@register
def day_04() -> None:
    """Day 04."""
    with open(INPUT_FOLDER / "04.txt", encoding="utf-8") as file_obj:
        lines = file_obj.readlines()
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
    print(f"Day 04, Part 1: {total_overlaps}")
    print(f"Day 04, Part 2: {partial_overlaps}")


@register
def day_05() -> None:
    """Day 05."""
    with open(INPUT_FOLDER / "05.txt", encoding="utf-8") as file_obj:
        raw = file_obj.read()
    [raw_stacks, raw_moves] = raw.split("\n\n")

    stacks = defaultdict(list)
    levels = raw_stacks.split("\n")[-2::-1]
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

    print(f"Day 05, Part 1: {crate_mover_9000()}")
    print(f"Day 05, Part 2: {crate_mover_9001()}")


@register
def day_06() -> None:
    """Day 06."""
    with open(INPUT_FOLDER / "06.txt", encoding="utf-8") as file_obj:
        code = file_obj.readline()

    def day_06_helper(repeat_len: int) -> int:
        for end in range(repeat_len, len(code) + 1):
            if len(set(code[end - repeat_len : end])) == repeat_len:
                break
        else:
            raise ValueError
        return end

    print(f"Day 06, Part 1: {day_06_helper(4)}")
    print(f"Day 06, Part 2: {day_06_helper(14)}")


@register
def day_07() -> None:
    """Day 07."""
    with open(INPUT_FOLDER / "07.txt", encoding="utf-8") as file_obj:
        lines = file_obj.read().splitlines()

    threshold = 100000
    total_space = 70000000
    required_space = 30000000

    class Directory:
        """
        A directory.
        Could probably use a bunch of dicts instead of classes, but
        this makes more sense to read.
        """

        def __init__(self, name: str, parent: Directory | None = None):
            self.name = name
            self.parent = parent
            self.subdirectories: dict[str, Directory] = {}
            self.files: dict[str, int] = {}
            self.size = 0

    root = Directory(name="/")
    all_directories: set[Directory] = set()
    location = root

    # This approach assumes that the user creating the data traversed
    # directories in a depth-first search, allowing us to compute the
    # total directory sizes on "$ cd .." operations.
    # Otherwise, would have to keep lists of seen files or re-traverse a
    # second time to totalize sizes.
    for line in lines:
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

    # The data doesn't cd back to root at the end; we need to do this to
    # correctly compute totals for all directories back up to root.
    while location != root:
        if location.parent:
            location.parent.size += location.size
            location = location.parent

    total = sum([dir.size for dir in all_directories if dir.size <= threshold])
    print(f"Day 07, Part 1: {total}")

    min_del = max(0, required_space - (total_space - root.size))
    del_size = min(dir.size for dir in all_directories if dir.size >= min_del)
    print(f"Day 07, Part 2: {del_size}")


if __name__ == "__main__":
    run_all()
