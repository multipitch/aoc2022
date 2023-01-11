"""
main.py

Advent of Code 2022
"""

from __future__ import annotations

import heapq
import itertools
import math
import re
from ast import literal_eval
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass
from operator import add, mul
from pathlib import Path
from typing import Any, Callable

from advent_of_code_ocr import convert_array_6 as ocr

from data import DAY_02_SCORES_1, DAY_02_SCORES_2, DAY_03_SCORES, DAY_09_MOVES

ALL_PROBLEMS = []


def register(func: Callable[..., Any]) -> Callable[..., Any]:
    """Register functions."""
    ALL_PROBLEMS.append(func)
    return func


def run_problem(func: Callable[[str], Any]) -> None:
    """Run a problem on a downloaded dataset."""
    day = str(func.__doc__)[4:6]
    with open(Path("inputs") / f"{day}.txt", encoding="utf-8") as file_obj:
        raw_string = file_obj.read()
    ret = func(raw_string)
    if isinstance(ret, tuple):
        print(f"Day {day}: Part 1 = {ret[0]}")
        print(f"        Part 2 = {ret[1]}")
    elif isinstance(ret, str) or isinstance(ret, int):
        print(f"Day {day}: Part 1 = {ret}")


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
    part_1 = sum(DAY_02_SCORES_1[play] for play in plays)
    part_2 = sum(DAY_02_SCORES_2[play] for play in plays)
    return part_1, part_2


@register
def day_03(data: str) -> tuple[int, int]:
    """Day 03."""
    lines = data.splitlines()
    common_1 = [set(i[: len(i) // 2]) & set(i[len(i) // 2 :]) for i in lines]
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
            case "$", "cd", "/":
                location = root
            case "$", "cd", "..":
                if location.parent:
                    location.parent.size += location.size
                    location = location.parent
                else:
                    raise ValueError
            case "$", "cd", subdirectory:
                location = location.subdirectories[subdirectory]
            case "$", "ls":
                continue
            case size, filename if size.isnumeric():
                location.files[filename] = int(size)
                location.size += int(size)
            case type_, subdir_name if type_ == "dir":
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
def day_09(data: str) -> tuple[int, int]:
    """Day 09."""

    directions = {"R": (1, 0), "L": (-1, 0), "D": (0, -1), "U": (0, 1)}

    def day_09_inner(length: int) -> int:
        knots_x = [0] * length
        knots_y = [0] * length
        seen = set()
        for direc, distance in [line.split() for line in data.splitlines()]:
            head_dx, head_dy = directions[direc]
            for _ in range(int(distance)):
                knots_x[0] += head_dx
                knots_y[0] += head_dy
                for i in range(1, length):
                    dx = knots_x[i] - knots_x[i - 1]
                    dy = knots_y[i] - knots_y[i - 1]
                    (dx, dy) = DAY_09_MOVES[dx, dy]
                    knots_x[i] += dx
                    knots_y[i] += dy
                seen.add((knots_x[-1], knots_y[-1]))
        return len(seen)

    return (day_09_inner(2), day_09_inner(10))


@register
def day_10(data: str) -> tuple[int, str]:
    """Day 10."""

    cycle = 1
    cycles_of_interest = [20, 60, 100, 140, 180, 220]
    sprite_centers = [1]
    width = 40
    height = 6
    pixels = [["."] * width for _ in range(height)]

    for line in data.splitlines():
        sprite_centers.append(sprite_centers[-1])
        cycle += 1
        if line[:4] == "addx":
            cycle += 1
            sprite_centers.append(sprite_centers[-1] + int(line[5:]))

    strength = sum([i * sprite_centers[i - 1] for i in cycles_of_interest])

    for cycle, sprite_center in enumerate(sprite_centers, start=1):
        pixel_x = (cycle - 1) % width
        if abs(sprite_center - pixel_x) <= 1:
            pixels[(cycle - 1) // width % height][pixel_x] = "#"

    return strength, ocr(pixels, fill_pixel="#", empty_pixel=".")


@register
def day_11(data: str) -> tuple[int, int]:
    """Day 11."""

    def day_11_inner(rounds: int, divisor: int) -> int:
        items = []
        ops = []
        args = []
        tests = []
        dests: list[list[int]] = [[], []]

        for row in data.splitlines():
            match row.split():
                case "Starting", _, *_items:
                    __items = deque(int(item.strip(",")) for item in _items)
                    items.append(__items)
                case "Operation:", _, _, _, operator, arg:
                    ops.append(mul if operator == "*" else add)
                    args.append(None if arg == "old" else int(arg))
                case "Test:", _, _, test_divisor:
                    tests.append(int(test_divisor))
                case "If", "true:", _, _, _, destination:
                    dests[0].append(int(destination))
                case "If", "false:", _, _, _, destination:
                    dests[1].append(int(destination))

        inspected = [0 for _ in items]
        lcm = math.lcm(*tests)

        for _ in range(rounds):
            for i, monkey_items in enumerate(items):
                while monkey_items:
                    item = monkey_items.popleft()
                    inspected[i] += 1
                    other = item if args[i] is None else args[i]
                    item = ops[i](item, other) % lcm // divisor
                    items[dests[item % tests[i] != 0][i]].append(item)

        return math.prod(sorted(inspected)[-2:])

    return day_11_inner(20, 3), day_11_inner(10000, 1)


# @register
# TODO: This works, but is very slow for Part 2.
#       Problem name mentions hill climb; probably a simpler approach
#       available.
def day_12(data: str) -> tuple[int, int]:
    """Day 12."""

    nodes = {}
    start = (-1, -1)
    starts = []
    end = (-1, -1)
    for y, row in enumerate(data.splitlines()):
        for x, height in enumerate(row):
            if height == "S":
                start = (x, y)
                height = "a"
            elif height == "E":
                end = (x, y)
                height = "z"
            elif height == "a":
                starts.append((x, y))
            nodes[x, y] = ord(height) - 97

    edges = set()
    for (x, y) in nodes:
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            dest = (x + dx, y + dy)
            if dest in nodes and nodes[dest] - nodes[x, y] <= 1:
                edges.add(((x, y), dest))

    class Removed:
        """A removed task."""

    removed = Removed()

    def run(start: tuple[int, int]) -> int:
        """Find shortes path between start and destination."""
        queue: list[tuple[int, int, tuple[int, int] | Removed]] = []
        entry_finder: dict[
            tuple[int, int] | Removed,
            tuple[int, int, tuple[int, int] | Removed],
        ] = {}
        counter = itertools.count()

        def add_task(task: tuple[int, int], priority: int = 0) -> None:
            """Add a new task or update the priority of an existing task."""
            if task in entry_finder:
                entry = entry_finder.pop(task)
                entry = (entry[0], entry[1], removed)

            count = next(counter)
            entry = (priority, count, task)
            entry_finder[task] = entry
            heapq.heappush(queue, entry)

        def pop_task() -> tuple[int, int]:
            """Remove and return the lowest priority task."""
            while queue:
                _, _, task = heapq.heappop(queue)
                if not isinstance(task, Removed):
                    del entry_finder[task]
                    return task
            raise KeyError("pop from an empty priority queue")

        permanent_set: set[tuple[int, int]] = {start}
        distances: dict[tuple[int, int], int] = {}
        predecessors = {}
        spst = set()

        for node in nodes:
            if node == start:
                distances[node] = 0
            else:
                if (start, node) in edges:
                    distances[node] = 1
                else:
                    distances[node] = 1000000000
                predecessors[node] = start
                add_task(node, distances[node])

        while queue:
            try:
                source = pop_task()
            except KeyError:
                break  # not sure why last one is throwing error

            if source in permanent_set:
                continue

            permanent_set.add(source)
            spst.add((predecessors[source], source))

            for _, _, dest_ in queue:
                if isinstance(dest_, Removed) or dest_ in permanent_set:
                    continue
                if (source, dest_) in edges:
                    if distances[dest_] > distances[source] + 1:
                        distances[dest_] = distances[source] + 1
                        predecessors[dest_] = source
                        add_task(dest_, distances[dest_])
        return distances[end]

    return run(start), min(run(s) for s in starts)


@register
def day_13(data: str) -> tuple[int, int]:
    """Day 13."""

    def compare(a: list[Any] | int, b: list[Any] | int) -> str:
        if isinstance(a, int) and isinstance(b, int):
            if a == b:
                return "eq"
            if a < b:
                return "lt"
            return "gt"
        if isinstance(a, list) and isinstance(b, list):
            for i, j in zip(a, b):
                if isinstance(i, int) and isinstance(j, list):
                    i = [i]
                if isinstance(i, list) and isinstance(j, int):
                    j = [j]
                    comp = compare(i, j)
                    if comp != "eq":
                        return comp
                comp = compare(i, j)
                if comp != "eq":
                    return comp
            if len(a) == len(b):
                return "eq"
            if len(a) < len(b):
                return "lt"
            return "gt"
        raise ValueError

    @dataclass
    class Packet:
        """A packet"""

        packet: list[Any] | int

        def __lt__(self, other: Any) -> bool:
            if isinstance(other, Packet):
                return compare(self.packet, other.packet) == "lt"
            else:
                raise NotImplementedError

        def __le__(self, other: Any) -> bool:
            if isinstance(other, Packet):
                return compare(self.packet, other.packet) in ["lt", "eq"]
            else:
                raise NotImplementedError

    packets = [
        Packet(literal_eval(x))
        for i, x in enumerate(data.splitlines())
        if i % 3 != 2
    ]
    pairs = [(a, packets[2 * i + 1]) for i, a in enumerate(packets[::2])]
    part_1 = sum([(a <= b) * (i + 1) for i, (a, b) in enumerate(pairs)])

    indices = [Packet([[2]]), Packet([[6]])]
    packets.extend(indices)
    packets.sort()
    part_2 = (packets.index(indices[0]) + 1) * (packets.index(indices[1]) + 1)

    return part_1, part_2


@register
def day_14(data: str) -> tuple[int, int]:
    """Day 14."""

    source = [500, 0]
    cells = defaultdict(lambda: True)
    for shape in [
        [[int(k) for k in j.split(",")] for j in i.split(" -> ")]
        for i in data.splitlines()
    ]:
        for i, (x_2, y_2) in enumerate(shape[1:]):
            x_1, y_1 = shape[i]
            for x in range(min(x_1, x_2), max(x_1, x_2) + 1):
                for y in range(min(y_1, y_2), max(y_1, y_2) + 1):
                    cells[x, y] = False

    x_min = min(coord[0] for coord in cells)
    x_max = max(coord[0] for coord in cells)
    y_max = max(coord[1] for coord in cells)
    is_part_1 = True
    part_1_result = 0
    sand = 0
    while [x, y] != source:
        sand += 1
        x, y = source
        halt_part_1 = False
        while True:
            if cells[x, y + 1] and y < y_max + 1:
                y += 1
            elif cells[x - 1, y + 1] and y < y_max + 1:
                x -= 1
                y += 1
            elif cells[x + 1, y + 1] and y < y_max + 1:
                x += 1
                y += 1
            else:
                break
            if is_part_1 and (y > y_max or x < x_min or x > x_max):
                halt_part_1 = True
                break
        cells[x, y] = False
        if is_part_1 and halt_part_1:
            cells[x, y] = True
            sand -= 1
            part_1_result = sand
            is_part_1 = False

    return part_1_result, sand


@register
def day_15(data: str) -> tuple[int, int]:
    """Day 15."""

    row = 2000000
    limit = 4000000

    sensors: list[tuple[int, int]] = []
    beacons: list[tuple[int, int]] = []

    for sensor_x, sensor_y, beacon_x, beacon_y in (
        [int(s) for s in re.findall(r"-?\d+", i)] for i in data.splitlines()
    ):
        sensors.append((sensor_x, sensor_y))
        beacons.append((beacon_x, beacon_y))

    radii = [
        abs(beacons[i][0] - sensor[0]) + abs(beacons[i][1] - sensor[1])
        for i, sensor in enumerate(sensors)
    ]

    # Probably a faster way of doing this that doesn't require whole
    # ranges of integers to be stored
    no_beacons: set[int] = set()
    confirmed_beacons = set()
    for i, (x, y) in enumerate(sensors):
        radius = abs(beacons[i][0] - x) + abs(beacons[i][1] - y) - abs(row - y)
        if radius >= 1:
            no_beacons.update(range(x - radius, x + radius + 1))
        if beacons[i][1] == row:
            confirmed_beacons.add(beacons[i][0])

    part_1 = len(no_beacons - confirmed_beacons)

    def must_contain_beacon(point: tuple[int, int]) -> bool:
        """Return treu if location contains an undiscovered beacon."""
        return point not in beacons and all(
            abs(point[0] - sensor[0]) + abs(point[1] - sensor[1]) > radii[i]
            for i, sensor in enumerate(sensors)
        )

    def perimeter(x: int, y: int, radius: int) -> list[tuple[int, int]]:
        """Return all perimeter points as a list."""
        points = []
        for r in range(radius):
            s = radius - r
            points.append((x + r, y + s))
            points.append((x + s, y - r))
            points.append((x - r, y - s))
            points.append((x - s, y + r))
        return points

    # For part 2, location must be just outside one or more radii, so
    # try only points that lie just outside perimeters and are within
    # limit range.
    def get_tuning_freq(limit: int) -> int:
        """Find beacon location and return its tuning frequency."""
        for i, sensor in enumerate(sensors):
            for point in perimeter(sensor[0], sensor[1], radii[i] + 1):
                if (
                    0 <= point[0] <= limit
                    and 0 <= point[1] <= limit
                    and must_contain_beacon(point)
                ):
                    return 4000000 * point[0] + point[1]
        raise KeyError("Location not found")

    part_2 = get_tuning_freq(limit)

    return part_1, part_2


@register
def day_16(data: str) -> tuple[int, int]:
    """Day 16."""

    # data = (
    #     "Valve AA has flow rate=0; tunnels lead to valves DD, II, BB\n"
    #     "Valve BB has flow rate=13; tunnels lead to valves CC, AA\n"
    #     "Valve CC has flow rate=2; tunnels lead to valves DD, BB\n"
    #     "Valve DD has flow rate=20; tunnels lead to valves CC, AA, EE\n"
    #     "Valve EE has flow rate=3; tunnels lead to valves FF, DD\n"
    #     "Valve FF has flow rate=0; tunnels lead to valves EE, GG\n"
    #     "Valve GG has flow rate=0; tunnels lead to valves FF, HH\n"
    #     "Valve HH has flow rate=22; tunnel leads to valve GG\n"
    #     "Valve II has flow rate=0; tunnels lead to valves AA, JJ\n"
    #     "Valve JJ has flow rate=21; tunnel leads to valve II\n"
    # )

    nodes = set()
    rates = {}
    edges = {}
    for line in data.splitlines():
        match line.split():
            case _, source, _, _, rate, _, _, _, _, *dests:
                nodes.add(source)
                if (rate_ := int(rate[5:-1])) > 0:
                    rates[source] = rate_
                edges[source] = set(dest.strip(",") for dest in dests)
            case _:
                raise ValueError

    # Note: Very small no of valves with flow; assume all need to be used
    # Find shortest paths between each pair of valves
    # Figure out ordering that gives best result? 8.7e9 options
    return 0, 0


@register
def day_17(data: str) -> tuple[int, int]:
    """Day 17."""
    jets = data.strip()
    width = 7
    start_dy = 4
    shapes = [
        [[2, 0], [3, 0], [4, 0], [5, 0]],
        [[2, 1], [3, 0], [3, 1], [3, 2], [4, 1]],
        [[2, 0], [3, 0], [4, 0], [4, 1], [4, 2]],
        [[2, 0], [2, 1], [2, 2], [2, 3]],
        [[2, 0], [2, 1], [3, 0], [3, 1]],
    ]

    def move_left(
        rock: list[list[int]], static: set[tuple[int, int]]
    ) -> list[list[int]]:
        """Move rock one place to the left, if possible."""
        if all(x > 0 and (x - 1, y) not in static for (x, y) in rock):
            return [[x - 1, y] for x, y in rock]
        return rock

    def move_right(
        rock: list[list[int]], static: set[tuple[int, int]]
    ) -> list[list[int]]:
        """Move rock one place to the right, if possible."""
        if all(x < width - 1 and (x + 1, y) not in static for (x, y) in rock):
            return [[x + 1, y] for x, y in rock]
        return rock

    def move_down(
        rock: list[list[int]], static: set[tuple[int, int]]
    ) -> list[list[int]]:
        """Move rock one place down, if possible."""
        if all((x, y - 1) not in static for (x, y) in rock):
            return [[x, y - 1] for x, y in rock]
        raise ValueError

    n_shapes = len(shapes)
    n_jets = len(jets)
    move = {"<": move_left, ">": move_right}

    def solve(n_rocks: int) -> int:
        """Solve for height."""

        static = set((i, 0) for i in range(width))
        tops = [0] * width
        base_height = 0
        frame_height = 0
        rocks_released = 0
        jet_index = 0
        height_deltas = [0]
        last_height = 0

        big_problem = n_rocks > n_shapes * n_jets

        n_rocks_ = n_shapes * n_jets if big_problem else n_rocks

        while rocks_released < n_rocks_:

            # Can trim search space if tops make convex shape.
            max_tops = max(tops)
            if max_tops - min(tops) <= 1:
                base_height += max_tops
                tops = [i - max_tops for i in tops]
                frame_height = 0
                static = set(enumerate(tops))

            rock = [
                [x, y + frame_height + start_dy]
                for x, y in shapes[rocks_released % n_shapes]
            ]
            rocks_released += 1

            while True:
                rock = move[jets[jet_index]](rock, static)
                jet_index = (jet_index + 1) % n_jets
                try:
                    rock = move_down(rock, static)
                except ValueError:
                    for (x, y) in rock:
                        static.add((x, y))
                        tops[x] = max(tops[x], y)
                        frame_height = max(tops)
                    break

            height = base_height + frame_height
            height_deltas.append(height - last_height)
            last_height = height

        if not big_problem:
            return height

        search_size = len(height_deltas) // 2
        for repeat_length in reversed(range(0, search_size, n_shapes)):
            if (
                height_deltas[-2 * repeat_length : -repeat_length]
                == height_deltas[-repeat_length:]
            ):
                break
        else:
            raise KeyError("Repeating unit not found; try a longer run.")

        repeat_height = sum(height_deltas[-repeat_length:])

        initial = height
        repeating = ((n_rocks - n_rocks_) // repeat_length) * repeat_height
        end = sum(
            height_deltas[
                -repeat_length : -repeat_length
                + (n_rocks - n_rocks_) % repeat_length :
            ]
        )

        return initial + repeating + end

    return solve(2022), solve(1000000000000)


@register
def day_18(data: str) -> tuple[int, int]:
    """Day 18."""

    # Part 2: 2836 is too high

    cubes: set[tuple[int, int, int]] = set(
        tuple(int(i) for i in j.split(",")) for j in data.splitlines()
    )

    moves = [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
    ]

    def surface_area(cubes: set[tuple[int, int, int]]) -> int:
        """Total internal and external surface area."""
        all_surf = sum(
            1
            for x, y, z in cubes
            for i, j, k in moves
            if (x + i, y + j, z + k) not in cubes
        )
        return all_surf

    [(x_min, x_max), (y_min, y_max), (z_min, z_max)] = [
        (min(i) - 1, max(i) + 2) for i in list(zip(*cubes))
    ]

    # bfs
    seen: set[tuple[int, int, int]] = {(x_min, y_min, z_min)}
    queue: deque[tuple[int, int, int]] = deque()
    queue.append((x_min, y_min, z_min))
    while queue:
        here = queue.popleft()
        for move in moves:
            adjacent = (
                here[0] + move[0],
                here[1] + move[1],
                here[2] + move[2],
            )
            print(adjacent)
            if (
                adjacent not in seen
                and adjacent not in cubes
                and x_min <= adjacent[0] <= x_max
                and y_min <= adjacent[1] <= y_max
                and z_min <= adjacent[2] <= z_max
            ):
                seen.add(adjacent)
                queue.append(adjacent)

    mould_area = surface_area(seen)
    width = x_max - x_min + 1
    height = y_max - y_min + 1
    depth = z_max - z_min + 1
    shape_outer_area = mould_area - 2 * (
        width * height + width * depth + height * depth
    )

    return surface_area(cubes), shape_outer_area


def day_20(data: str) -> tuple[int, int]:
    """Day 20."""

    # data = "1\n2\n-3\n3\n-2\n0\n4\n"

    @dataclass
    class Point:
        """A point."""

        value: int

    points = [Point(int(i)) for i in data.splitlines()]
    size = len(points)
    zero = next(point for point in points if point.value == 0)
    spool = deque(points)

    for point in points:
        spool.rotate(-spool.index(point))
        spool.popleft()
        spool.rotate(-(point.value % (size - 1)))
        spool.append(point)

    spool.rotate(-spool.index(zero))
    part_1 = 0
    for _ in range(3):
        spool.rotate(-1000)
        part_1 += spool[0].value

    return part_1, 0


@register
def day_25(data: str) -> str:
    """Day 25."""

    def s2d(snafu: str) -> int:
        """Convert SNAFU to decimal."""
        trans = {"=": -2, "-": -1, "0": 0, "1": 1, "2": 2}
        return sum(5**i * trans[d] for i, d in enumerate(snafu[::-1]))

    def d2s(dec: int) -> str:
        """Convert decimal to SNAFU."""
        trans = {0: "0", 1: "1", 2: "2", 3: "=", 4: "-"}
        s = []
        while dec:
            mod5 = dec % 5
            dec = dec // 5 + 1 if mod5 > 2 else dec // 5
            s.append(trans[mod5])
        return "".join(s)[::-1]

    return d2s(sum(s2d(s) for s in data.splitlines()))


if __name__ == "__main__":
    pass
    # run_all()
    run_problem(day_18)
