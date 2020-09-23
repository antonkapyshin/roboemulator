import re
import sys
import argparse

from enum import Enum
from itertools import zip_longest
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


IDENTIFIER_REGEX = re.compile("[a-zA-Z][a-zA-Z0-9]*")

# Keywords
KEYWORD_FUNCTION = "prikaz"
KEYWORD_END_FUNCTION = "*prikaz"

KEYWORD_HAS_BLOCK = "je tehla"
KEYWORD_NOT = "nie"

KEYWORD_PUT = "poloz"
KEYWORD_PICK = "zober"

KEYWORD_LEFT = "vlavo"
KEYWORD_RIGHT = "vpravo"
KEYWORD_MOVE = "krok"

KEYWORD_IF = "ak"
KEYWORD_ELSE = "inak"
KEYWORD_END_IF = "*ak"


# World
class WorldObject:
    pass


World = List[List[WorldObject]]


@dataclass
class Block(WorldObject):
    value: int

    def __str__(self):
        return str(self.value)

    __repr__ = __str__


class FreeSpace(WorldObject):

    def __str__(self):
        return "."

    __repr__ = __str__


@dataclass
class NestedObject(WorldObject):
    objects: List[WorldObject]

    def __str__(self):
        return str(self.objects[0])

    __repr__ = __str__


class Karel(WorldObject, Enum):
    up = "^"
    left = "<"
    right = ">"
    down = "v"

    def turn_left(self):
        return {
            Karel.up.value: Karel.left,
            Karel.left.value: Karel.down,
            Karel.down.value: Karel.right,
            Karel.right.value: Karel.up,
        }[self.value]

    def turn_right(self):
        return {
            Karel.up.value: Karel.right,
            Karel.left.value: Karel.up,
            Karel.down.value: Karel.left,
            Karel.right.value: Karel.down,
        }[self.value]

    def __str__(self):
        return self.value

    __repr__ = __str__


# Expressions
class Expression:
    pass


class HasBlock(Expression):
    pass


@dataclass
class Not(Expression):
    subexpression: Expression


# Commands
class Command:
    pass


@dataclass
class FunctionCall(Command):
    name: str


@dataclass
class Move(Command):
    pass


@dataclass
class Turn(Command, Enum):
    left = KEYWORD_LEFT
    right = KEYWORD_RIGHT


@dataclass
class Put(Command):
    pass


@dataclass
class Pick(Command):
    pass


@dataclass
class CreateFunction(Command):
    name: str
    body: List[Command]


@dataclass
class IfElse(Command):
    condition: Expression
    if_body: List[Command]
    else_body: List[Command] = field(default_factory=list)


def error(msg: str):
    print(msg)
    sys.exit(1)


class Interpreter:

    def eval_program(self, program: List[Command], world: World) -> World:
        self.world = world
        self.env: Dict[str, CreateFunction] = {}
        self.bag: int = 0
        self.set_initial_pos()
        self.eval_body(program)
        return self.world

    def set_initial_pos(self):
        for i, row in enumerate(self.world):
            for j, obj in enumerate(row):
                if isinstance(obj, Karel):
                    self.row = i
                    self.col = j
                    return

    def get_front(self, get_position: bool = False):
        karel = self.world[self.row][self.col]
        row = self.row
        col = self.col
        if karel == Karel.up:
            if self.row == 0:
                return None

            row = self.row - 1
        elif karel == Karel.down:
            if self.row == len(self.world) - 1:
                return None

            row = self.row + 1
        elif karel == Karel.right:
            if self.col == len(self.world[self.row]) - 1:
                return None

            col = self.col + 1
        elif karel == Karel.left:
            if self.col == 0:
                return None

            col = self.col - 1

        if get_position:
            return row, col
        return self.world[row][col]

    def eval_expression(self, expression: Expression) -> Optional[bool]:
        if isinstance(expression, Not):
            subexpression = self.eval_expression(expression.subexpression)
            if not isinstance(subexpression, bool):
                error("not used on non-bool expression")
            return not subexpression
        elif isinstance(expression, HasBlock):
            front = self.get_front()
            return not front or isinstance(front, Block)
        else:
            error(f"unknown expression object '{expression}'")

    def resolve_object_move(self, current: WorldObject, front: WorldObject) -> Tuple[WorldObject, WorldObject]:
        """Return (front, current)."""
        if isinstance(current, Karel):
            if isinstance(front, FreeSpace):
                return current, front
            elif isinstance(front, Block):
                # Karel is the first one
                return NestedObject([current, front]), FreeSpace()
            else:
                assert isinstance(front, NestedObject)
                # Karel is the first one
                front.objects = [current] + front.objects
                return front, FreeSpace()
        else:
            assert isinstance(current, NestedObject)
            karel = current.objects[0]
            current.objects = current.objects[1:]

            # No more need for nesting
            if len(current.objects) == 1:
                current = current.objects[0]

            if isinstance(front, FreeSpace):
                return karel, current
            elif isinstance(front, Block):
                return NestedObject([karel, front]), current
            error(f"unexpected object '{front}'")

    def eval_command(self, command: Command):
        if isinstance(command, CreateFunction):
            self.env[command.name] = command
        elif isinstance(command, IfElse):
            condition = self.eval_expression(command.condition)
            if condition:
                self.eval_body(command.if_body)
            else:
                self.eval_body(command.else_body)
        elif isinstance(command, FunctionCall):
            function = self.env.get(command.name)
            if not function:
                error(f"unknown function '{command.name}'")
            return self.eval_body(function.body)
        elif isinstance(command, Move):
            front_pos = self.get_front(get_position=True)
            if not front_pos:
                error(f"cannot move!")

            row, col = self.row, self.col
            current_obj = self.world[self.row][self.col]
            self.row, self.col = front_pos
            front_obj = self.world[self.row][self.col]

            new_front_obj, new_current_obj = self.resolve_object_move(current_obj, front_obj)

            self.world[self.row][self.col] = new_front_obj
            self.world[row][col] = new_current_obj
        elif isinstance(command, Turn):
            karel = self.world[self.row][self.col]
            if command == Turn.left:
                karel = karel.turn_left()
            else:
                karel = karel.turn_right()
            self.world[self.row][self.col] = karel
        elif isinstance(command, Pick):
            front = self.get_front()
            if not isinstance(front, Block):
                error("cannot pick: it is empty!")
            front.value -= 1
            if front.value == 0:
                row, col = self.get_front(get_position=True)
                self.world[row][col] = FreeSpace()
            self.bag += 1
        elif isinstance(command, Put):
            front_pos = self.get_front(get_position=True)
            if not front_pos:
                error("cannot put: it is an edge!")
            row, col = front_pos
            front_obj = self.world[row][col]
            if isinstance(front_obj, Block):
                front_obj.value += self.bag
                self.bag = 0
            elif isinstance(front_obj, FreeSpace):
                self.world[row][col] = Block(self.bag)
                self.bag = 0
            else:
                error(f"unknown world object to put on '{front_obj}'")
        else:
            error(f"unknown command '{command}'")

    def eval_body(self, body: List[Command]):
        for command in body:
            self.eval_command(command)


def process_object(obj: str) -> WorldObject:
    try:
        return Block(int(obj))
    except ValueError:
        pass

    try:
        return Karel(obj)
    except ValueError:
        pass

    return FreeSpace()


def load_world(content: str) -> World:
    """
    Example of a valid world:
     . . . .
     . 4 . .
     1 ^ 3 .
     . 2 . .
     . . . .
    """
    world = []

    for row in content.splitlines():
        world.append([process_object(obj) for obj in row.split(" ")])

    return world


class Parser:
    def parse_raw(self, string: str) -> bool:
        pos = self.pos
        for expected, got in zip_longest(string, self.content[self.pos:]):
            if expected is None:
                break
            elif got is None:
                return False
            elif expected != got:
                return False
            pos += 1
        self.pos = pos
        return True

    def parse_keyword(self, keyword: str) -> bool:
        return self.parse_raw(keyword) and self.content[self.pos].isspace()

    def parse_function_creation(self) -> Optional[CreateFunction]:
        self.spaces()
        if not self.parse_keyword(KEYWORD_FUNCTION):
            return None

        self.spaces()
        name = self.parse_function_call()
        if not name:
            error("expected function name")
        self.spaces()

        body, _ = self.parse_body([KEYWORD_END_FUNCTION])
        return CreateFunction(name.name, body)

    def parse_expression(self) -> Optional[Expression]:
        if not self.parse_keyword(KEYWORD_NOT):
            return self.parse_has_block()
        self.spaces()
        subexpression = self.parse_expression()
        if subexpression is None:
            return None
        return Not(subexpression)

    def parse_has_block(self) -> Optional[HasBlock]:
        if not self.parse_keyword(KEYWORD_HAS_BLOCK):
            return None
        return HasBlock()

    def parse_if_else(self) -> Optional[IfElse]:
        self.spaces()
        if not self.parse_keyword(KEYWORD_IF):
            return None

        self.spaces()
        condition = self.parse_expression()
        if not condition:
            return None
        self.spaces()

        if_body, parsed_keyword_n = self.parse_body([KEYWORD_END_IF, KEYWORD_ELSE])
        if parsed_keyword_n == 0:
            return IfElse(condition, if_body)

        self.spaces()
        else_body, _ = self.parse_body([KEYWORD_END_IF])
        return IfElse(condition, if_body, else_body)

    def parse_function_call(self) -> Optional[FunctionCall]:
        self.spaces()
        match = IDENTIFIER_REGEX.match(self.content[self.pos:])
        if match is None:
            return None
        match_length = len(match[0])
        self.pos += match_length
        return FunctionCall(match[0])

    def parse_builtin_command(self) -> Optional[Command]:
        self.spaces()
        if self.parse_keyword(KEYWORD_LEFT):
            return Turn.left

        if self.parse_keyword(KEYWORD_RIGHT):
            return Turn.right

        if self.parse_keyword(KEYWORD_MOVE):
            return Move()

        if self.parse_keyword(KEYWORD_PICK):
            return Pick()

        if self.parse_keyword(KEYWORD_PUT):
            return Put()
        return None

    parsers = [parse_function_creation, parse_if_else, parse_builtin_command, parse_function_call]

    def parse_body(self, end_keywords: List[str]) -> Tuple[List[Command], int]:
        def parse_kwds() -> Tuple[bool, int]:
            for i, keyword in enumerate(end_keywords):
                if self.parse_keyword(keyword):
                    return True, i
            return False, -1

        body = []
        self.spaces()
        parsed, i = parse_kwds()
        while not parsed:
            any_parser = False
            for parser in self.parsers:
                result = parser(self)
                if result:
                    any_parser = True
                    body.append(result)
                    break
            if not any_parser:
                error(f"cannot parse starting from '{self.content[self.pos:self.pos+20]}'")
            self.spaces()
            parsed, i = parse_kwds()
        return body, i

    def spaces(self):
        for char in self.content[self.pos:]:
            if not char.isspace():
                break
            self.pos += 1

    def parse_program(self, content: str) -> List[Command]:
        self.content = content
        self.pos = 0
        max_pos = len(content)
        program: List[Command] = []
        while self.pos < max_pos:
            any_parser = False
            for parser in self.parsers:
                result = parser(self)
                if result:
                    any_parser = True
                    program.append(result)
                    break
            if not any_parser:
                break
        return program


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("program")
    parser.add_argument("--world", required=True)
    args = parser.parse_args()

    with open(args.program) as file:
        program = Parser().parse_program(file.read())

    with open(args.world) as file:
        world = load_world(file.read())

    print('\n'.join(' '.join(map(str, row)) for row in Interpreter().eval_program(program, world)))


if __name__ == "__main__":
    main()
