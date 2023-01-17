from sys import stdin
from contextlib import contextmanager
from textwrap import indent
from typing import Union, Optional, Callable, Any, Iterable, Tuple
import traceback

TRUE = '#t'
FALSE = '#f'

class UnmatchedParenthesesError(RuntimeError):
  def __init__(self, message: str, *, depth: int):
    self.message = message
    self.depth = depth

class Symbol:
  def __init__(self, value: str):
    self.value = value
  
  def __repr__(self) -> str:
    return f'Symbol("{self.value}")'
  
  def __hash__(self) -> int:
    return hash(self.value)
  
  def __eq__(self, o: object) -> bool:
    if not isinstance(o, Symbol):
      return False
    return o.value == self.value
  
  def external(self) -> str:
    return self.value

ValueType = Union[str, bool, int, float]

class Value:
  def __init__(self, value: ValueType):
    self.value = value
  
  def __repr__(self) -> str:
    return f'Value({repr(self.value)})'

  def __eq__(self, o: object) -> bool:
    if not isinstance(o, Value):
      return False
    return o.value == self.value
  
  def external(self) -> str:
    if isinstance(self.value, str):
      return f'"{self.value}"'
    if isinstance(self.value, bool):
      return TRUE if self.value else FALSE
    return str(self.value)

class NativeFunction:
  def __init__(self, fn: Callable[..., Any]):
    self.fn = fn
  
  def __repr__(self) -> str:
    return f'NativeFunction({self.fn.__name__})'

  def external(self) -> str:
    return f'<builtin:{self.fn.__name__}>'

Atom = Union[Symbol, Value, NativeFunction]
Sexpr = Union[Atom, 'Pair']

class Pair:
  def __init__(self, car: Optional[Sexpr]=None, cdr: Optional[Sexpr]=None):
    self.car = car
    self.cdr = cdr
  
  def __repr__(self) -> str:
    return f'Pair({repr(self.car), repr(self.cdr)})'

  def __eq__(self, o: object) -> bool:
    if not isinstance(o, Pair):
      return False
    return o.car == self.car and o.cdr == self.cdr
  
  def __iter__(self):
    p: Optional[Sexpr] = self
    elem = self.car
    while elem is not None:
      yield elem
      p = p.cdr
      if p is None:
        break
      elem = assert_pair(p).car
  
  def __len__(self) -> int:
    counter = 0
    for _ in self:
      counter += 1
    return counter
  
  def __getitem__(self, i: int) -> Sexpr:
    if i < 0:
      raise IndexError(f'negative index {i}')
    counter = 0
    for el in self:
      if counter == i:
        return el
      counter += 1
    raise IndexError(f'index {i} greater than length {counter}')
  
  def external(self) -> str:
    inner = ' '.join(x.external() for x in self)
    return f'({inner})'

class EvalError(RuntimeError):
  def __init__(self, message: str, *, callstack: list[Tuple['Env', Sexpr]]):
    self.message = message
    self.callstack = callstack

class Env:
  names: dict[Symbol, Sexpr]
  callstack: list[Tuple['Env', Sexpr]]

  def __init__(self, *, names: dict[Symbol, Sexpr]={}, callstack: list[Tuple['Env', Sexpr]]=[]):
    self.names = names
    self.callstack = callstack

  def copy(self):
    return Env(names=self.names.copy(), callstack=self.callstack)

  def define(self, symbol: Symbol, value: Sexpr):
    self.names[symbol] = value
    return self

  def resolve(self, sexpr: Sexpr) -> Sexpr:
    if isinstance(sexpr, Symbol) and sexpr in self.names:
      return self.names[sexpr]
    return sexpr
  
  @contextmanager
  def log_call(self, sexpr: Sexpr):
    self.callstack.append((self, sexpr))
    try:
      yield self
    except EvalError as e:
      raise e
    except Exception as e:
      raise EvalError(str(e), callstack=self.callstack.copy()) from e
    finally:
      self.callstack.pop()
  
  @staticmethod
  def from_functions(functions: dict[str, Callable[..., Any]]) -> 'Env':
    names: dict[Symbol, Sexpr] = {Symbol(k): NativeFunction(v) for k, v in functions.items()}
    return Env(names=names)

def assert_pair(x: Optional[Sexpr]) -> Pair:
  if not isinstance(x, Pair):
    raise Exception(f'{x} must be a Pair')
  return x

def assert_symbol(x: Optional[Sexpr]) -> Symbol:
  if not isinstance(x, Symbol):
    raise Exception(f'{x} must be a Symbol')
  return x

def car(pair: Pair):
  if is_null(pair):
    raise Exception('pair is null so it has no car')
  return pair.car

def cdr(pair: Pair):
  if is_null(pair):
    raise Exception('pair is null so it has no cdr')
  return pair.cdr

def cons(a: Sexpr, pair: Pair) -> Pair:
  return Pair(a, pair)

def is_atom(x: Sexpr) -> bool:
  return isinstance(x, Symbol) or isinstance(x, Value) or isinstance(x, NativeFunction)

def is_list(x: Sexpr) -> bool:
  return isinstance(x, Pair)
  
def is_null(x: Sexpr) -> bool:
  return x.car is None if isinstance(x, Pair) else False

def is_lambda(x: Sexpr) -> bool:
  return isinstance(x, Pair) and not is_null(x) and Symbol('lambda') == car(x)

def is_truthy(x: Sexpr) -> bool:
  if Value(False) == x:
    return False
  return True

def string_value(x: Sexpr) -> str:
  if not isinstance(x, Value) and not isinstance(x, Symbol):
    raise Exception(f'argument must be Value or Symbol, was {x}')
  if not isinstance(x.value, str):
    raise Exception(f'argument must have a string value, was {type(x)}')
  return x.value

def numeric_value(x: Sexpr) -> Union[int, float]:
  if not isinstance(x, Value):
    raise Exception(f'argument must be Value, was {x}')
  if not isinstance(x.value, int) and not isinstance(x.value, float):
    raise Exception(f'argument must have a numeric value, was {type(x)}')
  return x.value

def reverse(p: Pair) -> Pair:
  result = Pair()
  for elem in p:
    result = cons(elem, result)
  return result

def is_eq(a: Sexpr, b: Sexpr):
  if not is_atom(a):  
    raise Exception(f'Argument {a} is not an atom')
  if not is_atom(b):  
    raise Exception(f'Argument {b} is not an atom')

  return a == b

def atom_fn(env: Env, a: Sexpr):
  return env, Value(is_atom(a))

def null_fn(env: Env, a: Sexpr):
  return env, Value(is_null(a))

def eq_fn(env: Env, a: Sexpr, b: Sexpr):
  return env, Value(is_eq(a, b))

def car_fn(env: Env, a: Sexpr):
  return env, car(assert_pair(a))

def cdr_fn(env: Env, a: Sexpr):
  return env, cdr(assert_pair(a))

def cons_fn(env: Env, a: Sexpr, b: Sexpr):
  return env, cons(a, assert_pair(b))

def load_fn(env: Env, path: Sexpr):
  with open(string_value(path), 'r') as f:
    env, result = run(f.read(), env)
    return env, result

def define_fn(env: Env, name: Sexpr, sexpr: Sexpr):
  if not isinstance(name, Symbol):
    raise Exception(f'name must be Symbol, was {name}')

  return env.copy().define(name, sexpr), None

def or_fn(env: Env, a: Sexpr, b: Sexpr):
  if is_truthy(a):
    return env, a
  if is_truthy(b):
    return env, b
  return env, Value(False)

def and_fn(env: Env, a: Sexpr, b: Sexpr):
  if is_truthy(a) and is_truthy(b):
    return env, b
  return env, Value(False)

def python_fn(env: Env, bindings: Sexpr, source: Sexpr):
  if not is_list(bindings):
    raise Exception(f'bindings must be a list, was {bindings}')
  
  def get_value(sexpr: Sexpr):
    nonlocal env
    env, result = seval(env, sexpr)
    return result
  
  py_globals: dict[str, Any] = {assert_symbol(sexpr).value: get_value(sexpr) for sexpr in assert_pair(bindings)}
  py_globals["_env"] = env
  py_result = eval(string_value(source), py_globals)
  return env, py_result

def abort_fn(_):
  raise Exception('Aborted!')

builtin_env = Env.from_functions({
  'abort': abort_fn,
  'python': python_fn,
  'load': load_fn,
  'car': car_fn,
  'cdr': cdr_fn,
  'cons': cons_fn,
  'define': define_fn,
  'atom?': atom_fn,
  'null?': null_fn,
  'eq?': eq_fn,
  'or': or_fn,
  'and': and_fn,
  '+': lambda env, a, b: (env, Value(numeric_value(a) + numeric_value(b))),
  '-': lambda env, a, b: (env, Value(numeric_value(a) - numeric_value(b))),
  '*': lambda env, a, b: (env, Value(numeric_value(a) * numeric_value(b))),
  '/': lambda env, a, b: (env, Value(numeric_value(a) / numeric_value(b))),
  'add1': lambda env, a: (env, Value(numeric_value(a) + 1)),
  'sub1': lambda env, a: (env, Value(numeric_value(a) - 1)),
  'zero?': lambda env, a: (env, Value(numeric_value(a) == 0)),
})

def parse(src: Iterable[str]) -> Pair:
  MODE_NORMAL = 'normal'
  MODE_STRING = 'string'
  MODE_ESCAPE = 'escape'
  MODE_ESCAPE_HEX = 'escape_hex'
  SIMPLE_ESCAPES = {
    '"': '"',
    '\\': '\\',
    't': '\t',
    'r': '\r',
    'n': '\n',
  }

  depth = 0
  buffer: list[str] = []
  escape: list[str] = []
  sexpr = Pair()
  mode = MODE_NORMAL

  def flush_buffer(string: bool=False):
    nonlocal sexpr
    if len(buffer) > 0:
      if string:
        result = Value(''.join(buffer))
      else:
        if buffer[0] == '(':
          result = parse(buffer[1:-1])
        else:
          s = ''.join(buffer)
          try:
            result = Value(int(s))
          except:
            try:
              result = Value(float(s))
            except:
              result = Symbol(s)
      sexpr = cons(result, sexpr)
      buffer.clear()

  for c in src:
    if mode == MODE_NORMAL:
      if depth == 0 and (c == ' ' or c == '\n'):
        flush_buffer()
      elif depth == 0 and c == '"':
        mode = MODE_STRING
      elif c == '(':
        depth += 1
        buffer.append(c)
      elif c == ')':
        if depth == 0:
          raise UnmatchedParenthesesError('Unmatched parenthesis: missing (', depth=depth)
        depth -= 1
        buffer.append(c)
      else:
        buffer.append(c)
    elif mode == MODE_STRING:
      if c == '\\':
        mode = MODE_ESCAPE
      elif c == '"':
        flush_buffer(string=True)
        mode = MODE_NORMAL
      else:
        buffer.append(c)
    elif mode == MODE_ESCAPE:
      if c in SIMPLE_ESCAPES:
        buffer.append(SIMPLE_ESCAPES[c])
        mode = MODE_STRING
      elif c == 'x':
        mode = MODE_ESCAPE_HEX
    elif mode == MODE_ESCAPE_HEX:
      if c == ';':
        escape_char = chr(int(''.join(escape), 16))
        escape.clear()
        buffer.append(escape_char)
        mode = MODE_STRING
      else:
        escape.append(c)

  if depth > 0:
    raise UnmatchedParenthesesError('Unmatched parenthesis: missing )', depth=depth)

  flush_buffer()

  return reverse(sexpr)

def stringify(env: Env, sexpr: Optional[Sexpr]) -> str:
  if sexpr is None:
    return 'None'
  return sexpr.external()

def stringify_bindings(env: Env):
  lines = [f'{name.value}: {stringify(env, value)}' 
    for name, value in env.names.items() 
    if not isinstance(value, NativeFunction) and not is_lambda(value)]
  return '\n'.join(lines)

def stringify_callstack(callstack: list[Tuple[Env, Sexpr]], *, include_bindings: bool=False) -> str:
  lines: list[str] = []
  for i, [env, sexpr] in enumerate(callstack):
    lines.append(f"{i: >4}. {stringify(env, sexpr)}")
    if include_bindings:
      lines.append(indent(stringify_bindings(env), ' ' * 6))
      lines.append('')
  return '\n'.join(lines)

def seval(env: Env, sexpr: Sexpr) -> Tuple[Env, Sexpr]:
  with env.log_call(sexpr) as env:
    if isinstance(sexpr, Symbol):
      sexpr = env.resolve(sexpr)
    if isinstance(sexpr, Pair) and not is_null(sexpr):
      first = car(sexpr)
      if first is None: # null list
        return env, sexpr
      if isinstance(first, Symbol):
        first = env.resolve(first)
      if Symbol('quote') == first:
        args = cdr(sexpr)
        if args is None:
          raise Exception('quote requires an argument')
        return env, args
      if Symbol('cond') == first:
        args = assert_pair(cdr(sexpr))
        for arg in args:
          arg = assert_pair(arg)
          predicate = car(arg)
          if predicate is None:
            raise Exception(f'missing predicate in cond clause: {arg}')
          env, predicate_result = seval(env, predicate)
          if is_truthy(predicate_result) or Symbol('else') == predicate_result:
            expressions = assert_pair(cdr(arg))
            result = None
            for expression in expressions:
              env, result = seval(env, expression)
            if result is None:
              raise Exception(f'no expressions in cond clause: {arg}')
            return env, result
      if isinstance(first, NativeFunction):
        args = assert_pair(cdr(sexpr))
        evaled_args: list[Sexpr] = []
        temp_env = env.copy()
        for arg in args:
          temp_env, result = seval(temp_env, arg)
          evaled_args.append(result)
        return first.fn(env, *evaled_args)
      if is_lambda(first):
        if not isinstance(first, Pair):
          raise Exception('invariant violation: lambda is not Pair')
        args = assert_pair(cdr(sexpr))
        params = assert_pair(first[1])
        body = first[2]
        lambda_env = env.copy()
        for param, arg in zip(params, args):
          if not isinstance(param, Symbol):
            raise Exception(f'lambda param {param} is not a symbol')
          lambda_env, arg = seval(lambda_env, arg)
          lambda_env.define(param, arg)
        return env, seval(lambda_env, body)[1]
    return env, sexpr

def seval_multiple(env: Env, sexprs: Pair) -> Tuple[Env, Pair]:
  with env.log_call(sexprs) as env:
    results = Pair()
    for sexpr in sexprs:
      env, result = seval(env, sexpr)
      results = cons(result, results)
    return env, reverse(results)

def run(src: Iterable[str], env: Optional[Env]=None):
  env = env or builtin_env.copy()
  parsed = parse(src)
  env, results = seval_multiple(env, parsed)
  return env, results

def main():
  env = builtin_env.copy()
  input_buffer = ''
  input_depth = 0

  while True:
    prompt = '> ' if input_depth == 0 else '  ' * input_depth + '  '
    print(prompt, end='', flush=True)
    input_buffer += stdin.readline()
    try:
      try:
        env, results = run(input_buffer, env)
        print(' '.join(stringify(env, result) for result in results))
        print()
        input_buffer = ''
        input_depth = 0
      except UnmatchedParenthesesError as e:
        if e.depth > 0:
          input_depth = e.depth
        else:
          raise e
    except EvalError as e:
      print(stringify_callstack(e.callstack, include_bindings=False))
      print(f"Error: {e}")
      print('  Traceback shown above.')
      print()
      input_buffer = ''
      input_depth = 0
    except Exception as e:
      traceback.print_exc()
      print()
      input_buffer = ''
      input_depth = 0

if __name__ == '__main__':
  main()
