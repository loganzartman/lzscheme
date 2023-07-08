import re
from sys import stdin
from collections import OrderedDict
from contextlib import contextmanager
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
  
  def __str__(self) -> str:
    return self.external()
  
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

  def __str__(self) -> str:
    return self.external()

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

  def __str__(self) -> str:
    return self.external()

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

  def __str__(self) -> str:
    return self.external()

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
  parent: Optional['Env']

  def __init__(self, *, names: dict[Symbol, Sexpr]={}, callstack: list[Tuple['Env', Sexpr]]=[], parent: Optional['Env'] = None):
    self.names = names
    self.callstack = callstack
    self.parent = parent

  def copy(self):
    return Env(names={}, callstack=self.callstack, parent=self)

  def define(self, symbol: Symbol, value: Sexpr):
    self.names[symbol] = value
    return self

  def resolve(self, sexpr: Sexpr) -> Sexpr:
    if isinstance(sexpr, Symbol):
      env = self
      while env is not None:
        if sexpr in env.names:
          return env.names[sexpr]
        env = env.parent
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

def assert_not_none(x: Optional[Sexpr]) -> Sexpr:
  if x is None:
    raise Exception(f'expression did not evaluate to any value')
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

def is_sexpr(x: Any) -> bool:
  return is_atom(x) or is_list(x)
  
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
    raise Exception(f'argument must have a numeric value, was {type(x.value)}')
  return x.value

def reverse(p: Pair) -> Pair:
  result = Pair()
  for elem in p:
    result = cons(elem, result)
  return result

def build_list(l: list[Sexpr]) -> Pair:
  result = Pair()
  for x in reversed(l):
    result = cons(x, result)
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

def not_fn(env: Env, a: Sexpr):
  if is_truthy(a):
    return env, Value(False)
  return env, Value(True)

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
  
  bindings_map: dict[str, Any] = {assert_symbol(sexpr).value: get_value(sexpr) for sexpr in assert_pair(bindings)}
  bindings_map["_env"] = env

  py_result = eval(string_value(source), {**globals(), **bindings_map})

  if not is_sexpr(py_result):
    py_result = Value(py_result)

  return env, py_result

def raise_fn(env: Env, a: Any="Error occurred."):
  raise Exception(string_value(a))

builtin_env = Env.from_functions({
  'raise': raise_fn,
  'python': python_fn,
  'load': load_fn,
  'car': car_fn,
  'cdr': cdr_fn,
  'cons': cons_fn,
  'atom?': atom_fn,
  'null?': null_fn,
  'eq?': eq_fn,
  'not': not_fn,
  'or': or_fn,
  # 'and': and_fn,
  'list': lambda env, *items: (env, build_list(items)),
  '+': lambda env, a, b: (env, Value(numeric_value(a) + numeric_value(b))),
  '-': lambda env, a, b: (env, Value(numeric_value(a) - numeric_value(b))),
  '*': lambda env, a, b: (env, Value(numeric_value(a) * numeric_value(b))),
  '/': lambda env, a, b: (env, Value(numeric_value(a) / numeric_value(b))),
  'add1': lambda env, a: (env, Value(numeric_value(a) + 1)),
  'sub1': lambda env, a: (env, Value(numeric_value(a) - 1)),
  'zero?': lambda env, a: (env, Value(numeric_value(a) == 0)),
})

token_patterns: OrderedDict[str, str] = OrderedDict([
  ('whitespace', r'[\s\r\n]*'),
  ('comment', r';[^\r\n]*(?:\r\n|\r|\n)'),
  ('open', r'[(\[{]'),
  ('close', r'[)\]}]'),
  ('quote', r"'"),
  ('quasiquote', r'`'),
  ('unquote', r','),
  ('unquote-splicing', r',@'),
  ('string', r'"(?:\\"|[^"])*"'),
  ('literal', r'''[^"'`,@\s()\[\]{}]+'''),
])

Token = tuple[str, str]

def tokenize(src: Iterable[str]) -> Iterable[Token]:
  src = ''.join(src)

  while len(src):
    longest_match_name = None
    longest_match = None
    for name, pattern in token_patterns.items():
      match = re.match(pattern, src)
      if match is not None:
        if longest_match is None or len(match[0]) > len(longest_match[0]):
          longest_match_name = name
          longest_match = match
    
    if not longest_match:
      raise Exception(f'cannot tokenize input: {src}')
    assert longest_match_name is not None
    yield (longest_match_name, longest_match[0])

    src = src[len(longest_match[0]):]

pattern_hex_escape = re.compile(r'\\x([a-eA-E0-9]+);')
pattern_other_escape = re.compile(r'\\(\S)')
escape_table = {
  '"': '"',
  '\\': '\\',
  't': '\t',
  'r': '\r',
  'n': '\n',
}
def parse_string(src: str) -> str:
  value = src[1:-1]
  value = re.sub(pattern_hex_escape, lambda match: chr(int(match[1], 16)), value)
  value = re.sub(pattern_other_escape, lambda match: escape_table[match[1]] if match[1] in escape_table else match[1], value)
  return value

def parse_apply_quotes(stack: list[Sexpr]):
  top = stack.pop()
  while len(stack):
    if Symbol("'") == stack[-1]:
      top = Pair(Symbol('quote'), Pair(top))
    elif Symbol("`") == stack[-1]:
      top = Pair(Symbol('quasiquote'), Pair(top))
    elif Symbol(",") == stack[-1]:
      top = Pair(Symbol('unquote'), Pair(top))
    elif Symbol(",@") == stack[-1]:
      top = Pair(Symbol('unquote-splicing'), Pair(top))
    else:
      break
    stack.pop()
  stack.append(top)

def parse_tokens(src: Iterable[Token]) -> Optional[Sexpr]:
  stack: list[Sexpr] = []
  for name, value in src:
    if name == 'whitespace':
      pass
    elif name == 'comment':
      pass
    elif name in {'quote', 'quasiquote', 'unquote', 'unquote-splicing'}:
      stack.append(Symbol(value)) 
    elif name == 'open':
      stack.append(Symbol('('))
    elif name == 'close':
      result = Pair()

      while Symbol('(') != stack[-1]:
        result = cons(stack.pop(), result)
        if not len(stack):
          raise UnmatchedParenthesesError('Umatched closing parenthesis', depth=0)
      stack.pop()

      stack.append(result)
      parse_apply_quotes(stack)
    elif name == 'literal':
      if value == TRUE:
        result = Value(True)
      elif value == FALSE:
        result = Value(False)
      else:
        try:
          result = Value(int(value))
        except:
          try:
            result = Value(float(value))
          except:
            result = Symbol(value)
      stack.append(result)
      parse_apply_quotes(stack)
    elif name == 'string':
      stack.append(Value(parse_string(value)))
      parse_apply_quotes(stack)
    else:
      raise Exception(f'unsupported token type: {name}')
  
  # raise if any open parens remain
  depth = 0
  for sexpr in stack:
    if Symbol('(') == sexpr:
      depth += 1
  if depth > 0:
    raise UnmatchedParenthesesError('Unmatched opening parenthesis', depth=depth)

  if len(stack) > 1:
    # convert multiple bare sexprs into a begin block
    return Pair(Symbol('begin'), build_list(stack))
  elif len(stack):
    return stack[0]
  else:
    return None

def parse(src: Iterable[str]) -> Optional[Sexpr]:
  return parse_tokens(tokenize(src))

def stringify(env: Env, sexpr: Optional[Sexpr]) -> str:
  if sexpr is None:
    return 'None'
  return sexpr.external()

def stringify_bindings(env: Env, *, include_lambdas: bool) -> list[str]:
  names = [name.value for name in env.names.keys()]
  if len(names):
    longest_name_len = max(len(name) for name in names)
    lines = [f'{name.value:>{longest_name_len}}: {stringify(env, value)}' 
      for name, value in env.names.items()
      if not is_lambda(value) or include_lambdas]
    return lines
  return []

def stringify_callstack(callstack: list[Tuple[Env, Sexpr]], *, include_bindings: bool=False, include_lambdas: bool=False) -> str:
  lines: list[str] = []
  base_indent = len(str(len(callstack)))
  for i, [env, sexpr] in enumerate(callstack):
    if include_bindings:
      bindings_lines = [' ' * (2 + base_indent) + x for x in stringify_bindings(env, include_lambdas=include_lambdas)]
      if len(bindings_lines):
        lines.append('')
        lines += bindings_lines
        lines.append('')
    lines.append(f"{i: >{base_indent}}. {stringify(env, sexpr)}")
  return '\n'.join(lines)

def seval(env: Env, sexpr: Optional[Sexpr]) -> Tuple[Env, Optional[Sexpr]]:
  if sexpr is None:
    return env, None

  with env.log_call(sexpr) as env:
    if isinstance(sexpr, Symbol):
      sexpr = env.resolve(sexpr)

    if isinstance(sexpr, Pair) and not is_null(sexpr):
      first = car(sexpr)

      if first is None: # null list
        return env, sexpr

      env, first = seval(env, first)
      first = assert_not_none(first)

      if Symbol('quote') == first:
        args = cdr(sexpr)
        if args is None:
          raise Exception('quote requires an argument')
        args = assert_pair(args)
        if len(args) != 1:
          raise Exception('quote requires exactly one argument')
        return env, args[0]
      
      if Symbol('quasiquote') == first:
        args = cdr(sexpr)
        if args is None:
          raise Exception('quasiquote requires an argument')
        args = assert_pair(args)
        if len(args) != 1:
          raise Exception('quasiquote requires exactly one argument')
        
        def visit(env: Env, node: Sexpr) -> tuple[Env, Optional[Sexpr]]:
          if is_list(node):
            node = assert_pair(node)
            first = car(node)
            if Symbol('unquote') == first:
              return seval(env, car(assert_pair(cdr(node))))
            else:
              result = Pair()
              for item in node:
                env, item_result = visit(env, item)
                result = cons(assert_not_none(item_result), result)
              return env, reverse(result)
          else:
            return env, node

        return visit(env, args[0])
      
      if Symbol('unquote') == first:
        raise Exception('unquote found outside of quasiquote')

      if Symbol('cond') == first:
        args = assert_pair(cdr(sexpr))
        for arg in args:
          arg = assert_pair(arg)
          predicate = car(arg)
          if predicate is None:
            raise Exception(f'missing predicate in cond clause: {arg}')
          env, predicate_result = seval(env, predicate)
          if is_truthy(assert_not_none(predicate_result)) or Symbol('else') == predicate_result:
            expressions = assert_pair(cdr(arg))
            result = None
            for expression in expressions:
              env, result = seval(env, expression)
            if result is None:
              raise Exception(f'no expressions in cond clause: {arg}')
            return env, result

      if Symbol('define') == first:
        name, value = assert_pair(cdr(sexpr))
        env, value = seval(env, value)
        env.define(assert_symbol(name), assert_not_none(value))
        return env, None

      if Symbol('begin') == first:
        expressions = assert_pair(cdr(sexpr))
        result = None
        for expression in expressions:
          env, result = seval(env, expression)
        return env, result
      
      if Symbol('apply') == first:
        rest = cdr(assert_pair(sexpr))
        fn = car(assert_pair(rest))
        env, fn = seval(env, fn)
        args = car(assert_pair(cdr(assert_pair(rest))))
        env, args = seval(env, args)
        call_sexpr = cons(assert_not_none(fn), assert_pair(args))
        return seval(env, call_sexpr)

      if isinstance(first, NativeFunction):
        args = assert_pair(cdr(sexpr))
        evaled_args: list[Sexpr] = []
        temp_env = env.copy()
        for arg in args:
          temp_env, result = seval(temp_env, arg)
          evaled_args.append(assert_not_none(result))
        return first.fn(env, *evaled_args)

      if is_lambda(first):
        if not isinstance(first, Pair):
          raise Exception('invariant violation: lambda is not Pair')
        args = assert_pair(cdr(sexpr))
        params = assert_pair(first[1])
        body = first[2]
        lambda_env = env.copy()

        args_iter = iter(args)
        params_iter = iter(params)

        for param in params_iter:
          # varargs
          if param == Symbol('.'):
            param = next(params_iter, None)
            if param is not None:
              lambda_env.define(assert_symbol(param), build_list(list(args_iter)))
            else:
              raise Exception(f'Unexpected symbol after . in parameters: {param}')
            break
          # positional args
          lambda_env, arg = seval(lambda_env, next(args_iter))
          lambda_env.define(assert_symbol(param), assert_not_none(arg))

        return env, seval(lambda_env, body)[1]

    return env, sexpr

def run(src: Iterable[str], env: Optional[Env]=None):
  env = env or builtin_env.copy()
  parsed = parse(src)
  env, result = seval(env, parsed)
  return env, result

def repl():
  env = builtin_env.copy()
  input_buffer = ''
  input_depth = 0

  while True:
    prompt = '> ' if input_depth == 0 else '  ' * input_depth + '  '
    print(prompt, end='', flush=True)
    input_buffer += stdin.readline()
    try:
      try:
        env, result = run(input_buffer, env)
        if result is not None:
          print(stringify(env, result))
        print()
        input_buffer = ''
        input_depth = 0
      except UnmatchedParenthesesError as e:
        if e.depth > 0:
          input_depth = e.depth
        else:
          raise e
    except EvalError as e:
      print(stringify_callstack(e.callstack, include_bindings=True, include_lambdas=False))
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

def main():
  if stdin.isatty():
    repl()
  else:
    env, result = run(stdin.read())
    print(stringify(env, result))

if __name__ == '__main__':
  main()
