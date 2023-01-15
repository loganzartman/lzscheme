from sys import stdin
from contextlib import contextmanager
import traceback

TRUE = '#t'
FALSE = '#f'

class UnmatchedParenthesesError(RuntimeError):
  def __init__(self, message, *, depth):
    self.message = message
    self.depth = depth

class Env:
  def __init__(self, *, names={}, callstack=[]):
    self.names = names
    self.callstack = callstack

  def copy(self):
    return Env(names=self.names.copy(), callstack=self.callstack)

  def define(self, name, value):
    self.names[name] = value
    return self

  def resolve(self, name):
    if isinstance(name, str) and name in self.names:
      return self.names[name]
    return name
  
  @contextmanager
  def log_call(self, sexpr):
    self.callstack.append((self, sexpr))
    try:
      yield self
    except Exception as e:
      if not hasattr(e, '_callstack'):
        e._callstack = self.callstack.copy()
      raise e
    finally:
      self.callstack.pop()

def is_list(l):
  return isinstance(l, list)

def list_fn(env, l):
  env, l = seval(env, l)
  return env, TRUE if is_list(l) else FALSE

def is_atom(src):
  return not is_list(src)

def atom_fn(env, sexpr):
  env, sexpr = seval(env, sexpr)
  return env, TRUE if is_atom(sexpr) else FALSE

def is_null(l):
  if not is_list(l):
    raise Exception(f'Argument {l} is not a list')
  return len(l) == 0

def null_fn(env, l):
  env, l = seval(env, l)
  return env, TRUE if is_null(l) else FALSE

def is_eq(a, b):
  if not is_atom(a):  
    raise Exception(f'Argument {a} is not an atom')
  if not is_atom(b):  
    raise Exception(f'Argument {b} is not an atom')
  return a == b

def eq_fn(env, a, b):
  env, a = seval(env, a)
  env, b = seval(env, b)
  return env, TRUE if is_eq(a, b) else FALSE

def load_fn(env, path):
  env, path = seval(env, path)
  with open(path, 'r') as f:
    env, result = run(f.read(), env)
    return env, result

def quote_fn(env, x):
  return env, x

def car(env, l):
  env, l = seval(env, l)
  if not isinstance(l, list):
    raise Exception(f'Argument {l} is not a list')
  if not len(l) > 0:
    raise Exception(f'Argument is null list')
  return env, l[0]

def cdr(env, l):
  env, l = seval(env, l)
  if not isinstance(l, list):
    raise Exception(f'Argument {l} is not a list')
  if not len(l) > 0:
    raise Exception(f'Argument is null list')
  return env, l[1:]

def cons(env, a, l):
  env, a = seval(env, a)
  env, l = seval(env, l)
  if not isinstance(l, list):
    raise Exception(f'Argument {l} is not a list')
  return env, [a, *l]

def cond(env, *clauses):
  for [test, then] in clauses:
    env, test_result = seval(env, test)
    if test_result == TRUE or test_result == 'else':
      env, then_result = seval(env, then)
      return env, then_result
  return env, None

def define_fn(env, name, sexpr):
  return env.copy().define(name, sexpr), None

def lambda_fn(env, params, sexpr):
  if not is_list(params):
    raise Exception(f'Parameters must be a list, got: {params}')

  def f(f_env, *args):
    new_f_env = f_env.copy()
    for param, arg in zip(params, args):
      new_f_env, arg = seval(new_f_env, arg)
      new_f_env.define(param, arg)
    _, result = seval(new_f_env, sexpr)
    return f_env, result
  f._sexpr = ['lambda', params, sexpr]

  return env, f

def or_fn(env, a, b):
  env, a = seval(env, a)
  env, b = seval(env, b)
  return env, TRUE if a == TRUE or b == TRUE else FALSE

def python_fn(env, bindings, source):
  if not is_list(bindings):
    raise Exception('Bindings must be a list')
  if not isinstance(source, str):
    raise Exception('Source must be a string')
  
  def get_value(name):
    nonlocal env
    env, result = seval(env, name)
    return result
  
  py_globals = {name: get_value(name) for name in bindings}
  py_globals["_env"] = env
  py_result = eval(source, py_globals)
  return env, py_result

def seval_args(fn):
  def bound(env, *args):
    new_args = []
    for arg in args:
      env, evaled = seval(env, arg)
      new_args.append(evaled)
    return fn(env, *new_args)
  return bound

def abort_fn(_):
  raise Exception('Aborted!')

builtin_func_table = {
  'abort': abort_fn,
  'python': python_fn,
  'load': load_fn,
  'quote': quote_fn,
  'car': car,
  'cdr': cdr,
  'cons': cons,
  'cond': cond,
  'define': define_fn,
  'lambda': lambda_fn,
  'atom?': atom_fn,
  'null?': null_fn,
  'eq?': eq_fn,
  'or': or_fn,
  '+': seval_args(lambda env, a, b: (env, a + b)),
  '-': seval_args(lambda env, a, b: (env, a - b)),
  '*': seval_args(lambda env, a, b: (env, a * b)),
  '/': seval_args(lambda env, a, b: (env, a / b)),
  'add1': seval_args(lambda env, a: (env, a + 1)),
  'sub1': seval_args(lambda env, a: (env, a - 1)),
  'zero?': seval_args(lambda env, a: (env, TRUE if a == 0 else FALSE)),
}

def parse(src):
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
  buffer = []
  escape = []
  sexprs = []
  mode = MODE_NORMAL

  def flush_buffer(string=False):
    if len(buffer) > 0:
      if not string and buffer[0] == '(':
        result = parse(buffer[1:-1])
      else:
        result = ''.join(buffer)
        try:
          result = int(result)
        except:
          try:
            result = float(result)
          except:
            pass
      sexprs.append(result)
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

  return sexprs

def stringify(env, sexpr):
  if sexpr is None:
    return 'None'
  if hasattr(sexpr, '_sexpr'):
    return stringify(env, sexpr._sexpr)
  if callable(sexpr):
    return f'<builtin:{sexpr.__name__}>'
  if is_list(sexpr):
    return f'({" ".join(stringify(env, x) for x in sexpr)})'
  return str(sexpr)

def stringify_callstack(callstack):
  lines = []
  for i, [env, sexpr] in enumerate(callstack):
    lines.append(f"{i}: {stringify(env, sexpr)}")
  return '\n'.join(lines)

def seval_list(env, sexprs):
  if len(sexprs) > 0:
    env, func = seval(env, sexprs[0])
    if callable(func):
      args = sexprs[1:]
      env, result = func(env, *args)
      return env, result
  return env, sexprs

def seval(env, sexpr):
  with env.log_call(sexpr) as env:
    sexpr = env.resolve(sexpr)
    if is_list(sexpr):
      return seval_list(env, sexpr)
    return env, sexpr

def seval_multiple(env, sexprs):
  with env.log_call(sexprs) as env:
    results = []
    for sexpr in sexprs:
      env, result = seval(env, sexpr)
      results.append(result)
    return env, results

def run(src, env=None):
  env = env or Env(names=builtin_func_table.copy())
  parsed = parse(src)
  env, results = seval_multiple(env, parsed)
  return env, results

def main():
  env = Env(names=builtin_func_table.copy())
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
    except Exception as e:
      if hasattr(e, '_callstack'):
        print(f"Error: {e}")
        print(stringify_callstack(e._callstack))
      else:
        traceback.print_exc()
      print()
      input_buffer = ''
      input_depth = 0

if __name__ == '__main__':
  main()