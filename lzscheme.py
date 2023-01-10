from sys import stdin
import traceback

TRUE = '#t'
FALSE = '#f'

class UnmatchedParenthesesError(RuntimeError):
  def __init__(self, message, *, depth):
    self.message = message
    self.depth = depth

class Context:
  def __init__(self, *, names={}):
    self.names = names

  def copy(self):
    return Context(names=self.names.copy())

  def define(self, name, value):
    self.names[name] = value
    return self

  def resolve(self, name):
    if isinstance(name, str) and name in self.names:
      return self.names[name]
    return name

def is_list(l):
  return isinstance(l, list)

def list_fn(context, l):
  context, l = seval(context, l)
  return context, TRUE if is_list(l) else FALSE

def is_atom(src):
  return not is_list(src)

def atom_fn(context, sexpr):
  context, sexpr = seval(context, sexpr)
  return context, TRUE if is_atom(sexpr) else FALSE

def is_null(l):
  if not is_list(l):
    raise Exception(f'Argument {l} is not a list')
  return len(l) == 0

def null_fn(context, l):
  context, l = seval(context, l)
  return context, TRUE if is_null(l) else FALSE

def is_eq(a, b):
  if not is_atom(a):  
    raise Exception(f'Argument {a} is not an atom')
  if not is_atom(b):  
    raise Exception(f'Argument {b} is not an atom')
  return a == b

def eq_fn(context, a, b):
  context, a = seval(context, a)
  context, b = seval(context, b)
  return context, TRUE if is_eq(a, b) else FALSE

def car(context, l):
  context, l = seval(context, l)
  if not isinstance(l, list):
    raise Exception(f'Argument {l} is not a list')
  if not len(l) > 0:
    raise Exception(f'Argument is null list')
  return context, l[0]

def cdr(context, l):
  context, l = seval(context, l)
  if not isinstance(l, list):
    raise Exception(f'Argument {l} is not a list')
  if not len(l) > 0:
    raise Exception(f'Argument is null list')
  return context, l[1:]

def cons(context, a, l):
  context, a = seval(context, a)
  context, l = seval(context, l)
  if not isinstance(l, list):
    raise Exception(f'Argument {l} is not a list')
  return context, [a, *l]

def cond(context, *clauses):
  for [test, then] in clauses:
    context, test_result = seval(context, test)
    if test_result == TRUE or test_result == 'else':
      context, then_result = seval(context, then)
      return context, then_result
  return context, None

def define_fn(context, name, sexpr):
  return context.copy().define(name, sexpr), None

def lambda_fn(context, params, sexpr):
  if not is_list(params):
    raise Exception(f'Parameters must be a list, got: {params}')

  def f(f_context, *args):
    new_f_context = f_context.copy()
    for param, arg in zip(params, args):
      new_f_context, arg = seval(new_f_context, arg)
      new_f_context.define(param, arg)
    new_f_context, result = seval(new_f_context, sexpr)
    return new_f_context, result

  return context, f

def or_fn(context, a, b):
  context, a = seval(context, a)
  context, b = seval(context, b)
  return context, TRUE if a == TRUE or b == TRUE else FALSE

builtin_func_table = {
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
}

def parse(src):
  depth = 0
  buffer = []
  sexprs = []

  def flush_buffer():
    if len(buffer) > 0:
      if buffer[0] == '(':
        result = parse(buffer[1:-1])
      else:
        result = ''.join(buffer)
      sexprs.append(result)
      buffer.clear()

  for i, c in enumerate(src):
    if depth == 0 and (c == ' ' or c == '\n'):
      flush_buffer()
    else:
      if c == '(':
        depth += 1
      elif c == ')':
        if depth == 0:
          raise UnmatchedParenthesesError('Unmatched parenthesis: missing (', depth=depth)
        depth -= 1
      buffer.append(c)

  if depth > 0:
    raise UnmatchedParenthesesError('Unmatched parenthesis: missing )', depth=depth)

  flush_buffer()

  return sexprs

def stringify(context, sexprs):
  if not is_list(sexprs):
    return str(sexprs)
  return ' '.join(f'({stringify(context, x)})' if not is_atom(x) else str(x) for x in sexprs)

def seval_list(context, sexprs):
  if len(sexprs) > 0:
    context, func = seval(context, sexprs[0])
    if callable(func):
      args = sexprs[1:]
      context, result = func(context, *args)
      return context, result
  return context, sexprs

def seval(context, sexpr):
  sexpr = context.resolve(sexpr)
  if is_list(sexpr):
    return seval_list(context, sexpr)
  return context, sexpr

def seval_multiple(context, sexprs):
  results = []
  for sexpr in sexprs:
    context, result = seval(context, sexpr)
    results.append(result)
  return context, results

def run(src, context=None):
  context = context or Context(names=builtin_func_table.copy())
  parsed = parse(src)
  context, results = seval_multiple(context, parsed)
  return context, results

def main():
  context = Context(names=builtin_func_table.copy())
  input_buffer = ''
  input_depth = 0

  while True:
    prompt = '> ' if input_depth == 0 else '  ' * input_depth + '  '
    print(prompt, end='', flush=True)
    input_buffer += stdin.readline()
    try:
      try:
        context, result = run(input_buffer, context)
        print(stringify(context, result))
        print()
        input_buffer = ''
        input_depth = 0
      except UnmatchedParenthesesError as e:
        if e.depth > 0:
          input_depth = e.depth
        else:
          raise e
    except Exception as e:
      traceback.print_exc()
      print()
      input_buffer = ''
      input_depth = 0

if __name__ == '__main__':
  main()