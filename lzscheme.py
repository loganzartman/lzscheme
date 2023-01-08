from sys import stdin

class UnmatchedParenthesesError(RuntimeError):
  def __init__(self, message, *, depth):
    self.message = message
    self.depth = depth

def is_list(l):
  return isinstance(l, list)

def is_atom(src):
  return not is_list(src)

def is_null(l):
  if not is_list(l):
    raise Exception(f'Argument {l} is not a list')
  return len(l) == 0

def is_eq(a, b):
  if not is_atom(a):  
    raise Exception(f'Argument {a} is not an atom')
  if not is_atom(b):  
    raise Exception(f'Argument {b} is not an atom')
  return a == b

def car(l):
  if not isinstance(l, list):
    raise Exception(f'Argument {l} is not a list')
  if not len(l) > 0:
    raise Exception(f'Argument is null list')
  return l[0]

def cdr(l):
  if not isinstance(l, list):
    raise Exception(f'Argument {l} is not a list')
  if not len(l) > 0:
    raise Exception(f'Argument is null list')
  return l[1:]

def cons(a, l):
  if not isinstance(l, list):
    raise Exception(f'Argument {l} is not a list')
  return [a, *l]

func_table = {
  'car': car,
  'cdr': cdr,
  'cons': cons,
  'atom?': is_atom,
  'null?': is_null,
  'eq?': is_eq,
}

def eval_list(src):
  sexprs = eval(src[1:-1])
  if len(sexprs) > 0:
    func_name = sexprs[0]
    if is_atom(func_name) and func_name in func_table:
      args = sexprs[1:]
      print('  calling', func_name, args)
      res = func_table[func_name](*args)
      print('  result', res)
      return res
  return sexprs

def eval(src):
  depth = 0
  buffer = []
  sexprs = []

  def flush_buffer():
    if len(buffer) > 0:
      if buffer[0] == '(':
        result = eval_list(buffer)
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

def stringify(sexprs):
  return ' '.join(f'({stringify(x)})' if not is_atom(x) else str(x) for x in sexprs)

def main():
  input_buffer = ''
  input_depth = 0
  while True:
    prompt = '> ' if input_depth == 0 else '  ' * input_depth + '  '
    print(prompt, end='', flush=True)
    input_buffer += stdin.readline()
    try:
      print(stringify(eval(input_buffer)))
      print()
      input_buffer = ''
      input_depth = 0
    except UnmatchedParenthesesError as e:
      if e.depth == 0:
        print(e)
      else:
        input_depth = e.depth
    except Exception as e:
      print(e)
      print()
      input_buffer = ''
      input_depth = 0

if __name__ == '__main__':
  main()