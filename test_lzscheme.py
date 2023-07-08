from typing import Any, Optional
from lzscheme import parse as parse_internal, run, stringify_callstack
from lzscheme import Pair, Sexpr, Env, EvalError, NativeFunction

# make assertions easier
def pythonify(s: Optional[Sexpr]) -> Any:
  if s is None:
    return s
  if isinstance(s, Pair):
    return [pythonify(el) for el in s]
  if isinstance(s, NativeFunction):
    return s.fn
  return s.value

def parse(s: str):
  result = parse_internal(s)
  return pythonify(result)

def run_results(src: str, env: Optional[Env]=None):
  _, results = run(src, env)
  return pythonify(results)

def test_parse_atoms():
  assert parse('atom') == 'atom'
  assert parse('another_one') == 'another_one'
  assert parse('+') == '+'

def test_parse_lists():
  assert parse('( )') == []
  assert parse('( atom )') == ['atom']
  assert parse('(atom)') == ['atom']
  assert parse('(atoms are everywhere)') == ['atoms', 'are', 'everywhere']

def test_parse_nested():
  assert parse('(())') == [[]]
  assert parse('(atoms (and lists))') == ['atoms', ['and', 'lists']]
  assert parse('(nested (lists are (very) fun))') == ['nested', ['lists', 'are', ['very'], 'fun']]

def test_parse_string():
  assert parse('"string"') == 'string'
  assert parse('(a "b" c)') == ['a', 'b', 'c']
  assert parse('("apples" ("bananas" "and" "citrus"))') == ['apples', ['bananas', 'and', 'citrus']]

def test_parse_string_empty():
  assert parse('""') == ''

def test_parse_simple_escapes():
  assert parse(r'"string\"s"') == 'string"s'
  assert parse(r'"tab\ttab\ttab"') == 'tab\ttab\ttab'

def test_parse_hex_escapes():
  assert parse(r'"unicod\x65;"') == 'unicode'
  assert parse(r'"\x443;\x43a;\x440;\x430;\x457;\x43d;\x0456;"') == 'україні'

def test_parse_numbers():
  assert parse('14') == 14
  assert parse('-3') == -3
  assert parse('3.14159') == 3.14159
  assert parse('(a 14)') == ['a', 14]

def test_parse_quote_atom():
  assert parse("'a") == ['quote', 'a']

def test_parse_quote_list():
  assert parse("'(a b c)") == ['quote', ['a', 'b', 'c']]

def test_parse_quote_nested():
  assert parse("''a") == ['quote', ['quote', 'a']]
  assert parse("'''(a b)") == ['quote', ['quote', ['quote', ['a', 'b']]]]

def test_eval_atom():
  assert run_results('atom') == 'atom'

def test_eval_list():
  assert run_results('(atom)') == ['atom']

def test_car():
  assert run_results('(car (a b c))') == 'a'

def test_cdr():
  assert run_results('(cdr (a b c))') == ['b', 'c']

def test_cons():
  assert run_results('(cons a (b c))') == ['a', 'b', 'c']

def test_atom_fn():
  assert run_results('(atom? atom)') == True
  assert run_results('(atom? (atom))') == False

def test_null_fn():
  assert run_results('(null? ())') == True
  assert run_results('(null? (atom))') == False

def test_eq_fn():
  assert run_results('(eq? atom atom)') == True
  assert run_results('(eq? atom banana)') == False

def test_nested_call():
  assert run_results('(car (cdr (a b c)))') == 'b'

def test_argument_scoping():
  result = run_results('''
    (define test
      (lambda (a) a))
    (test 1234)
  ''')

  assert str(result) == '1234'

  result = run_results('''
    (define test
      (lambda (a) a))
    a
  ''')

  assert str(result) != '1234'
  assert result == 'a'

def test_higher_order_fn():
  result = run_results('''
    (define thunk
      (lambda () (lambda () "hello")))
    ((thunk))
  ''')
  assert result == 'hello'

def test_lat():
  lat = '''
    (define lat?
      (lambda (l)
        (cond
          ((null? l) #t)
          ((atom? (car l)) (lat? (cdr l)))
          (else #f))))
  '''

  assert run_results(f'''
    {lat}
    (lat? (bacon and eggs))
  ''') == True

  assert run_results(f'''
    {lat}
    (lat? (bacon (and eggs)))
  ''') == False

def test_member():
  member = '''
    (define member?
      (lambda (a lat)
        (cond
          ((null? lat) #f)
          (else (or (eq? (car lat) a) (member? a (cdr lat)))))))
  '''

  assert run_results(f'''
    {member}
    (member? meat (mashed potatoes and meat gravy))
  ''') == True

  assert run_results(f'''
    {member}
    (member? meat (mashed potatoes and meat gravy))
  ''') == True

def test_rember():
  rember = '''
    (define rember
      (lambda (a lat)
        (cond
          ((null? lat) ())
          ((eq? (car lat) a) (cdr lat))
          (else (cons (car lat) (rember a (cdr lat)))))))
  '''

  assert run_results(f'''
    {rember}
    (rember meat (mashed potatoes and meat gravy))
  ''') == ['mashed', 'potatoes', 'and', 'gravy']

  assert run_results(f'''
    {rember}
    (rember mint (lamb chops and mint flavored mint jelly))
  ''') == ['lamb', 'chops', 'and', 'flavored', 'mint', 'jelly']

  assert run_results(f'''
    {rember}
    (rember toast (bacon lettuce and tomato))
  ''') == ['bacon', 'lettuce', 'and', 'tomato']

def test_firsts():
  firsts = '''
    (define firsts
      (lambda (l)
        (cond
          ((null? l) ())
          (else (cons (car (car l)) (firsts (cdr l)))))))
  '''

  assert run_results(f'''
    {firsts}
    (firsts (
      (apple peach pumpkin)
      (plum pear cherry)
      (grape raisin pea)
      (bean carrot eggplant)))
  ''') == ['apple', 'plum', 'grape', 'bean']
  
  assert run_results(f'''
    {firsts}
    (firsts (
      (apple peach pumpkin)
      (plum pear cherry)
      (grape raisin pea)
      (bean carrot eggplant)))
  ''') == ['apple', 'plum', 'grape', 'bean']

def test_load():
  assert run_results('(load std.scm) (member? a (a b c))') == True

def test_math():
  assert run_results('(+ 1 2)') == 3
  assert run_results('(- 3 4)') == -1
  assert run_results('(* 5 6)') == 30
  assert run_results('(/ 6 2)') == 3

  assert run_results('(add1 67)') == 68
  assert run_results('(sub1 5)') == 4
  assert run_results('(zero? 0)') == True
  assert run_results('(zero? 1492)') == False

def test_traceback():
  try:
    run_results('(cons a (cons b (abort)))')
    assert False
  except EvalError as e:
    lines = stringify_callstack(e.callstack).split('\n')
    assert '0' in lines[0]
    assert '(cons a (cons b (abort)))' in lines[0]
    assert '(cons b (abort))' in lines[1]
    assert '(abort)' in lines[2]
    assert '2' in lines[2]
  except:
    assert False

def test_apply():
  assert run_results('(apply + (1 2))') == 3

def test_varargs():
  assert run_results('((lambda (. args) (list args)) 1 2 3)') == [[1, 2, 3]]
  assert run_results('((lambda (a . rest) (list a rest)) 1 2 3)') == [1, [2, 3]]
  assert run_results('((lambda (a b c . rest) (list a b c rest)) 1 2 3 4 5)') == [1, 2, 3, [4, 5]]
  assert run_results('((lambda (. rest) (list rest)))') == [[]]
  assert run_results('((lambda (a . rest) (list a rest)) 1)') == [1, []]
