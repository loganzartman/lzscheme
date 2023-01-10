from lzscheme import parse, run

def run_results(src, context=None):
  _, results = run(src, context)
  return results

def test_parse_atoms():
  assert parse('atom') == ['atom']
  assert parse('an atom') == ['an', 'atom']
  assert parse('so  many   atoms') == ['so', 'many', 'atoms']

def test_parse_lists():
  assert parse('( )') == [[]]
  assert parse('( atom )') == [['atom']]
  assert parse('(atom)') == [['atom']]
  assert parse('(atoms are everywhere)') == [['atoms', 'are', 'everywhere']]
  assert parse('(lists) (hold) (many things)') == [['lists'], ['hold'], ['many', 'things']]

def test_parse_nested():
  assert parse('(())') == [[[]]]
  assert parse('(atoms (and lists))') == [['atoms', ['and', 'lists']]]
  assert parse('nested (lists are (very (fun)))') == ['nested', ['lists', 'are', ['very', ['fun']]]]

def test_eval_atom():
  assert run_results('atom') == ['atom']

def test_eval_list():
  assert run_results('(atom)') == [['atom']]

def test_car():
  assert run_results('(car (a b c))') == ['a']

def test_cdr():
  assert run_results('(cdr (a b c))') == [['b', 'c']]

def test_cons():
  assert run_results('(cons a (b c))') == [['a', 'b', 'c']]

def test_atom_fn():
  assert run_results('(atom? atom)') == ['#t']
  assert run_results('(atom? (atom))') == ['#f']

def test_null_fn():
  assert run_results('(null? ())') == ['#t']
  assert run_results('(null? (atom))') == ['#f']

def test_eq_fn():
  assert run_results('(eq? atom atom)') == ['#t']
  assert run_results('(eq? atom banana)') == ['#f']

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
  ''') == [None, '#t']

  assert run_results(f'''
    {lat}
    (lat? (bacon (and eggs)))
  ''') == [None, '#f']

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
  ''') == [None, '#t']

  assert run_results(f'''
    {member}
    (member? meat (mashed potatoes and meat gravy))
  ''') == [None, '#t']

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
  ''') == [None, ['mashed', 'potatoes', 'and', 'gravy']]

  assert run_results(f'''
    {rember}
    (rember mint (lamb chops and mint flavored mint jelly))
  ''') == [None, ['lamb', 'chops', 'and', 'flavored', 'mint', 'jelly']]

  assert run_results(f'''
    {rember}
    (rember toast (bacon lettuce and tomato))
  ''') == [None, ['bacon', 'lettuce', 'and', 'tomato']]

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
  ''') == [None, ['apple', 'plum', 'grape', 'bean']]
  
  assert run_results(f'''
    {firsts}
    (firsts (
      (apple peach pumpkin)
      (plum pear cherry)
      (grape raisin pea)
      (bean carrot eggplant)))
  ''') == [None, ['apple', 'plum', 'grape', 'bean']]
