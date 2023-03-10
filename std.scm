(define lat?
  (lambda (l)
    (cond
      ((null? l) #t)
      ((atom? (car l)) (lat? (cdr l)))
    (else #f))))

(define member?
  (lambda (a lat)
    (cond
      ((null? lat) #f)
      (else (or (eq? (car lat) a) (member? a (cdr lat)))))))

(define rember
    (lambda (a lat)
      (cond
        ((null? lat) ())
        ((eq? (car lat) a) (cdr lat))
        (else (cons (car lat) (rember a (cdr lat)))))))

(define firsts
    (lambda (l)
      (cond
        ((null? l) ())
        (else (cons (car (car l)) (firsts (cdr l)))))))
