import os
import sys
import re
import math
import numpy as np
import numexpr as ne
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from sympy.calculus.util import AccumBounds
from collections import OrderedDict
import errno
import signal
from functools import wraps, partial


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

SPECIAL_WORDS = ['<s>', '</s>', '<pad>', '(', ')']
SPECIAL_WORDS = SPECIAL_WORDS + [f'<SPECIAL_{i}>' for i in range(len(SPECIAL_WORDS), 10)]
EXP_OPERATORS = {'exp', 'sinh', 'cosh'}
EVAL_VALUES = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 2.1, 3.1]
EVAL_VALUES = EVAL_VALUES + [-x for x in EVAL_VALUES]
EVAL_SYMBOLS = {'x', 'y', 'z', 'a0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'a7', 'a8', 'a9'}

class ValueErrorExpression(Exception):
    pass


class UnknownSymPyOperator(Exception):
    pass


class InvalidPrefixExpression(Exception):

    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)
class TimeoutError(BaseException):
    pass

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    """
    Sets a timeout to terminate the current process. This is for making SymPy to avoid an infinite loop
    """
    def decorator(func):

        def _handle_timeout(repeat_id, signum, frame):
            # logger.warning(f"Catched the signal ({repeat_id}) Setting signal handler {repeat_id + 1}")
            signal.signal(signal.SIGALRM, partial(_handle_timeout, repeat_id + 1))
            signal.alarm(seconds)
            raise TimeoutError(error_message)

        def wrapper(*args, **kwargs):
            old_signal = signal.signal(signal.SIGALRM, partial(_handle_timeout, 0))
            old_time_left = signal.alarm(seconds)
            assert type(old_time_left) is int and old_time_left >= 0
            if 0 < old_time_left < seconds:  # do not exceed previous timer
                signal.alarm(old_time_left)
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
            finally:
                if old_time_left == 0:
                    signal.alarm(0)
                else:
                    sub = time.time() - start_time
                    signal.signal(signal.SIGALRM, old_signal)
                    signal.alarm(max(0, math.ceil(old_time_left - sub)))
            return result

        return wraps(func)(wrapper)

    return decorator

def to_cuda(config, *args):
    """
    Move tensors to CUDA.
    """
    if config.device.type == 'cpu':
        return args
    return [None if x is None else x.to(config.device) for x in args]

def count_nested_exp(s):
    """
    Return the maximum number of nested exponential functions in an infix expression.
    """
    stack = []
    count = 0
    max_count = 0
    for v in re.findall('[+-/*//()]|[a-zA-Z0-9]+', s):
        if v == '(':
            stack.append(v)
        elif v == ')':
            while True:
                x = stack.pop()
                if x in EXP_OPERATORS:
                    count -= 1
                if x == '(':
                    break
        else:
            stack.append(v)
            if v in EXP_OPERATORS:
                count += 1
                max_count = max(max_count, count)
    assert len(stack) == 0
    return max_count

## Got it
def is_valid_expr(s):
    """
    Check that we are able to evaluate an expression (and that it will not blow in SymPy evaluation).
    """
    s = s.replace('Derivative(f(x),x)', '1')
    s = s.replace('Derivative(1,x)', '1')
    s = s.replace('(E)', '(exp(1))')
    s = s.replace('(I)', '(1)')
    s = s.replace('(pi)', '(1)')
    s = re.sub(r'(?<![a-z])(f|g|h|Abs|sign|ln|sin|cos|tan|sec|csc|cot|asin|acos|atan|asec|acsc|acot|tanh|sech|csch|coth|asinh|acosh|atanh|asech|acoth|acsch)\(', '(', s)
    count = count_nested_exp(s)
    if count >= 4:
        return False
    for v in EVAL_VALUES:
        try:
            local_dict = {s: (v + 1e-4 * i) for i, s in enumerate(EVAL_SYMBOLS)}
            value = ne.evaluate(s, local_dict=local_dict).item()
            if not (math.isnan(value) or math.isinf(value)):
                return True
        except (FloatingPointError, ZeroDivisionError, TypeError, MemoryError):
            continue
    return False

class EnvHndler():
    """
    This class contains the necessary tools and attribute for the current experiment like the vocabulary and symbol tokenizer.
    """
    SYMPY_OPERATORS = {
        # Elementary functions
        sp.Add: 'add',
        sp.Mul: 'mul',
        sp.Pow: 'pow',
        sp.exp: 'exp',
        sp.log: 'ln',
        sp.Abs: 'abs',
        sp.sign: 'sign',
        # Trigonometric Functions
        sp.sin: 'sin',
        sp.cos: 'cos',
        sp.tan: 'tan',
        sp.cot: 'cot',
        sp.sec: 'sec',
        sp.csc: 'csc',
        # Trigonometric Inverses
        sp.asin: 'asin',
        sp.acos: 'acos',
        sp.atan: 'atan',
        sp.acot: 'acot',
        sp.asec: 'asec',
        sp.acsc: 'acsc',
        # Hyperbolic Functions
        sp.sinh: 'sinh',
        sp.cosh: 'cosh',
        sp.tanh: 'tanh',
        sp.coth: 'coth',
        sp.sech: 'sech',
        sp.csch: 'csch',
        # Hyperbolic Inverses
        sp.asinh: 'asinh',
        sp.acosh: 'acosh',
        sp.atanh: 'atanh',
        sp.acoth: 'acoth',
        sp.asech: 'asech',
        sp.acsch: 'acsch',
        # Derivative
        sp.Derivative: 'derivative',
    }
    OPERATORS = {
        # Elementary functions
        'add': 2,
        'sub': 2,
        'mul': 2,
        'div': 2,
        'pow': 2,
        'rac': 2,
        'inv': 1,
        'pow2': 1,
        'pow3': 1,
        'pow4': 1,
        'pow5': 1,
        'sqrt': 1,
        'exp': 1,
        'ln': 1,
        'abs': 1,
        'sign': 1,
        # Trigonometric Functions
        'sin': 1,
        'cos': 1,
        'tan': 1,
        'cot': 1,
        'sec': 1,
        'csc': 1,
        # Trigonometric Inverses
        'asin': 1,
        'acos': 1,
        'atan': 1,
        'acot': 1,
        'asec': 1,
        'acsc': 1,
        # Hyperbolic Functions
        'sinh': 1,
        'cosh': 1,
        'tanh': 1,
        'coth': 1,
        'sech': 1,
        'csch': 1,
        # Hyperbolic Inverses
        'asinh': 1,
        'acosh': 1,
        'atanh': 1,
        'acoth': 1,
        'asech': 1,
        'acsch': 1,
        # Derivative
        'derivative': 2,
        # custom functions
        'f': 1,
        'g': 2,
        'h': 3,
    }
    def __init__(self, config):
        self.operators = sorted(list(self.OPERATORS.keys()))
        self.clean_prefix_expr = config.clean_prefix_expr
        # symbols / elements
        self.constants = ['pi', 'E']
        self.variables = OrderedDict({
            'x': sp.Symbol('x', real=True, nonzero=True),  # , positive=True
            'y': sp.Symbol('y', real=True, nonzero=True),  # , positive=True
            'z': sp.Symbol('z', real=True, nonzero=True),  # , positive=True
            't': sp.Symbol('t', real=True, nonzero=True),  # , positive=True
        })
        self.coefficients = OrderedDict({
            f'a{i}': sp.Symbol(f'a{i}', real=True)
            for i in range(10)
        })
        self.functions = OrderedDict({
            'f': sp.Function('f', real=True, nonzero=True),
            'g': sp.Function('g', real=True, nonzero=True),
            'h': sp.Function('h', real=True, nonzero=True),
        })
        self.symbols = ['I', 'INT+', 'INT-', 'INT', 'FLOAT', '-', '.', '10^', 'Y', "Y'", "Y''"]
        self.elements = [str(i) for i in range(10)]
        self.local_dict = {}
        for k, v in list(self.variables.items()) + list(self.coefficients.items()) + list(self.functions.items()):
            assert k not in self.local_dict
            self.local_dict[k] = v
        # vocabulary
        self.words = SPECIAL_WORDS + self.constants + list(self.variables.keys()) + list(self.coefficients.keys()) + self.operators + self.symbols + self.elements
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))
        self.n_words = config.n_words = len(self.words)
        self.eos_index = config.eos_index
        self.pad_index = config.pad_index

        # rewrite expressions
        self.rewrite_functions = [x for x in config.rewrite_functions.split(',') if x != '']
        assert len(self.rewrite_functions) == len(set(self.rewrite_functions))
        assert all(x in ['expand', 'factor', 'expand_log', 'logcombine', 'powsimp', 'simplify'] for x in self.rewrite_functions)

    def write_infix(self, token, args):
        """
        Infix representation.
        Convert prefix expressions to a format that SymPy can parse.
        """
        if token == 'add':
            return f'({args[0]})+({args[1]})'
        elif token == 'sub':
            return f'({args[0]})-({args[1]})'
        elif token == 'mul':
            return f'({args[0]})*({args[1]})'
        elif token == 'div':
            return f'({args[0]})/({args[1]})'
        elif token == 'pow':
            return f'({args[0]})**({args[1]})'
        elif token == 'rac':
            return f'({args[0]})**(1/({args[1]}))'
        elif token == 'abs':
            return f'Abs({args[0]})'
        elif token == 'inv':
            return f'1/({args[0]})'
        elif token == 'pow2':
            return f'({args[0]})**2'
        elif token == 'pow3':
            return f'({args[0]})**3'
        elif token == 'pow4':
            return f'({args[0]})**4'
        elif token == 'pow5':
            return f'({args[0]})**5'
        elif token in ['sign', 'sqrt', 'exp', 'ln', 'sin', 'cos', 'tan', 'cot', 'sec', 'csc', 'asin', 'acos', 'atan', 'acot', 'asec', 'acsc', 'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh', 'acoth', 'asech', 'acsch']:
            return f'{token}({args[0]})'
        elif token == 'derivative':
            return f'Derivative({args[0]},{args[1]})'
        elif token == 'f':
            return f'f({args[0]})'
        elif token == 'g':
            return f'g({args[0]},{args[1]})'
        elif token == 'h':
            return f'h({args[0]},{args[1]},{args[2]})'
        elif token.startswith('INT'):
            return f'{token[-1]}{args[0]}'
        else:
            return token
        raise InvalidPrefixExpression(f"Unknown token in prefix expression: {token}, with arguments {args}")

    def parse_int(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = 10
        balanced = False
        val = 0
        if not (balanced and lst[0] == 'INT' or base >= 2 and lst[0] in ['INT+', 'INT-'] or base <= -2 and lst[0] == 'INT'):
            raise InvalidPrefixExpression(f"Invalid integer in prefix expression")
        i = 0
        for x in lst[1:]:
            if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
                break
            val = val * base + int(x)
            i += 1
        if base > 0 and lst[0] == 'INT-':
            val = -val
        return val, i + 1

    def _prefix_to_infix(self, expr):
        """
        Parse an expression in prefix mode, and output it in either:
          - infix mode (returns human readable string)
          - develop mode (returns a dictionary with the simplified expression)
        """
        if len(expr) == 0:
            raise InvalidPrefixExpression("Empty prefix list.")
        t = expr[0]
        if t in self.operators:
            args = []
            l1 = expr[1:]
            for _ in range(self.OPERATORS[t]):
                i1, l1 = self._prefix_to_infix(l1)
                args.append(i1)
            return self.write_infix(t, args), l1
        elif t in self.variables or t in self.coefficients or t in self.constants or t == 'I':
            return t, expr[1:]
        else:
            val, i = self.parse_int(expr)
            return str(val), expr[i:]
    ## Got it
    def prefix_to_infix(self, expr):
        """
        Prefix to infix conversion.
        """
        p, r = self._prefix_to_infix(expr)
        if len(r) > 0:
            raise InvalidPrefixExpression(f"Incorrect prefix expression \"{expr}\". \"{r}\" was not parsed.")
        return f'({p})'

    def write_int(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        In balanced bases (positive), digits range from -(base-1)//2 to (base-1)//2
        """
        base = 10
        balanced = False
        res = []
        max_digit = abs(base)
        if balanced:
            max_digit = (base - 1) // 2
        else:
            if base > 0:
                neg = val < 0
                val = -val if neg else val
        while True:
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
            if val == 0:
                break
        if base < 0 or balanced:
            res.append('INT')
        else:
            res.append('INT-' if neg else 'INT+')
        return res[::-1]

    def _sympy_to_prefix(self, op, expr):
        """
        Parse a SymPy expression given an initial root operator.
        """
        n_args = len(expr.args)

        # derivative operator
        if op == 'derivative':
            assert n_args >= 2
            assert all(len(arg) == 2 and str(arg[0]) in self.variables and int(arg[1]) >= 1 for arg in expr.args[1:]), expr.args
            parse_list = self.sympy_to_prefix(expr.args[0])
            for var, degree in expr.args[1:]:
                parse_list = ['derivative' for _ in range(int(degree))] + parse_list + [str(var) for _ in range(int(degree))]
            return parse_list

        assert (op == 'add' or op == 'mul') and (n_args >= 2) or (op != 'add' and op != 'mul') and (1 <= n_args <= 2)

        # square root
        if op == 'pow' and isinstance(expr.args[1], sp.Rational) and expr.args[1].p == 1 and expr.args[1].q == 2:
            return ['sqrt'] + self.sympy_to_prefix(expr.args[0])

        # parse children
        parse_list = []
        for i in range(n_args):
            if i == 0 or i < n_args - 1:
                parse_list.append(op)
            parse_list += self.sympy_to_prefix(expr.args[i])

        return parse_list
    ## Got it
    ## This will converst a SymPy expression to a prefix expression
    def sympy_to_prefix(self, expr):
        """
        Convert a SymPy expression to a prefix one.
        """
        if isinstance(expr, sp.Symbol):
            return [str(expr)]
        elif isinstance(expr, sp.Integer):
            return self.write_int(int(str(expr)))
        elif isinstance(expr, sp.Rational):
            return ['div'] + self.write_int(int(expr.p)) + self.write_int(int(expr.q))
        elif expr == sp.E:
            return ['E']
        elif expr == sp.pi:
            return ['pi']
        elif expr == sp.I:
            return ['I']
        # SymPy operator
        for op_type, op_name in self.SYMPY_OPERATORS.items():
            if isinstance(expr, op_type):
                return self._sympy_to_prefix(op_name, expr)
        # environment function
        for func_name, func in self.functions.items():
            if isinstance(expr, func):
                return self._sympy_to_prefix(func_name, expr)
        # unknown operator
        raise UnknownSymPyOperator(f"Unknown SymPy operator: {expr}")


    def infix_to_sympy(self, infix):
        """
        Convert an infix expression to SymPy.
        """
        if not is_valid_expr(infix):
            raise ValueErrorExpression
        expr = parse_expr(infix, evaluate=True, local_dict=self.local_dict)
        if expr.has(sp.I) or expr.has(AccumBounds):
            raise ValueErrorExpression

        return expr

    def clean_prefix(self, expr):
        """
        Clean prefix expressions before they are converted to PyTorch.
        "f x" -> "Y"
        "derivative f x x" -> "Y'"
        """
        if not self.clean_prefix_expr:
            return expr
        expr = " ".join(expr)
        expr = expr.replace("f x", "Y")
        expr = expr.replace("derivative Y x", "Y'")
        expr = expr.replace("derivative Y' x", "Y''")
        expr = expr.split()
        return expr


if '__name__' == '__main__' :
    env = EnvHndler(config)
