from environments.medcalc_bench.medcalc_bench.tools import ReplSession


def test_repl_print_is_returned() -> None:
    s = ReplSession()
    assert s.run('print("Hello World!")') == "Hello World!\n"


def test_repl_print_does_not_repeat_on_next_run() -> None:
    s = ReplSession()
    assert s.run('print("A")') == "A\n"
    assert s.run("x = 1") == ""


def test_repl_common_builtins_exist() -> None:
    s = ReplSession()
    assert s.run("print(min(3, 1, 2))") == "1\n"
    assert s.run("print(max([3, 1, 2]))") == "3\n"
    assert s.run("print(sum([1, 2, 3]))") == "6\n"
    assert s.run("print(list(enumerate([10, 20])))") == "[(0, 10), (1, 20)]\n"
    assert s.run("print(any([0, 0, 1]))") == "True\n"
    assert s.run("print(all([1, 2, 3]))") == "True\n"
    assert s.run("print(list(reversed([1, 2, 3])))") == "[3, 2, 1]\n"
    assert s.run("print(list(map(lambda x: x * 2, [1, 2, 3])))") == "[2, 4, 6]\n"
    assert s.run("print(list(filter(lambda x: x % 2 == 1, [1, 2, 3, 4])))") == "[1, 3]\n"


def test_repl_inplace_ops_work() -> None:
    s = ReplSession()
    assert s.run("x = 1\nx += 2\nprint(x)") == "3\n"
    assert s.run("y = 10\ny //= 3\nprint(y)") == "3\n"
