"""
╔══════════════════════════════════════════════════════╗
║         MATRIX OPERATIONS TOOL  — NumPy Edition      ║
╚══════════════════════════════════════════════════════╝
Requirements: numpy  →  pip install numpy
"""

import numpy as np
import os
import sys


# ──────────────────────────────────────────────
#  ANSI colours (fall back gracefully on Windows)
# ──────────────────────────────────────────────
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7
    )
except Exception:
    pass

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA= "\033[95m"
WHITE  = "\033[97m"
BG_DARK= "\033[40m"


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────

def clear():
    os.system("cls" if os.name == "nt" else "clear")


def banner():
    print(f"""
{CYAN}{BOLD}╔══════════════════════════════════════════════════════════╗
║   ███╗   ███╗ █████╗ ████████╗██████╗ ██╗██╗  ██╗        ║
║   ████╗ ████║██╔══██╗╚══██╔══╝██╔══██╗██║╚██╗██╔╝        ║
║   ██╔████╔██║███████║   ██║   ██████╔╝██║ ╚███╔╝         ║
║   ██║╚██╔╝██║██╔══██║   ██║   ██╔══██╗██║ ██╔██╗         ║
║   ██║ ╚═╝ ██║██║  ██║   ██║   ██║  ██║██║██╔╝ ██╗        ║
║   ╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝        ║
║                                                           ║
║         O P E R A T I O N S   T O O L   v1.0             ║
║                  Powered by  NumPy                        ║
╚══════════════════════════════════════════════════════════╝{RESET}
""")


def separator(char="─", width=62, colour=DIM):
    print(f"{colour}{char * width}{RESET}")


def success(msg): print(f"\n{GREEN}{BOLD}  ✔  {msg}{RESET}")
def error(msg):   print(f"\n{RED}{BOLD}  ✘  {msg}{RESET}")
def info(msg):    print(f"{YELLOW}  ▸  {msg}{RESET}")


def fmt_num(v):
    """Pretty-print a single number: int-like → no decimal, else 4 dp."""
    if isinstance(v, complex):
        r = f"{v.real:.4g}" if v.real != 0 else ""
        i = f"{v.imag:+.4g}j" if v.imag != 0 else ""
        return (r + i) or "0"
    if np.isreal(v):
        v = v.real
    return f"{v:.0f}" if float(v) == int(float(v)) else f"{v:.4g}"


def print_matrix(mat, label="Matrix", colour=CYAN):
    """Render a NumPy 2-D array as a nicely bordered table."""
    mat = np.atleast_2d(mat)
    rows, cols = mat.shape
    # Build cell strings
    cells = [[fmt_num(mat[r, c]) for c in range(cols)] for r in range(rows)]
    col_widths = [max(len(cells[r][c]) for r in range(rows)) for c in range(cols)]

    print(f"\n{colour}{BOLD}  {label}  ({rows}×{cols}){RESET}")
    separator("─", sum(col_widths) + cols * 3 + 2, colour)

    for r in range(rows):
        row_str = "  │ "
        for c in range(cols):
            cell = cells[r][c].rjust(col_widths[c])
            row_str += f"{WHITE}{cell}{RESET}  "
        row_str += f"{colour}│{RESET}"
        print(row_str)

    separator("─", sum(col_widths) + cols * 3 + 2, colour)


def input_matrix(label="A"):
    """Prompt user to enter a matrix row by row."""
    print(f"\n{YELLOW}{BOLD}  ─── Enter Matrix {label} ───{RESET}")
    while True:
        try:
            rows = int(input(f"{CYAN}  Rows    : {RESET}"))
            cols = int(input(f"{CYAN}  Columns : {RESET}"))
            if rows < 1 or cols < 1:
                raise ValueError
            break
        except ValueError:
            error("Please enter positive integers for dimensions.")

    print(f"{DIM}  Enter each row as space-separated numbers (e.g.  1 2 3){RESET}")
    data = []
    for r in range(rows):
        while True:
            try:
                raw = input(f"{MAGENTA}  Row {r+1:>2}  : {RESET}").split()
                if len(raw) != cols:
                    raise ValueError(f"Expected {cols} values, got {len(raw)}.")
                data.append([float(x) for x in raw])
                break
            except ValueError as e:
                error(str(e))

    mat = np.array(data)
    print_matrix(mat, f"Matrix {label}")
    return mat


# ──────────────────────────────────────────────
#  Operations
# ──────────────────────────────────────────────

def op_addition():
    info("Matrix Addition  →  A + B  (must be same shape)")
    A = input_matrix("A")
    B = input_matrix("B")
    if A.shape != B.shape:
        error(f"Shape mismatch: {A.shape} ≠ {B.shape}")
        return
    result = np.add(A, B)
    success("Addition complete!")
    print_matrix(result, "A + B", GREEN)


def op_subtraction():
    info("Matrix Subtraction  →  A − B  (must be same shape)")
    A = input_matrix("A")
    B = input_matrix("B")
    if A.shape != B.shape:
        error(f"Shape mismatch: {A.shape} ≠ {B.shape}")
        return
    result = np.subtract(A, B)
    success("Subtraction complete!")
    print_matrix(result, "A − B", GREEN)


def op_multiplication():
    print(f"\n{YELLOW}{BOLD}  ─── Multiplication type ───{RESET}")
    print(f"  {CYAN}1{RESET}  Element-wise (Hadamard)  — same shape required")
    print(f"  {CYAN}2{RESET}  Dot product (matrix mul) — cols(A) = rows(B)")
    print(f"  {CYAN}3{RESET}  Scalar multiplication")
    choice = input(f"\n{CYAN}  Choose [1/2/3]: {RESET}").strip()

    if choice == "1":
        A = input_matrix("A")
        B = input_matrix("B")
        if A.shape != B.shape:
            error(f"Shape mismatch: {A.shape} ≠ {B.shape}")
            return
        result = np.multiply(A, B)
        success("Element-wise multiplication complete!")
        print_matrix(result, "A ⊙ B", GREEN)

    elif choice == "2":
        A = input_matrix("A")
        B = input_matrix("B")
        if A.shape[1] != B.shape[0]:
            error(f"Incompatible shapes: cols(A)={A.shape[1]} ≠ rows(B)={B.shape[0]}")
            return
        result = np.dot(A, B)
        success("Dot-product multiplication complete!")
        print_matrix(result, "A · B", GREEN)

    elif choice == "3":
        A = input_matrix("A")
        while True:
            try:
                scalar = float(input(f"{CYAN}  Scalar value: {RESET}"))
                break
            except ValueError:
                error("Please enter a valid number.")
        result = A * scalar
        success("Scalar multiplication complete!")
        print_matrix(result, f"{fmt_num(scalar)} × A", GREEN)

    else:
        error("Invalid choice.")


def op_transpose():
    info("Transpose  →  Aᵀ")
    A = input_matrix("A")
    result = A.T
    success("Transpose complete!")
    print_matrix(result, "Aᵀ", GREEN)


def op_determinant():
    info("Determinant  →  det(A)  (square matrix only)")
    A = input_matrix("A")
    if A.shape[0] != A.shape[1]:
        error(f"Matrix must be square; got {A.shape}.")
        return
    det = np.linalg.det(A)
    success("Determinant computed!")
    print(f"\n{GREEN}{BOLD}  det(A)  =  {fmt_num(det)}{RESET}\n")


def op_inverse():
    info("Inverse  →  A⁻¹  (square, non-singular)")
    A = input_matrix("A")
    if A.shape[0] != A.shape[1]:
        error("Matrix must be square.")
        return
    if abs(np.linalg.det(A)) < 1e-12:
        error("Matrix is singular (det ≈ 0); inverse does not exist.")
        return
    result = np.linalg.inv(A)
    success("Inverse complete!")
    print_matrix(result, "A⁻¹", GREEN)


def op_eigenvalues():
    info("Eigenvalues & Eigenvectors  (square matrix only)")
    A = input_matrix("A")
    if A.shape[0] != A.shape[1]:
        error("Matrix must be square.")
        return
    vals, vecs = np.linalg.eig(A)
    success("Eigen-decomposition complete!")
    print(f"\n{CYAN}{BOLD}  Eigenvalues:{RESET}")
    for i, v in enumerate(vals):
        print(f"    λ{i+1}  =  {fmt_num(v)}")
    print(f"\n{CYAN}{BOLD}  Eigenvectors (columns):{RESET}")
    print_matrix(vecs, "V", MAGENTA)


def op_rank_trace():
    info("Rank & Trace")
    A = input_matrix("A")
    rank = np.linalg.matrix_rank(A)
    success("Done!")
    print(f"\n{GREEN}{BOLD}  Rank   =  {rank}{RESET}")
    if A.shape[0] == A.shape[1]:
        trace = np.trace(A)
        print(f"{GREEN}{BOLD}  Trace  =  {fmt_num(trace)}{RESET}\n")
    else:
        info("Trace is only defined for square matrices.")


# ──────────────────────────────────────────────
#  Main menu
# ──────────────────────────────────────────────

MENU = [
    ("Addition",                  op_addition),
    ("Subtraction",               op_subtraction),
    ("Multiplication",            op_multiplication),
    ("Transpose",                 op_transpose),
    ("Determinant",               op_determinant),
    ("Inverse",                   op_inverse),
    ("Eigenvalues & Eigenvectors",op_eigenvalues),
    ("Rank & Trace",              op_rank_trace),
]


def main_menu():
    while True:
        clear()
        banner()
        print(f"{BOLD}  SELECT AN OPERATION{RESET}\n")
        for idx, (name, _) in enumerate(MENU, 1):
            print(f"  {CYAN}{idx:>2}{RESET}  {name}")
        print(f"\n  {RED} 0{RESET}  Exit\n")
        separator()

        choice = input(f"\n{YELLOW}  ▶  Your choice: {RESET}").strip()

        if choice == "0":
            clear()
            print(f"\n{CYAN}{BOLD}  Thank you for using Matrix Operations Tool!{RESET}\n")
            sys.exit(0)

        if choice.isdigit() and 1 <= int(choice) <= len(MENU):
            idx = int(choice) - 1
            clear()
            banner()
            print(f"{BOLD}{CYAN}  ── {MENU[idx][0].upper()} ──{RESET}\n")
            try:
                MENU[idx][1]()
            except KeyboardInterrupt:
                info("Operation cancelled.")
            except Exception as exc:
                error(f"Unexpected error: {exc}")
        else:
            error(f"Enter a number between 0 and {len(MENU)}.")

        input(f"\n{DIM}  Press ENTER to return to the menu …{RESET}")


if __name__ == "__main__":
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("\n  NumPy not found. Install it with:\n\n    pip install numpy\n")
        sys.exit(1)
    main_menu()
