"""
tools/prepare_meditations.py

Turns the Project Gutenberg plain-text Meditations (eBook #2680, Meric Casaubon's
translation) into a folder of cosi-ready markdown notes:

    datasets/meditations/book_01.md ... book_12.md

Each passage keeps its traditional number as a label at the start of its paragraph
(e.g. "II.1"), so search results cite the passage the way the book is cited. The
introduction, appendix, notes and glossary are dropped; only Books I-XII are kept.

Some passages begin mid-line in the source (e.g. "...their motions, and desires. V. For
not observing..."). Splits are only accepted when the Roman numeral is the next
expected number in that book, so ordinary text such as "I." can't create a false split.

Usage:
    python tools/prepare_meditations.py path/to/pg2680.txt [--out datasets/meditations]
"""

import argparse
import os
import re
import sys

BOOK_NAMES = ["FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH", "SIXTH", "SEVENTH",
              "EIGHTH", "NINTH", "TENTH", "ELEVENTH", "TWELFTH"]
ROMAN = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"),
         (50, "L"), (40, "XL"), (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]


def to_roman(n):
    out = ""
    for value, sym in ROMAN:
        while n >= value:
            out += sym
            n -= value
    return out


def extract_books(text):
    text = text.replace("\r\n", "\n")
    starts = [m.start() for m in (re.search(rf"^THE {name} BOOK\s*$", text, re.M) for name in BOOK_NAMES) if m]
    if len(starts) != 12:
        sys.exit(f"Expected 12 book headings, found {len(starts)}. Is this Gutenberg eBook #2680?")
    end = re.search(r"^APPENDIX\s*$", text[starts[-1]:], re.M)
    if not end:
        sys.exit("Couldn't find the APPENDIX heading that follows Book XII.")
    bounds = starts + [starts[-1] + end.start()]
    books = []
    for i in range(12):
        body = text[bounds[i]:bounds[i + 1]].split("\n", 1)[1]  # drop the heading line
        books.append(re.sub(r"\s+", " ", body).strip())
    return books


def split_passages(body, max_skip=2):
    """
    Split one book's text into (number, passage) pairs. Only the next expected numeral
    (or one of the next `max_skip` after it, since the source occasionally skips a
    number) can start a passage; the earliest such marker wins.
    """
    passages = []
    expected = 1
    current = None
    pos = 0
    while True:
        best = None
        for n in range(expected, expected + 1 + max_skip):
            # The source occasionally drops the period ("XXIII Consider ..."); allow that only
            # for numerals of 4+ letters before a capital, where a false match is implausible.
            dot = r"(?:\. |\.? (?=[A-Z]))" if len(to_roman(n)) >= 4 else r"\. "
            m = re.compile(rf"(?:^|(?<=[\s\"'.?!:;)]))({to_roman(n)}){dot}").search(body, pos)
            if m and (best is None or m.start() < best[1].start()):
                best = (n, m)
        if best is None:
            break
        n, m = best
        if current is not None:
            passages.append((current, body[pos:m.start()].strip()))
        current, pos, expected = n, m.end(), n + 1
    if current is not None:
        passages.append((current, body[pos:].strip()))
    return passages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="Gutenberg plain-text file (pg2680.txt)")
    ap.add_argument("--out", default=os.path.join("datasets", "meditations"))
    args = ap.parse_args()

    with open(args.source, encoding="utf-8") as f:
        books = extract_books(f.read())

    os.makedirs(args.out, exist_ok=True)
    total = 0
    for b, body in enumerate(books, 1):
        passages = split_passages(body)
        total += len(passages)
        book = to_roman(b)
        lines = [f"# Meditations, Book {book}", ""]
        for n, p in passages:
            lines += [f"{book}.{n} {p}", ""]
        with open(os.path.join(args.out, f"book_{b:02d}.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Book {book:>4}: {len(passages):3d} passages")
    print(f"Wrote {total} passages to {args.out}/")


if __name__ == "__main__":
    main()
