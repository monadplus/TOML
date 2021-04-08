(TeX-add-style-hook
 "report"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("article" "12pt" "a4paper")))
   (TeX-run-style-hooks
    "latex2e"
    "preamble"
    "article"
    "art12"
    "fancyhdr")
   (LaTeX-add-labels
    "sec:introduction"))
 :latex)

