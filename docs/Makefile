DOC = architecture_review

# Prefer a full TeX Live toolchain if present; otherwise fall back to tectonic.
LATEXMK := $(shell command -v latexmk 2>/dev/null)
TECTONIC := $(shell command -v tectonic 2>/dev/null || echo $$HOME/bin/tectonic)

.PHONY: pdf clean distclean

pdf: $(DOC).pdf

$(DOC).pdf: $(DOC).tex refs/$(DOC).bib
ifneq ($(LATEXMK),)
	latexmk -pdf -bibtex -interaction=nonstopmode $(DOC).tex
else
	$(TECTONIC) --keep-intermediates --reruns 3 $(DOC).tex
endif

# Remove intermediate build artifacts, keep the PDF.
clean:
	rm -f $(DOC).aux $(DOC).bbl $(DOC).bcf $(DOC).blg $(DOC).log \
	      $(DOC).out $(DOC).run.xml $(DOC).toc $(DOC).lof $(DOC).lot \
	      $(DOC).fls $(DOC).fdb_latexmk $(DOC).synctex.gz \
	      $(DOC).xdv $(DOC)-blx.bib

# Remove everything including the PDF.
distclean: clean
	rm -f $(DOC).pdf
