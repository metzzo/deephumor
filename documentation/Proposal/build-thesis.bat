rem Copyright (C) 2014-2017 by Thomas Auzinger <thomas@auzinger.name>

@echo off
rem Replace the 'x' in the next line with the name of the thesis' main LaTeX document without the '.tex' extension
set SOURCE=x
@echo on

rem Build the thesis document
pdflatex %SOURCE%
bibtex   %SOURCE%
pdflatex %SOURCE%
pdflatex %SOURCE%
makeindex -t %SOURCE%.glg -s %SOURCE%.ist -o %SOURCE%.gls %SOURCE%.glo
makeindex -t %SOURCE%.alg -s %SOURCE%.ist -o %SOURCE%.acr %SOURCE%.acn
makeindex -t %SOURCE%.ilg -o %SOURCE%.ind %SOURCE%.idx
pdflatex %SOURCE%
pdflatex %SOURCE%

@echo off
echo.
echo.
echo Thesis document compiled.
pause
