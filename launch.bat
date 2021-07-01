@echo off
title UK_ACCIDENTS_EXERCISE

call conda env create -f requirements/uk_accidents.yaml
call conda activate uk_accidents

python -m ipykernel install --user --name=uk_accidents

jupyter notebook
