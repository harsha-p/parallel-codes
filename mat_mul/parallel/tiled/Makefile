#==============================================================
#
# SAMPLE SOURCE CODE - SUBJECT TO THE TERMS OF SAMPLE CODE LICENSE AGREEMENT,
# http://software.intel.com/en-us/articles/intel-sample-source-code-license-agreement/
#
# Copyright 2005-2018 Intel Corporation
#
# THIS FILE IS PROVIDED "AS IS" WITH NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO ANY IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE, NON-INFRINGEMENT OF INTELLECTUAL PROPERTY RIGHTS.
#
# =============================================================

CC := icc
SRCDIR := src
BUILDDIR := release
LIBFLAGS := -qopenmp -qopenmp-simd -O3 -qopt-report=5 -qopt-report-phase=vec -xCORE-AVX2 -D LEVEL1
all: $(BUILDDIR)/single_node
run: run_single_node
$(BUILDDIR)/%: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $< $(LIBFLAGS) -o $@

run_single_node: $(BUILDDIR)/single_node
	./$(BUILDDIR)/single_node

clean:
	@echo " Cleaning..."
	@rm -fr $(BUILDDIR) 2>/dev/null || true

.PHONY: clean
.PRECIOUS: $(BUILDDIR)/%.o
