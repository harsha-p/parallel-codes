CC := icc
SRCDIR := src
BUILDDIR := release
LIBFLAGS := -qopenmp -qopenmp-simd -Ofast -qopt-report=5 -qopt-report-phase=vec,openmp,par
LIBAVX512 := -xCORE-AVX512 -qopt-zmm-usage=high
LIBAVX2 := -xCORE-AVX2
all: $(BUILDDIR)/single_node_256 $(BUILDDIR)/single_node_512
	

$(BUILDDIR)/single_node_256: $(SRCDIR)/single_node_256.cpp
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $< $(LIBFLAGS) $(LIBAVX2) -o $@


$(BUILDDIR)/single_node_512: $(SRCDIR)/single_node_512.cpp
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $< $(LIBFLAGS) $(LIBAVX512) -o $@

clean:
	@echo "Cleaning"
	@rm -fr $(BUILDDIR) 2>/dev/null || true

.PHONY: clean
.PRECIOUS: $(BUILDDIR)/%.o
