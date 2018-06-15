dlbench: dlbench.cpp kenrels.cpp
	$(HCC) $(CXXFLAGS) $(MEM) $(RW) -DHC  $^  $(LDFLAGS) $(EXTRA_FLAGS) -o $@

.PHONY: clean
clean:
	rm -f hc-stream
