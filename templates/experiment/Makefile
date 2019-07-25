CONTIKI={{ contiki }}
TARGET=cooja

APPS += powertrace
include $(CONTIKI)/Makefile.include

run:
	java -mx10000m -jar $(CONTIKI)/tools/cooja/dist/cooja.jar -hidden=simulation.csc -contiki=$(CONTIKI) & echo $$! > .$(TASK)

cooja:
	make simulation.csc & echo $$! > .$(TASK)
	rm .$(TASK)
