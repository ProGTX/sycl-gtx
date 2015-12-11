CC=clang++
TARGET=Release

INC=-I$(INTELOCLSDKROOT)include -I../../sycl-gtx
CFLAGS=-c -Wall -std=c++11 $(INC)

ifeq ($(TARGET),Release)
# TODO: Clang 3.7.0 fails with -O2 on Windows
CFLAGS+= -O1
else
CFLAGS+= -O0
endif

SOURCES= 						\
	../main.cpp					\
	../org_single_precision.cpp	\
	../original.cpp				\
	../sycl_gtx.cpp				\

OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=smallpt

ifeq ($(OS),Windows_NT)
RM=del
OBJECTS_RM=$(subst /,\, $(OBJECTS))
else
RM=rm
OBJECTS_RM=$(OBJECTS)
endif

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)

.cpp.o:
	@echo --- Building target: $@. First dependency is: $<
	$(CC) $(CFLAGS) $< -o $@

clean:
	$(RM) $(EXECUTABLE)
	$(RM) $(OBJECTS_RM)
